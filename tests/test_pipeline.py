import numpy as np
import pandas as pd
import pytest

from src.artifacts import load_artifacts
from src.config import SimConfig, som
from src.data import Basket, build_dataset, clean_transactions, load_transactions
from src.experiment import REFERENCE, make_policies, run_simulation
from src.models import (Ctx, ExpectedValuePolicy, FixedListPolicy, PairRules, SequentialModel, accept_prob, finalize)
from src.simulator import (_accept_stage, _subsample, build_sessions, calibrate_stop_slope,
                           run_policy)
from fake_data import generate
from src.training import train

ALL_MODELS = ["Random", "Fixed list", "Association rules", "Sequential",
              "Contextual bandit (cold start)", "Contextual bandit (warm start)"]


@pytest.fixture(scope="module")
def raw():
    return generate(n_items=120, n_cats=10, n_customers=400, n_invoices=3000, seed=1)


@pytest.fixture(scope="module")
def trained(raw, tmp_path_factory):
    out = tmp_path_factory.mktemp("art")
    train(clean_transactions(raw), out, top_k=100, fx=16100.0)
    return load_artifacts(out)


def _ctx(b, cart, customer=-1, ts=0):
    return Ctx(customer, ts, b.ctxb.build(cart, ts))


def test_cleaning_drops_cancellations(raw):
    c = clean_transactions(raw)
    assert not c.invoice.str.startswith("C").any()
    assert (c.qty > 0).all() and (c.price > 0).all()
    assert raw.Invoice.str.startswith("C").any()


def test_merge_removes_overlap(raw, tmp_path):
    raw.iloc[:6000].to_csv(tmp_path / "a.csv", index=False)
    raw.iloc[4000:].to_csv(tmp_path / "b.csv", index=False)      # 2000 overlapping rows
    df, files = load_transactions(tmp_path)
    assert len(files) == 2
    assert not df.duplicated(["invoice", "item", "qty", "price", "date", "customer"]).any()
    assert len(df) == len(clean_transactions(raw))


def test_so_m_conversion_and_split(raw):
    df = clean_transactions(raw)
    gbp = build_dataset(df, 100, fx=1.0, round_to=0.01)
    uzs = build_dataset(df, 100, fx=16100.0, round_to=100)
    assert (uzs.prices % 100 == 0).all() and (uzs.prices >= 100).all()
    assert np.allclose(uzs.prices, gbp.prices * 16100, rtol=0.05, atol=100)
    assert uzs.test[0].ts >= uzs.train[-1].ts                      # chronological, no leakage
    assert som(1234567) == "1 234 567 so'm"
    assert any(b.customer >= 0 for b in uzs.test)                  # customers are kept


def test_pair_rules():
    train_b = [Basket(i, np.array([0, 1])) for i in range(20)] + \
              [Basket(i, np.array([2, 3])) for i in range(20, 40)]
    sc = PairRules.fit(train_b, 4, min_count=5, min_conf=0.1).scores([0])
    assert sc[1] > 0.9 and sc[0] == 0 and sc[2] == 0


def test_sequential_uses_customer_history():
    # customer 7 always buys 0 -> 1 -> 2 over time
    train_b = [Basket(t, np.array([0, 1, 2]), 7) for t in range(10)]
    m = SequentialModel.fit(train_b, 4)
    assert m.scores(np.array([0]), 7)[1] > 0
    assert m.scores(np.array([0]), 7)[0] == 0                       # cart item never suggested
    assert m.scores(np.array([3]), -1).sum() == 0                   # unknown item, guest: no signal


def test_finalize_filters_and_keeps_model_order():
    prices = np.array([500., 100., 300., 900., 200., 700.])
    out = finalize([3, 5, 0, 2, 1, 4], cart=[0], prices=prices, limit=800, n=3,
                   fallback=[0, 1, 2, 3, 4, 5])
    # ranking 3,5,0,2,1,4 -> 3 is above the limit, 0 is in the cart -> first 3 kept: 5, 2, 1
    assert out == [5, 2, 1]                              # the model's own order is kept
    assert 3 not in out and 0 not in out


def test_all_models_respect_the_rules(trained):
    b = trained
    pols = make_policies(b, SimConfig())
    assert [p.name for p in pols] == ALL_MODELS
    cart = b.test[0].items[:3]
    for pol in pols:
        recs = pol.recommend(cart, 10, 2.0 * b.cat.ref, _ctx(b, cart, b.test[0].customer, b.test[0].ts))
        assert 0 < len(recs) <= 10 and len(set(recs)) == len(recs), pol.name
        assert not set(recs) & set(cart.tolist()), pol.name               # cart items removed
        assert all(b.cat.prices[i] <= 2.0 * b.cat.ref for i in recs), pol.name


def test_fixed_list_is_top_popular(trained):
    assert FixedListPolicy(trained.cat).items == [int(i) for i in trained.cat.pop_order[:10]]


def test_expected_value_example():
    assert 0.7 * 5 < 0.3 * 30                                              # the pricier item is worth more
    assert accept_prob(1.0, 0.0, 1.0, 0.5, 0.0, 0.5) == 0.5
    assert accept_prob(1.0, 4.0, 1.0, 0.5, 0.0, 0.5) < 0.5


class _Silent:
    warm_start = False

    def reset(self, seed=0):
        pass

    def update(self, ctx, shown, accepted):
        pass


def test_nothing_suggested_means_no_change(trained):
    class Nothing(_Silent):
        name = "None"

        def recommend(self, cart, n, limit=0, ctx=None):
            return []

    cfg = SimConfig()
    sess = build_sessions(_subsample(trained.test, 400), trained.cat.prices, cfg, 0, trained.ctxb)
    m, _ = run_policy(Nothing(), sess, trained.cat, cfg, 0.5)
    assert m["RPS_uplift_pct"] == pytest.approx(0) and m["AOV_uplift_pct"] == pytest.approx(0)
    assert m["finish_rate"] == m["finish_rate_base"]


def test_oracle_beats_fixed(trained):
    cfg = SimConfig(position_bias=False)
    sess = build_sessions(_subsample(trained.test, 500), trained.cat.prices, cfg, 0, trained.ctxb)
    hidden = {id(s.cart): s.hidden for s in sess}

    class Oracle(_Silent):
        name = "oracle"

        def recommend(self, cart, n, limit=0, ctx=None):
            return list(hidden[id(cart)])[:n]

    mo, _ = run_policy(Oracle(), sess, trained.cat, cfg, 0.0)
    mf, _ = run_policy(FixedListPolicy(trained.cat), sess, trained.cat, cfg, 0.0)
    assert mo["conversion"] > mf["conversion"] and mo["relevance_rate"] == pytest.approx(1.0)


def test_bandit_learns_from_feedback(trained):
    cold = next(p for p in make_policies(trained, SimConfig()) if "cold" in p.name)
    cart = trained.test[0].items[:3]
    ctx = _ctx(trained, cart)
    shown = cold.recommend(cart, 3, float("inf"), ctx)
    for _ in range(30):
        cold.update(ctx, shown, [True] + [False] * (len(shown) - 1))      # only the first is accepted
    assert cold.b[cold.arm_pos[int(shown[0])]].any()                      # the accepted arm learned something
    assert cold.recommend(cart, 10, float("inf"), ctx)
    cold.reset()
    assert not cold.b.any()                                                # reset forgets everything


def test_stop_penalty_hurts_revenue(trained):
    out = {}
    for stop in (0.20, 0.60):
        s, _ = run_simulation(trained, SimConfig(stop_after_accept=stop), seeds=2, max_orders=800,
                              max_warm=300)
        out[stop] = s.set_index("policy").loc[REFERENCE, "RPS_uplift_pct"]
    assert out[0.20] > out[0.60]                                           # more stopping -> less revenue


def test_comparison_table_has_every_model(trained):
    s, bands = run_simulation(trained, SimConfig(), seeds=1, max_orders=300, max_warm=300)
    assert list(s["policy"]) == ALL_MODELS
    assert {"AOV_uplift_pct", "extra_per_order", "RPS_uplift_pct", "conversion"} <= set(s.columns)
    assert len(bands) == 4 * len(ALL_MODELS)


def test_calibration_hits_target(trained):
    cfg = SimConfig(stop_after_accept=0.30)
    ref = next(p for p in make_policies(trained, cfg) if p.name == REFERENCE)
    sess = build_sessions(_subsample(trained.test, 800), trained.cat.prices, cfg, 0, trained.ctxb)
    k = calibrate_stop_slope(sess, ref, trained.cat, cfg)
    ratios = []
    for ss in sess:
        acc, add, *_ = _accept_stage(ref.recommend(ss.cart, 10, np.inf, ss.ctx), ss, trained.cat, cfg)
        if acc:
            ratios.append(add / ss.base_value)
    assert 0.2 + k * np.mean(ratios) == pytest.approx(0.30, abs=1e-6)


def test_deterministic(trained):
    a, _ = run_simulation(trained, SimConfig(), seeds=1, max_orders=300, max_warm=300)
    b, _ = run_simulation(trained, SimConfig(), seeds=1, max_orders=300, max_warm=300)
    pd.testing.assert_frame_equal(a, b)


def test_artifacts_roundtrip(raw, trained, tmp_path):
    again = train(clean_transactions(raw), tmp_path, top_k=100, fx=16100.0)
    b2 = load_artifacts(tmp_path)
    cart = trained.test[0].items[:3]
    assert np.allclose(trained.rules.scores(cart), b2.rules.scores(cart))
    cu = trained.test[0].customer
    assert np.allclose(trained.seq.scores(cart, cu), b2.seq.scores(cart, cu))
    assert np.allclose(trained.ctxb.emb, b2.ctxb.emb)
    assert np.array_equal(trained.cat.prices, b2.cat.prices)
    assert len(trained.test) == len(b2.test) and len(trained.warm) == len(b2.warm)
    assert again["fx_to_som"] == 16100.0


def test_kaggle_excel_two_sheets_cached(raw, tmp_path, monkeypatch):
    """online_retail_II.xlsx: two sheets, Kaggle headers; read once, then served from cache."""
    import src.data as D
    monkeypatch.setattr(D, "CACHE_DIR", tmp_path / "cache")
    x = tmp_path / "online_retail_II.xlsx"
    with pd.ExcelWriter(x) as w:
        raw.iloc[:3000].to_excel(w, sheet_name="Year 2009-2010", index=False)
        raw.iloc[2000:5000].to_excel(w, sheet_name="Year 2010-2011", index=False)   # overlap
    df, files = load_transactions(x)
    expected = clean_transactions(raw.iloc[:5000])
    assert len(files) == 1 and len(df) == len(expected)          # both sheets, overlap removed
    assert list((tmp_path / "cache").glob("*.parquet"))           # cache written
    df2, _ = load_transactions(x)                                 # second run: from the cache
    pd.testing.assert_frame_equal(df, df2)


def test_all_products_used_by_default(raw):
    df = clean_transactions(raw)
    small, full = build_dataset(df, 50), build_dataset(df)          # top_k=None -> every product
    assert full.V > small.V and full.info["vocab_size"] == full.V
    assert full.info["row_coverage"] > small.info["row_coverage"]


def test_expected_value_does_not_just_pick_the_priciest(trained):
    """With price sensitivity 0 the policy must still look at relevance, not only at price."""
    b = trained
    cfg = SimConfig(price_sensitivity=0.0)
    ev = ExpectedValuePolicy(b.cat, b.rules, s=0.0)
    cart = b.test[0].items[:3]
    chance, value = ev.expected(cart)
    assert chance.min() < cfg.p_miss                       # no flat "luck" floor any more
    recs = ev.recommend(cart, 10, float("inf"), None)
    priciest = set(np.argsort(-b.cat.prices)[:10].tolist())
    assert set(recs) != priciest
    rel_picked = np.mean([value[i] for i in recs])
    assert rel_picked >= np.median(value)                  # picks are above-median value

# ----------------------------------------------------------------- AOV-first additions
from src.experiment import calibrate_accept                      # noqa: E402
from src.explain import sequential_contributions, why_shown      # noqa: E402
from src.simulator import simulate_cart                          # noqa: E402
from src.models import relevance                                 # noqa: E402


def test_calibration_is_per_shopper_and_reachable(trained):
    fixed = tuple(int(i) for i in trained.cat.pop_order[:10])
    p_hit, p_miss, reached = calibrate_accept(trained, fixed, target=0.15)
    assert 0 < p_hit <= 1 and reached == pytest.approx(0.15, abs=0.03)


def test_summary_has_aov_drivers(trained):
    s, _ = run_simulation(trained, SimConfig(), seeds=1, max_orders=300, max_warm=300)
    assert {"orders_with_accept", "items_added_per_order", "finish_rate_accepters", "relevance_rate",
            "AOV_policy", "extra_per_order"} <= set(s.columns)


def test_simulate_cart_deterministic_and_sane(trained):
    b, cfg = trained, SimConfig()
    cart = b.test[0].items[:3]
    rel = relevance(b.rules, b.cat, cart)
    pol = next(p for p in make_policies(b, cfg) if p.name == "Association rules")
    shown = pol.recommend(cart, 10, np.inf, None)
    a = simulate_cart(b.cat, rel, cart, shown, cfg, 0.5)
    assert a == simulate_cart(b.cat, rel, cart, shown, cfg, 0.5)
    assert a["aov_before"] == pytest.approx(b.cat.prices[cart].sum())
    assert a["aov_after"] >= a["aov_before"] and 0 <= a["accept_share"] <= 1
    empty = simulate_cart(b.cat, rel, cart, [], cfg, 0.5)
    assert empty["aov_change"] == 0
    # higher leave slope -> fewer finished orders
    assert simulate_cart(b.cat, rel, cart, shown, cfg, 2.0)["finish_rate"] <= a["finish_rate"]


def test_why_shown_differs_by_model_and_matches_scores(trained):
    b, cfg = trained, SimConfig()
    cart = b.test[0].items[:3]
    pols = {p.name: p for p in make_policies(b, cfg)}
    reasons = {}
    for name, p in pols.items():
        ctx = _ctx(b, cart)
        recs = p.recommend(cart, 10, np.inf, ctx)
        r = why_shown(p, b.cat, cart, recs, ctx, -1)
        assert len(r) == len(recs) and all(isinstance(x, str) and x for x in r)
        reasons[name] = r
    assert len({tuple(v) for v in reasons.values()}) > 3          # not the same text everywhere
    seq = pols["Sequential"]
    recs = seq.recommend(cart, 10, np.inf, None)
    _, contrib = sequential_contributions(seq, cart, -1, recs)
    assert np.allclose(contrib.sum(axis=0), seq.model.scores(cart, -1)[recs])


def test_cart_changes_the_top_slots(trained):
    """Cart-aware models must move their first slots when the cart changes (not frozen cheap items)."""
    b = trained
    pols = {p.name: p for p in make_policies(b, SimConfig())}
    firsts = {n: set() for n in ("Association rules", "Sequential")}
    for o in b.test[:60]:
        cart = o.items[:3]
        ctx = _ctx(b, cart, o.customer, o.ts)
        for n in firsts:
            firsts[n].add(tuple(pols[n].recommend(cart, 10, float("inf"), ctx)[:5]))
    assert all(len(v) > 20 for v in firsts.values())


def test_prepare_bundle_hits_target_aov(trained):
    import copy
    from src.experiment import prepare_bundle
    b = copy.deepcopy(trained)
    info = prepare_bundle(b, target_aov=120_000, order_items=(3, 8))
    sess = build_sessions(b.test, b.cat.prices, SimConfig(), 0)
    assert np.mean([s.base_value for s in sess]) == pytest.approx(120_000, rel=0.06)
    assert all(3 <= len(o.items) <= 8 for o in b.test) and 0 < info["share"] <= 1


def test_random_picks_are_rarely_accepted(trained):
    """Impulse accept chance scales with popularity: random (obscure) picks << best sellers."""
    s, _ = run_simulation(trained, SimConfig(p_hit=0.6, p_miss=0.06), seeds=1, max_orders=400, max_warm=300)
    t = s.set_index("policy")["orders_with_accept"]
    assert t["Random"] < t["Fixed list"]
