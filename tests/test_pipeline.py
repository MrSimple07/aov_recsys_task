import numpy as np
import pytest

from src.config import SimConfig
from src.data import build_dataset, clean_transactions
from src.experiment import aggregate, evaluate, prepare
from src.models import LinUCBPolicy, Session
from src.models.base import Policy
from src.simulator import build_sessions, run_policy, summarize
from src.synthetic import generate


@pytest.fixture(scope="module")
def raw():
    return generate(n_items=120, n_customers=300, n_invoices=3000, seed=1)


@pytest.fixture(scope="module")
def ds(raw):
    return build_dataset(clean_transactions(raw), top_k=120)


def test_cleaning_drops_cancellations(raw):
    clean = clean_transactions(raw)
    assert not clean["invoice"].str.startswith("C").any()
    assert (clean["qty"] > 0).all() and (clean["price"] > 0).all()


def test_split_is_chronological(ds):
    assert max(b.ts for b in ds.train) <= min(b.ts for b in ds.test)
    assert ds.prices.shape == (ds.V,) and (ds.prices > 0).all()


class Oracle(Policy):
    """Cheats: recommends a latent want. Must beat random by a wide margin."""
    name = "oracle"
    def __init__(self, sessions):
        self.map = {id(s.session): next(iter(s.hidden)) for s in sessions}
    def recommend(self, s, k):
        return [self.map[id(s)]]


def test_simulator_oracle_beats_random(ds):
    from src.features import ContextBuilder, fit_item_embeddings
    from src.models import RandomPolicy
    cfg = SimConfig()
    ctxb = ContextBuilder(fit_item_embeddings(ds.train, ds.V), ds.prices)
    sess = build_sessions(ds.test, ctxb, ds.prices, cfg, np.random.default_rng(0))
    o = summarize(run_policy(Oracle(sess), sess, ds.prices, cfg))
    r = summarize(run_policy(RandomPolicy(ds.V), sess, ds.prices, cfg))
    assert o["uplift_abs"] > 3 * r["uplift_abs"]


def test_linucb_learns_best_arm():
    prices = np.ones(5)
    pol = LinUCBPolicy(arms=[0, 1, 2], prices=prices, ctx_dim=2, alpha=0.1)
    s = Session(np.array([4]), -1, None, np.array([1.0, 0.0]))
    for _ in range(300):
        for arm, acc in ((0, False), (1, True), (2, False)):
            pol.update(s, [arm], [acc])
    assert pol.recommend(s, 1) == [1]


def test_full_evaluation_runs_and_is_deterministic(ds):
    prep = prepare(ds, min_support=0.01, n_arms=30)
    d1, _ = evaluate(prep, SimConfig(), seeds=2, max_test=800, max_warm=1500)
    d2, _ = evaluate(prep, SimConfig(), seeds=2, max_test=800, max_warm=1500)
    assert np.allclose(d1["uplift_abs"], d2["uplift_abs"])
    s = aggregate(d1).set_index("policy")
    assert s.loc["Association Rules", "uplift_pct"] > s.loc["Random", "uplift_pct"]
