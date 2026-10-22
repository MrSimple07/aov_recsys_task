"""Glue between saved artifacts and the simulator (shared by the CLI and the app)."""
from __future__ import annotations

from dataclasses import replace

import numpy as np

from .artifacts import Bundle
from .config import MISS_RATIO, SIM_ORDER_ITEMS, TARGET_ACCEPT, TARGET_AOV, SimConfig
from .models import (FixedListPolicy, FrequentlyBoughtTogether, LinUCBPolicy, RandomPolicy,
                     SequentialPolicy, relevance)
from .simulator import _accept_stage, _subsample, build_sessions, calibrate_stop_slope, evaluate

REFERENCE = "Association rules"    # the stop-after-accept slope is calibrated on this policy


def make_policies(b: Bundle, cfg: SimConfig, fixed_items=None) -> list:
    """All models, in the order they appear in the comparison table."""
    cat = b.cat
    arms = cat.pop_order[:100]
    return [
        RandomPolicy(cat),
        FixedListPolicy(cat, fixed_items),
        FrequentlyBoughtTogether(cat, b.rules),
        SequentialPolicy(cat, b.seq),
        LinUCBPolicy(cat, arms, b.ctxb, name="Contextual bandit (cold start)"),
        LinUCBPolicy(cat, arms, b.ctxb, name="Contextual bandit (warm start)", warm_start=True),
    ]


def run_simulation(b: Bundle, cfg: SimConfig, seeds: int = 3, max_orders: int = 3000,
                   fixed_items=None, max_warm: int = 3000):
    pols = make_policies(b, cfg, fixed_items)
    ref = next(p for p in pols if p.name == REFERENCE)
    return evaluate(b.cat, b.test, pols, cfg, ref, seeds=seeds, max_orders=max_orders,
                    ctxb=b.ctxb, warm=b.warm, max_warm=max_warm)


def estimate_stop_slope(b: Bundle, cfg: SimConfig, n: int = 500, seed: int = 0) -> float:
    """Quick calibration used by the live cart view so it matches the simulation."""
    pols = make_policies(b, cfg)
    ref = next(p for p in pols if p.name == REFERENCE)
    sess = build_sessions(_subsample(b.test, n), b.cat.prices, cfg, seed, b.ctxb)
    return calibrate_stop_slope(sess, ref, b.cat, cfg)


def calibrate_accept(b: Bundle, fixed_items, target: float = TARGET_ACCEPT, n: int = 800,
                     seed: int = 1):
    """Find the shopper's "accept a wanted product" chance so that the Fixed list gets accepted
    by about `target` of SHOPPERS (share of orders where at least one suggestion is accepted,
    default settings: Low price sensitivity, no price limit).

    Why per shopper and not per suggestion: with 10 slots, position decay and only ~5% of the
    best sellers actually wanted, a per-suggestion rate of 24% is unreachable - the search used to
    saturate at p_hit = 1 and the Fixed list landed at ~4%.

    -> (p_hit, p_miss, reached). If even p_hit = 1 cannot reach the target, `reached` is lower."""
    base = SimConfig()
    cat = b.cat
    pol = FixedListPolicy(cat, fixed_items)
    sess = build_sessions(_subsample(b.test, n), cat.prices, base, seed)
    recs = [pol.recommend(s.cart, base.n_slots, base.price_limit, s.ctx) for s in sess]

    def share(p: float) -> float:
        cfg = replace(base, p_hit=p, p_miss=MISS_RATIO * p)
        hit = sum(_accept_stage(r, s, cat, cfg)[0] for s, r in zip(sess, recs))
        return hit / max(len(sess), 1)

    top = share(1.0)
    if top <= target:
        return 1.0, MISS_RATIO, top
    lo, hi = 0.02, 1.0
    for _ in range(14):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if share(mid) < target else (lo, mid)
    p = (lo + hi) / 2
    return p, MISS_RATIO * p, share(p)


def prepare_bundle(b: Bundle, target_aov: float = TARGET_AOV, order_items=SIM_ORDER_ITEMS,
                   cart_frac: float = SimConfig.cart_frac, rescale: bool = True) -> dict:
    """Make the data look like an app: keep only small orders and rescale prices (once!) so the
    average cart value equals `target_aov`. -> dict(scale, share, n_test)."""
    lo, hi = order_items
    keep = lambda xs: [o for o in xs if lo <= len(o.items) <= hi]       # noqa: E731
    test, warm = keep(b.test), keep(b.warm)
    share = len(test) / max(len(b.test), 1)
    if len(test) >= 50:                                 # tiny datasets: keep everything
        b.test, b.warm = test, (warm or b.warm)
    else:
        share = 1.0
    vals = []
    for o in b.test:
        n = len(o.items)
        k = int(np.clip(round(n * cart_frac), 1, max(n - 1, 1)))
        vals.append(b.cat.prices[o.items].sum() * k / n)       # expected cart value
    scale = float(np.mean(vals)) / target_aov if rescale else 1.0
    if rescale:
        b.cat.rescale(scale)
    return dict(scale=scale, share=share, n_test=len(b.test))


def calibrate_relevance(b: Bundle, cfg: SimConfig, n: int = 600, seed: int = 3) -> float:
    """The cart view guesses P(wanted) from rule confidence / popularity, which counts the WHOLE
    order. In the simulation only the hidden half of an order is wanted, so the raw guess is too
    high. Scale = (products really wanted among the shown ones) / (sum of the raw guesses)."""
    sess = build_sessions(_subsample(b.test, n), b.cat.prices, cfg, seed, b.ctxb)
    hits = raw = 0.0
    for p in make_policies(b, cfg):
        if p.name not in ("Fixed list", "Association rules", "Sequential"):
            continue
        for ss in sess:
            rel = relevance(b.rules, b.cat, ss.cart)
            for it in p.recommend(ss.cart, cfg.n_slots, cfg.price_limit, ss.ctx):
                hits += it in ss.hidden
                raw += rel[it]
    return float(np.clip(hits / raw, 0.05, 1.0)) if raw > 0 else 1.0
