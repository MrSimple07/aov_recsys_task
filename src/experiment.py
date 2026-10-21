"""Glue between saved artifacts and the simulator (shared by the CLI and the app)."""
from __future__ import annotations

from .artifacts import Bundle
from .config import SimConfig
from .models import (ExpectedValuePolicy, FixedListPolicy, FrequentlyBoughtTogether, LinUCBPolicy,
                     RandomPolicy, SequentialPolicy)
from .simulator import _subsample, build_sessions, calibrate_stop_slope, evaluate

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
        ExpectedValuePolicy(cat, b.rules, p_hit=cfg.p_hit, s=cfg.price_sensitivity,
                            remove_hit=cfg.remove_hit, remove_miss=cfg.remove_miss,
                            finish_base=cfg.finish_base),
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