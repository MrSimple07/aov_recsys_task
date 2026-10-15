"""Train the three models once, then evaluate all policies under a SimConfig."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .config import SimConfig
from .data import Dataset
from .features import ContextBuilder, fit_item_embeddings
from .models import (AssociationRulesPolicy, LinUCBPolicy, MarkovPolicy,
                     PopularityPolicy, RandomPolicy, popularity_order)
from .simulator import build_sessions, run_policy, summarize


@dataclass
class Prepared:
    ds: Dataset
    ctxb: ContextBuilder
    pop_order: np.ndarray
    ar: AssociationRulesPolicy
    seq: MarkovPolicy
    arms: np.ndarray


def prepare(ds: Dataset, min_support=0.005, min_confidence=0.1, n_arms=100, emb_dim=16) -> Prepared:
    pop = popularity_order(ds.train, ds.V)
    ctxb = ContextBuilder(fit_item_embeddings(ds.train, ds.V, emb_dim), ds.prices)
    ar = AssociationRulesPolicy(ds.prices, pop, min_support, min_confidence).fit(ds.train, ds.V)
    seq = MarkovPolicy(ds.prices, pop).fit(ds.train, ds.V)
    print(f"[prepare] vocab={ds.V} train_baskets={len(ds.train)} test_baskets={len(ds.test)} "
          f"rules={ar.n_rules}")
    return Prepared(ds, ctxb, pop, ar, seq, pop[:n_arms])


def _subsample(items, cap):
    if len(items) <= cap:
        return items
    idx = np.linspace(0, len(items) - 1, cap).astype(int)  # keeps chronological order
    return [items[i] for i in idx]


def evaluate(prep: Prepared, cfg: SimConfig, seeds=5, alpha=0.1, max_test=10000, max_warm=20000):
    """Returns (per-seed metrics DataFrame, {policy: [added-value arrays per seed]})."""
    ds, rows, curves = prep.ds, [], {}
    test_b = _subsample(ds.test, max_test)
    warm_b = _subsample(ds.train, max_warm)
    for seed in range(seeds):
        rng = np.random.default_rng(seed)
        test = build_sessions(test_b, prep.ctxb, ds.prices, cfg, rng)
        warm = build_sessions(warm_b, prep.ctxb, ds.prices, cfg, np.random.default_rng(10_000 + seed))

        cold = LinUCBPolicy(prep.arms, ds.prices, prep.ctxb.dim, alpha, name="LinUCB (cold start)")
        warmed = LinUCBPolicy(prep.arms, ds.prices, prep.ctxb.dim, alpha, name="LinUCB (warm start)")
        run_policy(warmed, warm, ds.prices, cfg, learn=True)  # offline pre-training in the simulator

        policies = [RandomPolicy(ds.V, seed), PopularityPolicy(prep.pop_order),
                    prep.ar, prep.seq, cold, warmed]
        for pol in policies:
            res = run_policy(pol, test, ds.prices, cfg, learn=True)
            rows.append({"policy": pol.name, "seed": seed, **summarize(res)})
            curves.setdefault(pol.name, []).append(res["added"])
    return pd.DataFrame(rows), curves


def aggregate(df: pd.DataFrame) -> pd.DataFrame:
    g = df.drop(columns="seed").groupby("policy", sort=False)
    mean, std = g.mean(), g.std(ddof=1).fillna(0)
    n = df["seed"].nunique()
    out = mean.copy()
    out["uplift_pct_ci95"] = 1.96 * std["uplift_pct"] / np.sqrt(n)
    return out.reset_index()
