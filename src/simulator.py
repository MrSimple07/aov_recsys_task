"""Offline user-response simulator.

Each test basket becomes a checkout session: a random subset (cart_frac) is the CART,
the remaining items are LATENT wants. A policy shows k items. An item is accepted with
probability p_hit if it is a latent want else p_miss, damped by price. Accepted items add
their price to the order. Uniform draws are fixed per session -> all policies face
identical randomness (common random numbers).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import SimConfig
from .models.base import Session


@dataclass
class SimSession:
    session: Session
    hidden: set
    u: np.ndarray
    base_value: float


def build_sessions(baskets, ctxb, prices, cfg: SimConfig, rng, min_items=3):
    out = []
    for b in baskets:
        n = len(b.items)
        if n < min_items:
            continue
        m = min(max(1, int(np.ceil(cfg.cart_frac * n))), n - 1)
        mask = np.zeros(n, bool)
        mask[rng.choice(n, m, replace=False)] = True
        cart, hidden = b.items[mask], b.items[~mask]
        out.append(SimSession(Session(cart, b.customer, b.ts, ctxb.build(cart, b.ts)),
                              set(hidden.tolist()), rng.random(cfg.slots),
                              float(prices[cart].sum())))
    return out


def run_policy(policy, sessions, prices, cfg: SimConfig, learn=True) -> dict:
    ref = float(np.median(prices))
    n = len(sessions)
    added, accepted_n, relevant_n = np.zeros(n), np.zeros(n), np.zeros(n)
    base = np.array([s.base_value for s in sessions])
    for t, ss in enumerate(sessions):
        shown = policy.recommend(ss.session, cfg.slots)
        acc = []
        for j, item in enumerate(shown):
            hit = item in ss.hidden
            p = (cfg.p_hit if hit else cfg.p_miss) * np.exp(-cfg.price_sensitivity * prices[item] / ref)
            ok = bool(ss.u[j] < p)
            acc.append(ok)
            relevant_n[t] += hit
            if ok:
                added[t] += prices[item]
                accepted_n[t] += 1
        if learn:
            policy.update(ss.session, shown, acc)
    return {"base": base, "added": added, "accepted": accepted_n,
            "relevant": relevant_n, "shown": cfg.slots}


def summarize(res: dict) -> dict:
    shown_total = res["shown"] * len(res["base"])
    aov_base = res["base"].mean()
    return {
        "AOV_base": aov_base,
        "AOV_policy": (res["base"] + res["added"]).mean(),
        "uplift_abs": res["added"].mean(),
        "uplift_pct": 100 * res["added"].mean() / aov_base,
        "accept_rate": res["accepted"].sum() / shown_total,
        "relevance_rate": res["relevant"].sum() / shown_total,
    }
