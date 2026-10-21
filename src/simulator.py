"""Shopper simulation: suggest -> accept -> (maybe remove) -> finish or stop.

All shopper behaviour is SYNTHETIC (there are no accept/reject logs in the data):
  1. Half of a real test order is the cart; the other half is what the shopper "wants".
  2. Each shown item is accepted with p_hit (wanted) or p_miss (unrelated), times a
     price penalty exp(-s*price/median) and a position weight 1/log2(rank+1).
  3. Accepted items may be removed again (more often when unrelated).
  4. Learning models (the bandits) see which shown items were accepted and update.
  5. Finishing: with nothing accepted, 80% finish. After accepting, the chance to stop
     grows with how expensive the added items are relative to the cart ("too pricey").
     The slope is calibrated once so ~30% of accepters stop on average, and the SAME
     slope is used for every policy, so expensive policies are penalised fairly.
Random draws are shared across policies (common random numbers).
"""
from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
import pandas as pd

from .config import MAX_SLOTS, SimConfig
from .models import Catalog, Ctx


@dataclass
class Session:
    cart: np.ndarray
    hidden: frozenset
    base_value: float        # so'm value of the cart (1 unit per item)
    u_acc: np.ndarray        # uniform draws, one per slot
    u_rem: np.ndarray
    u_fin: float
    ctx: Ctx | None = None   # customer, time, context vector


def build_sessions(baskets, prices, cfg: SimConfig, seed: int, ctxb=None) -> list:
    rng = np.random.default_rng(seed)
    out = []
    for b in baskets:
        items = rng.permutation(b.items)
        k = int(np.clip(round(len(items) * cfg.cart_frac), 1, len(items) - 1))
        cart = items[:k]
        ctx = Ctx(b.customer, b.ts, ctxb.build(cart, b.ts) if ctxb is not None else None)
        out.append(Session(cart, frozenset(int(i) for i in items[k:]),
                           float(prices[cart].sum()),
                           rng.random(MAX_SLOTS), rng.random(MAX_SLOTS), float(rng.random()), ctx))
    return out


def _accept_stage(recs, ss: Session, cat: Catalog, cfg: SimConfig, bands=None):
    """-> (anything accepted?, so'm kept in cart, accepted count, kept count, hits shown,
    accepted flag per shown item)."""
    any_acc, add, acc_n, kept_n, hits, flags = False, 0.0, 0, 0, 0, []
    for j, it in enumerate(recs[:MAX_SLOTS]):
        hit = it in ss.hidden
        hits += hit
        price = cat.prices[it]
        p = (cfg.p_hit if hit else cfg.p_miss) * np.exp(-cfg.price_sensitivity * price / cat.ref)
        if cfg.position_bias:
            p /= np.log2(j + 2)
        b = cat.band[it]
        if bands is not None:
            bands[0][b] += 1
        flags.append(bool(ss.u_acc[j] < p))
        if ss.u_acc[j] < p:
            any_acc = True
            acc_n += 1
            if bands is not None:
                bands[1][b] += 1
            if ss.u_rem[j] >= (cfg.remove_hit if hit else cfg.remove_miss):
                add += price
                kept_n += 1
                if bands is not None:
                    bands[2][b] += price
    return any_acc, add, acc_n, kept_n, hits, flags


def calibrate_stop_slope(sessions, policy, cat: Catalog, cfg: SimConfig) -> float:
    """Slope k so that, for the reference policy, accepters stop ~stop_after_accept on avg."""
    ratios = []
    for ss in sessions:
        recs = policy.recommend(ss.cart, cfg.n_slots, cfg.price_limit, ss.ctx)
        acc, add, *_ = _accept_stage(recs, ss, cat, cfg)
        if acc:
            ratios.append(add / ss.base_value)
    mean_ratio = float(np.mean(ratios)) if ratios else 0.0
    if mean_ratio <= 0:
        return 0.0
    return max(0.0, (cfg.stop_after_accept - (1 - cfg.finish_base)) / mean_ratio)


def run_policy(policy, sessions, cat: Catalog, cfg: SimConfig, slope: float):
    n = len(sessions)
    base = np.array([s.base_value for s in sessions])
    added, any_acc, finished = np.zeros(n), np.zeros(n, bool), np.zeros(n, bool)
    bands = (np.zeros(4), np.zeros(4), np.zeros(4))   # shown, accepted, kept so'm
    shown = acc_n = kept_n = hits = 0
    for t, ss in enumerate(sessions):
        recs = policy.recommend(ss.cart, cfg.n_slots, cfg.price_limit, ss.ctx)
        shown += len(recs)
        a, add, an, kn, h, flags = _accept_stage(recs, ss, cat, cfg, bands)
        policy.update(ss.ctx, recs, flags)                # only the bandits actually learn
        any_acc[t], added[t] = a, add
        acc_n += an
        kept_n += kn
        hits += h
        if a:
            stop = min(cfg.max_stop, (1 - cfg.finish_base) + slope * add / ss.base_value)
            p_fin = 1 - stop
        else:
            p_fin = cfg.finish_base
        finished[t] = ss.u_fin < p_fin
    fin0 = np.array([s.u_fin < cfg.finish_base for s in sessions])   # no suggestions at all
    aov0 = base[fin0].mean()
    aov1 = (base + added)[finished].mean() if finished.any() else float("nan")
    rps0, rps1 = (base * fin0).mean(), ((base + added) * finished).mean()
    metrics = dict(
        AOV_base=aov0, AOV_policy=aov1, extra_per_order=aov1 - aov0,
        AOV_uplift_pct=100 * (aov1 / aov0 - 1),
        RPS_base=rps0, RPS_policy=rps1, RPS_uplift_pct=100 * (rps1 / rps0 - 1),
        extra_per_1000_started=(rps1 - rps0) * 1000,
        conversion=acc_n / max(shown, 1), orders_with_accept=any_acc.mean(),
        kept_share=kept_n / max(acc_n, 1), value_per_suggestion=added.sum() / max(shown, 1),
        finish_rate=finished.mean(), finish_rate_base=fin0.mean(),
        relevance_rate=hits / max(shown, 1), n_orders=n)
    return metrics, bands


def _subsample(items, k: int, seed: int = 0):
    if len(items) <= k:
        return list(items)
    idx = np.random.default_rng(seed).choice(len(items), size=k, replace=False)
    return [items[i] for i in sorted(idx)]


def evaluate(cat: Catalog, test, policies: list, cfg: SimConfig, reference,
             seeds: int = 3, max_orders: int = 3000, ctxb=None, warm=(), max_warm: int = 3000):
    """-> (summary per policy, price-band table). Mean over `seeds` repeats.

    Every repeat: fresh sessions, fresh random draws, bandits reset; "warm start" policies are
    first pre-trained on `warm` (past orders) with the same shopper assumptions."""
    test = _subsample(test, max_orders)
    warm = _subsample(list(warm), max_warm) if len(warm) else []
    rows, band_acc = [], {p.name: np.zeros((3, 4)) for p in policies}
    for seed in range(seeds):
        sess = build_sessions(test, cat.prices, cfg, seed, ctxb)
        warm_sess = build_sessions(warm, cat.prices, cfg, 10_000 + seed, ctxb)
        slope = calibrate_stop_slope(sess, reference, cat, cfg)
        for pol in policies:
            pol.reset(seed)
            if hasattr(pol, "set_stop_slope"):        # the business knows how price scares shoppers
                pol.set_stop_slope(slope)
            if pol.warm_start and warm_sess:
                run_policy(pol, warm_sess, cat, cfg, slope)
            m, b = run_policy(pol, sess, cat, cfg, slope)
            rows.append(dict(policy=pol.name, seed=seed, stop_slope=slope, **m))
            band_acc[pol.name] += np.vstack(b)
    df = pd.DataFrame(rows)
    order = [p.name for p in policies]
    summary = df.drop(columns="seed").groupby("policy", sort=False).mean().reset_index()
    for col in ("RPS_uplift_pct", "AOV_uplift_pct"):
        sd = df.groupby("policy", sort=False)[col].std(ddof=1).fillna(0.0).to_numpy()
        summary[col + "_ci95"] = 1.96 * sd / np.sqrt(seeds)
    summary["policy"] = pd.Categorical(summary["policy"], order, ordered=True)
    summary = summary.sort_values("policy").reset_index(drop=True)
    summary["policy"] = summary["policy"].astype(str)
    labels = cat.band_labels()
    brow = []
    for name, (shown, acc, val) in band_acc.items():
        for k in range(4):
            brow.append(dict(policy=name, band=labels[k], shown=shown[k],
                             conversion=acc[k] / max(shown[k], 1),
                             value_per_suggestion=val[k] / max(shown[k], 1)))
    return summary, pd.DataFrame(brow)
