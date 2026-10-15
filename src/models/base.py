from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class Session:
    """What a policy is allowed to see when it must recommend."""
    cart: np.ndarray
    customer: int
    ts: pd.Timestamp
    ctx: np.ndarray


class Policy:
    name = "policy"

    def reset(self):
        pass

    def recommend(self, s: Session, k: int) -> list:
        raise NotImplementedError

    def update(self, s: Session, shown: list, accepted: list):
        pass


def popularity_order(train, V) -> np.ndarray:
    counts = np.bincount(np.concatenate([b.items for b in train]), minlength=V)
    return np.argsort(-counts, kind="stable")


def fill_popular(chosen: list, cart, order, k: int) -> list:
    banned = set(int(x) for x in cart) | set(chosen)
    for i in order:
        if len(chosen) >= k:
            break
        if int(i) not in banned:
            chosen.append(int(i))
    return chosen


class PopularityPolicy(Policy):
    name = "Popularity"

    def __init__(self, order):
        self.order = order

    def recommend(self, s, k):
        return fill_popular([], s.cart, self.order, k)


class RandomPolicy(Policy):
    name = "Random"

    def __init__(self, V, seed=0):
        self.V, self.seed = V, seed
        self.reset()

    def reset(self):
        self.rng = np.random.default_rng(self.seed)

    def recommend(self, s, k):
        cand = np.setdiff1d(np.arange(self.V), s.cart)
        return self.rng.choice(cand, k, replace=False).tolist()


class ScorePolicy(Policy):
    """Policy that scores every item; picks by score (x price if price_aware)."""

    def __init__(self, prices, pop_order, price_aware=True, shortlist=20):
        self.prices, self.pop_order = prices, pop_order
        self.price_aware, self.shortlist = price_aware, shortlist

    def scores(self, s: Session) -> np.ndarray:
        raise NotImplementedError

    def recommend(self, s, k):
        sc = np.asarray(self.scores(s), dtype=float).copy()
        sc[s.cart] = 0
        pos = np.flatnonzero(sc > 0)
        chosen = []
        if len(pos):
            if len(pos) > self.shortlist:
                pos = pos[np.argpartition(-sc[pos], self.shortlist)[: self.shortlist]]
            key = sc[pos] * (self.prices[pos] if self.price_aware else 1.0)
            chosen = pos[np.argsort(-key)][:k].tolist()
        return fill_popular(chosen, s.cart, self.pop_order, k)  # fallback when no rule fires
