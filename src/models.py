"""Recommenders, each explainable in one sentence:

* RandomPolicy             - random products (a floor to beat)
* FixedListPolicy          - the same list for everyone (default: top-10 best sellers)
* FrequentlyBoughtTogether - "Association rules": items bought together with the cart
* SequentialPolicy         - next product from the customer's past orders + the cart
* LinUCBPolicy             - contextual bandit, learns from accept/reject (cold / warm start)
* ExpectedValuePolicy      - ranks by  chance to accept  x  price  (business value)

Every policy returns the final list the shopper sees: cart items removed, items above
the price limit removed, at most n items, SORTED BY PRICE (cheapest first).
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import chain

import numpy as np
from scipy.sparse import coo_matrix, csr_matrix, diags
from scipy.sparse.linalg import svds


# ----------------------------------------------------------------------------- catalog
@dataclass
class Catalog:
    item_ids: list
    names: list
    prices: np.ndarray        # so'm
    train_counts: np.ndarray
    n_train: int

    def __post_init__(self):
        self.prices = np.asarray(self.prices, float)
        self.V = len(self.prices)
        self.ref = float(np.median(self.prices))                 # "typical" price
        self.pop_order = np.argsort(-self.train_counts, kind="stable")
        self.pop_share = self.train_counts / max(self.n_train, 1)  # P(item is in an order)
        self.band_edges = np.quantile(self.prices, [0.25, 0.5, 0.75])
        self.band = np.searchsorted(self.band_edges, self.prices, side="right")

    def label(self, i: int) -> str:
        return (self.names[i] or self.item_ids[i]).strip().title()

    def band_labels(self) -> list:
        edges = [self.prices.min(), *self.band_edges, self.prices.max()]
        f = lambda x: f"{x:,.0f}".replace(",", " ")  # noqa: E731
        return [f"{k + 1}. {f(edges[k])}-{f(edges[k + 1])}" for k in range(4)]


# ------------------------------------------------------------- frequently bought together
class PairRules:
    """conf[i, j] = P(j in order | i in order), kept only when the pair is reliable."""

    def __init__(self, conf: csr_matrix):
        self.conf = conf.tocsr()

    @property
    def n_rules(self) -> int:
        return int(self.conf.nnz)

    @staticmethod
    def fit(train, V: int, min_count: int = 5, min_conf: float = 0.05,
            min_lift: float = 1.0, top_per_item: int = 50) -> "PairRules":
        N = len(train)
        sizes = np.fromiter((len(b.items) for b in train), dtype=np.int64, count=N)
        rows = np.repeat(np.arange(N), sizes)
        cols = np.concatenate([b.items for b in train])
        X = csr_matrix((np.ones(len(rows), np.float32), (rows, cols)), shape=(N, V))
        cnt = np.asarray(X.sum(axis=0)).ravel()
        co = (X.T @ X).tocoo()
        m = (co.row != co.col) & (co.data >= min_count)
        r, c, d = co.row[m], co.col[m], co.data[m]
        conf = d / cnt[r]
        lift = conf / (cnt[c] / N)
        k = (conf >= min_conf) & (lift >= min_lift)
        r, c, conf = r[k], c[k], conf[k]
        o = np.lexsort((-conf, r))                       # per antecedent, best first
        r, c, conf = r[o], c[o], conf[o]
        if len(r):
            first = np.r_[0, np.flatnonzero(np.diff(r)) + 1]
            rank = np.arange(len(r)) - np.repeat(first, np.diff(np.r_[first, len(r)]))
            s = rank < top_per_item
            r, c, conf = r[s], c[s], conf[s]
        return PairRules(csr_matrix((conf.astype(np.float32), (r, c)), shape=(V, V)))

    def scores(self, cart) -> np.ndarray:
        cart = np.asarray(cart, dtype=int)
        V = self.conf.shape[0]
        if len(cart) == 0:
            return np.zeros(V)
        out = self.conf[cart].max(axis=0).toarray().ravel().astype(float)
        out[cart] = 0.0
        return out


# ---------------------------------------------------------------------- shared helpers
def relevance(rules: PairRules, cat: Catalog, cart) -> np.ndarray:
    """Rough P(shopper wants the item): rule confidence, or its popularity if higher."""
    cart = np.asarray(cart, dtype=int)
    r = np.maximum(rules.scores(cart), cat.pop_share)
    r[cart] = 0.0
    return np.clip(r, 0.0, 1.0)


def accept_prob(rel, price, ref, p_hit, p_miss, s):
    """Chance to accept = (wanted / unrelated mix) x price penalty exp(-s * price / ref)."""
    return (rel * p_hit + (1.0 - rel) * p_miss) * np.exp(-s * np.asarray(price) / ref)


def finalize(ranked, cart, prices, limit, n, fallback) -> list:
    """Filter first, then sort: drop cart items + items above the price limit, take the
    first n of the ranking (popular items fill gaps), show CHEAPEST FIRST."""
    banned = set(int(i) for i in cart)
    out: list = []
    for i in chain(ranked, fallback):
        i = int(i)
        if i in banned or prices[i] > limit:
            continue
        out.append(i)
        banned.add(i)
        if len(out) >= n:
            break
    out.sort(key=lambda i: prices[i])          # stable: equal prices keep rank order
    return out


# ------------------------------------------------------------------------------ policies
@dataclass
class Ctx:
    """What the shop knows besides the cart: who (customer), when (ns), context vector."""
    customer: int = -1
    ts: int = 0
    vec: np.ndarray | None = None


class _Stateless:
    warm_start = False

    def reset(self, seed: int = 0):
        pass

    def update(self, ctx, shown, accepted):
        pass


class FixedListPolicy(_Stateless):
    name = "Fixed list"
    blurb = "The same best sellers for everyone (editable)."

    def __init__(self, cat: Catalog, items=None, size: int = 10):
        self.cat = cat
        self.items = [int(i) for i in (items if items is not None else cat.pop_order[:size])]

    def recommend(self, cart, n: int, limit: float = float("inf"), ctx=None) -> list:
        return finalize(self.items, cart, self.cat.prices, limit, n, self.cat.pop_order)


class FrequentlyBoughtTogether(_Stateless):
    name = "Association rules"
    blurb = "Products often bought together with the items in the cart."

    def __init__(self, cat: Catalog, rules: PairRules):
        self.cat, self.rules = cat, rules

    def recommend(self, cart, n: int, limit: float = float("inf"), ctx=None) -> list:
        sc = self.rules.scores(cart)
        pos = np.flatnonzero(sc > 0)
        ranked = pos[np.argsort(-sc[pos], kind="stable")][:300]
        return finalize(ranked, cart, self.cat.prices, limit, n, self.cat.pop_order)

    
class ExpectedValuePolicy(_Stateless):
    """Ranks by money, not by relevance alone:

        value = chance to accept x price x (1 - removal risk), minus the risk that a pricey
                add-on makes the shopper stop the whole order.

    The chance counts only EVIDENCE of interest (bought-together rules, popularity) times the
    price effect. It does not plan on lucky impulse buys of unrelated items - otherwise, when
    price does not matter, it would simply suggest the most expensive products.
    A 70%-likely 5-unit item is worth 3.5; a 30%-likely 30-unit item is worth 9 - unless it is so
    expensive relative to the cart that the shopper walks away. stop_slope = 0 -> no leave risk.
    """
    name = "Expected value"
    blurb = ("Picks by chance to accept x price (real interest only), minus the risk that a "
             "high price makes shoppers leave.")

    def __init__(self, cat: Catalog, rules: PairRules, p_hit=0.5, s=0.15,
                 remove_hit=0.05, remove_miss=0.30, finish_base=0.80, stop_slope=0.0):
        self.cat, self.rules = cat, rules
        self.p_hit, self.s = p_hit, s
        self.remove_hit, self.remove_miss = remove_hit, remove_miss
        self.finish_base, self.stop_slope = finish_base, stop_slope

    def set_stop_slope(self, slope: float):
        self.stop_slope = float(slope)

    def expected(self, cart):
        """-> (chance to accept, expected value in so'm) for every product."""
        cart = np.asarray(cart, dtype=int)
        rel = relevance(self.rules, self.cat, cart)
        price = self.cat.prices
        p = rel * self.p_hit * np.exp(-self.s * price / self.cat.ref)
        keep = 1.0 - (rel * self.remove_hit + (1.0 - rel) * self.remove_miss)
        B = max(float(price[cart].sum()), 1e-9)
        gain = price * (self.finish_base - self.stop_slope * (B + price) / B)
        return p, p * keep * gain

    def recommend(self, cart, n: int, limit: float = float("inf"), ctx=None) -> list:
        _, ev = self.expected(cart)
        ranked = np.argsort(-ev, kind="stable")[:300]
        return finalize(ranked, cart, self.cat.prices, limit, n, self.cat.pop_order)


class RandomPolicy(_Stateless):
    name = "Random"
    blurb = "Random products - the floor every model must beat."

    def __init__(self, cat: Catalog):
        self.cat = cat
        self.reset(0)

    def reset(self, seed: int = 0):
        self.rng = np.random.default_rng(seed)

    def recommend(self, cart, n: int, limit: float = float("inf"), ctx=None) -> list:
        return finalize(self.rng.permutation(self.cat.V), cart, self.cat.prices, limit, n,
                        self.cat.pop_order)


# ------------------------------------------------------------------- sequential (Markov)
class SequentialModel:
    """Decayed Markov chain over each customer's purchase stream: P[i, j] ~ 'j comes after i'."""

    def __init__(self, P: csr_matrix, history: dict):
        self.P, self.history = P.tocsr(), history

    @staticmethod
    def fit(train, V: int, window: int = 3, gamma: float = 0.7, smooth: float = 1.0):
        per_cust, guests = {}, []
        for b in train:                                   # train is chronological
            (per_cust.setdefault(b.customer, []) if b.customer >= 0 else guests).append(b.items)
        history = {c: np.concatenate(v) for c, v in per_cust.items()}
        r, c, v = [], [], []
        for seq in list(history.values()) + guests:
            for d in range(1, window + 1):
                if len(seq) > d:
                    r.append(seq[:-d]); c.append(seq[d:])
                    v.append(np.full(len(seq) - d, gamma ** (d - 1)))
        if not r:
            return SequentialModel(csr_matrix((V, V)), history)
        r, c, v = np.concatenate(r), np.concatenate(c), np.concatenate(v)
        k = r != c
        T = coo_matrix((v[k], (r[k], c[k])), shape=(V, V)).tocsr()
        rowsum = np.asarray(T.sum(axis=1)).ravel()
        return SequentialModel((diags(1.0 / (rowsum + smooth)) @ T).tocsr(), history)

    def scores(self, cart, customer: int, lam=0.8, hist_len=8, max_state=10) -> np.ndarray:
        cart = np.asarray(cart, dtype=int)
        hist = self.history.get(customer)
        seq = cart if hist is None else np.concatenate([hist[-hist_len:], cart])
        seq = seq[-max_state:]
        out = self.P[seq].T.dot(lam ** np.arange(len(seq) - 1, -1, -1))
        out[cart] = 0.0
        return out


class SequentialPolicy(_Stateless):
    name = "Sequential"
    blurb = "Predicts the next product from the customer's past orders and the cart."

    def __init__(self, cat: Catalog, model: SequentialModel):
        self.cat, self.model = cat, model

    def recommend(self, cart, n: int, limit: float = float("inf"), ctx=None) -> list:
        sc = self.model.scores(cart, -1 if ctx is None else ctx.customer)
        pos = np.flatnonzero(sc > 0)
        ranked = pos[np.argsort(-sc[pos], kind="stable")][:300]
        return finalize(ranked, cart, self.cat.prices, limit, n, self.cat.pop_order)


# ----------------------------------------------------------------- contextual bandit
def fit_embeddings(train, V: int, dim: int = 16) -> np.ndarray:
    """SVD of the order x product matrix -> one unit-length vector per product."""
    rows = np.concatenate([np.full(len(b.items), i) for i, b in enumerate(train)])
    cols = np.concatenate([b.items for b in train])
    X = csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(len(train), V))
    k = max(1, min(dim, V - 1, len(train) - 1))
    _, _, vt = svds(X, k=k, random_state=0)
    emb = vt.T
    return emb / (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-9)


class ContextBuilder:
    """Context vector = [bias, cart size, cart value, day-of-week, hour] + mean cart embedding."""

    def __init__(self, emb: np.ndarray, prices: np.ndarray):
        self.emb, self.prices = emb, np.asarray(prices, float)
        self.ref = float(np.median(self.prices))
        self.dim = 7 + emb.shape[1]

    def build(self, cart, ts_ns: int) -> np.ndarray:
        cart = np.asarray(cart, dtype=int)
        dow = 2 * np.pi * (((ts_ns // 86_400_000_000_000) + 3) % 7) / 7      # Monday = 0
        hr = 2 * np.pi * ((ts_ns // 3_600_000_000_000) % 24) / 24
        head = [1.0, np.log1p(len(cart)) / 3, np.log1p(self.prices[cart].sum() / self.ref) / 3,
                np.sin(dow), np.cos(dow), np.sin(hr), np.cos(hr)]
        return np.concatenate([head, self.emb[cart].mean(axis=0)])


class LinUCBPolicy:
    """Disjoint LinUCB. Arm = product, reward = accepted x price (scaled).
    Cold start begins knowing nothing; warm start is pre-trained on past orders first."""
    
    blurb = "Learns what shoppers accept while running."  # Default fallback class attribute

    def __init__(self, cat: Catalog, arms, ctxb: ContextBuilder, alpha=0.1, l2=1.0,
                 name=None, warm_start=False):
        self.cat, self.arms, self.ctxb = cat, np.asarray(arms), ctxb
        self.arm_pos = {int(a): i for i, a in enumerate(self.arms)}
        self.alpha, self.l2, self.warm_start = alpha, l2, warm_start
        self.cap = float(np.percentile(cat.prices, 95))

        # Assign specific blurb based on warm_start status
        if self.warm_start:
            self.blurb = "Pre-trained on historical orders; adapts online to shopper feedback."
            default_name = "Contextual bandit (warm start)"
        else:
            self.blurb = "Starts with zero historical data; learns what shoppers accept online."
            default_name = "Contextual bandit (cold start)"

        self.name = name if name is not None else default_name

        self.reset()

    def reset(self, seed: int = 0):
        M, d = len(self.arms), self.ctxb.dim
        self.A_inv = np.tile(np.eye(d) / self.l2, (M, 1, 1))
        self.b = np.zeros((M, d))

    def recommend(self, cart, n: int, limit: float = float("inf"), ctx=None) -> list:
        if ctx is None or ctx.vec is None:
            raise ValueError("The contextual bandit needs a context (Ctx.vec)")
        x = ctx.vec
        theta = np.einsum("mij,mj->mi", self.A_inv, self.b)
        Ax = self.A_inv @ x
        ucb = theta @ x + self.alpha * np.sqrt(np.maximum(Ax @ x, 0))
        ucb[np.isin(self.arms, cart)] = -np.inf
        ranked = self.arms[np.argsort(-ucb, kind="stable")]
        return finalize(ranked, cart, self.cat.prices, limit, n, self.cat.pop_order)

    def update(self, ctx, shown, accepted):
        x = ctx.vec
        for item, acc in zip(shown, accepted):
            a = self.arm_pos.get(int(item))
            if a is None:
                continue
            r = float(acc) * min(self.cat.prices[item], self.cap) / self.cap
            Ax = self.A_inv[a] @ x
            self.A_inv[a] -= np.outer(Ax, Ax) / (1.0 + x @ Ax)
            self.b[a] += r * x