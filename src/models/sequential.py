"""Model 2: sequential model = decayed Markov chain over each customer's purchase stream."""
import numpy as np
from scipy.sparse import coo_matrix, diags

from .base import ScorePolicy


class MarkovPolicy(ScorePolicy):
    name = "Sequential (Markov)"

    def __init__(self, prices, pop_order, window=3, gamma=0.7, lam=0.8,
                 hist_len=8, max_state=10, smooth=1.0, **kw):
        super().__init__(prices, pop_order, **kw)
        self.window, self.gamma, self.lam = window, gamma, lam
        self.hist_len, self.max_state, self.smooth = hist_len, max_state, smooth

    def fit(self, train, V):
        per_cust, guests = {}, []
        for b in train:  # train is chronological
            (per_cust.setdefault(b.customer, []) if b.customer >= 0 else guests).append(b.items)
        self.history = {c: np.concatenate(v) for c, v in per_cust.items()}
        streams = list(self.history.values()) + guests

        r, c, v = [], [], []
        for seq in streams:
            for d in range(1, self.window + 1):
                if len(seq) > d:
                    r.append(seq[:-d]); c.append(seq[d:])
                    v.append(np.full(len(seq) - d, self.gamma ** (d - 1)))
        r, c, v = np.concatenate(r), np.concatenate(c), np.concatenate(v)
        keep = r != c  # ignore self transitions
        T = coo_matrix((v[keep], (r[keep], c[keep])), shape=(V, V)).tocsr()
        rowsum = np.asarray(T.sum(axis=1)).ravel()
        self.P = (diags(1.0 / (rowsum + self.smooth)) @ T).tocsr()
        return self

    def scores(self, s):
        hist = self.history.get(s.customer)
        seq = s.cart if hist is None else np.concatenate([hist[-self.hist_len:], s.cart])
        seq = seq[-self.max_state:]
        w = self.lam ** np.arange(len(seq) - 1, -1, -1)
        return self.P[seq].T.dot(w)
