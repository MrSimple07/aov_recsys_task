"""Model 1: association rules (FP-Growth) -> recommend the consequent of matching rules."""
import itertools

import numpy as np
import pandas as pd
from mlxtend.frequent_patterns import fpgrowth

from .base import ScorePolicy


class AssociationRulesPolicy(ScorePolicy):
    name = "Association Rules"

    def __init__(self, prices, pop_order, min_support=0.005, min_confidence=0.1,
                 min_lift=1.0, max_len=3, max_cart=12, **kw):
        super().__init__(prices, pop_order, **kw)
        self.min_support, self.min_confidence = min_support, min_confidence
        self.min_lift, self.max_len, self.max_cart = min_lift, max_len, max_cart
        self.rules1, self.rules2 = {}, {}
        self.n_rules = 0

    def fit(self, train, V):
        X = np.zeros((len(train), V), dtype=bool)
        for r, b in enumerate(train):
            X[r, b.items] = True
        freq = fpgrowth(pd.DataFrame(X), min_support=self.min_support,
                        use_colnames=True, max_len=self.max_len)
        sup = {tuple(sorted(int(i) for i in fs)): float(s)
               for fs, s in zip(freq["itemsets"], freq["support"])}
        self.rules1, self.rules2, self.n_rules = {}, {}, 0
        for itemset, s in sup.items():
            if len(itemset) < 2:
                continue
            for c in itemset:
                ant = tuple(x for x in itemset if x != c)
                conf = s / sup[ant]
                lift = conf / sup[(c,)]
                if conf >= self.min_confidence and lift >= self.min_lift:
                    store = self.rules1 if len(ant) == 1 else self.rules2
                    store.setdefault(ant[0] if len(ant) == 1 else ant, []).append((c, conf))
                    self.n_rules += 1
        return self

    def scores(self, s):
        out = np.zeros(len(self.prices))
        cart = sorted(set(int(x) for x in s.cart[-self.max_cart:]))
        for i in cart:
            for c, conf in self.rules1.get(i, ()):
                if conf > out[c]:
                    out[c] = conf
        if self.rules2:
            for pair in itertools.combinations(cart, 2):
                for c, conf in self.rules2.get(pair, ()):
                    if conf > out[c]:
                        out[c] = conf
        return out
