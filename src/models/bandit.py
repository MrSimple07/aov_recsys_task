"""Model 3: contextual bandit (disjoint LinUCB). Arm = item, reward = accepted * price."""
import numpy as np

from .base import Policy


class LinUCBPolicy(Policy):
    def __init__(self, arms, prices, ctx_dim, alpha=0.1, l2=1.0, name="LinUCB"):
        self.arms = np.asarray(arms)
        self.arm_pos = {int(a): i for i, a in enumerate(self.arms)}
        self.prices, self.d, self.alpha, self.l2 = prices, ctx_dim, alpha, l2
        self.cap = float(np.percentile(prices, 95))  # reward scaling
        self.name = name
        self.reset()

    def reset(self):
        M = len(self.arms)
        self.A_inv = np.tile(np.eye(self.d) / self.l2, (M, 1, 1))
        self.b = np.zeros((M, self.d))

    def recommend(self, s, k):
        x = s.ctx
        theta = np.einsum("mij,mj->mi", self.A_inv, self.b)
        Ax = self.A_inv @ x
        ucb = theta @ x + self.alpha * np.sqrt(np.maximum(Ax @ x, 0))
        ucb[np.isin(self.arms, s.cart)] = -np.inf
        return self.arms[np.argsort(-ucb)[:k]].tolist()

    def update(self, s, shown, accepted):
        x = s.ctx
        for item, acc in zip(shown, accepted):
            a = self.arm_pos[int(item)]
            r = float(acc) * min(self.prices[item], self.cap) / self.cap
            Ax = self.A_inv[a] @ x
            self.A_inv[a] -= np.outer(Ax, Ax) / (1.0 + x @ Ax)
            self.b[a] += r * x
