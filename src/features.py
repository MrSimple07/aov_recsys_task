"""Context features for the contextual bandit (cart + time)."""
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.decomposition import TruncatedSVD


def fit_item_embeddings(train, V, dim=16, seed=0) -> np.ndarray:
    """SVD of the basket x item matrix -> one unit-length vector per item."""
    rows = np.concatenate([np.full(len(b.items), i) for i, b in enumerate(train)])
    cols = np.concatenate([b.items for b in train])
    X = csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(len(train), V))
    svd = TruncatedSVD(n_components=min(dim, V - 1), random_state=seed).fit(X)
    emb = svd.components_.T
    return emb / (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-9)


class ContextBuilder:
    def __init__(self, emb: np.ndarray, prices: np.ndarray):
        self.emb, self.prices = emb, prices
        self.dim = 7 + emb.shape[1]

    def build(self, cart: np.ndarray, ts) -> np.ndarray:
        dow, hr = 2 * np.pi * ts.dayofweek / 7, 2 * np.pi * ts.hour / 24
        head = [1.0, np.log1p(len(cart)) / 3, np.log1p(self.prices[cart].sum()) / 7,
                np.sin(dow), np.cos(dow), np.sin(hr), np.cos(hr)]
        return np.concatenate([head, self.emb[cart].mean(axis=0)])
