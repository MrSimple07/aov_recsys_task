"""Plain-language reason a model shows each product ("why shown" column in the cart view).

Every model has its own reason, so the column changes with the model, the cart and (for the
Sequential model) the shopper's history:

* Association rules - which cart item it is bought with, how often, and the lift
* Sequential        - which earlier item (cart or past order) it usually follows
* Contextual bandit - learned reward + exploration bonus for this cart
* Fixed list        - its slot on the list
* Random            - random pick
"""
from __future__ import annotations

import numpy as np

from .models import (FixedListPolicy, FrequentlyBoughtTogether, LinUCBPolicy, RandomPolicy,
                     SequentialPolicy)

FILLER = "Popular filler (no signal for this cart)"


def _short(cat, i: int, width: int = 30) -> str:
    n = " ".join(cat.label(int(i)).split()).replace(" ,", ",")
    return n if len(n) <= width else n[: width - 1].rstrip() + "…"


def sequential_contributions(pol: SequentialPolicy, cart, customer: int, recs, lam=0.8,
                             hist_len=8, max_state=10):
    """-> (state items, contribution matrix [state x recs]). Mirrors SequentialModel.scores."""
    cart = np.asarray(cart, dtype=int)
    hist = pol.model.history.get(customer)
    seq = cart if hist is None else np.concatenate([hist[-hist_len:], cart])
    seq = seq[-max_state:]
    w = lam ** np.arange(len(seq) - 1, -1, -1)
    P = pol.model.P[seq][:, list(recs)].toarray()
    return seq, P * w[:, None]


def why_shown(pol, cat, cart, recs, ctx=None, customer: int = -1) -> list:
    """One short sentence per recommended product, in the order of `recs`."""
    cart = np.asarray(cart, dtype=int)
    recs = [int(i) for i in recs]
    if not recs:
        return []

    if isinstance(pol, FrequentlyBoughtTogether):
        conf = pol.rules.conf[cart][:, recs].toarray()
        out = []
        for k, j in enumerate(recs):
            col = conf[:, k]
            d = int(col.argmax())
            if col[d] <= 0:
                out.append(FILLER)
                continue
            lift = col[d] / max(float(cat.pop_share[j]), 1e-9)
            out.append(f"With {_short(cat, cart[d], 24)}: {100 * col[d]:.0f}% of orders ({lift:.1f}×)")
        return out

    if isinstance(pol, SequentialPolicy):
        seq, contrib = sequential_contributions(pol, cart, customer, recs)
        n_past = len(seq) - min(len(cart), len(seq))
        out = []
        for k in range(len(recs)):
            col = contrib[:, k]
            d = int(col.argmax())
            if col[d] <= 0:
                out.append(FILLER)
            else:
                src = "past order" if d < n_past else "cart"
                out.append(f"After {_short(cat, seq[d], 24)} ({src})")
        return out

    if isinstance(pol, LinUCBPolicy):
        if ctx is None or ctx.vec is None:
            return ["Bandit needs a context"] * len(recs)
        x = ctx.vec
        theta = np.einsum("mij,mj->mi", pol.A_inv, pol.b)
        out = []
        for j in recs:
            a = pol.arm_pos.get(j)
            if a is None:
                out.append(FILLER)
                continue
            est = float(theta[a] @ x)
            bonus = pol.alpha * float(np.sqrt(max(x @ pol.A_inv[a] @ x, 0.0)))
            out.append(f"Learned reward {est:.3f} + exploration {bonus:.3f}")
        return out

    if isinstance(pol, FixedListPolicy):
        slot = {int(i): k + 1 for k, i in enumerate(pol.items)}
        return [f"Fixed list, slot {slot[j]}" if j in slot else "Best seller (fills the list)"
                for j in recs]

    if isinstance(pol, RandomPolicy):
        return ["Random pick"] * len(recs)

    return [""] * len(recs)
