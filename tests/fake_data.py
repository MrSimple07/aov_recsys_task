"""TEST FIXTURE ONLY - tiny fake transactions in the Online Retail II layout.

Not used by train.py / the app (those only read your real files). It exists so the unit
tests run in seconds without the 90 MB dataset.

Structure planted in the data: item bundles inside categories (co-purchase signal),
customer-specific favourite categories, and a category-to-next-category chain
(cross-basket sequential signal). Results on this data are NOT scientifically meaningful.
"""
import numpy as np
import pandas as pd


def generate(n_items=300, n_cats=20, n_customers=1500, n_invoices=20000, seed=0,
             start="2020-01-01", days=365 * 3, price_scale=1.0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    item_cat = np.arange(n_items) % n_cats
    pop = 1.0 / (1 + rng.permutation(n_items)) ** 0.8
    cat_price = np.exp(rng.normal(1.2, 0.6, n_cats))
    price = cat_price[item_cat] * np.exp(rng.normal(0, 0.3, n_items)) * price_scale
    price = (np.round(price, -2).clip(100) if price_scale >= 100 else np.round(price, 2).clip(0.2))
    bundles = {c: [rng.choice(np.flatnonzero(item_cat == c), 4, replace=False) for _ in range(3)]
               for c in range(n_cats)}
    next_cat = rng.permutation(n_cats)
    fav = rng.integers(0, n_cats, size=(n_customers, 2))
    last_cat = {}

    start = pd.Timestamp(start)
    secs = np.sort(rng.integers(0, days * 24 * 3600, n_invoices))
    rows = []
    for k, sec in enumerate(secs):
        ts = start + pd.Timedelta(seconds=int(sec))
        ts = ts.replace(hour=int(rng.integers(8, 19)))
        guest = rng.random() < 0.15
        cust = -1 if guest else int(rng.integers(0, n_customers))
        if not guest and cust in last_cat and rng.random() < 0.6:
            cat = next_cat[last_cat[cust]]
        elif not guest:
            cat = fav[cust, rng.integers(0, 2)]
        else:
            cat = rng.integers(0, n_cats)
        size = 2 + rng.poisson(6)
        basket = set()
        if rng.random() < 0.6:
            for it in bundles[cat][rng.integers(0, 3)]:
                if rng.random() < 0.85:
                    basket.add(int(it))
        pool = np.flatnonzero(item_cat == cat)
        w = pop[pool] / pop[pool].sum()
        while len(basket) < size:
            basket.add(int(rng.choice(pool, p=w)) if rng.random() < 0.8 else int(rng.choice(n_items, p=pop / pop.sum())))
        if not guest:
            last_cat[cust] = cat
        items = list(basket)
        rng.shuffle(items)
        for it in items:
            rows.append((f"{500000 + k}", f"{10000 + it}", f"ITEM C{item_cat[it]} #{it}",
                         int(rng.integers(1, 7)), ts, price[it],
                         np.nan if guest else 12000.0 + cust, "United Kingdom"))
        if rng.random() < 0.02:  # a cancellation row, to exercise the cleaner
            rows.append((f"C{500000 + k}", f"{10000 + items[0]}", "CANCELLED", -1, ts, price[items[0]],
                         12000.0 + cust if not guest else np.nan, "United Kingdom"))
    return pd.DataFrame(rows, columns=["Invoice", "StockCode", "Description", "Quantity",
                                       "InvoiceDate", "Price", "Customer ID", "Country"])
