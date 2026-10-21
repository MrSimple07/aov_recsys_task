"""Train once, save to artifacts/. Used by train.py and the tests."""
from __future__ import annotations

import time
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import scipy

from .artifacts import save_artifacts
from .config import ROUND_TO
from .data import build_dataset, even_subsample
from .models import PairRules, SequentialModel, fit_embeddings


def train(df: pd.DataFrame, out="artifacts", top_k: int | None = None, train_frac: float = 0.7,
          fx: float = 1.0, currency: str = "GBP", min_count: int | None = None,
          min_conf: float = 0.05, files=()) -> dict:
    t0 = time.time()
    ds = build_dataset(df, top_k=top_k, train_frac=train_frac, fx=fx, round_to=ROUND_TO)
    print(f"  dataset: {ds.V:,} products covering {100 * ds.info['row_coverage']:.0f}% of purchased "
          f"items, {len(ds.train):,} train / {len(ds.test):,} test orders ({time.time() - t0:.0f}s)")
    print(f"  (the files hold {ds.info['rows']:,} rows = line items, but only "
          f"{ds.info['orders']:,} orders)")
    n_train = len(ds.train)
    mc = min_count if min_count is not None else max(5, int(0.0005 * n_train))
    t1 = time.time()
    rules = PairRules.fit(ds.train, ds.V, min_count=mc, min_conf=min_conf)
    print(f"  pair rules: {rules.n_rules:,} rules, min_count={mc} ({time.time() - t1:.0f}s)")
    t2 = time.time()
    seq = SequentialModel.fit(ds.train, ds.V)
    emb = fit_embeddings(ds.train, ds.V)
    print(f"  sequential model + bandit embeddings ({time.time() - t2:.0f}s)")
    warm = even_subsample(ds.train, 5000)
    meta = dict(
        created=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        source_currency=currency, fx_to_som=fx, price_unit="so'm",
        files=[f"{getattr(f, 'name', f)}" for f in files],
        top_k=top_k, train_frac=train_frac, min_count=mc, min_conf=min_conf,
        n_rules=rules.n_rules, train_seconds=round(time.time() - t0, 1),
        median_price=float(np.median(ds.prices)),
        versions=dict(numpy=np.__version__, pandas=pd.__version__, scipy=scipy.__version__),
        **ds.info)
    save_artifacts(out, ds, rules, seq, emb, warm, meta)
    print(f"  saved to {out}/ in {time.time() - t0:.0f}s total")
    return meta
