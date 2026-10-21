"""Save / load everything the app and the simulation need (no pickle, no retraining)."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import load_npz, save_npz

from .data import Basket, Dataset
from .models import Catalog, ContextBuilder, PairRules, SequentialModel

SCHEMA = 2


@dataclass
class Bundle:
    meta: dict
    cat: Catalog
    rules: PairRules
    seq: SequentialModel
    ctxb: ContextBuilder
    test: list
    warm: list          # a sample of past (train) orders, used to pre-train the warm bandit


def _save_baskets(path, baskets):
    flat = np.concatenate([b.items for b in baskets]).astype(np.int32)
    offs = np.r_[0, np.cumsum([len(b.items) for b in baskets])].astype(np.int64)
    np.savez_compressed(path, flat=flat, offsets=offs,
                        ts=np.array([b.ts for b in baskets], dtype=np.int64),
                        customer=np.array([b.customer for b in baskets], dtype=np.int64))


def _load_baskets(path):
    z = np.load(path)
    flat, offs, ts, cu = z["flat"], z["offsets"], z["ts"], z["customer"]
    return [Basket(int(ts[i]), flat[offs[i]:offs[i + 1]].astype(int), int(cu[i]))
            for i in range(len(ts))]


def save_artifacts(out, ds: Dataset, rules: PairRules, seq: SequentialModel, emb: np.ndarray,
                   warm: list, meta: dict) -> Path:
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"item_id": ds.item_ids, "name": ds.names, "price": ds.prices,
                  "train_count": ds.train_counts}).to_csv(out / "catalog.csv", index=False)
    save_npz(out / "pair_rules.npz", rules.conf)
    save_npz(out / "sequential.npz", seq.P)
    cust = np.array(sorted(seq.history), dtype=np.int64)
    parts = [seq.history[c] for c in cust]
    np.savez_compressed(out / "history.npz", customers=cust,
                        offsets=np.r_[0, np.cumsum([len(p) for p in parts])].astype(np.int64),
                        flat=(np.concatenate(parts) if parts else np.array([], dtype=np.int64)))
    np.save(out / "embeddings.npy", emb)
    _save_baskets(out / "test_baskets.npz", ds.test)
    _save_baskets(out / "warm_baskets.npz", warm)
    (out / "meta.json").write_text(json.dumps({"schema": SCHEMA, **meta}, indent=2, default=str))
    return out


def load_artifacts(path) -> Bundle:
    p = Path(path)
    meta = json.loads((p / "meta.json").read_text())
    if meta.get("schema") != SCHEMA:
        raise ValueError(f"Artifacts schema {meta.get('schema')} != {SCHEMA}: re-run train.py")
    c = pd.read_csv(p / "catalog.csv", dtype={"item_id": str, "name": str}, keep_default_na=False)
    cat = Catalog(c["item_id"].tolist(), c["name"].tolist(), c["price"].to_numpy(float),
                  c["train_count"].to_numpy(np.int64), int(meta["n_train"]))
    h = np.load(p / "history.npz")
    custs, offs, flat = h["customers"], h["offsets"], h["flat"].astype(int)   # read each ONCE
    history = {int(cu): flat[offs[i]:offs[i + 1]] for i, cu in enumerate(custs)}
    seq = SequentialModel(load_npz(p / "sequential.npz"), history)
    ctxb = ContextBuilder(np.load(p / "embeddings.npy"), cat.prices)
    return Bundle(meta, cat, PairRules(load_npz(p / "pair_rules.npz")), seq, ctxb,
                  _load_baskets(p / "test_baskets.npz"), _load_baskets(p / "warm_baskets.npz"))
