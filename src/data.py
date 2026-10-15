"""Load Online Retail II style transactions and turn them into baskets."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

ALIASES = {
    "invoice": ["invoice", "invoiceno", "invoice_no"],
    "item": ["stockcode", "stock_code", "item_id", "product_id"],
    "desc": ["description"],
    "qty": ["quantity"],
    "date": ["invoicedate", "invoice_date", "date", "timestamp"],
    "price": ["price", "unitprice", "unit_price"],
    "customer": ["customer id", "customerid", "customer_id", "user_id"],
}


@dataclass
class Basket:
    invoice: str
    customer: int          # -1 = guest checkout
    ts: pd.Timestamp
    items: np.ndarray      # vocabulary indices, unique, in row order


@dataclass
class Dataset:
    item_ids: list
    names: list
    prices: np.ndarray     # median unit price per vocabulary item
    train: list
    test: list

    @property
    def V(self) -> int:
        return len(self.item_ids)


def find_data_file(folder="data/raw") -> str:
    """Return the first .csv/.xlsx in data/raw (so the file name doesn't matter)."""
    files = sorted(list(Path(folder).glob("*.csv")) + list(Path(folder).glob("*.xlsx")))
    if not files:
        raise FileNotFoundError(f"No .csv/.xlsx file found in {folder}/ - put the Kaggle file there.")
    return str(files[0])


def load_raw(path) -> pd.DataFrame:
    path = Path(path)
    if path.suffix.lower() in (".xlsx", ".xls"):
        sheets = pd.read_excel(path, sheet_name=None, dtype=str)
        return pd.concat(sheets.values(), ignore_index=True)
    try:
        return pd.read_csv(path, dtype=str)
    except UnicodeDecodeError:
        return pd.read_csv(path, dtype=str, encoding="latin-1")


def clean_transactions(df: pd.DataFrame, filter_non_products: bool = True) -> pd.DataFrame:
    df = df.copy()
    df.columns = [c.strip().lower() for c in df.columns]
    rename = {}
    for target, names in ALIASES.items():
        for n in names:
            if n in df.columns:
                rename[n] = target
                break
    df = df.rename(columns=rename)
    missing = [c for c in ["invoice", "item", "qty", "date", "price"] if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns {missing}. Found: {list(df.columns)}")

    df["qty"] = pd.to_numeric(df["qty"], errors="coerce")
    df["price"] = pd.to_numeric(df["price"], errors="coerce")
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    before = len(df)
    df = df.dropna(subset=["invoice", "item", "qty", "price", "date"])
    if len(df) < before:
        print(f"[clean] dropped {before - len(df):,} rows with missing/unparseable values")
    df["invoice"] = df["invoice"].astype(str)
    df["item"] = df["item"].astype(str).str.strip()

    df = df[~df["invoice"].str.upper().str.startswith("C")]   # cancellations
    df = df[(df["qty"] > 0) & (df["price"] > 0)]
    if filter_non_products:  # POST, DOT, M, BANK CHARGES, ... are not products
        df = df[df["item"].str.match(r"^\d{5}")]

    if "customer" in df.columns:
        df["customer"] = pd.to_numeric(df["customer"], errors="coerce").fillna(-1).astype(int)
    else:
        df["customer"] = -1
    if "desc" not in df.columns:
        df["desc"] = ""
    df["desc"] = df["desc"].fillna("").astype(str)
    return df.reset_index(drop=True)


def build_dataset(df: pd.DataFrame, top_k: int = 500, train_frac: float = 0.7,
                  min_test_items: int = 3) -> Dataset:
    """Chronological split by invoice; vocabulary = top-k items of the TRAIN period."""
    inv = df.groupby("invoice", sort=False).agg(ts=("date", "min"), customer=("customer", "first"))
    inv = inv.sort_values("ts", kind="stable")
    n_train = int(len(inv) * train_frac)
    train_inv = set(inv.index[:n_train])

    tr = df[df["invoice"].isin(train_inv)]
    counts = tr.drop_duplicates(["invoice", "item"])["item"].value_counts()
    vocab = counts.index[:top_k].tolist()
    index = {it: i for i, it in enumerate(vocab)}

    fallback = df["price"].median()
    prices = tr.groupby("item")["price"].median().reindex(vocab).fillna(fallback).to_numpy(float)
    names = df.drop_duplicates("item").set_index("item")["desc"].reindex(vocab).fillna("").tolist()

    d = df[df["item"].isin(index)].copy()
    d["idx"] = d["item"].map(index)
    grouped = {k: v.to_numpy() for k, v in d.groupby("invoice", sort=False)["idx"]}

    train, test = [], []
    for pos, (invoice, row) in enumerate(inv.iterrows()):
        arr = grouped.get(invoice)
        if arr is None:
            continue
        items = pd.unique(arr)  # dedupe, keep row order
        b = Basket(invoice, int(row["customer"]), row["ts"], items)
        if pos < n_train:
            if len(items) >= 2:
                train.append(b)
        elif len(items) >= min_test_items:
            test.append(b)
    return Dataset(vocab, names, prices, train, test)


def load_dataset(path, top_k=500, train_frac=0.7) -> Dataset:
    return build_dataset(clean_transactions(load_raw(path)), top_k, train_frac)
