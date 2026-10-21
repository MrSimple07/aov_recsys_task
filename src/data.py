"""Load + merge transaction files, clean them, build baskets and a time split."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import NamedTuple

import numpy as np
import pandas as pd

ALIASES = {
    "invoice": ["invoice", "invoiceno", "order_id", "orderid", "order"],
    "item": ["stockcode", "stock_code", "sku", "item", "product_id", "productid"],
    "desc": ["description", "product_name", "name"],
    "qty": ["quantity", "qty"],
    "date": ["invoicedate", "date", "timestamp", "order_date", "created_at"],
    "price": ["price", "unitprice", "unit_price"],
    "customer": ["customer id", "customerid", "customer_id", "client_id", "user_id"],
}
_WANTED = {a for names in ALIASES.values() for a in names}
CACHE_DIR = Path("data/cache")      # Excel is slow: it is converted once and cached here


def _norm(c) -> str:
    return str(c).replace("\ufeff", "").replace("ï»¿", "").strip().lower()


class Basket(NamedTuple):
    ts: int               # order timestamp, ns since epoch
    items: np.ndarray     # unique vocabulary indices
    customer: int = -1    # -1 = guest / unknown


@dataclass
class Dataset:
    item_ids: list
    names: list
    prices: np.ndarray        # so'm (after conversion)
    train_counts: np.ndarray  # number of train orders containing each item
    train: list               # list[Basket]
    test: list                # list[Basket]
    info: dict = field(default_factory=dict)

    @property
    def V(self) -> int:
        return len(self.prices)


def find_data_files(src) -> list[Path]:
    """A folder (all csv/xlsx/parquet inside), a single file, or a list of them."""
    srcs = src if isinstance(src, (list, tuple)) else [src]
    out: list[Path] = []
    for s in srcs:
        p = Path(s)
        if p.is_dir():
            out += sorted(f for f in p.iterdir()
                          if f.suffix.lower() in (".csv", ".xlsx", ".xls", ".parquet"))
        elif p.exists():
            out.append(p)
        else:
            raise FileNotFoundError(p)
    if not out:
        raise FileNotFoundError(f"No csv/xlsx/parquet files found in {src}")
    return out


def _read_excel_cached(path: Path) -> pd.DataFrame:
    """Read ALL sheets (online_retail_II.xlsx has 'Year 2009-2010' and 'Year 2010-2011').

    Reading ~1M Excel rows takes minutes, so the result is cached as parquet in data/cache/
    (keyed by file name, size and modified time) and later runs take seconds."""
    st = path.stat()
    cache = CACHE_DIR / f"{path.stem}_{st.st_size}_{int(st.st_mtime)}.parquet"
    if cache.exists():
        print(f"  {path.name}: using cache {cache}")
        return pd.read_parquet(cache)
    print(f"  {path.name}: reading Excel (slow - several minutes for the full file, "
          f"cached afterwards) ...")
    keep = lambda c: _norm(c) in _WANTED  # noqa: E731
    sheets = pd.read_excel(path, sheet_name=None, dtype=str, usecols=keep)
    for name, sh in sheets.items():
        print(f"    sheet '{name}': {len(sh):,} rows")
    df = pd.concat(sheets.values(), ignore_index=True)
    try:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        df.to_parquet(cache, index=False)
    except Exception as e:  # pyarrow missing / read-only disk: just skip the cache
        print(f"    (could not write cache: {e})")
    return df


def load_raw(path) -> pd.DataFrame:
    """Read one file; only the columns we know are kept (saves memory on big files)."""
    path = Path(path)
    suf = path.suffix.lower()
    if suf == ".parquet":
        df = pd.read_parquet(path)
    elif suf in (".xlsx", ".xls"):
        df = _read_excel_cached(path)
    else:
        keep = lambda c: _norm(c) in _WANTED  # noqa: E731
        try:
            df = pd.read_csv(path, dtype=str, usecols=keep, encoding="utf-8-sig")
        except UnicodeDecodeError:
            df = pd.read_csv(path, dtype=str, usecols=keep, encoding="latin-1")
    df.columns = [_norm(c) for c in df.columns]
    return df


def clean_transactions(df: pd.DataFrame, item_regex: str | None = r"^\d{5}",
                       cancel_prefix: str | None = "C") -> pd.DataFrame:
    """Standard names, drop cancellations / non-products / non-positive rows.

    item_regex=None keeps every item code (use for non-Online-Retail data).
    cancel_prefix=None keeps invoices starting with 'C'.
    """
    df = df.copy()
    df.columns = [_norm(c) for c in df.columns]
    rename = {}
    for std, names in ALIASES.items():
        for n in names:
            if n in df.columns:
                rename[n] = std
                break
    df = df.rename(columns=rename)
    missing = [c for c in ("invoice", "item", "qty", "date", "price") if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns {missing}; found {list(df.columns)}")
    if "customer" not in df.columns:
        df["customer"] = np.nan
    if "desc" not in df.columns:
        df["desc"] = ""
    df = df[["invoice", "item", "desc", "qty", "date", "price", "customer"]].copy()
    df["invoice"] = df["invoice"].astype(str).str.strip()
    df["item"] = df["item"].astype(str).str.strip()
    df["desc"] = df["desc"].fillna("").astype(str).str.strip()
    df["qty"] = pd.to_numeric(df["qty"], errors="coerce")
    df["price"] = pd.to_numeric(df["price"], errors="coerce")
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["customer"] = pd.to_numeric(df["customer"], errors="coerce").fillna(-1).astype(int)
    df = df.dropna(subset=["invoice", "item", "qty", "price", "date"])
    if cancel_prefix:
        df = df[~df["invoice"].str.upper().str.startswith(cancel_prefix.upper())]
    df = df[(df["qty"] > 0) & (df["price"] > 0)]
    if item_regex:
        df = df[df["item"].str.match(item_regex)]
    return df.reset_index(drop=True)


def load_transactions(src, since: str | None = None, **clean_kw):
    """Load every file, clean, MERGE, and drop rows duplicated between files.

    The two Online Retail II files overlap (Dec 2010), so exact duplicate rows are removed.
    """
    files = find_data_files(src)
    frames = []
    for p in files:
        c = clean_transactions(load_raw(p), **clean_kw)
        print(f"  {p.name}: {len(c):,} clean rows, {c.invoice.nunique():,} orders, "
              f"{c.date.min().date()} -> {c.date.max().date()}")
        frames.append(c)
    df = pd.concat(frames, ignore_index=True)
    n0 = len(df)
    df = df.drop_duplicates(["invoice", "item", "qty", "price", "date", "customer"])
    if n0 - len(df):
        print(f"  removed {n0 - len(df):,} duplicate rows shared between files")
    if since:
        df = df[df["date"] >= pd.Timestamp(since)]
        print(f"  kept orders since {since}: {len(df):,} rows")
    if df.empty:
        raise ValueError("No data left after cleaning / date filter")
    return df.reset_index(drop=True), files


def build_dataset(df: pd.DataFrame, top_k: int | None = None, train_frac: float = 0.7,
                  min_test_items: int = 3, fx: float = 1.0, round_to: int = 100) -> Dataset:
    """Orders -> baskets of the top_k products. Chronological split (no leakage).

    top_k=None (default) uses EVERY product seen in the training orders; a number keeps only
    the most popular ones (faster, but the carts lose the other products - see row_coverage).

    Prices: per-item median unit price over TRAIN orders, times fx, rounded to
    `round_to` (so'm). Pass fx=1 when the data is already in so'm.
    """
    inv = df.groupby("invoice", sort=False)["date"].min().sort_values(kind="stable")
    n_train = int(len(inv) * train_frac)
    rank = pd.Series(np.arange(len(inv)), index=inv.index)
    tr = df[df["invoice"].isin(set(inv.index[:n_train]))]

    counts = tr.drop_duplicates(["invoice", "item"])["item"].value_counts()
    vocab = counts.index[:top_k].tolist()
    index = {it: i for i, it in enumerate(vocab)}

    raw_p = tr.groupby("item")["price"].median().reindex(vocab).fillna(df["price"].median())
    prices = np.maximum(np.round(raw_p.to_numpy(float) * fx / round_to) * round_to, round_to)
    names = (df.drop_duplicates("item").set_index("item")["desc"]
             .reindex(vocab).fillna("").tolist())

    d = df.loc[df["item"].isin(index), ["invoice", "item"]].copy()
    d["idx"] = d["item"].map(index)
    d["rank"] = d["invoice"].map(rank)
    d = d.sort_values("rank", kind="stable")
    idx, rk = d["idx"].to_numpy(), d["rank"].to_numpy()
    cuts = np.flatnonzero(np.diff(rk)) + 1
    starts, ends = np.r_[0, cuts], np.r_[cuts, len(rk)]
    ts_ns = inv.to_numpy().astype("datetime64[ns]").astype("int64")
    cust = df.groupby("invoice", sort=False)["customer"].first().reindex(inv.index).to_numpy()

    train, test = [], []
    for s, e in zip(starts, ends):
        r = int(rk[s])
        arr = idx[s:e]
        _, first = np.unique(arr, return_index=True)
        items = arr[np.sort(first)].astype(np.int32)
        if r < n_train:
            if len(items) >= 2:
                train.append(Basket(int(ts_ns[r]), items, int(cust[r])))
        elif len(items) >= min_test_items:
            test.append(Basket(int(ts_ns[r]), items, int(cust[r])))
    if not train or not test:
        raise ValueError(f"Need train and test orders, got {len(train)} / {len(test)}")

    train_counts = np.bincount(np.concatenate([b.items for b in train]),
                               minlength=len(vocab)).astype(np.int64)
    info = dict(rows=len(df), orders=int(len(inv)), products_total=int(df["item"].nunique()),
                date_min=str(df["date"].min().date()), date_max=str(df["date"].max().date()),
                n_train=len(train), n_test=len(test), vocab_size=len(vocab),
                row_coverage=round(len(d) / len(df), 4),
                test_start=str(pd.Timestamp(test[0].ts).date()), fx=fx)
    return Dataset(vocab, names, prices, train_counts, train, test, info)


def even_subsample(items, k: int):
    """Keep at most k items, evenly spread (keeps chronological order)."""
    if len(items) <= k:
        return list(items)
    return [items[i] for i in np.linspace(0, len(items) - 1, k).astype(int)]
