"""Train ONCE and save the models. The app and run_experiment.py only load them.

  python train.py                                   # reads every file in data/raw/ and merges them
  python train.py --data data/raw/online_retail_II.xlsx   # the Kaggle Excel file (2 sheets)
  python train.py --fx 16200                        # use another so'm per GBP rate
  python train.py --data my_orders.parquet --currency UZS --no-item-filter   # your own so'm data
"""
import argparse
import time

from src.config import FX_GBP_UZS, SOURCE_CURRENCY
from src.data import load_transactions
from src.training import train


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--data", nargs="+", default=["data/raw"], help="folder(s) or file(s)")
    ap.add_argument("--out", default="artifacts")
    ap.add_argument("--currency", default=SOURCE_CURRENCY, choices=["GBP", "UZS"],
                    help="currency of the raw prices (UZS = already in so'm, no conversion)")
    ap.add_argument("--fx", type=float, default=FX_GBP_UZS, help="so'm per 1 GBP")
    ap.add_argument("--since", default=None, help="keep orders from this date, e.g. 2020-01-01")
    ap.add_argument("--top-k", type=int, default=None,
                    help="keep only the K most popular products (default: ALL products)")
    ap.add_argument("--train-frac", type=float, default=0.7)
    ap.add_argument("--min-count", type=int, default=None, help="min orders per product pair")
    ap.add_argument("--no-item-filter", action="store_true",
                    help="keep all item codes (default keeps 5-digit product codes)")
    ap.add_argument("--no-cancel-filter", action="store_true",
                    help="keep invoices that start with 'C'")
    a = ap.parse_args()
    fx = 1.0 if a.currency == "UZS" else a.fx
    t = time.time()
    print("Loading and merging data ...")
    df, files = load_transactions(
        a.data, since=a.since, item_regex=None if a.no_item_filter else r"^\d{5}",
        cancel_prefix=None if a.no_cancel_filter else "C")
    print(f"Training (prices x {fx:,.0f} -> so'm) ...")
    train(df, a.out, top_k=a.top_k, train_frac=a.train_frac, fx=fx, currency=a.currency,
          min_count=a.min_count, files=files)
    print(f"Done in {time.time() - t:.0f}s. Next: streamlit run app.py")


if __name__ == "__main__":
    main()
