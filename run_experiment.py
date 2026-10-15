"""Run the full comparison.  Example:
    python -m src.synthetic                       # only if you have no real data yet
    python run_experiment.py --data data/raw/online_retail_II.csv
"""
import argparse
from dataclasses import replace
from pathlib import Path

import pandas as pd

from src.config import SimConfig
from src.data import find_data_file, load_dataset
from src.experiment import aggregate, evaluate, prepare
from src.plots import bar_uplift, learning_curves, sensitivity


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default=None, help="CSV/XLSX path (default: first file in data/raw)")
    p.add_argument("--out", default="results")
    p.add_argument("--top-k", type=int, default=500, help="vocabulary size (most frequent items)")
    p.add_argument("--seeds", type=int, default=5)
    p.add_argument("--min-support", type=float, default=0.005)
    p.add_argument("--min-confidence", type=float, default=0.1)
    p.add_argument("--arms", type=int, default=100, help="bandit arms = top-N popular items")
    p.add_argument("--alpha", type=float, default=0.1, help="LinUCB exploration strength")
    p.add_argument("--max-test", type=int, default=10000)
    p.add_argument("--slots", type=int, default=1)
    p.add_argument("--p-hit", type=float, default=0.5)
    p.add_argument("--p-miss", type=float, default=0.03)
    p.add_argument("--price-sensitivity", type=float, default=0.15)
    p.add_argument("--sensitivity", action="store_true", help="also run one-at-a-time sensitivity")
    a = p.parse_args()

    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    ds = load_dataset(a.data or find_data_file(), a.top_k)
    prep = prepare(ds, a.min_support, a.min_confidence, a.arms)
    cfg = SimConfig(p_hit=a.p_hit, p_miss=a.p_miss, price_sensitivity=a.price_sensitivity, slots=a.slots)

    df, curves = evaluate(prep, cfg, a.seeds, a.alpha, a.max_test)
    summary = aggregate(df)
    df.to_csv(out / "per_seed.csv", index=False)
    summary.to_csv(out / "summary.csv", index=False)
    try:
        md = summary.round(4).to_markdown(index=False)
    except ImportError:  # tabulate not installed
        md = summary.round(4).to_string(index=False)
    (out / "summary.md").write_text(md)
    bar_uplift(summary, out / "aov_uplift.png")
    learning_curves(curves, out / "learning_curves.png")
    print(summary.round(3).to_string(index=False))

    if a.sensitivity:
        grid = {"p_hit": [0.3, 0.5, 0.7], "p_miss": [0.01, 0.03, 0.08], "price_sensitivity": [0.0, 0.15, 0.4]}
        rows = []
        for param, values in grid.items():
            for v in values:
                d, _ = evaluate(prep, replace(cfg, **{param: v}), max(2, a.seeds // 2), a.alpha, a.max_test)
                s = aggregate(d); s["param"], s["value"] = param, v
                rows.append(s)
        sens = pd.concat(rows); sens.to_csv(out / "sensitivity.csv", index=False)
        sensitivity(sens, out / "sensitivity.png")
        print("Saved sensitivity analysis.")


if __name__ == "__main__":
    main()
