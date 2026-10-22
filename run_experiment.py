"""Run the simulation from SAVED models (train.py first). Writes CSVs and charts to results/.

  python run_experiment.py --slots 10 --price-limit 60000 --s 0.15
"""
import argparse
from pathlib import Path

from src.artifacts import load_artifacts
from src.config import SimConfig, som
from src.experiment import run_simulation
from src.plots import band_charts, uplift_bar


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifacts", default="artifacts")
    ap.add_argument("--out", default="results")
    ap.add_argument("--slots", type=int, default=10)
    ap.add_argument("--price-limit", type=float, default=float("inf"), help="so'm")
    ap.add_argument("--s", type=float, default=0.15, help="price sensitivity")
    ap.add_argument("--p-hit", type=float, default=0.5)
    ap.add_argument("--p-miss", type=float, default=0.03)
    ap.add_argument("--finish", type=float, default=0.80, help="orders that finish w/o accepts")
    ap.add_argument("--stop", type=float, default=0.30, help="accepters who stop (average)")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--max-orders", type=int, default=3000)
    a = ap.parse_args()

    b = load_artifacts(a.artifacts)
    cfg = SimConfig(n_slots=a.slots, price_limit=a.price_limit, price_sensitivity=a.s,
                    p_hit=a.p_hit, p_miss=a.p_miss, finish_base=a.finish, stop_after_accept=a.stop)
    summary, bands = run_simulation(b, cfg, a.seeds, a.max_orders)
    out = Path(a.out)
    out.mkdir(exist_ok=True)
    summary.to_csv(out / "summary.csv", index=False)
    bands.to_csv(out / "price_bands.csv", index=False)
    uplift_bar(summary, out / "uplift.png")
    band_charts(bands, out / "price_bands.png")
    print(f"Test orders: {min(a.max_orders, len(b.test)):,} x {a.seeds} repeats | "
          f"{a.slots} suggestions | price limit "
          f"{'none' if a.price_limit == float('inf') else som(a.price_limit)} | s={a.s}")
    for _, r in summary.iterrows():
        print(f"{r.policy:28s} AOV {som(r.AOV_base)} -> {som(r.AOV_policy)} "
              f"({r.AOV_uplift_pct:+.1f}%) | accept {100 * r.orders_with_accept:.1f}% "
              f"| finish {100 * r.finish_rate:.1f}% | revenue/started order {r.RPS_uplift_pct:+.1f}%")
    print(f"Saved to {out}/")


if __name__ == "__main__":
    main()
