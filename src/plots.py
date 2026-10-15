import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def bar_uplift(summary, path):
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(summary["policy"], summary["uplift_pct"], yerr=summary["uplift_pct_ci95"], capsize=4,
           color="#4c72b0")
    ax.set_ylabel("AOV uplift vs. no recommendation (%)")
    ax.set_title("Average order value uplift by policy (95% CI over seeds)")
    plt.setp(ax.get_xticklabels(), rotation=25, ha="right")
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def learning_curves(curves, path, window=500):
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for name, arrs in curves.items():
        m = np.mean(arrs, axis=0)
        if len(m) <= window:
            continue
        roll = np.convolve(m, np.ones(window) / window, mode="valid")
        ax.plot(np.arange(window, len(m) + 1), roll, label=name)
    ax.set_xlabel("Session (chronological)"); ax.set_ylabel(f"Rolling mean added value ({window} sessions)")
    ax.set_title("Online performance over time"); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def sensitivity(df, path):
    params = list(df["param"].unique())
    fig, axes = plt.subplots(1, len(params), figsize=(4.2 * len(params), 4), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, p in zip(axes, params):
        sub = df[df["param"] == p]
        for name, g in sub.groupby("policy", sort=False):
            ax.plot(g["value"], g["uplift_pct"], marker="o", label=name)
        ax.set_title(p); ax.set_xlabel("value")
    axes[0].set_ylabel("AOV uplift (%)"); axes[-1].legend(fontsize=7)
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)
