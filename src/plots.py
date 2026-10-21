import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def uplift_bar(summary, path):
    fig, ax = plt.subplots(figsize=(7, 3.5))
    ax.bar(summary["policy"], summary["RPS_uplift_pct"], yerr=summary["RPS_uplift_pct_ci95"],
           color="#2a9d8f", capsize=4)
    ax.axhline(0, color="k", lw=0.8)
    ax.set_ylabel("Revenue per started order, uplift %")
    ax.set_title("Business result (includes orders that were not finished)")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def band_charts(bands, path):
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    for ax, col, title in ((axes[0], "conversion", "Conversion by price band"),
                           (axes[1], "value_per_suggestion", "Value per suggestion (so'm)")):
        pv = bands.pivot(index="band", columns="policy", values=col)
        pv.plot.bar(ax=ax, rot=15)
        ax.set_title(title)
        ax.set_xlabel("price band (so'm, quartiles)")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
