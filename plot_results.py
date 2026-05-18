import warnings
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
from matplotlib.ticker import MultipleLocator

warnings.filterwarnings("ignore")
matplotlib.rcParams.update({
    "font.family":       "DejaVu Sans",
    "font.size":         14,
    "axes.titlesize":    16,
    "axes.labelsize":    15,
    "axes.labelcolor":   "black",
    "axes.edgecolor":    "black",
    "axes.linewidth":    1.4,
    "xtick.labelsize":   13,
    "ytick.labelsize":   13,
    "xtick.color":       "black",
    "ytick.color":       "black",
    "xtick.major.width": 1.2,
    "ytick.major.width": 1.2,
    "xtick.major.size":  5,
    "ytick.major.size":  5,
    "legend.fontsize":   12,
    "legend.frameon":    True,
    "legend.edgecolor":  "black",
    "legend.framealpha": 0.95,
    "figure.dpi":        150,
    "savefig.dpi":       300,
    "savefig.bbox":      "tight",
})

_C4_KEY = "Real+Synthetic"

BLUES = {
    "Real→Cross":  "#1f4e79",
    "Interpolation": "#2e75b6",
    "VAE":         "#5ba3d9",
    "GAN":         "#9dc3e6",
    _C4_KEY:       "#70ad47",
    "Interp + DA": "#c6a600",
}

DATA_FILE = "cross_dataset_results_synthetic_agg.csv"


def load() -> pd.DataFrame:
    df = pd.read_csv(DATA_FILE)
    if df.empty:
        raise FileNotFoundError(f"{DATA_FILE} is empty")
    return df


def fig1_model_comparison(df: pd.DataFrame):
    rc  = df[df["Experiment_Type"] == "Real→Cross"]
    sc  = df[df["Experiment_Type"] == "Synthetic→Cross"]

    conditions = [
        ("Real→Cross",    rc,                                     BLUES["Real→Cross"]),
        ("Interpolation", sc[sc["Synth_Method"] == "Interpolation"], BLUES["Interpolation"]),
        ("VAE",           sc[sc["Synth_Method"] == "VAE"],           BLUES["VAE"]),
        ("GAN",           sc[sc["Synth_Method"] == "GAN"],           BLUES["GAN"]),
    ]

    model_order = ["LDA", "Logistic\nRegression", "Random\nForest", "SVM-RBF", "XGBoost"]
    model_keys  = ["LDA", "Logistic Regression",  "Random Forest",  "SVM-RBF", "XGBoost"]
    n_models  = len(model_keys)
    n_conds   = len(conditions)
    bar_w     = 0.18
    group_gap = 0.04
    offsets   = np.linspace(
        -(n_conds - 1) / 2 * (bar_w + group_gap),
         (n_conds - 1) / 2 * (bar_w + group_gap),
        n_conds
    )
    x = np.arange(n_models)

    fig, ax = plt.subplots(figsize=(13, 7))

    for offset, (label, sub, color) in zip(offsets, conditions):
        stats = sub.groupby("Model")["AUC"].agg(["mean", "sem"]).reindex(model_keys)
        means = stats["mean"].values
        errs  = stats["sem"].values
        bars  = ax.bar(
            x + offset, means, bar_w,
            color=color, edgecolor="black", linewidth=0.8,
            label=label, zorder=3
        )
        ax.errorbar(
            x + offset, means, yerr=errs,
            fmt="none", ecolor="black", elinewidth=1.2,
            capsize=4, capthick=1.2, zorder=4
        )

    ax.axhline(0.5, color="black", linestyle="--", linewidth=1.2,
               alpha=0.6, zorder=2, label="Chance (AUC = 0.50)")

    ax.set_xticks(x)
    ax.set_xticklabels(model_order, fontsize=13, color="black")
    ax.set_ylabel("Mean AUC (cross-dataset)", fontsize=15, color="black")
    ax.set_xlabel("Classifier", fontsize=15, color="black")
    ax.set_title(
        "Classifier Performance by Training Condition\n"
        "(Cross-Dataset Transfer, All Source–Target Pairs)",
        fontsize=16, color="black", pad=14
    )
    ax.set_ylim(0, 0.65)
    ax.yaxis.set_minor_locator(MultipleLocator(0.01))
    ax.yaxis.set_major_locator(MultipleLocator(0.05))
    ax.grid(axis="y", which="major", linestyle="--", linewidth=0.6,
            alpha=0.5, zorder=1)
    ax.set_axisbelow(True)

    legend = ax.legend(
        title="Training Data",
        title_fontsize=13,
        loc="upper left",
        bbox_to_anchor=(1.01, 1.0),
        borderaxespad=0,
        frameon=True,
    )
    legend.get_title().set_fontweight("bold")

    plt.tight_layout()
    out = "fig_model_comparison.png"
    plt.savefig(out)
    plt.close()
    print(f"Saved → {out}")


def _best_per_pair(df_sub):
    """Mean ± SEM of best-model AUC per source-target pair, matching Table 3 aggregation."""
    best = df_sub.groupby(["Train_Dataset", "Test_Dataset"])["AUC"].max()
    return best.mean(), best.sem()


def fig2_condition_comparison(df: pd.DataFrame):
    rc  = df[df["Experiment_Type"] == "Real→Cross"]
    sc  = df[df["Experiment_Type"] == "Synthetic→Cross"]
    aug = df[df["Experiment_Type"] == "Real+Synthetic→Cross"]
    da  = df[df["Experiment_Type"] == "Real+SyntheticDA→Cross"]

    conditions = [
        ("Real→Cross\n(C2 Baseline)",       rc,                                        BLUES["Real→Cross"]),
        ("Interpolation\n(C3)",             sc[sc["Synth_Method"] == "Interpolation"], BLUES["Interpolation"]),
        ("VAE\n(C3)",                       sc[sc["Synth_Method"] == "VAE"],           BLUES["VAE"]),
        ("GAN\n(C3)",                       sc[sc["Synth_Method"] == "GAN"],           BLUES["GAN"]),
        ("Real+Synthetic\n(C4)",            aug,                                       BLUES[_C4_KEY]),
        ("Real+DA-Synthetic\n(C5)",         da,                                        BLUES["Interp + DA"]),
    ]

    labels             = [c[0] for c in conditions]
    means, sems        = zip(*[_best_per_pair(c[1]) for c in conditions])
    means, sems        = list(means), list(sems)
    colors             = [c[2] for c in conditions]

    x     = np.arange(len(conditions))
    bar_w = 0.55

    _, ax = plt.subplots(figsize=(16, 8))

    ax.bar(x, means, bar_w, color=colors, edgecolor="black", linewidth=1.0, zorder=3)
    ax.errorbar(x, means, yerr=sems,
                fmt="none", ecolor="black", elinewidth=1.4,
                capsize=6, capthick=1.4, zorder=4)

    for xi, (m, s) in enumerate(zip(means, sems)):
        ax.text(xi, m + s + 0.003, f"{m:.3f}",
                ha="center", va="bottom", fontsize=12,
                fontweight="bold", color="black")

    ax.axhline(0.5, color="black", linestyle="--", linewidth=1.3,
               alpha=0.7, zorder=2)

    baseline = means[0]
    vae_mean = means[2]
    delta    = vae_mean - baseline
    bx, bx2  = x[0], x[2]
    by = max(means[:3]) + max(sems[:3]) + 0.022
    ax.annotate("", xy=(bx2, by), xytext=(bx, by),
                arrowprops={"arrowstyle": "<->", "color": "black", "lw": 1.2})
    ax.text((bx + bx2) / 2, by + 0.003,
            f"ΔAUC = {delta:+.3f}", ha="center", va="bottom",
            fontsize=11, color="black")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=13, color="black",
                       fontweight="bold", rotation=20, ha="right",
                       rotation_mode="anchor")
    ax.set_ylabel("Mean AUC (cross-dataset)", fontsize=18, color="black", fontweight="bold")
    ax.set_xlabel("Training Condition", fontsize=18, color="black", fontweight="bold")
    ax.set_title(
        "Cross-Dataset AUC by Training Condition\n"
        "(Best model per source–target pair, mean ± SEM across 12 pairs)",
        fontsize=20, color="black", fontweight="bold", pad=16
    )
    ax.set_ylim(0, 0.70)
    ax.yaxis.set_minor_locator(MultipleLocator(0.01))
    ax.yaxis.set_major_locator(MultipleLocator(0.05))
    ax.grid(axis="y", which="major", linestyle="--", linewidth=0.6,
            alpha=0.5, zorder=1)
    ax.set_axisbelow(True)

    patches = [
        mpatches.Patch(facecolor=BLUES["Real→Cross"],      edgecolor="black", label="C2 — Real"),
        mpatches.Patch(facecolor=BLUES["Interpolation"],   edgecolor="black", label="C3 — Synthetic (Interpolation)"),
        mpatches.Patch(facecolor=BLUES["VAE"],             edgecolor="black", label="C3 — Synthetic (VAE)"),
        mpatches.Patch(facecolor=BLUES["GAN"],             edgecolor="black", label="C3 — Synthetic (GAN)"),
        mpatches.Patch(facecolor=BLUES[_C4_KEY],           edgecolor="black", label="C4 — Real + Synthetic"),
        mpatches.Patch(facecolor=BLUES["Interp + DA"],     edgecolor="black", label="C5 — Real + Domain-Adapted Synthetic"),
        mlines.Line2D([], [], color="black", linestyle="--", linewidth=1.3,
                      alpha=0.7, label="Chance (AUC = 0.50)"),
    ]
    legend = ax.legend(
        handles=patches,
        title="Training Condition",
        title_fontsize=13,
        loc="upper left",
        bbox_to_anchor=(1.01, 1.0),
        borderaxespad=0,
        frameon=True,
    )
    legend.get_title().set_fontweight("bold")

    plt.tight_layout()
    out = "fig_condition_comparison.png"
    plt.savefig(out)
    plt.close()
    print(f"Saved → {out}")


if __name__ == "__main__":
    df = load()
    fig1_model_comparison(df)
    fig2_condition_comparison(df)
    print("Done.")
