import warnings
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
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

BLUES = {
    "Real→Cross":          "#1f4e79",
    "Interpolation":       "#2e75b6",
    "VAE":                 "#5ba3d9",
    "GAN":                 "#9dc3e6",
    "Interp + DA":         "#c6a600",
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


def fig2_condition_comparison(df: pd.DataFrame):
    rc  = df[df["Experiment_Type"] == "Real→Cross"]
    sc  = df[df["Experiment_Type"] == "Synthetic→Cross"]
    da  = df[df["Experiment_Type"] == "Real+SyntheticDA→Cross"]

    conditions = [
        ("Real→Cross\n(Baseline)",         rc,                                        BLUES["Real→Cross"]),
        ("Interpolation\n→Cross",           sc[sc["Synth_Method"] == "Interpolation"], BLUES["Interpolation"]),
        ("VAE\n→Cross",                     sc[sc["Synth_Method"] == "VAE"],           BLUES["VAE"]),
        ("GAN\n→Cross",                     sc[sc["Synth_Method"] == "GAN"],           BLUES["GAN"]),
        ("Real + Interpolation\n+DA→Cross", da[da["Synth_Method"] == "Prototype-DA"], BLUES["Interp + DA"]),
    ]

    labels = [c[0] for c in conditions]
    means  = [c[1]["AUC"].mean() for c in conditions]
    sems   = [c[1]["AUC"].sem()  for c in conditions]
    colors = [c[2] for c in conditions]

    x = np.arange(len(conditions))
    bar_w = 0.55

    fig, ax = plt.subplots(figsize=(12, 7))

    bars = ax.bar(
        x, means, bar_w,
        color=colors, edgecolor="black", linewidth=1.0,
        zorder=3
    )
    ax.errorbar(
        x, means, yerr=sems,
        fmt="none", ecolor="black", elinewidth=1.4,
        capsize=6, capthick=1.4, zorder=4
    )

    for xi, (m, s) in enumerate(zip(means, sems)):
        ax.text(xi, m + s + 0.003, f"{m:.3f}",
                ha="center", va="bottom", fontsize=12,
                fontweight="bold", color="black")

    ax.axhline(0.5, color="black", linestyle="--", linewidth=1.3,
               alpha=0.7, zorder=2)
    ax.text(len(conditions) - 0.5 + 0.05, 0.5 + 0.002,
            "Chance", ha="right", va="bottom",
            fontsize=11, color="black", style="italic")

    baseline = means[0]
    vae_mean = means[2]
    delta    = vae_mean - baseline
    bx, bx2 = x[0], x[2]
    by = max(means[:3]) + max(sems[:3]) + 0.022
    ax.annotate(
        "", xy=(bx2, by), xytext=(bx, by),
        arrowprops=dict(arrowstyle="<->", color="black", lw=1.2)
    )
    ax.text((bx + bx2) / 2, by + 0.003,
            f"ΔAUC = {delta:+.3f}", ha="center", va="bottom",
            fontsize=11, color="black")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=12, color="black")
    ax.set_ylabel("Mean AUC (cross-dataset)", fontsize=15, color="black")
    ax.set_xlabel("Training Condition", fontsize=15, color="black")
    ax.set_title(
        "Cross-Dataset AUC by Training Condition\n"
        "(Mean ± SEM over All Source–Target Pairs and Classifiers)",
        fontsize=16, color="black", pad=14
    )
    ax.set_ylim(0, 0.65)
    ax.yaxis.set_minor_locator(MultipleLocator(0.01))
    ax.yaxis.set_major_locator(MultipleLocator(0.05))
    ax.grid(axis="y", which="major", linestyle="--", linewidth=0.6,
            alpha=0.5, zorder=1)
    ax.set_axisbelow(True)

    patches = [
        mpatches.Patch(facecolor=BLUES["Real→Cross"],   edgecolor="black", label="Real training data"),
        mpatches.Patch(facecolor=BLUES["Interpolation"],edgecolor="black", label="Synthetic — Interpolation"),
        mpatches.Patch(facecolor=BLUES["VAE"],          edgecolor="black", label="Synthetic — VAE"),
        mpatches.Patch(facecolor=BLUES["GAN"],          edgecolor="black", label="Synthetic — GAN"),
        mpatches.Patch(facecolor=BLUES["Interp + DA"],  edgecolor="black", label="Synthetic — Interpolation + Prototype-DA"),
    ]
    legend = ax.legend(
        handles=patches,
        title="Training Data Type",
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
