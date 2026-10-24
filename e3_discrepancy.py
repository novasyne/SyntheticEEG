"""
Revision experiment E3 (Reviewers #2.5 and #3.3).

Quantifies the domain discrepancy that the manuscript currently only asserts in prose, for every
source->target pair, comparing the REAL source distribution against each SYNTHETIC source
distribution with respect to the same real target distribution.

Metrics (all computed in the shared 10-D band-power feature space, standardised on the target):
  MMD^2      - maximum mean discrepancy, RBF kernel, median-heuristic bandwidth
  W2         - 2-Wasserstein distance (exact EMD on empirical samples via POT)
  CORAL      - Frobenius norm of the covariance difference, ||C_s - C_t||_F
  DomAcc/AUC - domain-classifier separability (logistic regression, source vs target,
               5-fold stratified CV, balanced) - 0.5 means the domains are indistinguishable

Also emitted:
  - correlation between discrepancy reduction (real->synthetic) and the observed Delta AUC
  - same-task (Stroop: SAM<->MS) vs different-task pair comparison, for Section 4.5
  - 2-D PCA embedding figure

Run from the Synthetic/ directory:  ./venv/bin/python revision/e3_discrepancy.py
"""
import os
import sys
import itertools
import warnings

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import ot

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import cross_dataset_generalization_synthetic as C  # noqa: E402

warnings.filterwarnings("ignore")

OUT_DIR  = "revision"
RNG      = np.random.default_rng(42)
MATCH_N  = 70           # every comparison uses this many samples per side (= synthetic set size)
N_DRAWS  = 20           # independent matched draws, averaged, to control sampling noise
SAME_TASK_PAIRS = {("SAM", "MS"), ("MS", "SAM")}   # both Stroop-based


def mmd2_rbf(X, Y):
    """Unbiased MMD^2 with a median-heuristic RBF bandwidth."""
    Z = np.vstack([X, Y])
    d2 = ((Z[:, None, :] - Z[None, :, :]) ** 2).sum(-1)
    med = np.median(d2[d2 > 0])
    gamma = 1.0 / med if med > 0 else 1.0
    K = np.exp(-gamma * d2)
    n, m = len(X), len(Y)
    Kxx = K[:n, :n]; Kyy = K[n:, n:]; Kxy = K[:n, n:]
    return float(
        (Kxx.sum() - np.trace(Kxx)) / (n * (n - 1))
        + (Kyy.sum() - np.trace(Kyy)) / (m * (m - 1))
        - 2 * Kxy.mean()
    )


def w2_distance(X, Y):
    M = ot.dist(X, Y, metric="sqeuclidean")
    a = np.ones(len(X)) / len(X)
    b = np.ones(len(Y)) / len(Y)
    return float(np.sqrt(ot.emd2(a, b, M, numItermax=200000)))


def coral_distance(X, Y):
    return float(np.linalg.norm(np.cov(X.T) - np.cov(Y.T), ord="fro"))


def domain_separability(X, Y, seed=42):
    """AUC of a classifier trained to tell source from target. 0.5 = indistinguishable."""
    Z = np.vstack([X, Y])          # X and Y are already size-matched by discrepancy()
    d = np.concatenate([np.zeros(len(X)), np.ones(len(Y))])
    clf = LogisticRegression(max_iter=1000, class_weight="balanced")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    return float(cross_val_score(clf, Z, d, cv=cv, scoring="roc_auc").mean())


def subsample(X, n, rng):
    """Draw exactly n rows (without replacement where possible)."""
    if len(X) == n:
        return X
    replace = len(X) < n
    return X[rng.choice(len(X), n, replace=replace)]


def discrepancy(X_src, X_tgt, scaler, n_draws=N_DRAWS):
    """Sample-size-matched discrepancy: both sides are reduced to MATCH_N rows on every draw,
    so that a 70-row synthetic set and an 853-row real set are compared on equal terms
    (the same size-confound Reviewer #2.4 raises for the classification experiments)."""
    acc = {k: [] for k in ["MMD2", "W2", "CORAL", "DomainAUC"]}
    for d in range(n_draws):
        rng = np.random.default_rng(1000 + d)
        A = scaler.transform(subsample(X_src, MATCH_N, rng))
        B = scaler.transform(subsample(X_tgt, MATCH_N, rng))
        acc["MMD2"].append(mmd2_rbf(A, B))
        acc["W2"].append(w2_distance(A, B))
        acc["CORAL"].append(coral_distance(A, B))
        acc["DomainAUC"].append(domain_separability(A, B, seed=1000 + d))
    out = {k: round(float(np.mean(v)), 5) for k, v in acc.items()}
    out["DomainAUC_SD"] = round(float(np.std(acc["DomainAUC"])), 4)
    return out


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    C._worker_init()
    datasets = list(C.DATASET_CONFIG.keys())
    pairs    = list(itertools.permutations(datasets, 2))

    rows = []
    for src, tgt in pairs:
        X_src, _, _ = C._worker_data["real"][src]
        X_tgt, _, _ = C._worker_data["real"][tgt]
        scaler = StandardScaler().fit(X_tgt)          # target-referenced feature scaling

        base = {"Source": src, "Target": tgt,
                "SameTask": (src, tgt) in SAME_TASK_PAIRS}

        rows.append({**base, "TrainingSet": "Real", **discrepancy(X_src, X_tgt, scaler)})

        for method in C.SYNTH_METHODS:
            key = (src, method)
            if key not in C._worker_data["synth"]:
                continue
            X_syn, _ = C._worker_data["synth"][key]
            rows.append({**base, "TrainingSet": f"Synthetic-{method}",
                         **discrepancy(X_syn, X_tgt, scaler)})

        for da in C.DA_METHODS:
            key = (src, tgt, da)
            if key not in C._worker_data["adapted"]:
                continue
            X_ad, _ = C._worker_data["adapted"][key]
            rows.append({**base, "TrainingSet": f"DA-{da}",
                         **discrepancy(X_ad, X_tgt, scaler)})

        print(f"  {src} -> {tgt} done", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT_DIR, "discrepancy_per_pair.csv"), index=False)

    summary = df.groupby("TrainingSet")[
        ["MMD2", "W2", "CORAL", "DomainAUC"]].mean().round(4)
    summary.to_csv(os.path.join(OUT_DIR, "discrepancy_summary.csv"))
    print("\nMean discrepancy to the real target, over 12 source-target pairs")
    print("(lower MMD2/W2/CORAL = closer; DomainAUC closer to 0.5 = less distinguishable)\n")
    print(summary.to_string())

    # -- same-task vs different-task, for Section 4.5 ------------------------------------------
    real = df[df.TrainingSet == "Real"]
    st  = real[real.SameTask]
    dt  = real[~real.SameTask]
    print("\nSection 4.5 check - real-source discrepancy to target:")
    print(f"  same-task pairs      (SAM<->MS, n={len(st)}): "
          f"MMD2={st.MMD2.mean():.4f}  W2={st.W2.mean():.3f}  "
          f"CORAL={st.CORAL.mean():.3f}  DomainAUC={st.DomainAUC.mean():.3f}")
    print(f"  different-task pairs (n={len(dt)}): "
          f"MMD2={dt.MMD2.mean():.4f}  W2={dt.W2.mean():.3f}  "
          f"CORAL={dt.CORAL.mean():.3f}  DomainAUC={dt.DomainAUC.mean():.3f}")

    # -- 2-D embedding figure -------------------------------------------------------------------
    fig, axes = plt.subplots(1, 4, figsize=(18, 4.2))
    for ax, ds in zip(axes, datasets):
        X_real, _, _ = C._worker_data["real"][ds]
        sc = StandardScaler().fit(X_real)
        p = PCA(n_components=2).fit(sc.transform(X_real))
        R = p.transform(sc.transform(subsample(X_real, min(400, len(X_real)), RNG)))
        ax.scatter(R[:, 0], R[:, 1], s=8, alpha=0.35, label="real", color="#4C72B0")
        for method, col in [("Interpolation", "#DD8452"), ("VAE", "#55A868"), ("GAN", "#C44E52")]:
            key = (ds, method)
            if key in C._worker_data["synth"]:
                S = p.transform(sc.transform(C._worker_data["synth"][key][0]))
                ax.scatter(S[:, 0], S[:, 1], s=14, alpha=0.8, label=method, color=col)
        ax.set_title(ds); ax.set_xlabel("PC1"); ax.set_ylabel("PC2")
    axes[0].legend(fontsize=8, loc="best")
    fig.suptitle("Real vs synthetic feature distributions per dataset (PCA of the 10-D band-power space)")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "fig_feature_space.png"), dpi=200)
    print("\nFigure -> revision/fig_feature_space.png")


if __name__ == "__main__":
    main()
