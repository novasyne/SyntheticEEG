"""
Revision item 1 — synthetic sample-size sweep under target-free selection.

The manuscript cites a sweep over n in {30, 50, 70, 100, 150, 200} to justify n = 70, but the
original outputs are absent from the repository and the original summary selected the best model per
pair on the TARGET test AUC (sensitivity_sweep.py, summarise(): idxmax on mean_auc) — the same flaw
E1 identified. Re-running it with the oracle rule would cite an oracle-selected curve in a paper
that argues against oracle selection.

This version records, for every fit, the same two target-free signals used elsewhere in the
revision, so the curve can be summarised without touching target labels:
  Source_CV_AUC  - 3-fold stratified CV AUC on the synthetic training matrix
  SourceReal_AUC - AUC on the real source dataset (out-of-sample for synthetic-only training)
Selection for the reported curve uses SourceReal_AUC (protocol P6 for synthetic-only conditions).

Run from the Synthetic/ directory:  ./venv/bin/python revision/e8_sensitivity_sweep.py
"""
import os
import sys
import time
import itertools
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.base import clone
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cross_dataset_generalization_synthetic as C   # noqa: E402
import sensitivity_sweep as SW                       # noqa: E402
import e1_e2_rerun as R                              # noqa: E402

warnings.filterwarnings("ignore")
matplotlib.rcParams.update({"figure.dpi": 150, "savefig.dpi": 300,
                            "savefig.bbox": "tight", "font.size": 12})

OUT_DIR   = "revision"
OUT_ROWS  = os.path.join(OUT_DIR, "sensitivity_sweep_targetfree.csv")
OUT_SUM   = os.path.join(OUT_DIR, "sensitivity_sweep_summary_targetfree.csv")
OUT_PLOT  = os.path.join(OUT_DIR, "fig_sensitivity_sweep.png")
N_SWEEP   = SW.N_SWEEP
SEEDS     = C.SEEDS
MAX_WORKERS = 8


def task(n, seed, source, target, model_name):
    X_src, y_src, _ = C._worker_data["real"][source]
    X_tgt, y_tgt, _ = C._worker_data["real"][target]

    rng = np.random.default_rng(abs(hash(())) if False else (seed * 1000 + n))
    X_syn, y_syn = SW.generate_interp_synthetic(X_src, y_src, n, rng)
    if len(np.unique(y_syn)) < 2 or len(np.unique(y_tgt)) < 2:
        return None

    scaler = StandardScaler().fit(X_syn)
    X_tr   = scaler.transform(X_syn)

    base = R.get_models(seed)[model_name]
    model, cv_auc = R.tune_and_fit_cv(model_name, clone(base), X_tr, y_syn, seed)

    tgt_auc = float(roc_auc_score(y_tgt, model.predict_proba(scaler.transform(X_tgt))[:, 1]))
    try:
        src_auc = float(roc_auc_score(y_src, model.predict_proba(scaler.transform(X_src))[:, 1]))
    except Exception:
        src_auc = np.nan

    return {"n": n, "Seed": seed, "Source": source, "Target": target, "Model": model_name,
            "Target_AUC": round(tgt_auc, 4),
            "Source_CV_AUC": round(cv_auc, 4) if cv_auc == cv_auc else np.nan,
            "SourceReal_AUC": round(src_auc, 4) if src_auc == src_auc else np.nan}


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    C._worker_init()
    datasets = list(C.DATASET_CONFIG.keys())
    pairs    = list(itertools.permutations(datasets, 2))
    models   = list(R.get_models(SEEDS[0]).keys())

    tasks = [(n, s, src, tgt, m)
             for n in N_SWEEP for s in SEEDS for src, tgt in pairs for m in models]
    print(f"Sensitivity sweep, target-free selection | {len(tasks)} runs")
    print(f"n = {N_SWEEP} | seeds {SEEDS} | models {models}", flush=True)

    t0, rows = time.time(), []
    with ProcessPoolExecutor(max_workers=MAX_WORKERS, initializer=C._worker_init) as ex:
        futs = {ex.submit(task, *t): t for t in tasks}
        done = 0
        for f in as_completed(futs):
            try:
                r = f.result()
            except Exception as exc:
                print(f"  [!] {futs[f]}: {exc}", flush=True)
                r = None
            if r:
                rows.append(r)
            done += 1
            if done % 200 == 0 or done == len(tasks):
                print(f"  {done}/{len(tasks)}  ({(time.time()-t0)/60:.1f} min)", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(OUT_ROWS, index=False)
    print(f"\n{len(df)} rows -> {OUT_ROWS}\n")

    # ---- summarise under both rules so the contrast is explicit --------------------------------
    out = []
    for n in N_SWEEP:
        d = df[df.n == n]
        row = {"n": n}
        # oracle (as the original script did)
        orc = (d.groupby(["Source", "Target", "Model"])["Target_AUC"].mean()
                 .groupby(level=[0, 1]).max())
        row["Oracle_mean"] = round(float(orc.mean()), 4)
        row["Oracle_SEM"]  = round(float(orc.std(ddof=1) / np.sqrt(len(orc))), 4)
        # target-free (P6: select on real-source AUC)
        sel = (d.dropna(subset=["SourceReal_AUC"])
                 .sort_values(["SourceReal_AUC", "Model"], ascending=[False, True])
                 .groupby(["Source", "Target", "Seed"]).head(1))
        tf = sel.groupby(["Source", "Target"])["Target_AUC"].mean()
        row["TargetFree_mean"] = round(float(tf.mean()), 4)
        row["TargetFree_SEM"]  = round(float(tf.std(ddof=1) / np.sqrt(len(tf))), 4)
        # stress-only subset
        S = {"EDMA", "MS", "SAM"}
        tf_s = tf.loc[[p for p in tf.index if p[0] in S and p[1] in S]]
        row["TargetFree_stress_only"] = round(float(tf_s.mean()), 4)
        row["Inflation"] = round(row["Oracle_mean"] - row["TargetFree_mean"], 4)
        out.append(row)
    summary = pd.DataFrame(out)
    summary.to_csv(OUT_SUM, index=False)
    print(summary.to_string(index=False))

    fig, ax = plt.subplots(figsize=(7, 4.4))
    ax.errorbar(summary.n, summary.Oracle_mean, yerr=summary.Oracle_SEM,
                marker="o", capsize=3, color="#C44E52",
                label="Best model chosen on target AUC (oracle)")
    ax.errorbar(summary.n, summary.TargetFree_mean, yerr=summary.TargetFree_SEM,
                marker="s", capsize=3, color="#4C72B0",
                label="Best model chosen on source data (target-free)")
    ax.plot(summary.n, summary.TargetFree_stress_only, marker="^", linestyle="--",
            color="#55A868", label="Target-free, stress datasets only")
    ax.axhline(0.5, ls=":", c="grey", lw=1)
    ax.text(summary.n.max(), 0.503, "chance", ha="right", va="bottom",
            fontsize=9, color="grey")
    ax.set_xlabel("Synthetic samples generated per source dataset ($n$)")
    ax.set_ylabel("Mean cross-dataset AUC")
    ax.set_title("Cross-dataset transfer vs. synthetic sample size")
    ax.legend(fontsize=9, loc="best")
    fig.tight_layout()
    fig.savefig(OUT_PLOT)
    print(f"\nFigure -> {OUT_PLOT}")


if __name__ == "__main__":
    main()
