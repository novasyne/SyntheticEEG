"""
Revision experiment E7 — within-dataset LOSO under target-free (nested) model selection.

Why this is needed: Table 2 and Table B.7 of the submitted manuscript report "best model per
dataset", selected on the held-out LOSO fold — the same target-side selection flaw that E1 found
in the cross-dataset results. Under the paper's actual objective (does synthetic EEG data train
models that generalize ACROSS datasets?) the within-dataset result is the POSITIVE CONTROL: it
establishes whether synthesis preserves class structure at all. That control has to be clean.

Protocol here is properly nested:
  outer loop  - leave-one-subject-out over the real data (the evaluation)
  inner loop  - 3-fold stratified CV on the held-in subjects only (the selection signal)
Model selection uses ONLY the inner CV score, never the held-out subject. The held-out subject's
AUC is recorded for every model so the oracle rule can also be reported for comparison.

Conditions:
  Real-Within       - train on held-in subjects' real trials
  Synthetic-Within  - train on 70 interpolation-generated samples built from held-in subjects only
                      (identical to the manuscript's Table B.7 procedure)

Run from the Synthetic/ directory:  ./venv/bin/python revision/e7_within_loso.py
"""
import os
import sys
import time
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, StratifiedKFold, LeaveOneGroupOut
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cross_dataset_generalization_synthetic as C  # noqa: E402
import e1_e2_rerun as R                             # noqa: E402

warnings.filterwarnings("ignore")

OUT_DIR  = "revision"
OUT_FILE = os.path.join(OUT_DIR, "within_loso_targetfree.csv")
SEEDS    = C.SEEDS
N_SYNTH  = 70
MAX_WORKERS = 8


def run_one(dataset, model_name, seed, condition):
    """One dataset x model x seed: full LOSO, pooled predictions, nested selection."""
    X, y, groups = C._worker_data["real"][dataset]
    logo = LeaveOneGroupOut()

    true_all, proba_all, inner_scores = [], [], []

    for tr_idx, te_idx in logo.split(X, y, groups):
        X_in, y_in = X[tr_idx], y[tr_idx]
        X_te, y_te = X[te_idx], y[te_idx]
        if len(np.unique(y_in)) < 2 or len(np.unique(y_te)) < 2:
            continue

        if condition == "Synthetic-Within":
            X_tr_raw, y_tr = C._generate_interpolation_synthetic(
                X_in, y_in, n=N_SYNTH, seed=seed
            )
            if len(X_tr_raw) == 0 or len(np.unique(y_tr)) < 2:
                continue
        else:
            X_tr_raw, y_tr = X_in, y_in

        scaler = StandardScaler().fit(X_tr_raw)
        X_tr   = scaler.transform(X_tr_raw)
        X_tes  = scaler.transform(X_te)

        base = R.get_models(seed)[model_name]
        model, inner = R.tune_and_fit_cv(model_name, clone(base), X_tr, y_tr, seed)

        true_all.extend(y_te)
        proba_all.extend(model.predict_proba(X_tes)[:, 1])
        if inner == inner:
            inner_scores.append(inner)

    if not true_all or len(np.unique(true_all)) < 2:
        return None

    return {
        "Condition":      condition,
        "Dataset":        dataset,
        "Model":          model_name,
        "Seed":           seed,
        "N_Folds":        len(inner_scores),
        "Inner_CV_AUC":   round(float(np.mean(inner_scores)), 4) if inner_scores else np.nan,
        "Heldout_AUC":    round(float(roc_auc_score(true_all, proba_all)), 4),
        "N_Test":         len(true_all),
    }


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    C._worker_init()
    datasets    = list(C.DATASET_CONFIG.keys())
    model_names = list(R.get_models(SEEDS[0]).keys())
    conditions  = ["Real-Within", "Synthetic-Within"]

    tasks = [(ds, m, s, c)
             for c in conditions for ds in datasets
             for m in model_names for s in SEEDS]
    print(f"E7 within-dataset LOSO, target-free nested selection")
    print(f"{len(tasks)} runs | models {model_names} | seeds {SEEDS}", flush=True)

    t0 = time.time()
    rows = []
    with ProcessPoolExecutor(max_workers=MAX_WORKERS, initializer=C._worker_init) as ex:
        futs = {ex.submit(run_one, *t): t for t in tasks}
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
            if done % 20 == 0 or done == len(tasks):
                print(f"  {done}/{len(tasks)}  ({(time.time()-t0)/60:.1f} min)", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(OUT_FILE, index=False)
    print(f"\n{len(df)} rows -> {OUT_FILE}\n")

    # ---- selection protocols, mirroring the cross-dataset analysis ------------------------------
    print("=" * 78)
    print("Within-dataset LOSO AUC by selection protocol")
    print("=" * 78)
    out = []
    for cond in conditions:
        d = df[df.Condition == cond]
        for proto, key in [("oracle (as published)", "Heldout_AUC"),
                           ("target-free (nested)",  "Inner_CV_AUC")]:
            per_seed = (d.dropna(subset=[key])
                          .sort_values([key, "Model"], ascending=[False, True])
                          .groupby(["Dataset", "Seed"]).head(1))
            per_ds = per_seed.groupby("Dataset")["Heldout_AUC"].mean()
            row = {"Condition": cond, "Protocol": proto}
            row.update({k: round(v, 4) for k, v in per_ds.items()})
            row["Mean"] = round(float(per_ds.mean()), 4)
            row["Mean_stress_only"] = round(
                float(per_ds[[i for i in per_ds.index if i != "MW"]].mean()), 4)
            out.append(row)
    t = pd.DataFrame(out)
    print(t.to_string(index=False))
    t.to_csv(os.path.join(OUT_DIR, "table_within_loso_protocols.csv"), index=False)

    print("\nPer-model mean held-out AUC (no selection at all):")
    print(df.pivot_table(index="Model", columns=["Condition"],
                         values="Heldout_AUC", aggfunc="mean").round(3).to_string())


if __name__ == "__main__":
    main()
