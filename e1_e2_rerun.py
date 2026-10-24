"""
Revision experiment E1 + E2 + E6 (single pass).

E1 (Reviewer #2.1): records a source-side validation signal for every fitted model so that
    model/synthesis-method selection can be performed WITHOUT touching the target test set.
    Two target-free signals are stored per run:
      Source_CV_AUC   - mean AUC of 3-fold stratified CV on the training matrix (GridSearchCV
                        best_score_). Defined for every condition.
      SourceReal_AUC  - AUC of the fitted model on the real SOURCE dataset. For synthetic-only
                        training this is an out-of-sample, target-free validation signal; for
                        real-source training it is training-set performance and is not used
                        for selection.
E2 (Reviewer #2.4): adds condition Real-Sub70 - training on 70 real source samples drawn with
    the source class proportions, repeated over 10 independent draws, matching the synthetic
    training-set size exactly.
E6 (Reviewer #3.1, decision D2b): adds an MLP to the classifier pool on the same 10-D features.

Run from the Synthetic/ directory:  ./venv/bin/python revision/e1_e2_rerun.py
Per-seed checkpoints are written to revision/checkpoints/ and are skipped on restart.
"""
import os
import sys
import time
import zlib
import itertools
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import cross_dataset_generalization_synthetic as C  # noqa: E402

warnings.filterwarnings("ignore")

OUT_DIR   = "revision"
CKPT_DIR  = os.path.join(OUT_DIR, "checkpoints")
OUT_FILE  = os.path.join(OUT_DIR, "revision_results.csv")

SEEDS        = C.SEEDS
N_BOOTSTRAP  = 1000      # headline conditions
N_SUB_DRAWS  = 10        # E2 / decision D4
N_SUB        = 70        # matches the synthetic training-set size
MAX_WORKERS  = 8

MLP_GRID = {
    "hidden_layer_sizes": [(32,), (64, 32)],
    "alpha":              [1e-3, 1e-1],
}


def get_models(seed: int) -> dict:
    models = C.get_base_models(seed)
    models["MLP"] = MLPClassifier(
        max_iter=1000, random_state=seed, learning_rate_init=1e-3,
    )
    return models


def get_grid(model_name: str) -> dict:
    if model_name == "MLP":
        return MLP_GRID
    return C.PARAM_GRIDS.get(model_name, {})


def tune_and_fit_cv(model_name, base_model, X_train, y_train, seed):
    """As C.tune_and_fit, but also returns the source-side CV AUC used for selection."""
    grid = get_grid(model_name)
    min_class = np.bincount(y_train).min() if len(np.unique(y_train)) > 1 else 0
    if not grid or min_class < 3:
        base_model.fit(X_train, y_train)
        return base_model, np.nan
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=seed)
    search = GridSearchCV(clone(base_model), grid, cv=cv,
                          scoring="roc_auc", n_jobs=1, refit=True)
    try:
        search.fit(X_train, y_train)
        return search.best_estimator_, float(search.best_score_)
    except Exception:
        base_model.fit(X_train, y_train)
        return base_model, np.nan


def _evaluate(exp_type, source, target, model_name, seed, X_train, y_train,
              synth_method, n_boot, extra=None):
    """Train, score on the real target, and record both target-free validation signals."""
    X_test, y_test, _ = C._worker_data["real"][target]
    X_src,  y_src,  _ = C._worker_data["real"][source]

    if len(np.unique(y_train)) < 2 or len(np.unique(y_test)) < 2:
        return None

    scaler = StandardScaler().fit(X_train)
    X_tr   = scaler.transform(X_train)
    X_te   = scaler.transform(X_test)

    base_model = get_models(seed)[model_name]
    model, cv_auc = tune_and_fit_cv(model_name, clone(base_model), X_tr, y_train, seed)

    y_pred  = model.predict(X_te)
    y_proba = model.predict_proba(X_te)[:, 1]
    point   = C.compute_metrics(y_test, y_pred, y_proba)

    if n_boot > 0:
        ci = C.bootstrap_ci(y_test, y_pred, y_proba, seed, n_boot=n_boot)
    else:
        ci = {m: (np.nan, np.nan) for m in
              ["Accuracy", "AUC", "F1", "Precision", "Recall"]}

    try:
        src_proba = model.predict_proba(scaler.transform(X_src))[:, 1]
        src_auc   = float(roc_auc_score(y_src, src_proba))
    except Exception:
        src_auc = np.nan

    row = C.format_row(
        exp_type=exp_type, train_ds=source, test_ds=target, model_name=model_name,
        adaptation="Baseline", n_train=len(y_train), n_test=len(y_test),
        n_synth_samples=extra.get("n_synth", 0) if extra else 0,
        n_features=X_train.shape[1], point=point, ci=ci,
        synth_method=synth_method, seed=seed,
    )
    row["Source_CV_AUC"]  = round(cv_auc, 4) if cv_auc == cv_auc else np.nan
    row["SourceReal_AUC"] = round(src_auc, 4) if src_auc == src_auc else np.nan
    row["Draw"]           = extra.get("draw", -1) if extra else -1
    return row


# ----------------------------------------------------------------------------- task entry points
def task_real(source, target, model_name, seed):
    X_train, y_train, _ = C._worker_data["real"][source]
    return _evaluate("Real→Cross", source, target, model_name, seed,
                     X_train, y_train, "N/A", N_BOOTSTRAP)


def task_real_sub(source, target, model_name, seed, draw):
    """E2: class-proportion-matched subsample of n=70 real source trials."""
    X_all, y_all, _ = C._worker_data["real"][source]
    # deterministic across processes and runs (Python hash() is per-process salted)
    tag = f"{source}|{target}|{seed}|{draw}".encode()
    rng = np.random.default_rng(zlib.crc32(tag))

    counts = np.bincount(y_all, minlength=2)
    props  = counts / counts.sum()
    n_per  = np.round(props * N_SUB).astype(int)
    n_per[-1] = N_SUB - n_per[:-1].sum()

    idx = []
    for cls in [0, 1]:
        pool = np.where(y_all == cls)[0]
        k    = min(int(n_per[cls]), len(pool))
        if k > 0:
            idx.extend(rng.choice(pool, k, replace=False))
    idx = np.array(idx)
    if len(np.unique(y_all[idx])) < 2:
        return None

    return _evaluate("RealSub70→Cross", source, target, model_name, seed,
                     X_all[idx], y_all[idx], "N/A", 0, extra={"draw": draw})


def task_synth(source, target, model_name, seed, method):
    key = (source, method)
    if key not in C._worker_data["synth"]:
        return None
    X_train, y_train = C._worker_data["synth"][key]
    return _evaluate("Synthetic→Cross", source, target, model_name, seed,
                     X_train, y_train, method, N_BOOTSTRAP,
                     extra={"n_synth": len(y_train)})


def task_aug(source, target, model_name, seed, method):
    key = (source, method)
    if key not in C._worker_data["synth"]:
        return None
    X_real, y_real, _ = C._worker_data["real"][source]
    X_syn,  y_syn     = C._worker_data["synth"][key]
    return _evaluate("Real+Synthetic→Cross", source, target, model_name, seed,
                     np.vstack([X_real, X_syn]), np.concatenate([y_real, y_syn]),
                     method, N_BOOTSTRAP, extra={"n_synth": len(y_syn)})


def task_da(source, target, model_name, seed, da_method):
    key = (source, target, da_method)
    if key not in C._worker_data["adapted"]:
        return None
    X_real, y_real, _ = C._worker_data["real"][source]
    X_syn,  y_syn     = C._worker_data["adapted"][key]
    return _evaluate("Real+SyntheticDA→Cross", source, target, model_name, seed,
                     np.vstack([X_real, X_syn]), np.concatenate([y_real, y_syn]),
                     da_method, N_BOOTSTRAP, extra={"n_synth": len(y_syn)})


def build_tasks(seed, model_names, pairs):
    tasks = []
    for src, tgt in pairs:
        for m in model_names:
            tasks.append((task_real, (src, tgt, m, seed)))
            for d in range(N_SUB_DRAWS):
                tasks.append((task_real_sub, (src, tgt, m, seed, d)))
            for sm in C.SYNTH_METHODS:
                tasks.append((task_synth, (src, tgt, m, seed, sm)))
                tasks.append((task_aug,   (src, tgt, m, seed, sm)))
            for da in C.DA_METHODS:
                tasks.append((task_da, (src, tgt, m, seed, da)))
    return tasks


def main():
    os.makedirs(CKPT_DIR, exist_ok=True)
    datasets    = list(C.DATASET_CONFIG.keys())
    pairs       = list(itertools.permutations(datasets, 2))
    model_names = list(get_models(SEEDS[0]).keys())

    print("=" * 74)
    print("REVISION RERUN  E1 (source-only selection) + E2 (n=70 real control) + E6 (MLP)")
    print(f"Models ({len(model_names)}): {model_names}")
    print(f"Pairs: {len(pairs)}   Seeds: {SEEDS}   Subsample draws: {N_SUB_DRAWS}")
    print("=" * 74, flush=True)

    t_start = time.time()
    executor = ProcessPoolExecutor(max_workers=MAX_WORKERS, initializer=C._worker_init)
    all_rows = []

    for seed in SEEDS:
        ckpt = os.path.join(CKPT_DIR, f"seed_{seed}.csv")
        if os.path.exists(ckpt):
            print(f"[seed {seed}] checkpoint found, skipping", flush=True)
            all_rows.append(pd.read_csv(ckpt, keep_default_na=False, na_values=[""]))
            continue

        tasks = build_tasks(seed, model_names, pairs)
        print(f"\n[seed {seed}] {len(tasks)} runs", flush=True)

        futures = {executor.submit(fn, *args): (fn.__name__, args) for fn, args in tasks}
        rows, done = [], 0
        for fut in as_completed(futures):
            try:
                r = fut.result()
            except Exception as exc:
                print(f"  [!] {futures[fut]}: {exc}", flush=True)
                r = None
            if r:
                rows.append(r)
            done += 1
            if done % 250 == 0 or done == len(tasks):
                el = time.time() - t_start
                print(f"  [seed {seed}] {done}/{len(tasks)}  ({el/60:.1f} min elapsed)",
                      flush=True)

        df_seed = pd.DataFrame(rows)
        df_seed.to_csv(ckpt, index=False)
        all_rows.append(df_seed)
        print(f"  [seed {seed}] -> {ckpt}  ({len(df_seed)} rows)", flush=True)

    executor.shutdown(wait=True)

    df = pd.concat(all_rows, ignore_index=True)
    df.to_csv(OUT_FILE, index=False)
    print(f"\nDone in {(time.time()-t_start)/60:.1f} min. {len(df)} rows -> {OUT_FILE}")


if __name__ == "__main__":
    main()
