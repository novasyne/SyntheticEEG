import os
import itertools
import warnings
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed
import pandas as pd
from sklearn.base import clone
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, StratifiedKFold, LeaveOneGroupOut
from sklearn.metrics import (
    roc_auc_score, f1_score, precision_score, recall_score, accuracy_score
)
import xgboost as xgb

warnings.filterwarnings("ignore")

DATA_DIR       = "Processed"
SYNTH_DIR      = "Synthetic"
OUTPUT_FILE    = "cross_dataset_results_synthetic.csv"
OUTPUT_AGG     = "cross_dataset_results_synthetic_agg.csv"
N_BOOTSTRAP    = 1000
SEEDS          = [42, 43, 44, 45, 46]

BANDS = ["Alpha", "Beta", "Delta", "Theta", "Gamma"]

COMMON_SYNTH_FEATURES = sorted(
    [f"{b} Frontal" for b in BANDS] + [f"{b} Parietal" for b in BANDS]
)


def _participant_edma(df):
    return df["Subject"].str.extract(r"EEGStress(?:Baseline|Stressed)_(\d+)_\d+")[0]


def _participant_subject_str(df):
    return df["Subject"].astype(str)


DATASET_CONFIG = {
    "EDMA": {
        "synth_file":  "Synthetic_Aggregated_EDMA.csv",
        "real_file":   "EEG_Dataset_EDMA.csv",
        "label_col":   "Label",
        "participant_fn": _participant_edma,
        "frontal_ch":  ["Fp1", "Fp2", "F3", "F4", "F7", "F8", "Fz"],
        "parietal_ch": ["P3", "P4", "Pz"],
    },
    "MS": {
        "synth_file":  "Synthetic_Aggregated_MS.csv",
        "real_file":   "EEG_Dataset_MS.csv",
        "label_col":   "Task",
        "participant_fn": _participant_subject_str,
        "frontal_ch":  ["F3", "F4"],
        "parietal_ch": ["Pz"],
    },
    "MW": {
        "synth_file":  "Synthetic_Aggregated_MW.csv",
        "real_file":   "EEG_Dataset_MW.csv",
        "label_col":   "Label",
        "participant_fn": _participant_subject_str,
        "frontal_ch":  ["AF3", "AF4", "F3", "F4", "F7", "F8"],
        "parietal_ch": ["P7", "P8"],
    },
    "SAM": {
        "synth_file":  "Synthetic_Aggregated_SAM.csv",
        "real_file":   "EEG_Dataset_SAM.csv",
        "label_col":   "Label",
        "participant_fn": _participant_subject_str,
        "frontal_ch":  ["F3", "F4", "F7", "F8", "Fz",
                        "FC1", "FC2", "FC5", "FC6", "FT9", "FT10"],
        "parietal_ch": ["P3", "P4", "Pz", "PO9", "PO10"],
    },
}

FEATURE_STRATEGIES = {
    "Baseline": False,
}

SYNTH_METHODS = {
    "Interpolation": "",
    "GAN":           "_GAN",
    "VAE":           "_VAE",
}

DA_METHODS = ["CORAL-DA", "OT-DA", "Prototype-DA"]

MERGED_SYNTH_CONFIG = {}


PARAM_GRIDS = {
    "SVM-RBF": {
        "C":     [0.1, 10],
        "gamma": ["scale", 0.1],
    },
    "XGBoost": {
        "n_estimators":  [100, 200],
        "max_depth":     [3, 6],
    },
    "LDA": {
        "solver":    ["lsqr"],
        "shrinkage": [None, "auto"],
    },
    "Random Forest": {
        "n_estimators":  [100, 200],
        "max_depth":     [None, 10],
    },
    "Logistic Regression": {
        "C": [0.1, 1, 10],
    },
}


def get_base_models(random_state: int) -> dict:
    return {
        "SVM-RBF": SVC(
            kernel="rbf", probability=True, random_state=random_state
        ),
        "XGBoost": xgb.XGBClassifier(
            random_state=random_state, verbosity=0, eval_metric="logloss",
            nthread=1,
        ),
        "LDA": LinearDiscriminantAnalysis(),
        "Random Forest": RandomForestClassifier(
            random_state=random_state, n_jobs=1
        ),
        "Logistic Regression": LogisticRegression(
            max_iter=1000, random_state=random_state, solver="lbfgs"
        ),
    }


_worker_data: dict = {}


def tune_and_fit(model_name: str, base_model, X_train: np.ndarray,
                 y_train: np.ndarray, seed: int):
    grid = PARAM_GRIDS.get(model_name, {})
    min_class = np.bincount(y_train).min() if len(np.unique(y_train)) > 1 else 0
    if not grid or min_class < 3:
        base_model.fit(X_train, y_train)
        return base_model

    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=seed)
    search = GridSearchCV(
        clone(base_model), grid, cv=cv,
        scoring="roc_auc", n_jobs=1, refit=True
    )
    try:
        search.fit(X_train, y_train)
        return search.best_estimator_
    except Exception:
        base_model.fit(X_train, y_train)
        return base_model


def _worker_init():
    global _worker_data
    datasets    = list(DATASET_CONFIG.keys())
    cross_pairs = list(itertools.permutations(datasets, 2))

    real = {}
    for ds in datasets:
        real[ds] = load_real_aggregated(ds)

    synth = {}
    for ds in datasets:
        for sm in SYNTH_METHODS:
            try:
                synth[(ds, sm)] = load_synthetic(ds, method=sm)
            except (FileNotFoundError, ValueError):
                pass

    adapted = {}
    for src, tgt in cross_pairs:
        for da_m in DA_METHODS:
            try:
                adapted[(src, tgt, da_m)] = load_synthetic_adapted(src, tgt, da_m)
            except (FileNotFoundError, ValueError):
                pass

    _worker_data["real"]    = real
    _worker_data["synth"]   = synth
    _worker_data["adapted"] = adapted


def prepare_features(X_train_raw: np.ndarray, X_test_raw: np.ndarray,
                     use_log_ratio: bool):
    scaler = StandardScaler()
    X_tr   = scaler.fit_transform(X_train_raw)
    X_te   = scaler.transform(X_test_raw)
    return X_tr, X_te


def load_synthetic(name: str, method: str = "Interpolation"):
    if name in MERGED_SYNTH_CONFIG:
        cfg  = MERGED_SYNTH_CONFIG[name]
        path = os.path.join(SYNTH_DIR, cfg["synth_file"])
    else:
        suffix = SYNTH_METHODS.get(method, "")
        fname  = f"Synthetic_Aggregated_{name}{suffix}.csv"
        path   = os.path.join(SYNTH_DIR, fname)
    df   = pd.read_csv(path)
    df   = df[df["Label"].isin([0, 1])].reset_index(drop=True)

    available = [f for f in COMMON_SYNTH_FEATURES if f in df.columns]
    missing   = [f for f in COMMON_SYNTH_FEATURES if f not in df.columns]
    if missing:
        raise ValueError(f"Synthetic {name} missing features: {missing}")

    X = df[COMMON_SYNTH_FEATURES].values.astype(float)
    y = df["Label"].values.astype(int)
    return X, y


def _region_mean(df: pd.DataFrame, band: str, channels: list) -> np.ndarray:
    cols = [f"{band}{ch}" for ch in channels if f"{band}{ch}" in df.columns]
    if not cols:
        raise ValueError(
            f"No columns found for band={band}, channels={channels}. "
            f"Available cols sample: {list(df.columns[:10])}"
        )
    return df[cols].mean(axis=1).values


def load_real_aggregated(name: str):
    cfg  = DATASET_CONFIG[name]
    path = os.path.join(DATA_DIR, cfg["real_file"])
    df   = pd.read_csv(path)
    df   = df[df[cfg["label_col"]].isin([0, 1])].reset_index(drop=True)

    frontal_ch  = cfg["frontal_ch"]
    parietal_ch = cfg["parietal_ch"]

    cols = {}
    for band in BANDS:
        cols[f"{band} Frontal"]  = _region_mean(df, band, frontal_ch)
        cols[f"{band} Parietal"] = _region_mean(df, band, parietal_ch)

    feat_df = pd.DataFrame(cols)[COMMON_SYNTH_FEATURES]

    feat_df = np.log1p(feat_df.clip(lower=0))

    groups_raw = cfg["participant_fn"](df).values
    feat_arr   = feat_df.values.astype(float)
    for subj in np.unique(groups_raw):
        mask = groups_raw == subj
        mu   = feat_arr[mask].mean(axis=0)
        sd   = feat_arr[mask].std(axis=0)
        sd[sd == 0] = 1
        feat_arr[mask] = (feat_arr[mask] - mu) / sd

    y      = df[cfg["label_col"]].values.astype(int)
    return feat_arr, y, groups_raw


def compute_metrics(y_true, y_pred, y_proba):
    return {
        "Accuracy":  accuracy_score(y_true, y_pred),
        "AUC":       roc_auc_score(y_true, y_proba),
        "F1":        f1_score(y_true, y_pred, zero_division=0),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall":    recall_score(y_true, y_pred, zero_division=0),
    }


def bootstrap_ci(y_true, y_pred, y_proba, seed: int,
                 n_boot=N_BOOTSTRAP, alpha=0.05):
    n    = len(y_true)
    boot = {m: [] for m in ["Accuracy", "AUC", "F1", "Precision", "Recall"]}
    rng  = np.random.default_rng(seed)

    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        yt, yp, ypr = y_true[idx], y_pred[idx], y_proba[idx]
        if len(np.unique(yt)) < 2:
            continue
        for k, v in compute_metrics(yt, yp, ypr).items():
            boot[k].append(v)

    ci = {}
    for metric, vals in boot.items():
        vals = np.array(vals)
        lo   = np.percentile(vals, 100 * alpha / 2)
        hi   = np.percentile(vals, 100 * (1 - alpha / 2))
        ci[metric] = (round(lo, 4), round(hi, 4))
    return ci


def format_row(exp_type, train_ds, test_ds, model_name, adaptation,
               n_train, n_test, n_synth_samples, n_features, point, ci,
               synth_method="Interpolation", seed: int = 0):
    row = {
        "Experiment_Type":    exp_type,
        "Train_Dataset":      train_ds,
        "Test_Dataset":       test_ds,
        "Model":              model_name,
        "Adaptation":         adaptation,
        "Synth_Method":       synth_method,
        "Seed":               seed,
        "N_Train_Samples":    n_train,
        "N_Test_Samples":     n_test,
        "N_Synth_Samples":    n_synth_samples,
        "N_Features":         n_features,
    }
    for metric in ["Accuracy", "AUC", "F1", "Precision", "Recall"]:
        row[metric]              = round(point[metric], 4)
        row[f"{metric}_CI_Low"]  = ci[metric][0]
        row[f"{metric}_CI_High"] = ci[metric][1]
    return row


def real_cross_evaluate(source, target, model_name, base_model,
                        strategy, use_log_ratio, seed: int):
    X_train, y_train, _ = _worker_data["real"][source]
    X_test,  y_test,  _ = _worker_data["real"][target]

    if len(np.unique(y_train)) < 2 or len(np.unique(y_test)) < 2:
        return None

    X_tr, X_te = prepare_features(X_train, X_test, use_log_ratio)
    model   = tune_and_fit(model_name, clone(base_model), X_tr, y_train, seed)
    y_pred  = model.predict(X_te)
    y_proba = model.predict_proba(X_te)[:, 1]

    return format_row(
        exp_type="Real→Cross",
        train_ds=source,
        test_ds=target,
        model_name=model_name,
        adaptation=strategy,
        n_train=len(y_train),
        n_test=len(y_test),
        n_synth_samples=0,
        n_features=X_train.shape[1],
        point=compute_metrics(y_test, y_pred, y_proba),
        ci=bootstrap_ci(y_test, y_pred, y_proba, seed),
        synth_method="N/A",
        seed=seed,
    )


def load_synthetic_adapted(source: str, target: str, da_method: str):
    tag   = da_method.replace("-DA", "").replace("-", "")
    fname = f"Synthetic_Adapted_{source}_to_{target}_{tag}.csv"
    path  = os.path.join(SYNTH_DIR, fname)
    df    = pd.read_csv(path)
    df    = df[df["Label"].isin([0, 1])].reset_index(drop=True)
    available = [f for f in COMMON_SYNTH_FEATURES if f in df.columns]
    missing   = [f for f in COMMON_SYNTH_FEATURES if f not in df.columns]
    if missing:
        raise ValueError(f"Adapted {source}→{target} ({da_method}) missing: {missing}")
    X = df[COMMON_SYNTH_FEATURES].values.astype(float)
    y = df["Label"].values.astype(int)
    return X, y


def augmented_cross_da_evaluate(source, target, model_name, base_model,
                                 strategy, use_log_ratio, da_method="CORAL-DA",
                                 seed: int = 0):
    adapted_key = (source, target, da_method)
    if adapted_key not in _worker_data["adapted"]:
        return None
    X_real,  y_real, _ = _worker_data["real"][source]
    X_synth, y_synth   = _worker_data["adapted"][adapted_key]

    X_train = np.vstack([X_real, X_synth])
    y_train = np.concatenate([y_real, y_synth])
    X_test, y_test, _ = _worker_data["real"][target]

    if len(np.unique(y_train)) < 2 or len(np.unique(y_test)) < 2:
        return None

    X_tr, X_te = prepare_features(X_train, X_test, use_log_ratio)
    model   = tune_and_fit(model_name, clone(base_model), X_tr, y_train, seed)
    y_pred  = model.predict(X_te)
    y_proba = model.predict_proba(X_te)[:, 1]

    return format_row(
        exp_type="Real+SyntheticDA→Cross",
        train_ds=source,
        test_ds=target,
        model_name=model_name,
        adaptation=strategy,
        n_train=len(y_train),
        n_test=len(y_test),
        n_synth_samples=len(y_synth),
        n_features=X_train.shape[1],
        point=compute_metrics(y_test, y_pred, y_proba),
        ci=bootstrap_ci(y_test, y_pred, y_proba, seed),
        synth_method=da_method,
        seed=seed,
    )


def augmented_cross_evaluate(source, target, model_name, base_model,
                              strategy, use_log_ratio, synth_method="Interpolation",
                              seed: int = 0):
    synth_key = (source, synth_method)
    if synth_key not in _worker_data["synth"]:
        return None
    X_real,  y_real,  _ = _worker_data["real"][source]
    X_synth, y_synth     = _worker_data["synth"][synth_key]

    X_train = np.vstack([X_real, X_synth])
    y_train = np.concatenate([y_real, y_synth])

    X_test, y_test, _ = _worker_data["real"][target]

    if len(np.unique(y_train)) < 2 or len(np.unique(y_test)) < 2:
        return None

    X_tr, X_te = prepare_features(X_train, X_test, use_log_ratio)
    model   = tune_and_fit(model_name, clone(base_model), X_tr, y_train, seed)
    y_pred  = model.predict(X_te)
    y_proba = model.predict_proba(X_te)[:, 1]

    return format_row(
        exp_type="Real+Synthetic→Cross",
        train_ds=source,
        test_ds=target,
        model_name=model_name,
        adaptation=strategy,
        n_train=len(y_train),
        n_test=len(y_test),
        n_synth_samples=len(y_synth),
        n_features=X_train.shape[1],
        point=compute_metrics(y_test, y_pred, y_proba),
        ci=bootstrap_ci(y_test, y_pred, y_proba, seed),
        synth_method=synth_method,
        seed=seed,
    )


def synthetic_to_real_evaluate(
    synth_source, real_target,
    model_name, base_model,
    strategy, use_log_ratio,
    synth_method="Interpolation",
    seed: int = 0,
):
    if synth_source in MERGED_SYNTH_CONFIG:
        exp_type = "Synthetic→Merged→Cross"
    elif synth_source == real_target:
        exp_type = "Synthetic→Within"
    else:
        exp_type = "Synthetic→Cross"

    synth_key = (synth_source, synth_method)
    if synth_key not in _worker_data["synth"]:
        return None
    X_train_raw, y_train   = _worker_data["synth"][synth_key]
    X_test_raw,  y_test, _ = _worker_data["real"][real_target]

    if len(np.unique(y_train)) < 2:
        print(f"    [!] Synthetic {synth_source}: <2 classes, skipping.")
        return None
    if len(np.unique(y_test)) < 2:
        print(f"    [!] Real {real_target}: <2 classes in test, skipping.")
        return None

    X_tr, X_te = prepare_features(X_train_raw, X_test_raw, use_log_ratio)
    model   = tune_and_fit(model_name, clone(base_model), X_tr, y_train, seed)
    y_pred  = model.predict(X_te)
    y_proba = model.predict_proba(X_te)[:, 1]

    n_feat = X_train_raw.shape[1]

    return format_row(
        exp_type=exp_type,
        train_ds=f"Synthetic_{synth_source}",
        test_ds=real_target,
        model_name=model_name,
        adaptation=strategy,
        n_train=len(y_train),
        n_test=len(y_test),
        n_synth_samples=len(y_train),
        n_features=n_feat,
        point=compute_metrics(y_test, y_pred, y_proba),
        ci=bootstrap_ci(y_test, y_pred, y_proba, seed),
        synth_method=synth_method,
        seed=seed,
    )


_SYNTH_CORR_THRESHOLD = 0.2
_SYNTH_NOISE_LEVEL    = 0.05
_SYNTH_N              = 70


def _generate_interpolation_synthetic(
    X_held_in: np.ndarray, y_held_in: np.ndarray,
    n: int = _SYNTH_N, seed: int = 42,
) -> tuple:
    rng = np.random.default_rng(seed)

    scaler   = StandardScaler()
    X_norm   = scaler.fit_transform(X_held_in)
    labels   = y_held_in.astype(int)

    class_counts = np.bincount(labels, minlength=2)
    class_probs  = class_counts / max(len(labels), 1)
    n_per_class  = np.round(class_probs * n).astype(int)
    n_per_class[-1] = n - n_per_class[:-1].sum()

    synth_X, synth_y = [], []

    for cls in [0, 1]:
        n_cls = int(n_per_class[cls])
        if n_cls == 0:
            continue
        idx   = np.where(labels == cls)[0]
        feats = X_norm[idx]

        if len(idx) < 2:
            for _ in range(n_cls):
                noise = rng.normal(0, _SYNTH_NOISE_LEVEL, feats.shape[1])
                synth_X.append(feats[0] + noise)
                synth_y.append(cls)
            continue

        corr = np.corrcoef(feats)
        np.fill_diagonal(corr, 0)
        pairs = np.argwhere(corr > _SYNTH_CORR_THRESHOLD)
        if len(pairs) == 0:
            nc    = len(feats)
            pairs = np.array([(i, j) for i in range(nc)
                              for j in range(nc) if i != j])

        for _ in range(n_cls):
            pick         = rng.integers(len(pairs))
            i1, i2      = pairs[pick]
            alpha        = rng.uniform(0.3, 0.7)
            new_norm     = feats[i1] + (feats[i2] - feats[i1]) * alpha
            noise_std    = np.std(np.vstack([feats[i1], feats[i2]]), axis=0)
            noise        = rng.normal(0, 1, new_norm.shape) * noise_std * _SYNTH_NOISE_LEVEL
            synth_X.append(new_norm + noise)
            synth_y.append(cls)

    if not synth_X:
        return np.empty((0, X_held_in.shape[1])), np.array([], dtype=int)

    X_synth = scaler.inverse_transform(np.array(synth_X))
    y_synth = np.array(synth_y, dtype=int)
    return X_synth, y_synth


def synthetic_within_loso_evaluate(
    dataset: str, model_name: str, base_model,
    strategy: str, use_log_ratio: bool, seed: int,
):
    X_real, y_real, groups = _worker_data["real"][dataset]
    logo = LeaveOneGroupOut()

    all_true, all_pred, all_proba = [], [], []

    for train_idx, test_idx in logo.split(X_real, y_real, groups):
        X_held_in, y_held_in = X_real[train_idx], y_real[train_idx]
        X_test,    y_test    = X_real[test_idx],  y_real[test_idx]

        if len(np.unique(y_held_in)) < 2 or len(np.unique(y_test)) < 2:
            continue

        X_synth, y_synth = _generate_interpolation_synthetic(
            X_held_in, y_held_in, n=_SYNTH_N, seed=seed
        )
        if len(X_synth) == 0 or len(np.unique(y_synth)) < 2:
            continue

        X_tr, X_te = prepare_features(X_synth, X_test, use_log_ratio)
        model   = tune_and_fit(model_name, clone(base_model), X_tr, y_synth, seed)
        y_pred  = model.predict(X_te)
        y_proba = model.predict_proba(X_te)[:, 1]

        all_true.extend(y_test)
        all_pred.extend(y_pred)
        all_proba.extend(y_proba)

    if not all_true or len(np.unique(all_true)) < 2:
        return None

    all_true  = np.array(all_true)
    all_pred  = np.array(all_pred)
    all_proba = np.array(all_proba)

    point = compute_metrics(all_true, all_pred, all_proba)
    ci    = bootstrap_ci(all_true, all_pred, all_proba, seed)

    return format_row(
        exp_type="Synthetic→Within",
        train_ds=f"Synthetic_{dataset}",
        test_ds=dataset,
        model_name=model_name,
        adaptation=strategy,
        n_train=_SYNTH_N,
        n_test=len(all_true),
        n_synth_samples=_SYNTH_N,
        n_features=X_real.shape[1],
        point=point,
        ci=ci,
        synth_method="Interpolation",
        seed=seed,
    )


def main():
    print("=" * 70)
    print("Cross-Dataset EEG Stress Classification — Synthetic Training")
    print(f"Common features ({len(COMMON_SYNTH_FEATURES)}): {COMMON_SYNTH_FEATURES}")
    print(f"Feature strategies: {list(FEATURE_STRATEGIES.keys())}")
    print(f"Seeds: {SEEDS}  |  Bootstrap samples per seed: {N_BOOTSTRAP}")
    print("=" * 70)

    datasets = list(DATASET_CONFIG.keys())

    cross_pairs  = list(itertools.permutations(datasets, 2))
    within_pairs = [(ds, ds) for ds in datasets]
    merged_pairs = [
        (src, tgt)
        for src, cfg in MERGED_SYNTH_CONFIG.items()
        for tgt in cfg["test_targets"]
    ]
    n_strat  = len(FEATURE_STRATEGIES)
    n_mod    = len(PARAM_GRIDS)
    n_smeth  = len(SYNTH_METHODS)
    n_da     = len(DA_METHODS)
    runs_per_seed = n_strat * n_mod * (
        len(within_pairs)
        + n_smeth * len(cross_pairs)
        + len(merged_pairs)
        + len(cross_pairs)
        + n_smeth * len(cross_pairs)
        + n_da * len(cross_pairs)
    )
    print(f"Runs per seed: {runs_per_seed}  |  Total: {runs_per_seed * len(SEEDS)}\n")

    all_results = []

    executor = ProcessPoolExecutor(max_workers=16, initializer=_worker_init)
    for seed in SEEDS:
        print(f"\n{'─'*70}")
        print(f"SEED {seed}")
        print(f"{'─'*70}")
        base_models = get_base_models(seed)

        tasks = []
        for strategy, use_lr in FEATURE_STRATEGIES.items():

            for ds in datasets:
                for model_name, base_model in base_models.items():
                    tasks.append((
                        f"Syn_{ds}→Real_{ds} [Interpolation/LOSO] {model_name}",
                        synthetic_within_loso_evaluate,
                        (ds, model_name, base_model, strategy, use_lr),
                        {"seed": seed},
                    ))

            for sm in SYNTH_METHODS:
                for src, tgt in cross_pairs:
                    for model_name, base_model in base_models.items():
                        tasks.append((
                            f"Syn_{src}→Real_{tgt} [{sm}] {model_name}",
                            synthetic_to_real_evaluate,
                            (src, tgt, model_name, base_model, strategy, use_lr),
                            {"synth_method": sm, "seed": seed},
                        ))

            for src, tgt in merged_pairs:
                for model_name, base_model in base_models.items():
                    tasks.append((
                        f"Syn_{src}→Real_{tgt} [Merged] {model_name}",
                        synthetic_to_real_evaluate,
                        (src, tgt, model_name, base_model, strategy, use_lr),
                        {"synth_method": "Interpolation", "seed": seed},
                    ))

            for src, tgt in cross_pairs:
                for model_name, base_model in base_models.items():
                    tasks.append((
                        f"Real_{src}→Real_{tgt} {model_name}",
                        real_cross_evaluate,
                        (src, tgt, model_name, base_model, strategy, use_lr, seed),
                        {},
                    ))

            for sm in SYNTH_METHODS:
                for src, tgt in cross_pairs:
                    for model_name, base_model in base_models.items():
                        tasks.append((
                            f"Aug_{src}→Real_{tgt} [{sm}] {model_name}",
                            augmented_cross_evaluate,
                            (src, tgt, model_name, base_model, strategy, use_lr),
                            {"synth_method": sm, "seed": seed},
                        ))

            for da_m in DA_METHODS:
                for src, tgt in cross_pairs:
                    for model_name, base_model in base_models.items():
                        tasks.append((
                            f"Aug_DA_{src}→Real_{tgt} [{da_m}] {model_name}",
                            augmented_cross_da_evaluate,
                            (src, tgt, model_name, base_model, strategy, use_lr),
                            {"da_method": da_m, "seed": seed},
                        ))

        results = []
        completed = 0
        futures = {
            executor.submit(fn, *args, **kwargs): label
            for label, fn, args, kwargs in tasks
        }
        for future in as_completed(futures):
            label = futures[future]
            try:
                row = future.result()
            except Exception as exc:
                print(f"  [!] {label}: {exc}")
                row = None
            if row:
                results.append(row)
            completed += 1
            if completed % 50 == 0 or completed == len(tasks):
                print(f"  [{seed}] {completed}/{len(tasks)} tasks done",
                      flush=True)

        all_results.extend(results)
        print(f"  → Seed {seed} done: {len(results)} rows")

    executor.shutdown(wait=True)

    df = pd.DataFrame(all_results)

    metric_cols = []
    for m in ["Accuracy", "AUC", "F1", "Precision", "Recall"]:
        metric_cols += [m, f"{m}_CI_Low", f"{m}_CI_High"]

    col_order = [
        "Experiment_Type", "Train_Dataset", "Test_Dataset",
        "Model", "Adaptation", "Synth_Method", "Seed",
        "N_Train_Samples", "N_Test_Samples", "N_Synth_Samples", "N_Features",
    ] + metric_cols
    df = df[[c for c in col_order if c in df.columns]]
    df.to_csv(OUTPUT_FILE, index=False)
    print(f"\nAll-seeds results saved → {OUTPUT_FILE}  ({len(df)} rows)")

    group_cols = ["Experiment_Type", "Train_Dataset", "Test_Dataset",
                  "Model", "Adaptation", "Synth_Method"]
    agg_parts = []
    for _, g in df.groupby(group_cols, sort=False):
        row_agg = {c: g[c].iloc[0] for c in group_cols}
        row_agg["N_Seeds"] = len(g)
        for m in ["Accuracy", "AUC", "F1", "Precision", "Recall"]:
            row_agg[m]             = round(g[m].mean(), 4)
            row_agg[f"{m}_Std"]    = round(g[m].std(ddof=1), 4)
            row_agg[f"{m}_CI_Low"] = round(g[f"{m}_CI_Low"].mean(), 4)
            row_agg[f"{m}_CI_High"]= round(g[f"{m}_CI_High"].mean(), 4)
        agg_parts.append(row_agg)

    df_agg = pd.DataFrame(agg_parts)
    df_agg.to_csv(OUTPUT_AGG, index=False)
    print(f"Aggregated results saved → {OUTPUT_AGG}  ({len(df_agg)} rows)")

    for exp_type in ["Synthetic→Within", "Synthetic→Cross", "Synthetic→Merged→Cross",
                     "Real→Cross", "Real+Synthetic→Cross", "Real+SyntheticDA→Cross"]:
        subset = df[df["Experiment_Type"] == exp_type]
        if subset.empty:
            continue
        print(f"\n{'='*70}")
        print(f"{exp_type}  |  AUC by Feature Strategy  (best model per cell)")
        print("=" * 70)

        rows_list = []
        for strat in FEATURE_STRATEGIES:
            sub = subset[subset["Adaptation"] == strat]
            if sub.empty:
                continue
            if exp_type == "Synthetic→Within":
                best = sub.loc[sub.groupby("Test_Dataset")["AUC"].idxmax()]
                row_dict = {"Strategy": strat}
                row_dict.update(dict(zip(best["Test_Dataset"], best["AUC"].round(3))))
            else:
                best = sub.loc[
                    sub.groupby(["Train_Dataset", "Test_Dataset"])["AUC"].idxmax()
                ]
                row_dict = {"Strategy": strat}
                for _, r in best.iterrows():
                    src = r["Train_Dataset"].replace("Synthetic_", "Syn_")
                    row_dict[f"{src}→{r['Test_Dataset']}"] = round(r["AUC"], 3)
            rows_list.append(row_dict)

        if rows_list:
            print(pd.DataFrame(rows_list).set_index("Strategy").to_string())

    print("\nDone.")


if __name__ == "__main__":
    main()
