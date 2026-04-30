import os
import warnings
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import (
    GridSearchCV, StratifiedKFold, LeaveOneGroupOut
)
from sklearn.metrics import (
    roc_auc_score, f1_score, precision_score, recall_score, accuracy_score
)
from sklearn.preprocessing import StandardScaler
import xgboost as xgb

warnings.filterwarnings("ignore")

DATA_DIR       = "Processed"
OUTPUT_FILE    = "within_loso_real_results.csv"
OUTPUT_SUMMARY = "within_loso_real_summary.csv"
SEEDS          = [42, 43, 44, 45, 46]
BANDS          = ["Alpha", "Beta", "Delta", "Theta", "Gamma"]
COMMON_FEATURES = sorted(
    [f"{b} Frontal" for b in BANDS] + [f"{b} Parietal" for b in BANDS]
)

DATASET_CONFIG = {
    "EDMA": {
        "real_file":   "EEG_Dataset_EDMA.csv",
        "label_col":   "Label",
        "frontal_ch":  ["Fp1", "Fp2", "F3", "F4", "F7", "F8", "Fz"],
        "parietal_ch": ["P3", "P4", "Pz"],
        "participant_fn": lambda df: df["Subject"].str.extract(
            r"EEGStress(?:Baseline|Stressed)_(\d+)_\d+"
        )[0].fillna("unknown"),
    },
    "MS": {
        "real_file":   "EEG_Dataset_MS.csv",
        "label_col":   "Task",
        "frontal_ch":  ["F3", "F4"],
        "parietal_ch": ["Pz"],
        "participant_fn": lambda df: df["Subject"].astype(str),
    },
    "MW": {
        "real_file":   "EEG_Dataset_MW.csv",
        "label_col":   "Label",
        "frontal_ch":  ["AF3", "AF4", "F3", "F4", "F7", "F8"],
        "parietal_ch": ["P7", "P8"],
        "participant_fn": lambda df: df["Subject"].astype(str),
    },
    "SAM": {
        "real_file":   "EEG_Dataset_SAM.csv",
        "label_col":   "Label",
        "frontal_ch":  ["F3", "F4", "F7", "F8", "Fz",
                        "FC1", "FC2", "FC5", "FC6", "FT9", "FT10"],
        "parietal_ch": ["P3", "P4", "Pz", "PO9", "PO10"],
        "participant_fn": lambda df: df["Subject"].astype(str),
    },
}

PARAM_GRIDS = {
    "SVM-RBF":            {"C": [0.1, 10], "gamma": ["scale", 0.1]},
    "XGBoost":            {"n_estimators": [100, 200], "max_depth": [3, 6]},
    "LDA":                {"solver": ["lsqr"], "shrinkage": [None, "auto"]},
    "Random Forest":      {"n_estimators": [100, 200], "max_depth": [None, 10]},
    "Logistic Regression":{"C": [0.1, 1, 10]},
}


def get_base_models(seed: int) -> dict:
    return {
        "SVM-RBF": SVC(kernel="rbf", probability=True, random_state=seed),
        "XGBoost": xgb.XGBClassifier(
            random_state=seed, verbosity=0, eval_metric="logloss", nthread=1
        ),
        "LDA": LinearDiscriminantAnalysis(),
        "Random Forest": RandomForestClassifier(random_state=seed, n_jobs=1),
        "Logistic Regression": LogisticRegression(
            max_iter=1000, random_state=seed, solver="lbfgs"
        ),
    }


def _region_mean(df: pd.DataFrame, band: str, channels: list) -> np.ndarray:
    cols = [f"{band}{ch}" for ch in channels if f"{band}{ch}" in df.columns]
    if not cols:
        return np.zeros(len(df))
    return df[cols].mean(axis=1).values


def load_real(name: str):
    cfg  = DATASET_CONFIG[name]
    path = os.path.join(DATA_DIR, cfg["real_file"])
    df   = pd.read_csv(path)
    df   = df[df[cfg["label_col"]].isin([0, 1])].reset_index(drop=True)

    feat_cols = {}
    for band in BANDS:
        feat_cols[f"{band} Frontal"]  = _region_mean(df, band, cfg["frontal_ch"])
        feat_cols[f"{band} Parietal"] = _region_mean(df, band, cfg["parietal_ch"])

    feat_df  = pd.DataFrame(feat_cols)[COMMON_FEATURES]
    feat_arr = np.log1p(feat_df.clip(lower=0).values.astype(float))

    groups = cfg["participant_fn"](df).values

    for subj in np.unique(groups):
        mask = groups == subj
        mu   = feat_arr[mask].mean(axis=0)
        sd   = feat_arr[mask].std(axis=0)
        sd[sd == 0] = 1
        feat_arr[mask] = (feat_arr[mask] - mu) / sd

    y = df[cfg["label_col"]].values.astype(int)
    return feat_arr, y, groups


def tune_and_fit(model_name: str, base_model, X_train: np.ndarray,
                 y_train: np.ndarray, seed: int):
    grid     = PARAM_GRIDS.get(model_name, {})
    min_cls  = np.bincount(y_train).min() if len(np.unique(y_train)) > 1 else 0
    if not grid or min_cls < 3:
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


def evaluate_within_loso(ds_name: str, seed: int) -> list:
    X, y, groups = load_real(ds_name)
    logo         = LeaveOneGroupOut()
    base_models  = get_base_models(seed)
    n_subjects   = len(np.unique(groups))

    rows = []
    for fold_idx, (train_idx, test_idx) in enumerate(
        logo.split(X, y, groups)
    ):
        X_tr_raw, X_te_raw = X[train_idx], X[test_idx]
        y_tr, y_te         = y[train_idx], y[test_idx]

        if len(np.unique(y_tr)) < 2 or len(np.unique(y_te)) < 2:
            continue

        scaler = StandardScaler()
        X_tr   = scaler.fit_transform(X_tr_raw)
        X_te   = scaler.transform(X_te_raw)

        held_out_subj = groups[test_idx[0]]

        for model_name, base_model in base_models.items():
            model   = tune_and_fit(model_name, clone(base_model), X_tr, y_tr, seed)
            y_pred  = model.predict(X_te)
            y_proba = model.predict_proba(X_te)[:, 1]

            if len(np.unique(y_te)) < 2:
                continue

            rows.append({
                "Dataset":         ds_name,
                "Fold":            fold_idx,
                "HeldOut_Subject": held_out_subj,
                "N_Subjects":      n_subjects,
                "Model":           model_name,
                "Seed":            seed,
                "N_Train":         len(y_tr),
                "N_Test":          len(y_te),
                "AUC":   round(roc_auc_score(y_te, y_proba), 4),
                "F1":    round(f1_score(y_te, y_pred, zero_division=0), 4),
                "Precision": round(precision_score(y_te, y_pred, zero_division=0), 4),
                "Recall":    round(recall_score(y_te, y_pred, zero_division=0), 4),
            })

    return rows


def main():
    print("Within-Dataset LOSO Evaluation — Real EEG Data")
    print(f"Datasets: {list(DATASET_CONFIG.keys())}")
    print(f"Seeds: {SEEDS}")
    print("=" * 60)

    all_rows = []
    for ds in DATASET_CONFIG:
        X, y, groups = load_real(ds)
        n_subj = len(np.unique(groups))
        print(f"\n{ds}: {len(y)} trials, {n_subj} unique subject IDs")
        for seed in SEEDS:
            rows = evaluate_within_loso(ds, seed)
            all_rows.extend(rows)
            n_folds = len({r["Fold"] for r in rows if r["Model"] == "LDA"})
            print(f"  seed={seed}: {n_folds} valid folds, {len(rows)} rows")

    df = pd.DataFrame(all_rows)
    df.to_csv(OUTPUT_FILE, index=False)
    print(f"\nFull results → {OUTPUT_FILE}  ({len(df)} rows)")

    print("\n" + "=" * 60)
    print("Summary (best model per seed, mean ± SEM across seeds):")
    print("=" * 60)

    summary_rows = []
    for ds in DATASET_CONFIG:
        sub = df[df["Dataset"] == ds]
        if sub.empty:
            continue

        model_seed = (
            sub.groupby(["Model", "Seed"])["AUC"]
            .mean()
            .reset_index(name="mean_fold_auc")
        )
        best_per_seed = model_seed.loc[
            model_seed.groupby("Seed")["mean_fold_auc"].idxmax()
        ].copy()

        mean_auc = best_per_seed["mean_fold_auc"].mean()
        sem_auc  = best_per_seed["mean_fold_auc"].sem()
        std_auc  = best_per_seed["mean_fold_auc"].std()
        best_model = (
            model_seed.groupby("Model")["mean_fold_auc"]
            .mean()
            .idxmax()
        )
        n_subj = sub["HeldOut_Subject"].nunique()

        print(
            f"  {ds:<6}: AUC = {mean_auc:.4f} ± {sem_auc:.4f} SEM  "
            f"(best: {best_model}, {n_subj} subjects)"
        )

        summary_rows.append({
            "Dataset":     ds,
            "N_Subjects":  n_subj,
            "N_Trials":    len(sub[sub["Seed"] == SEEDS[0]]["Fold"].map(
                lambda _: 1
            )) // len(PARAM_GRIDS),
            "Best_Model":  best_model,
            "Mean_AUC":    round(mean_auc, 4),
            "SEM_AUC":     round(sem_auc, 4),
            "Std_AUC":     round(std_auc, 4),
        })

    summary_df = pd.DataFrame(summary_rows)
    for i, row in summary_df.iterrows():
        ds  = row["Dataset"]
        sub = df[(df["Dataset"] == ds) & (df["Seed"] == SEEDS[0])]
        n_trials_total = sub["N_Train"].iloc[0] + sub["N_Test"].iloc[0] if not sub.empty else 0
        summary_df.at[i, "N_Trials"] = n_trials_total

    summary_df.to_csv(OUTPUT_SUMMARY, index=False)
    print(f"\nSummary → {OUTPUT_SUMMARY}")

    overall = summary_df["Mean_AUC"].mean()
    print(f"\nOverall mean AUC: {overall:.4f}")
    print("\nDone.")


if __name__ == "__main__":
    main()
