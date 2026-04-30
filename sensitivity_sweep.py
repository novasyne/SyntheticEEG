import os
import itertools
import warnings
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from sklearn.base import clone
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import roc_auc_score
import xgboost as xgb

warnings.filterwarnings("ignore")
matplotlib.rcParams.update({"figure.dpi": 150, "savefig.dpi": 300,
                             "savefig.bbox": "tight", "font.size": 13})

DATA_DIR  = "Processed"
SEEDS     = [42, 43, 44, 45, 46]
N_SWEEP   = [30, 50, 70, 100, 150, 200]

BANDS = ["Alpha", "Beta", "Delta", "Theta", "Gamma"]
COMMON_FEATURES = sorted(
    [f"{b} Frontal" for b in BANDS] + [f"{b} Parietal" for b in BANDS]
)

CORR_THRESH  = 0.2
NOISE_LEVEL  = 0.05

DATASET_CONFIG = {
    "EDMA": {
        "real_file":      "EEG_Dataset_EDMA.csv",
        "label_col":      "Label",
        "frontal_ch":     ["Fp1", "Fp2", "F3", "F4", "F7", "F8", "Fz"],
        "parietal_ch":    ["P3", "P4", "Pz"],
        "participant_fn": lambda df: df["Subject"].str.extract(
            r"EEGStress(?:Baseline|Stressed)_(\d+)_\d+"
        )[0].fillna("unknown"),
    },
    "MS": {
        "real_file":      "EEG_Dataset_MS.csv",
        "label_col":      "Task",
        "frontal_ch":     ["F3", "F4"],
        "parietal_ch":    ["Pz"],
        "participant_fn": lambda df: df["Subject"].astype(str),
    },
    "MW": {
        "real_file":      "EEG_Dataset_MW.csv",
        "label_col":      "Label",
        "frontal_ch":     ["AF3", "AF4", "F3", "F4", "F7", "F8"],
        "parietal_ch":    ["P7", "P8"],
        "participant_fn": lambda df: df["Subject"].astype(str),
    },
    "SAM": {
        "real_file":      "EEG_Dataset_SAM.csv",
        "label_col":      "Label",
        "frontal_ch":     ["F3", "F4", "F7", "F8", "Fz",
                           "FC1", "FC2", "FC5", "FC6", "FT9", "FT10"],
        "parietal_ch":    ["P3", "P4", "Pz", "PO9", "PO10"],
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

def _region_mean(df, band, channels):
    cols = [f"{band}{ch}" for ch in channels if f"{band}{ch}" in df.columns]
    return df[cols].mean(axis=1).values


def load_real(name):
    cfg  = DATASET_CONFIG[name]
    path = os.path.join(DATA_DIR, cfg["real_file"])
    df   = pd.read_csv(path)
    df   = df[df[cfg["label_col"]].isin([0, 1])].reset_index(drop=True)

    agg = {}
    for band in BANDS:
        agg[f"{band} Frontal"]  = _region_mean(df, band, cfg["frontal_ch"])
        agg[f"{band} Parietal"] = _region_mean(df, band, cfg["parietal_ch"])

    feat_arr = np.log1p(pd.DataFrame(agg)[COMMON_FEATURES].clip(lower=0).values.astype(float))

    groups = cfg["participant_fn"](df).values
    for subj in np.unique(groups):
        mask = groups == subj
        mu   = feat_arr[mask].mean(axis=0)
        sd   = feat_arr[mask].std(axis=0)
        sd[sd == 0] = 1.0
        feat_arr[mask] = (feat_arr[mask] - mu) / sd

    y = df[cfg["label_col"]].values.astype(int)
    return feat_arr, y

def generate_interp_synthetic(X_src, y_src, n, rng):
    scaler  = StandardScaler().fit(X_src)
    X_norm  = scaler.transform(X_src)

    counts   = np.bincount(y_src, minlength=2)
    props    = counts / len(y_src)
    n_per    = np.round(props * n).astype(int)
    n_per[-1] = n - n_per[:-1].sum()

    samples, labels = [], []
    for cls in [0, 1]:
        n_cls = n_per[cls]
        if n_cls == 0:
            continue
        idx  = np.where(y_src == cls)[0]
        feats = X_norm[idx]

        corr = np.corrcoef(feats)
        np.fill_diagonal(corr, 0)
        pairs = np.argwhere(corr > CORR_THRESH)
        if len(pairs) == 0:
            nc = len(feats)
            pairs = np.array([(i, j) for i in range(nc) for j in range(nc) if i != j])

        for _ in range(n_cls):
            i1, i2 = pairs[rng.integers(len(pairs))]
            alpha   = rng.uniform(0.3, 0.7)
            new     = feats[i1] + (feats[i2] - feats[i1]) * alpha
            noise   = (rng.standard_normal(new.shape)
                       * np.std(np.vstack([feats[i1], feats[i2]]), axis=0)
                       * NOISE_LEVEL)
            samples.append(new + noise)
            labels.append(cls)

    X_syn = scaler.inverse_transform(np.array(samples))
    y_syn = np.array(labels, dtype=int)
    return X_syn, y_syn

def get_models(seed):
    return {
        "SVM-RBF":            SVC(kernel="rbf", probability=True, random_state=seed),
        "XGBoost":            xgb.XGBClassifier(random_state=seed, verbosity=0,
                                                 eval_metric="logloss", nthread=1),
        "LDA":                LinearDiscriminantAnalysis(),
        "Random Forest":      RandomForestClassifier(random_state=seed, n_jobs=1),
        "Logistic Regression":LogisticRegression(max_iter=1000, random_state=seed,
                                                 solver="lbfgs"),
    }


def tune_and_fit(model_name, base_model, X_train, y_train, seed):
    grid = PARAM_GRIDS.get(model_name, {})
    if not grid or np.bincount(y_train).min() < 3:
        base_model.fit(X_train, y_train)
        return base_model
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=seed)
    search = GridSearchCV(clone(base_model), grid, cv=cv,
                          scoring="roc_auc", n_jobs=1, refit=True)
    try:
        search.fit(X_train, y_train)
        return search.best_estimator_
    except Exception:
        base_model.fit(X_train, y_train)
        return base_model


def evaluate_pair(X_train, y_train, X_test, y_test, model_name, base_model, seed):
    if len(np.unique(y_train)) < 2 or len(np.unique(y_test)) < 2:
        return None
    scaler = StandardScaler()
    X_tr   = scaler.fit_transform(X_train)
    X_te   = scaler.transform(X_test)
    model  = tune_and_fit(model_name, clone(base_model), X_tr, y_train, seed)
    proba  = model.predict_proba(X_te)[:, 1]
    return float(roc_auc_score(y_test, proba))

def run_sweep():
    datasets = list(DATASET_CONFIG.keys())
    pairs    = list(itertools.permutations(datasets, 2))

    print("Loading real datasets …")
    real = {ds: load_real(ds) for ds in datasets}
    for ds, (X, y) in real.items():
        print(f"  {ds}: {len(y)} epochs  classes={np.bincount(y).tolist()}")

    rows = []
    total = len(N_SWEEP) * len(SEEDS) * len(pairs) * len(PARAM_GRIDS)
    done  = 0

    for n in N_SWEEP:
        print(f"\n── n={n} ─────────────────────────────────────────────")
        for seed in SEEDS:
            rng = np.random.default_rng(seed)
            models = get_models(seed)

            synth = {}
            for src in datasets:
                X_src, y_src = real[src]
                synth[src] = generate_interp_synthetic(X_src, y_src, n, rng)

            for src, tgt in pairs:
                X_syn, y_syn = synth[src]
                X_tgt, y_tgt = real[tgt]

                for model_name, base_model in models.items():
                    auc = evaluate_pair(X_syn, y_syn, X_tgt, y_tgt,
                                        model_name, base_model, seed)
                    done += 1
                    if auc is not None:
                        rows.append({
                            "n":      n,
                            "seed":   seed,
                            "source": src,
                            "target": tgt,
                            "model":  model_name,
                            "auc":    round(auc, 4),
                        })
                    if done % 100 == 0:
                        print(f"  {done}/{total} runs completed …")

    return pd.DataFrame(rows)


def summarise(df):
    mean_per_model = (df.groupby(["n", "source", "target", "model"])["auc"]
                      .mean().reset_index(name="mean_auc"))

    best_idx = mean_per_model.groupby(["n", "source", "target"])["mean_auc"].idxmax()
    best     = mean_per_model.loc[best_idx].copy()

    summary = (best.groupby("n")["mean_auc"]
               .agg(["mean", "sem", "std"])
               .reset_index()
               .rename(columns={"mean": "mean_auc", "sem": "sem_auc", "std": "std_auc"}))
    summary["mean_auc"] = summary["mean_auc"].round(4)
    summary["sem_auc"]  = summary["sem_auc"].round(4)
    summary["std_auc"]  = summary["std_auc"].round(4)
    return summary


def plot_summary(summary, out="sensitivity_sweep_plot.png"):
    fig, ax = plt.subplots(figsize=(8, 5))

    ax.errorbar(summary["n"], summary["mean_auc"],
                yerr=summary["sem_auc"],
                marker="o", linewidth=2, capsize=5, color="#1f4e79",
                label="Mean AUC (best model per pair)")

    ref = summary[summary["n"] == 70]
    if not ref.empty:
        ax.axvline(70, color="grey", linestyle="--", linewidth=1.2, alpha=0.7)
        ax.text(71, ref["mean_auc"].values[0] + 0.003, "n=70\n(paper)", fontsize=10, color="grey")

    ax.set_xlabel("Number of synthetic samples (n)", fontsize=13)
    ax.set_ylabel("Mean AUC (Synthetic→Cross)", fontsize=13)
    ax.set_title("Sensitivity of Cross-Dataset AUC to Synthetic Sample Size\n"
                 "(Interpolation method, best model per pair, ±SEM over 12 pairs)",
                 fontsize=12)
    ax.set_xticks(summary["n"].tolist())
    ax.yaxis.set_minor_locator(matplotlib.ticker.MultipleLocator(0.01))
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    ax.legend()
    plt.tight_layout()
    plt.savefig(out)
    plt.close()
    print(f"Plot saved → {out}")


if __name__ == "__main__":
    import matplotlib.ticker

    print("=" * 60)
    print("Sensitivity Sweep: Synthetic Sample Size vs. AUC")
    print(f"n values: {N_SWEEP}  |  Seeds: {SEEDS}")
    print(f"12 source-target pairs  |  5 classifiers")
    print("=" * 60)

    results_df = run_sweep()
    results_df.to_csv("sensitivity_sweep_results.csv", index=False)
    print(f"\nDetailed results → sensitivity_sweep_results.csv  "
          f"({len(results_df)} rows)")

    summary_df = summarise(results_df)
    summary_df.to_csv("sensitivity_sweep_summary.csv", index=False)

    print("\nSummary — Mean AUC (best model per pair) vs. n:")
    print(summary_df.to_string(index=False))

    plot_summary(summary_df)
    print("\nDone.")
