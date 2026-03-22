import warnings
import os
import itertools
import numpy as np
import pandas as pd
from scipy.linalg import sqrtm
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import ot

warnings.filterwarnings("ignore")

DATA_DIR  = "Processed"
OUT_DIR   = "."

N_SYNTHETIC = 70
CORR_THRESH = 0.2
NOISE_LEVEL = 0.05
RANDOM_STATE = 42

OT_REG_E  = 0.5
OT_REG_CL = 0.1
OT_MAX_TGT = 500

PROTO_ALPHA = 0.5

BANDS = ["Alpha", "Beta", "Delta", "Theta", "Gamma"]
COMMON_FEATURES = [f"{b} {r}" for b in BANDS for r in ["Frontal", "Parietal"]]

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


def _region_mean(df, band, channels):
    cols = [f"{band}{ch}" for ch in channels if f"{band}{ch}" in df.columns]
    return df[cols].mean(axis=1).values


def load_real(name: str):
    cfg  = DATASET_CONFIG[name]
    path = os.path.join(DATA_DIR, cfg["real_file"])
    df   = pd.read_csv(path)
    df   = df[df[cfg["label_col"]].isin([0, 1])].reset_index(drop=True)

    agg = {}
    for band in BANDS:
        agg[f"{band} Frontal"]  = _region_mean(df, band, cfg["frontal_ch"])
        agg[f"{band} Parietal"] = _region_mean(df, band, cfg["parietal_ch"])
    feat_df = pd.DataFrame(agg)[COMMON_FEATURES]

    feat_arr = np.log1p(feat_df.clip(lower=0).values.astype(float))

    groups = cfg["participant_fn"](df).values
    for subj in np.unique(groups):
        mask = groups == subj
        mu   = feat_arr[mask].mean(axis=0)
        sd   = feat_arr[mask].std(axis=0)
        sd[sd == 0] = 1.0
        feat_arr[mask] = (feat_arr[mask] - mu) / sd

    y = df[cfg["label_col"]].values.astype(int)
    return feat_arr, y


def _interpolate_sample(s1, s2, noise_level):
    alpha = np.random.uniform(0.3, 0.7)
    new   = s1 + (s2 - s1) * alpha
    noise = (np.random.normal(0, 1, new.shape)
             * np.std(np.vstack([s1, s2]), axis=0)
             * noise_level)
    return new + noise


def generate_base_synthetic(X_src, y_src, n=N_SYNTHETIC):
    scaler = StandardScaler().fit(X_src)
    X_norm = scaler.transform(X_src)

    class_counts = np.bincount(y_src, minlength=2)
    class_probs  = class_counts / len(y_src)
    n_per_class  = np.round(class_probs * n).astype(int)
    n_per_class[-1] = n - n_per_class[:-1].sum()

    samples, labels = [], []
    for cls in [0, 1]:
        n_cls = n_per_class[cls]
        if n_cls == 0:
            continue
        cls_idx      = np.where(y_src == cls)[0]
        cls_features = X_norm[cls_idx]

        cls_corr = np.corrcoef(cls_features)
        np.fill_diagonal(cls_corr, 0)
        cls_pairs = np.argwhere(cls_corr > CORR_THRESH)

        if len(cls_pairs) == 0:
            n_c = len(cls_features)
            cls_pairs = np.array([(i, j)
                                   for i in range(n_c)
                                   for j in range(n_c) if i != j])

        for _ in range(n_cls):
            li, lj   = cls_pairs[np.random.randint(len(cls_pairs))]
            new_norm = _interpolate_sample(cls_features[li], cls_features[lj],
                                           NOISE_LEVEL)
            samples.append(new_norm)
            labels.append(cls)

    X_syn = scaler.inverse_transform(np.array(samples))
    y_syn = np.array(labels, dtype=int)
    return X_syn, y_syn


def coral_adapt(X_syn, X_tgt, reg=1e-5):
    d     = X_syn.shape[1]
    mu_s  = X_syn.mean(axis=0)
    mu_t  = X_tgt.mean(axis=0)
    C_syn = np.cov((X_syn - mu_s).T) + reg * np.eye(d)
    C_tgt = np.cov((X_tgt - mu_t).T) + reg * np.eye(d)
    A     = np.real(sqrtm(np.linalg.inv(C_syn))) @ np.real(sqrtm(C_tgt))
    return (X_syn - mu_s) @ A + mu_t


def ot_adapt(X_syn, y_syn, X_tgt, rng):
    if len(X_tgt) > OT_MAX_TGT:
        idx = rng.choice(len(X_tgt), OT_MAX_TGT, replace=False)
        X_tgt_sub = X_tgt[idx]
    else:
        X_tgt_sub = X_tgt

    transport = ot.da.SinkhornLpl1Transport(
        reg_e=OT_REG_E,
        reg_cl=OT_REG_CL,
        max_iter=200,
        verbose=False,
    )
    transport.fit(Xs=X_syn, ys=y_syn.astype(float), Xt=X_tgt_sub)
    X_adapted = transport.transform(Xs=X_syn)
    return X_adapted


def prototype_adapt(X_syn, y_syn, X_src, y_src, X_tgt, alpha=PROTO_ALPHA):
    src_means = np.stack([X_src[y_src == c].mean(axis=0)
                          for c in [0, 1]], axis=0)

    km = KMeans(n_clusters=2, random_state=RANDOM_STATE, n_init=20)
    km.fit(X_tgt)
    centroids = km.cluster_centers_

    dist_mat = np.linalg.norm(
        centroids[:, None, :] - src_means[None, :, :], axis=2
    )
    row_ind, col_ind = linear_sum_assignment(dist_mat)
    tgt_means = np.zeros_like(src_means)
    for r, c in zip(row_ind, col_ind):
        tgt_means[c] = centroids[r]

    X_adapted = X_syn.copy()
    for cls in [0, 1]:
        mask = y_syn == cls
        if mask.sum() == 0:
            continue
        X_adapted[mask] = X_syn[mask] + alpha * (tgt_means[cls] - src_means[cls])
    return X_adapted


def save_adapted(X_adapted, y_syn, src, tgt, method):
    df = pd.DataFrame(X_adapted, columns=COMMON_FEATURES)
    df["Label"] = y_syn
    fname = f"Synthetic_Adapted_{src}_to_{tgt}_{method}.csv"
    path  = os.path.join(OUT_DIR, fname)
    df.to_csv(path, index=False)
    lb = dict(zip(*np.unique(y_syn, return_counts=True)))
    print(f"    Saved {fname}  labels={lb}")
    return path


def main():
    np.random.seed(RANDOM_STATE)
    rng = np.random.default_rng(RANDOM_STATE)

    datasets = list(DATASET_CONFIG.keys())
    pairs    = list(itertools.permutations(datasets, 2))

    print("=" * 70)
    print("Domain-Adapted Synthetic EEG Generation")
    print(f"Pairs: {len(pairs)}  ×  Methods: CORAL, OT, Prototype")
    print(f"Synthetic samples per CSV: {N_SYNTHETIC}")
    print("=" * 70)

    print("\nLoading all real datasets …")
    real_data = {}
    for ds in datasets:
        X, y = load_real(ds)
        real_data[ds] = (X, y)
        print(f"  {ds}: {len(y)} epochs  label={dict(zip(*np.unique(y, return_counts=True)))}")

    print()
    for src, tgt in pairs:
        print(f"  [{src} → {tgt}]")

        X_src, y_src = real_data[src]
        X_tgt, _     = real_data[tgt]

        X_syn, y_syn = generate_base_synthetic(X_src, y_src, n=N_SYNTHETIC)

        try:
            X_coral = coral_adapt(X_syn, X_tgt)
            save_adapted(X_coral, y_syn, src, tgt, "CORAL")
        except Exception as e:
            print(f"    [CORAL] ERROR: {e}")

        try:
            X_ot = ot_adapt(X_syn, y_syn, X_tgt, rng)
            save_adapted(X_ot, y_syn, src, tgt, "OT")
        except Exception as e:
            print(f"    [OT]    ERROR: {e}")

        try:
            X_proto = prototype_adapt(X_syn, y_syn, X_src, y_src, X_tgt)
            save_adapted(X_proto, y_syn, src, tgt, "Prototype")
        except Exception as e:
            print(f"    [Proto] ERROR: {e}")

    total = len(pairs) * 3
    print(f"\nDone. {total} adapted synthetic CSVs written.")


if __name__ == "__main__":
    main()
