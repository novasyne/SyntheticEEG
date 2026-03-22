import warnings
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

np.random.seed(42)

INPUT_FILE = "Processed/EEG_Dataset_EDMA.csv"
OUTPUT_ORIGINAL_AGG = "Original_Aggregated_EDMA.csv"
OUTPUT_SYNTHETIC_AGG = "Synthetic_Aggregated_EDMA.csv"
FILE_SUFFIX = "EDMA"

N_SYNTHETIC_SAMPLES = 70
SYNTHESIS_CORR_THRESHOLD = 0.2
SYNTHESIS_NOISE_LEVEL = 0.05


def load_and_aggregate_data(filepath: str) -> pd.DataFrame:
    try:
        data = pd.read_csv(filepath)
    except FileNotFoundError:
        print(f"Error: Input file not found at {filepath}")
        return pd.DataFrame()

    if "Subject" in data.columns:
        participant = data["Subject"].str.extract(
            r"EEGStress(?:Baseline|Stressed)_(\d+)_\d+"
        )[0].fillna("unknown")
    else:
        participant = pd.Series(["all"] * len(data))

    if "Subject" in data.columns:
        data = data.drop(columns=["Subject"])
    if "Index" in data.columns:
        data = data.drop(columns=["Index"])

    data = data[data["Label"].isin([0, 1])].reset_index(drop=True)
    participant = participant[data.index].reset_index(drop=True)

    regions = {
        "Frontal":  ["Fp1", "Fp2", "F3", "F4", "F7", "F8", "Fz"],
        "Parietal": ["P3", "P4", "Pz"],
    }
    bands = ["Alpha", "Beta", "Delta", "Theta", "Gamma"]

    agg_df = pd.DataFrame()
    for band in bands:
        for region_name, region_channels in regions.items():
            cols = [f"{band}{ch}" for ch in region_channels if f"{band}{ch}" in data.columns]
            if cols:
                agg_df[f"{band} {region_name}"] = data[cols].mean(axis=1)

    agg_df["Label"] = data["Label"]

    feat_cols = [c for c in agg_df.columns if c != "Label"]

    agg_df[feat_cols] = np.log1p(agg_df[feat_cols].clip(lower=0))

    agg_df["_participant"] = participant.values
    for subj in agg_df["_participant"].unique():
        mask = agg_df["_participant"] == subj
        mu = agg_df.loc[mask, feat_cols].mean()
        sd = agg_df.loc[mask, feat_cols].std().replace(0, 1)
        agg_df.loc[mask, feat_cols] = (agg_df.loc[mask, feat_cols] - mu) / sd
    agg_df = agg_df.drop(columns=["_participant"])

    return agg_df


def generate_synthetic_sample(s1: np.ndarray, s2: np.ndarray, noise_level: float) -> np.ndarray:
    interpolation_factor = np.random.uniform(0.3, 0.7)
    new_sample = s1 + (s2 - s1) * interpolation_factor
    noise = (
        np.random.normal(0, 1, new_sample.shape)
        * np.std(np.vstack((s1, s2)), axis=0)
        * noise_level
    )
    synthetic_sample = new_sample + noise
    return synthetic_sample


def create_synthetic_dataset(original_df: pd.DataFrame, n_samples: int, task_cols: list) -> pd.DataFrame:
    meta_cols = ["Stress", "Trial"] + task_cols
    existing_meta_cols = [c for c in meta_cols if c in original_df.columns]

    labels = original_df["Label"].values.astype(int)
    features_df = original_df.drop(columns=existing_meta_cols)
    scaler = StandardScaler().fit(features_df)
    features_norm = scaler.transform(features_df)

    class_counts = np.bincount(labels, minlength=2)
    class_probs = class_counts / len(labels)
    n_per_class = np.round(class_probs * n_samples).astype(int)
    n_per_class[-1] = n_samples - n_per_class[:-1].sum()

    synthetic_samples = []
    synthetic_meta = []

    for cls in [0, 1]:
        n_cls = n_per_class[cls]
        if n_cls == 0:
            continue
        cls_idx = np.where(labels == cls)[0]
        cls_features = features_norm[cls_idx]

        cls_corr = np.corrcoef(cls_features)
        np.fill_diagonal(cls_corr, 0)
        cls_pairs = np.argwhere(cls_corr > SYNTHESIS_CORR_THRESHOLD)

        if len(cls_pairs) == 0:
            n_c = len(cls_features)
            cls_pairs = np.array([(i, j) for i in range(n_c) for j in range(n_c) if i != j])

        for _ in range(n_cls):
            local_i1, local_i2 = cls_pairs[np.random.randint(len(cls_pairs))]
            new_norm_sample = generate_synthetic_sample(
                cls_features[local_i1], cls_features[local_i2], SYNTHESIS_NOISE_LEVEL
            )
            synthetic_samples.append(new_norm_sample)
            parent_meta = original_df.iloc[cls_idx[local_i1]][existing_meta_cols].copy()
            parent_meta["Label"] = cls
            synthetic_meta.append(parent_meta.values)

    synthetic_features = scaler.inverse_transform(np.array(synthetic_samples))
    synthetic_df = pd.DataFrame(synthetic_features, columns=features_df.columns)
    meta_df = pd.DataFrame(synthetic_meta, columns=existing_meta_cols)
    synthetic_df = pd.concat([synthetic_df, meta_df], axis=1)
    if "Trial" in synthetic_df.columns:
        synthetic_df["Trial"] = synthetic_df["Trial"].astype(int)
    for task in task_cols:
        if task in synthetic_df.columns:
            synthetic_df[task] = synthetic_df[task].astype(int)

    return synthetic_df


def main():
    original_agg_df = load_and_aggregate_data(INPUT_FILE)
    if original_agg_df.empty:
        return

    synthetic_agg_df = create_synthetic_dataset(
        original_agg_df,
        n_samples=N_SYNTHETIC_SAMPLES,
        task_cols=["Label"],
    )

    original_agg_df.to_csv(OUTPUT_ORIGINAL_AGG, index=False)
    synthetic_agg_df.to_csv(OUTPUT_SYNTHETIC_AGG, index=False)


if __name__ == "__main__":
    main()
