import numpy as np
import pandas as pd
import warnings
import mne
from mne.preprocessing import ICA
from mne_icalabel import label_components
from pathlib import Path
from typing import List, Dict, Tuple
import os

warnings.simplefilter(action="ignore", category=FutureWarning)
mne.set_log_level(verbose="WARNING")
np.random.seed(42)

INPUT_DIR = Path("Data/MentalStress")
OUTPUT_CSV = Path("EEG_Dataset_MS.csv")
CHANNELS_LIST: List[str] = [
    "AF3", "AF4", "T7", "T8", "Pz",
]
CH_TYPES: List[str] = ["eeg"] * len(CHANNELS_LIST)
RAW_SFREQ: int = 128
REGION_PICKS: Dict[str, List[str]] = {
    "Frontal": ["F3", "F4"],
    "Parietal": ["Pz"],
    "Temporal": ["T7", "T8"],
}

FREQ_BANDS: Dict[str, Tuple[float, float]] = {
    "Alpha": (8.0, 13.0),
    "Beta": (13.0, 30.0),
    "Delta": (1.0, 4.0),
    "Theta": (4.0, 8.0),
    "Gamma": (30.0, 45.0),
}

FILTER_LOW: float = 1.0
FILTER_HIGH: float = 45.0
ICA_COMPONENTS: int = 4
RESAMPLE_FREQ: int = 128
EPOCH_DURATION_S: int = 10


def calculate_psd_for_epoch(epoch: mne.Epochs, region_picks: Dict[str, List[str]], freq_bands: Dict[str, Tuple[float, float]]) -> pd.DataFrame:
    all_bands_data = []

    spectrum = epoch.compute_psd(
        method="welch",
        fmin=FILTER_LOW,
        fmax=FILTER_HIGH,
        picks="eeg",
    )
    psds, freqs = spectrum.get_data(return_freqs=True)
    channel_names = spectrum.ch_names

    for band_name, (fmin, fmax) in freq_bands.items():
        freq_mask = (freqs >= fmin) & (freqs < fmax)

        for region_name, region_channels in region_picks.items():
            picks_for_region = [ch for ch in region_channels if ch in channel_names]
            channel_mask = [ch_name in picks_for_region for ch_name in channel_names]

            if not picks_for_region:
                continue

            psds_band_region = psds[:, channel_mask, :][:, :, freq_mask].mean(axis=-1)
            columns = [f"{band_name}{ch_name}" for ch_name in channel_names if ch_name in picks_for_region]

            band_df = pd.DataFrame(psds_band_region * 1e12, columns=columns)
            all_bands_data.append(band_df)

    return pd.concat(all_bands_data, axis=1)


def process_edf_file(filepath: Path) -> mne.Epochs:
    data_raw = mne.io.read_raw_edf(filepath, preload=True)

    data_raw.pick_channels(CHANNELS_LIST)

    montage = mne.channels.make_standard_montage("standard_1020")
    data_raw.set_montage(montage)
    data_raw.set_eeg_reference(ref_channels="average")

    data_filtered = data_raw.copy().filter(l_freq=FILTER_LOW, h_freq=FILTER_HIGH)

    ica = ICA(
        n_components=ICA_COMPONENTS,
        method="infomax",
        random_state=42,
        fit_params=dict(extended=True)
    )

    ica.fit(data_filtered)

    ic_labels = label_components(data_filtered, ica, method="iclabel")
    labels = ic_labels["labels"]

    exclude_idx = [
        idx for idx, label in enumerate(labels)
        if label not in ["brain", "other"]
    ]

    reconstructed = data_filtered.copy()
    ica.apply(reconstructed, exclude=exclude_idx)

    data_filtered = reconstructed.filter(l_freq=FILTER_LOW, h_freq=FILTER_HIGH)
    data_filtered = data_filtered.resample(sfreq=RESAMPLE_FREQ)

    data_filtered.rename_channels({"AF3": "F3", "AF4": "F4"})

    epochs = mne.make_fixed_length_epochs(
        data_filtered,
        duration=EPOCH_DURATION_S,
        preload=True
    )

    return epochs


def main():
    final_dataset = pd.DataFrame()
    tasks = ["Baseline", "Stroop"]

    for subject in range(1, 21):
        print(f"Processing Subject: {subject}")
        for task in tasks:
            filepath = os.path.join(INPUT_DIR, task, f"{subject}.edf")
            try:
                epochs = process_edf_file(filepath)

                if len(epochs) == 0:
                    print(f"  Skipping {filepath.name}: No epochs created.")
                    continue

                for i in range(len(epochs)):
                    single_epoch_obj = epochs[i]
                    psd_features = calculate_psd_for_epoch(single_epoch_obj, REGION_PICKS, FREQ_BANDS)
                    psd_features["Subject"] = str(subject)
                    psd_features["Index"] = str(i)
                    psd_features["Task"] = 0 if task == "Baseline" else 1 if task == "Stroop" else None
                    final_dataset = pd.concat([final_dataset, psd_features], axis=0, ignore_index=True)

            except Exception as e:
                print(f"  Error processing {filepath.name}: {e}")
                continue

    if not final_dataset.empty:
        eeg_cols = [col for col in final_dataset.columns if col in psd_features.columns and col not in ["Subject", "Index", "Task"]]
        meta_cols = [col for col in final_dataset.columns if col not in eeg_cols]
        final_dataset = final_dataset[eeg_cols + meta_cols]
        final_dataset.to_csv(OUTPUT_CSV, index=False)


if __name__ == "__main__":
    main()
