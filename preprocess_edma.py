import numpy as np
import pandas as pd
import mne
import warnings
from mne.preprocessing import ICA
from mne_icalabel import label_components
from pathlib import Path
from typing import List, Dict, Tuple

warnings.simplefilter(action="ignore", category=FutureWarning)
mne.set_log_level(verbose="WARNING")
np.random.seed(42)

INPUT_DIR = Path("Data/EDMA")
OUTPUT_CSV = Path("EEG_Dataset_EDMA.csv")

CHANNELS_TO_PICK: List[str] = [
    "Fp1", "Fp2", "F3", "F4", "C3", "C4", "P3", "P4",
    "O1", "O2", "F7", "F8", "Fz", "Pz", "Cz",
]

REGION_PICKS: Dict[str, List[str]] = {
    "Frontal": ["Fp1", "Fp2", "F3", "F4", "F7", "F8", "Fz"],
    "Central": ["C3", "C4", "Cz"],
    "Parietal": ["P3", "P4", "Pz"],
    "Occipital": ["O1", "O2"],
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
ICA_COMPONENTS: int = 10
RESAMPLE_FREQ: int = 128
EPOCH_DURATION_S: int = 10


def calculate_psd_for_epoch(epoch: mne.Epochs, region_picks: Dict[str, List[str]], freq_bands: Dict[str, Tuple[float, float]],) -> pd.DataFrame:
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
            channel_mask = [ch_name in region_channels for ch_name in channel_names]
            psds_band_region = psds[:, channel_mask, :][:, :, freq_mask].mean(axis=-1)
            columns = [f"{band_name}{ch_name}" for ch_name in channel_names if ch_name in region_channels]
            band_df = pd.DataFrame(psds_band_region * 1e12, columns=columns)
            all_bands_data.append(band_df)
    return pd.concat(all_bands_data, axis=1)


def process_raw_file(filepath: Path, condition_label: int, condition_name: str, subject_id: str) -> pd.DataFrame:
    raw = mne.io.read_raw_edf(filepath, infer_types=True, preload=True)
    if "ECG" in raw.ch_names:
        raw.set_channel_types(mapping={"ECG": "ecg"})
    raw = raw.pick_channels(CHANNELS_TO_PICK)

    montage = mne.channels.make_standard_montage("standard_1020")
    raw.set_montage(montage)
    raw.set_eeg_reference(ref_channels="average")

    data_filtered = raw.filter(l_freq=FILTER_LOW, h_freq=FILTER_HIGH)

    ica = ICA(
        n_components=ICA_COMPONENTS,
        method="infomax",
        random_state=42,
        fit_params=dict(extended=True),
    )
    ica.fit(data_filtered)
    ic_labels = label_components(data_filtered, ica, method="iclabel")
    labels = ic_labels["labels"]
    exclude_idx = [
        idx for idx, label in enumerate(labels) if label not in ["brain", "other"]
    ]

    reconstructed = data_filtered.copy()
    ica.apply(reconstructed, exclude=exclude_idx)

    data_filtered = reconstructed.copy().filter(l_freq=FILTER_LOW, h_freq=FILTER_HIGH)
    data_filtered = data_filtered.copy().resample(sfreq=RESAMPLE_FREQ)
    epochs = mne.make_fixed_length_epochs(
        data_filtered, duration=EPOCH_DURATION_S, preload=True
    )

    subject_epoch_data = []
    for i in range(len(epochs)):
        single_epoch_obj = epochs[i]
        psd_features = calculate_psd_for_epoch(
            single_epoch_obj, REGION_PICKS, FREQ_BANDS
        )
        psd_features["Subject"] = f"EEGStress{condition_name}_{subject_id}_{i}"
        psd_features["Label"] = condition_label
        subject_epoch_data.append(psd_features)

    if subject_epoch_data:
        return pd.concat(subject_epoch_data, axis=0, ignore_index=True)
    else:
        return pd.DataFrame()


def main():
    all_subject_data = []

    subjects = [f"{i:02d}" for i in range(36)]
    conditions = [
        ("1", 0, "Baseline"),
        ("2", 1, "Stressed"),
    ]

    for subject_id in subjects:
        print(f"Processing Subject: {subject_id}")

        for file_suffix, label, name in conditions:
            file_to_process = INPUT_DIR / name / f"Subject{subject_id}_{file_suffix}.edf"

            if not file_to_process.exists():
                print(f"File not found, skipping: {file_to_process}")
                continue

            try:
                subject_data = process_raw_file(
                    file_to_process, label, name, subject_id
                )
                if not subject_data.empty:
                    all_subject_data.append(subject_data)
            except Exception as e:
                print(f"Failed to process {file_to_process}: {e}")

    if not all_subject_data:
        return

    full_dataset = pd.concat(all_subject_data, axis=0, ignore_index=True)
    full_dataset.to_csv(OUTPUT_CSV, index=False)


if __name__ == "__main__":
    main()
