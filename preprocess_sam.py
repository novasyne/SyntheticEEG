import numpy as np
import pandas as pd
import warnings
import mne
from scipy.io import loadmat
from mne.preprocessing import ICA
from mne_icalabel import label_components
from pathlib import Path
from typing import List, Dict, Tuple

warnings.simplefilter(action="ignore", category=FutureWarning)
mne.set_log_level(verbose="WARNING")
np.random.seed(42)

INPUT_DIR = Path("Data/SAM40")
OUTPUT_CSV = Path("EEG_Dataset_SAM.csv")
CHANNELS_LIST: List[str] = [
    "Cz", "Fz", "Fp1", "F7", "F3", "FC1", "C3", "FC5", "FT9", "T7", "CP5", "CP1", "P3",
    "P7", "PO9", "O1", "Pz", "Oz", "O2", "PO10", "P8", "P4", "CP2", "CP6", "T8", "FT10",
    "FC6", "C4", "FC2", "F4", "F8", "Fp2",
]

CH_TYPES: List[str] = ["eeg"] * len(CHANNELS_LIST)
RAW_SFREQ: int = 128
REGION_PICKS: Dict[str, List[str]] = {
    "Frontal": [
        "Fp1", "Fp2", "Fz",
        "F3", "F4", "F7", "F8",
        "FC1", "FC2", "FC5", "FC6"
    ],
    "Central": [
        "Cz", "C3", "C4"
    ],
    "Parietal": [
        "Pz", "P3", "P4", "P7", "P8",
        "CP1", "CP2", "CP5", "CP6"
    ],
    "Occipital": [
        "Oz", "O1", "O2",
        "PO9", "PO10"
    ],
    "Temporal": [
        "T7", "T8",
        "FT9", "FT10"
    ],
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
ICA_COMPONENTS: int = 15
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


def preprocess_raw(filepath: Path) -> mne.io.RawArray:
    temp = loadmat(filepath, simplify_cells=True)
    df   = pd.DataFrame(temp["Data"].T, columns=CHANNELS_LIST)
    info     = mne.create_info(ch_names=CHANNELS_LIST, sfreq=RAW_SFREQ, ch_types=CH_TYPES)
    data_raw = mne.io.RawArray(df.T.values / 1e6, info)
    data_raw.set_montage(mne.channels.make_standard_montage("standard_1020"))
    data_raw.set_eeg_reference(ref_channels="average")
    data_filtered = data_raw.copy().filter(l_freq=FILTER_LOW, h_freq=FILTER_HIGH)

    ica = ICA(
        n_components=ICA_COMPONENTS,
        method="infomax",
        random_state=42,
        fit_params=dict(extended=True),
    )
    ica.fit(data_filtered)
    ic_labels   = label_components(data_filtered, ica, method="iclabel")
    exclude_idx = [
        idx for idx, label in enumerate(ic_labels["labels"])
        if label not in ["brain", "other"]
    ]
    reconstructed = data_filtered.copy()
    ica.apply(reconstructed, exclude=exclude_idx)
    data_filtered = reconstructed.filter(l_freq=FILTER_LOW, h_freq=FILTER_HIGH)
    data_filtered = data_filtered.resample(sfreq=RESAMPLE_FREQ)
    return data_filtered


def main():
    final_dataset = pd.DataFrame()
    tasks = ["Baseline", "Stroop"]

    for subject in range(1, 41):
        all_raws: List[mne.io.RawArray] = []
        task_boundaries: List[Tuple[int, int, int]] = []
        sample_offset = 0

        for task in tasks:
            task_label = 0 if task == "Baseline" else 1
            for trial in range(1, 4):
                if task == "Baseline":
                    filepath = INPUT_DIR / task / f"Relax_sub_{subject}_trial{trial}.mat"
                else:
                    filepath = INPUT_DIR / task / f"Stroop_sub_{subject}_trial{trial}.mat"

                try:
                    raw = preprocess_raw(filepath)
                except Exception as e:
                    print(f"  Error processing {filepath.name}: {e}")
                    continue

                task_boundaries.append((sample_offset, sample_offset + raw.n_times, task_label))
                sample_offset += raw.n_times
                all_raws.append(raw)

        if not all_raws:
            continue

        combined_raw = mne.concatenate_raws(all_raws)
        epochs = mne.make_fixed_length_epochs(combined_raw, duration=EPOCH_DURATION_S, preload=True)

        if len(epochs) == 0:
            print(f"  Skipping subject {subject}: No epochs created.")
            continue

        epoch_onsets = epochs.events[:, 0]
        for i, onset in enumerate(epoch_onsets):
            task_label = None
            for start, end, label in task_boundaries:
                if start <= onset < end:
                    task_label = label
                    break

            psd_features = calculate_psd_for_epoch(epochs[i], REGION_PICKS, FREQ_BANDS)
            psd_features["Subject"] = str(subject)
            psd_features["Label"] = task_label
            psd_features["Index"] = str(i)
            final_dataset = pd.concat([final_dataset, psd_features], axis=0, ignore_index=True)

    if not final_dataset.empty:
        eeg_cols = [col for col in final_dataset.columns if col not in ["Subject", "Index", "Label"]]
        final_dataset = final_dataset[eeg_cols + ["Subject", "Label", "Index"]]
        final_dataset.to_csv(OUTPUT_CSV, index=False)


if __name__ == "__main__":
    main()
