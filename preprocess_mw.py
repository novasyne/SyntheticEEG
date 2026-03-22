import numpy as np
import pandas as pd
import json
import warnings
import mne
from mne.preprocessing import ICA
from mne_icalabel import label_components
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Tuple

warnings.simplefilter(action="ignore", category=FutureWarning)
mne.set_log_level(verbose="WARNING")
np.random.seed(42)

INPUT_DIR = Path("Data/MentalWorkloadEEG/data_flight_simulator/eeg")
JSON_DIR = Path("Data/MentalWorkloadEEG/data_flight_simulator/perceived_difficulty")
OUTPUT_CSV = Path("EEG_Dataset_MW.csv")

CHANNELS_TO_PICK: List[str] = [
    "AF3", "AF4", "F3", "F4", "F7", "F8", "FC5",
    "FC6", "P7", "P8", "T7", "T8", "O1", "O2",
]
CH_TYPES: List[str] = ["eeg"] * len(CHANNELS_TO_PICK)
RAW_SFREQ: int = 128

REGION_PICKS: Dict[str, List[str]] = {
    "Frontal": ["AF3", "AF4", "F3", "F4", "F7", "F8"],
    "Central": ["FC5", "FC6"],
    "Parietal": ["P7", "P8"],
    "Occipital": ["O1", "O2"],
    "Temporal": ["T7", "T8"],
}

FREQ_BANDS: Dict[str, Tuple[float, float]] = {
    "Delta": (1.0, 4.0),
    "Theta": (4.0, 8.0),
    "Alpha": (8.0, 13.0),
    "Beta": (13.0, 30.0),
    "Gamma": (30.0, 45.0),
}

FILTER_LOW: float = 1.0
FILTER_HIGH: float = 45.0
ICA_COMPONENTS: int = 10
RESAMPLE_FREQ: int = 128
EPOCH_DURATION_S: int = 10

OUTPUT_COLUMNS = [
    "DeltaAF3", "DeltaAF4", "DeltaF3", "DeltaF4", "DeltaF7", "DeltaF8",
    "DeltaFC5", "DeltaFC6", "DeltaP7", "DeltaP8", "DeltaO1", "DeltaO2",
    "DeltaT7", "DeltaT8",
    "ThetaAF3", "ThetaAF4", "ThetaF3", "ThetaF4", "ThetaF7", "ThetaF8",
    "ThetaFC5", "ThetaFC6", "ThetaP7", "ThetaP8", "ThetaO1", "ThetaO2",
    "ThetaT7", "ThetaT8",
    "AlphaAF3", "AlphaAF4", "AlphaF3", "AlphaF4", "AlphaF7", "AlphaF8",
    "AlphaFC5", "AlphaFC6", "AlphaP7", "AlphaP8", "AlphaO1", "AlphaO2",
    "AlphaT7", "AlphaT8",
    "BetaAF3", "BetaAF4", "BetaF3", "BetaF4", "BetaF7", "BetaF8",
    "BetaFC5", "BetaFC6", "BetaP7", "BetaP8", "BetaO1", "BetaO2",
    "BetaT7", "BetaT8",
    "GammaAF3", "GammaAF4", "GammaF3", "GammaF4", "GammaF7", "GammaF8",
    "GammaFC5", "GammaFC6", "GammaP7", "GammaP8", "GammaO1", "GammaO2",
    "GammaT7", "GammaT8",
    "Subject", "Label", "Index",
]

JSON_FLIGHT_MAP = {
    "flight_1.json":   [1],
    "flight_2_4.json": [2, 4],
    "flight_3_5.json": [3, 5],
}


def parse_timestamp(ts_str: str) -> datetime:
    return datetime.fromisoformat(ts_str)


def load_flight_phases(json_dir: Path) -> Dict[int, List[Dict]]:
    flight_phases = {}

    for json_file, flight_numbers in JSON_FLIGHT_MAP.items():
        fpath = json_dir / json_file
        if not fpath.exists():
            print(f"Warning: {fpath} not found, skipping flights {flight_numbers}")
            continue

        with open(fpath) as f:
            data = json.load(f)

        phases = data["phases"]
        pair_idx = 0
        for flight_num in flight_numbers:
            baseline_phase = phases[pair_idx]
            game_phase = phases[pair_idx + 1]

            assert baseline_phase["type"] == "BASELINE", \
                f"Expected BASELINE at index {pair_idx} in {json_file}, got {baseline_phase['type']}"
            assert game_phase["type"] == "GAME", \
                f"Expected GAME at index {pair_idx+1} in {json_file}, got {game_phase['type']}"

            flight_phases[flight_num] = [
                {
                    "type": "BASELINE",
                    "start": parse_timestamp(baseline_phase["start"]),
                    "end": parse_timestamp(baseline_phase["end"]),
                    "duration_s": (parse_timestamp(baseline_phase["end"]) -
                                   parse_timestamp(baseline_phase["start"])).total_seconds(),
                },
                {
                    "type": "GAME",
                    "start": parse_timestamp(game_phase["start"]),
                    "end": parse_timestamp(game_phase["end"]),
                    "duration_s": (parse_timestamp(game_phase["end"]) -
                                   parse_timestamp(game_phase["start"])).total_seconds(),
                },
            ]
            pair_idx += 2

    return flight_phases


def calculate_psd_for_epoch(
    epoch: mne.Epochs,
    region_picks: Dict[str, List[str]],
    freq_bands: Dict[str, Tuple[float, float]],
) -> pd.DataFrame:
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


def preprocess_raw_eeg(eeg_subset: pd.DataFrame) -> mne.io.RawArray:
    eeg_values = eeg_subset[[f"EEG.{ch}" for ch in CHANNELS_TO_PICK]]
    eeg_values = pd.DataFrame.transpose(eeg_values)

    info = mne.create_info(ch_names=CHANNELS_TO_PICK, ch_types=CH_TYPES, sfreq=RAW_SFREQ)
    data_raw = mne.io.RawArray(eeg_values, info)
    montage = mne.channels.make_standard_montage("standard_1020")
    data_raw.set_montage(montage)
    data_raw.set_eeg_reference(ref_channels="average")

    data_filtered = data_raw.filter(l_freq=FILTER_LOW, h_freq=FILTER_HIGH)

    ica = ICA(
        n_components=ICA_COMPONENTS,
        method="infomax",
        random_state=42,
        fit_params=dict(extended=True),
    )
    ica.fit(data_filtered)
    ic_labels = label_components(data_filtered, ica, method="iclabel")
    labels = ic_labels["labels"]
    exclude_idx = [idx for idx, label in enumerate(labels) if label not in ["brain", "other"]]
    reconstructed = data_filtered.copy()
    ica.apply(reconstructed, exclude=exclude_idx)

    data_filtered = reconstructed.copy().filter(l_freq=FILTER_LOW, h_freq=FILTER_HIGH)
    data_filtered = data_filtered.copy().resample(sfreq=RESAMPLE_FREQ)

    return data_filtered


def extract_epochs_for_phase(
    data_filtered: mne.io.RawArray,
    recording_start: datetime,
    phase: Dict,
) -> mne.Epochs:
    tmin = (phase["start"] - recording_start).total_seconds()
    tmax = (phase["end"] - recording_start).total_seconds()
    raw_tmax = data_filtered.times[-1]
    tmin = max(0.0, tmin)
    tmax = min(raw_tmax, tmax)
    if tmax - tmin < EPOCH_DURATION_S:
        return None
    cropped = data_filtered.copy().crop(tmin=tmin, tmax=tmax)
    epochs = mne.make_fixed_length_epochs(cropped, duration=EPOCH_DURATION_S, preload=True)
    return epochs if len(epochs) > 0 else None


def main():
    all_rows = []
    flight_phases = load_flight_phases(JSON_DIR)
    print(f"Loaded phase definitions for flights: {sorted(flight_phases.keys())}")

    try:
        eeg_data = pd.read_parquet(
            INPUT_DIR / "eeg.parquet",
            engine="pyarrow",
        )
    except FileNotFoundError as e:
        print(f"Error loading EEG data: {e}")
        return

    subjects = sorted(eeg_data["subject"].unique())
    for subject in subjects:
        for flight in sorted(flight_phases.keys()):
            try:
                eeg_subset = eeg_data.loc[
                    (eeg_data["subject"] == subject) & (eeg_data["flight"] == flight)
                ]

                if eeg_subset.empty:
                    print(f"  Skipping Flight {flight}: no EEG data for subject {subject}")
                    continue

                phases = flight_phases[flight]
                recording_start = min(p["start"] for p in phases)
                data_filtered = preprocess_raw_eeg(eeg_subset)
                for phase in phases:
                    label = 0 if phase["type"] == "BASELINE" else 1

                    epochs = extract_epochs_for_phase(data_filtered, recording_start, phase)
                    if epochs is None:
                        print(f"    {phase['type']}: too short for {EPOCH_DURATION_S}s epochs, skipping")
                        continue
                    for i in range(len(epochs)):
                        single_epoch = epochs[i]
                        psd_features = calculate_psd_for_epoch(single_epoch, REGION_PICKS, FREQ_BANDS)
                        psd_features["Subject"] = subject
                        psd_features["Label"] = label
                        psd_features["Index"] = str(i)
                        all_rows.append(psd_features)

            except Exception as e:
                print(f"  Error processing Subject {subject}, Flight {flight}: {e}")
                continue

    final_dataset = pd.concat(all_rows, axis=0, ignore_index=True)
    final_dataset = final_dataset[OUTPUT_COLUMNS]
    final_dataset.to_csv(OUTPUT_CSV, index=False)


if __name__ == "__main__":
    main()
