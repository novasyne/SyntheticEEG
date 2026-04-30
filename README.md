# Synthetic EEG Data for Cross-Dataset Stress Classification

This repository investigates whether synthetically generated EEG data can substitute for or augment real training data in cross-dataset stress and mental workload classification. Four publicly available EEG datasets are used, and three synthesis strategies are evaluated: correlation-guided interpolation, conditional GAN (WGAN-GP), and conditional VAE. A domain-adaptation stage (CORAL, Optimal Transport, Prototype-shift) is also included.

---

## Repository Structure

```
.
├── Data/                          # Raw EEG datasets (user-provided, see below)
│   ├── EDMA/
│   ├── SAM40/
│   ├── MentalWorkloadEEG/
│   └── MentalStress/
├── Processed/                     # Preprocessed CSVs (ready-to-use, skip Step 1)
├── Synthetic/                     # Generated synthetic CSVs (ready-to-use, skip Step 2)
├── preprocess_edma.py
├── preprocess_sam.py
├── preprocess_mw.py
├── preprocess_ms.py
├── synthetic_edma.py
├── synthetic_sam.py
├── synthetic_mw.py
├── synthetic_ms.py
├── synthetic_gan_vae.py
├── synthetic_domain_adapted.py
├── cross_dataset_generalization_synthetic.py
├── within_loso_real.py
├── sensitivity_sweep.py
├── mw_exclusion.py
├── plot_results.py
└── requirements.txt
```

---

## Datasets

The four datasets must be downloaded separately and placed under `Data/`:

| Dataset | Folder | Format | Description |
|---------|--------|--------|-------------|
| [EDMA](https://zenodo.org/record/7318435) | `Data/EDMA/` | `.edf` | 36-subject stress/baseline EEG |
| [SAM40](https://www.kaggle.com/datasets/hmason123/sam40-eeg-dataset) | `Data/SAM40/` | `.mat` | 40-subject Stroop/relax EEG |
| [MentalWorkloadEEG](https://zenodo.org/record/7413447) | `Data/MentalWorkloadEEG/` | `.parquet` + `.json` | Flight simulator workload EEG |
| [MentalStress](https://physionet.org/content/eegmat/1.0.0/) | `Data/MentalStress/` | `.edf` | 20-subject Stroop stress EEG |

Expected layout within `Data/`:

```
Data/
├── EDMA/
│   ├── Baseline/
│   │   └── Subject00_1.edf  ...
│   └── Stressed/
│       └── Subject00_2.edf  ...
├── SAM40/
│   ├── Baseline/
│   │   └── Relax_sub_1_trial1.mat  ...
│   └── Stroop/
│       └── Stroop_sub_1_trial1.mat  ...
├── MentalWorkloadEEG/
│   └── data_flight_simulator/
│       ├── eeg/
│       │   └── eeg.parquet
│       └── perceived_difficulty/
│           ├── flight_1.json
│           ├── flight_2_4.json
│           └── flight_3_5.json
└── MentalStress/
    ├── Baseline/
    │   └── 1.edf  ...
    └── Stroop/
        └── 1.edf  ...
```

> **Note:** The `Processed/` and `Synthetic/` folders already contain fully preprocessed and generated data. You can skip Steps 1 and 2 entirely and proceed directly to Step 3 if you do not wish to reprocess from raw data.

---

## Installation

Python 3.12 is required.

```bash
# Clone the repository
git clone <repository-url>
cd <repository-folder>

# Create and activate a virtual environment
python3.12 -m venv venv
source venv/bin/activate        # Linux / macOS
# venv\Scripts\activate         # Windows

# Install all dependencies
pip install -r requirements.txt
```

---

## Run Sequence

### Step 1 — Preprocess raw EEG data

Each script reads from `Data/`, applies bandpass filtering, ICA artefact rejection, epoching, and PSD feature extraction, then writes a CSV to the project root (which should then be moved to `Processed/`).

> Skip this step if using the pre-built files in `Processed/`.

```bash
python preprocess_edma.py
python preprocess_sam.py
python preprocess_mw.py
python preprocess_ms.py
```

Output: `EEG_Dataset_EDMA.csv`, `EEG_Dataset_SAM.csv`, `EEG_Dataset_MW.csv`, `EEG_Dataset_MS.csv`

---

### Step 2 — Generate synthetic data

Run the interpolation scripts first (one per dataset), then the GAN/VAE script (covers all datasets), then the domain-adaptation script (covers all source→target pairs).

> Skip this step if using the pre-built files in `Synthetic/`.

```bash
# Correlation-guided interpolation (one CSV per dataset)
python synthetic_edma.py
python synthetic_sam.py
python synthetic_mw.py
python synthetic_ms.py

# Conditional GAN and VAE (generates _GAN and _VAE variants for all datasets)
python synthetic_gan_vae.py

# Domain-adapted synthesis: CORAL, OT, Prototype (all 12 source→target pairs)
python synthetic_domain_adapted.py
```

Output (written to `Synthetic/`):
- `Synthetic_Aggregated_{DS}.csv` — interpolation-based (DS = EDMA, SAM, MW, MS)
- `Synthetic_Aggregated_{DS}_GAN.csv` / `_VAE.csv` — deep generative models
- `Synthetic_Adapted_{SRC}_to_{TGT}_{METHOD}.csv` — domain-adapted variants (METHOD = CORAL, OT, Prototype)

---

### Step 3 — Cross-dataset evaluation

Trains classifiers (LDA, Logistic Regression, Random Forest, SVM-RBF, XGBoost) on each synthetic source and evaluates on real target data across all 12 source→target pairs and 5 random seeds. Experiment types covered:

- `Synthetic→Within` — proper LOSO: synthetic data generated from held-in subjects per fold, tested on held-out subject's real data (interpolation only)
- `Synthetic→Cross` — train on synthetic from source dataset, test on real target (all three synthesis methods)
- `Real→Cross` — train on real source, test on real target (baseline)
- `Real+Synthetic→Cross` — augment real source with synthetic, test on real target
- `Real+SyntheticDA→Cross` — augment real source with domain-adapted synthetic, test on real target

```bash
python cross_dataset_generalization_synthetic.py
```

Output:
- `cross_dataset_results_synthetic.csv` — full per-seed results
- `cross_dataset_results_synthetic_agg.csv` — aggregated mean ± std across seeds

---

### Step 4 — Within-dataset LOSO evaluation (real data)

Evaluates all five classifiers on each dataset under leave-one-subject-out cross-validation using only real EEG data. Provides the within-dataset real-data upper-bound reference (Table 2 replacement). Uses the same 10-dimensional band-power features, log transform, per-subject z-score normalisation, and hyperparameter grids as Step 3.

```bash
python within_loso_real.py
```

Output:
- `within_loso_real_results.csv` — per-fold, per-model, per-seed rows
- `within_loso_real_summary.csv` — per-dataset mean AUC ± SEM (best model per seed, averaged over 5 seeds)

---

### Step 5 — Sensitivity sweep: synthetic sample size

Sweeps the number of interpolation-based synthetic samples (`n` ∈ {30, 50, 70, 100, 150, 200}) across all 12 source→target pairs and 5 seeds. For each `n`, synthetic data is generated fresh from the real source data. The best-mean-AUC model is selected per pair per `n`, and results are averaged across pairs to produce a summary curve.

```bash
python sensitivity_sweep.py
```

Output:
- `sensitivity_sweep_results.csv` — one row per (n, seed, source, target, model)
- `sensitivity_sweep_summary.csv` — mean AUC per n (best model per pair, mean ± SEM over 12 pairs)
- `sensitivity_sweep_plot.png` — line plot of mean AUC vs. n with ±SEM error bars

---

### Step 6 — MW exclusion check

Re-evaluates all key conclusions using only the three non-workload datasets (EDMA, MS, SAM40), giving 6 source→target pairs instead of 12. Loads the aggregated results from Step 3 and filters out any row where the Mental Workload (MW) dataset appears as source or target. Reports mean AUC per condition for the full 12-pair and MW-excluded 6-pair sets, and checks whether the paper's main rankings hold.

> Requires `cross_dataset_results_synthetic_agg.csv` from Step 3.

```bash
python mw_exclusion.py
```

Output:
- `mw_exclusion_summary.csv` — mean AUC per condition (full vs. MW-excluded)
- `mw_exclusion_pairs.csv` — best-model AUC per source→target pair for each condition

---

### Step 7 — Plot results

Generates publication-quality figures from the aggregated results CSV.

> Requires `cross_dataset_results_synthetic_agg.csv` from Step 3.

```bash
python plot_results.py
```

Output:
- `fig_model_comparison.png` — mean cross-dataset AUC per classifier by training condition
- `fig_condition_comparison.png` — mean cross-dataset AUC per training condition with SEM bars

---

## Key Dependencies

| Package | Purpose |
|---------|---------|
| `mne` / `mne-icalabel` | EEG preprocessing, ICA artefact rejection |
| `numpy` / `pandas` / `scipy` | Numerical computing and data handling |
| `scikit-learn` | Classifiers, scaling, cross-validation |
| `xgboost` | Gradient boosting classifier |
| `torch` | Conditional GAN and VAE training |
| `POT` | Optimal Transport domain adaptation |
| `matplotlib` | Figure generation |
| `pyarrow` | Reading the MW dataset Parquet file |

Full pinned versions are listed in `requirements.txt`.
