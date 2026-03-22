# Cross-Dataset EEG Stress Classification via Synthetic Data Generation and Domain Adaptation

Cross-dataset generalization remains a critical limitation in EEG-based stress classification, primarily due to variability in acquisition hardware, subject populations, and experimental paradigms. While prior work has focused on improving within-dataset performance, the role of synthetic data in enabling robust cross-dataset transfer remains underexplored. This study presents a systematic evaluation of synthetic data generation and domain-adapted synthesis for cross-dataset EEG stress classification across four publicly available datasets using a unified band-power feature representation. Five experimental conditions were compared across all ordered source-target pairs, including real-data baselines, synthetic-only training, standard augmentation, and domain-adapted synthesis using covariance alignment, optimal transport, and prototype alignment. A substantial domain gap was observed, with cross-dataset performance falling consistently below within-dataset baselines. Training exclusively on synthetic data improved cross-dataset performance beyond real-data training in most source-target pairs, while standard augmentation yielded only marginal gains. Domain-adapted synthesis achieved the most consistent improvements, with the largest gains observed in transfers between datasets with systematic distributional offsets, particularly workload-to-stress transfers where the mismatch was structured and correctable. Analysis of transfer patterns revealed that shared task paradigm does not guarantee transferability: two datasets employing the same stress induction task produced among the worst cross-dataset results, while paradigm-mismatched pairs with different hardware configurations achieved the strongest transfer. These results indicate that hardware, electrode montage, and procedural differences can dominate paradigm similarity in determining cross-dataset performance, and that low channel-count source data may act as an implicit regularizer by forcing models to learn spectrally broad features. A source-target taxonomy based on paradigm relationship is proposed to guide the selection of domain adaptation strategies. The findings demonstrate that synthetic data can capture transferable representations that improve cross-dataset generalization, that domain adaptation is most effective when distributional mismatch is structured rather than noisy, and that empirical assessment of distributional alignment should take precedence over paradigm-level assumptions when designing cross-dataset EEG classification systems.

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

Trains classifiers (LDA, Logistic Regression, Random Forest, SVM-RBF, XGBoost) on each synthetic source and evaluates on real target data across all 12 source→target pairs and 5 random seeds.

```bash
python cross_dataset_generalization_synthetic.py
```

Output:
- `cross_dataset_results_synthetic.csv` — full per-seed results
- `cross_dataset_results_synthetic_agg.csv` — aggregated mean ± std across seeds

---

### Step 4 — Plot results

Generates publication-quality figures from the aggregated results CSV.

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
