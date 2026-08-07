# Echoes of War: Transferable Damage Monitoring with Space Radar

**Alex Shaw, MSc GIS, University of Edinburgh**

This repository adapts the open-source war damage mapping pipeline developed by [Dietrich et al. (2025)](https://www.nature.com/articles/s43247-025-02183-7) for Ukraine to the Gaza Strip, and tests whether the Gaza-trained model generalises to three further conflicts (Mosul, Raqqa, Aleppo) without retraining.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## Key findings

- The Gaza-trained model achieved **88.9% balanced accuracy** at a calibrated threshold of t=0.670 (90% precision)
- **151,368 buildings (68.5% of all studied)** were classified as damaged across Gaza
- Applied without retraining to Mosul, Raqqa and Aleppo, balanced accuracy held at **64.0–64.8%**
- Local retraining on Mosul data did not improve on zero-shot transfer once thresholds were properly calibrated


---

## Study area and data

| Component | Source | Details |
|---|---|---|
| SAR imagery | Sentinel-1 (Copernicus / ESA) | GRD Product, VV+VH, IW mode, via Google Earth Engine |
| Damage labels | UNOSAT Gaza Comprehensive Damage Assessments | 14 releases, Oct 2023–Oct 2025; 220,820 assessed structures |
| Building footprints | HOTOSM Gaza Buildings | 220,820 structures after 50 m² filter |
| Admin boundaries | OCHA COD-AB Palestine | Governorate level (admin2) |
| Transfer cities | UNOSAT products 1188 (Mosul), 1192 (Raqqa), 1118 (Aleppo) | Single release per city |

**Training areas (Gaza):** North Gaza, Gaza City  
**Test areas (Gaza):** Deir al-Balah, Khan Yunis, Rafah

**Training areas (Mosul retrain):** West bank (lon < 43.1262°E), 6,184 points  
**Test areas (Mosul retrain):** East bank (lon ≥ 43.1262°E), 7,250 points

---

## Key methodological adaptations from Dietrich et al. (2025)

| Adaptation | Rationale |
|---|---|
| Two-month assessment windows instead of three-month | Matched the frequency of UNOSAT Gaza releases |
| 14 UNOSAT assessment epochs (October 2023–October 2025) | Provided full conflict-period coverage and ~80 times as many damage observations |
| `first_severe` epoch-combining instead of most recent damage class | Gaza assessments captured worsening damage, not only refined classifications |
| HOTOSM building footprints instead of Overture Maps | Provided the largest pre-war Gaza building inventory |
| Sentinel-2 excluded | No performance improvement in original study |
| Feature extraction, training and inference moved to local HPC | Gaza's data volume exceeded GEE processing limits |
| scikit-learn Random Forest instead of GEE SMILE | Required for local model execution; same hyperparameters retained |
| Pixel-level ablation instead of point-level | Allows direct comparison with main results |
| Transfer evaluation (Mosul, Raqqa, Aleppo) | Tests Dietrich et al.'s (2025) claim their model "will adapt well to new areas" |
| Mosul local retraining comparison | Tests whether local retraining improves on zero-shot transfer |

---
## Project structure
```
gaza-damage-mapping/
├── check_environment.py              # Verifies all required packages are installed
├── reauth_gdrive.py                  # Refreshes Google Drive credentials
├── requirements.txt                  # Pinned direct dependencies
├── setup.py / setup.cfg / pyproject.toml  # Package metadata
├── LICENSE / README.md / .gitignore  # Licence, docs, ignored paths
├── secrets/.gitkeep                  # Placeholder for gitignored credentials
│
├── src/
│   ├── constants.py                  # Study areas, time periods, paths, thresholds
│   │
│   ├── classification/               # Model training, evaluation and ablation
│   │   ├── main_local.py             # Trains the production Random Forest
│   │   ├── models_local.py           # scikit-learn classifier factory
│   │   ├── dataset_local.py          # Train/test splits by governorate AOI
│   │   ├── metrics.py                # Shared evaluation metrics
│   │   ├── reducers.py               # SAR statistical reducers
│   │   ├── utils.py                  # Run name and feature name helpers
│   │   └── ablation/                 # Ablation and threshold calibration
│   │       ├── ablation_pixel_level.py  # n_trees, polarisation, statistics ablation
│   │       ├── ablation_mtry.py         # Full mtry sweep on complete training set
│   │       └── threshold_sweep.py       # Gaza threshold calibration (90% precision)
│   │
│   ├── data/                         # Data preparation and feature extraction
│   │   ├── unosat.py                 # Label loading, epoch combining
│   │   ├── quadkeys.py               # Quadkey tiling grid for GEE export
│   │   ├── utils.py                  # AOI and data helpers
│   │   ├── hotosm/                   # HOTOSM footprint download and filtering (2 scripts)
│   │   ├── sentinel1/                # S1 collection, time series extraction (5 scripts)
│   │   └── transfer_cities/          # Transfer city data and evaluation
│   │       ├── constants_transfer.py    # Transfer city settings and orbits
│   │       ├── preprocess_transfer_unosat.py  # Standardise UNOSAT fields
│   │       ├── pixel_inference/         # Zero-shot inference and evaluation (4 scripts)
│   │       └── retrain/                 # Mosul local retraining (5 scripts)
│   │
│   ├── inference/                    # Gaza pixel-level inference (3 scripts)
│   ├── postprocessing/               # Building-level classification (3 scripts)
│   ├── utils/                        # GEE, Drive, geometry, timing helpers (4 scripts)
│   └── visualisation/                # Figure generation
│       ├── plot/                     # Chart scripts (5 scripts)
│       └── prep/                     # Data preparation for figures (4 scripts)
│
└── test_sites/
    └── processed/                    # Preprocessed UNOSAT labels and AOIs
        ├── alp/                      # Aleppo
        ├── mos/                      # Mosul (includes east-bank split)
        └── raq/                      # Raqqa
```
---

## Prerequisites

- Python 3.12+
- Google account with access to Google Earth Engine
- Google Cloud Console project with the Google Drive API enabled

---

## Setup

Developed on the University of Edinburgh Forth HPC cluster (Python 3.12.3, Ubuntu 24.04).

### 1. Clone the repository

```bash
git clone https://github.com/alexjrshaw/gaza-damage-mapping.git
cd gaza-damage-mapping
```

### 2. Python environment

```bash
python3 -m venv alex
source alex/bin/activate
pip install -r requirements.txt
```

### 3. Verify your environment

```bash
python3 check_environment.py
```

### 4. Google Earth Engine

```bash
earthengine authenticate
earthengine set_project gaza-damage-mapping
```

Update `ASSETS_PATH` in `src/constants.py` to point to your own GEE project if needed.

### 5. Google Drive credentials

Required for downloading GEE feature raster exports.

1. Go to [Google Cloud Console](https://console.cloud.google.com) → Enable **Google Drive API**
2. Create an OAuth client ID (Desktop app) → download `client_secrets.json`
3. Place `client_secrets.json` in `secrets/`
4. Create `secrets/pydrive_settings.yaml`:

```yaml
client_config_backend: 'file'
client_config_file: secrets/client_secrets.json
save_credentials: True
save_credentials_backend: 'file'
save_credentials_file: secrets/pydrive_credentials.json
get_refresh_token: True
oauth_scope:
  - "https://www.googleapis.com/auth/drive"
```

Run `python3 reauth_gdrive.py` once on the login node to trigger the OAuth flow and cache credentials.

### 6. Data

**UNOSAT labels** — download all 14 Gaza releases from [unosat.org](https://unosat.org) (product IDs 3714–4213) and place GDB files in `data/raw/`. Then run:

```bash
python3 src/data/unosat.py
```

**HOTOSM buildings** — downloaded automatically:

```bash
python3 src/data/hotosm/preprocessing.py
```

**Admin boundaries** — downloaded automatically from OCHA HDX.

**Sentinel-1** — processed via Google Earth Engine (see pipeline below).

---

## Running the pipeline

### A. Gaza (main pipeline)

Steps 1–4 require internet access and should run interactively on the login node in a persistent `screen` session.

**1. Extract Sentinel-1 time series (GEE)**
```bash
python3 src/data/sentinel1/intermediate_data.py
```

**2. Download intermediate assets to Forth**
```bash
python3 src/data/sentinel1/download_intermediate_assets.py
```

**3. Compute SAR features locally**
```bash
python3 src/data/sentinel1/extract_features_local.py
```

**4. Train and evaluate classifier**
```bash
python3 src/classification/main_local.py
```

**5. Export feature rasters from GEE**
```bash
python3 src/inference/export_feature_rasters.py
```

**6. Download feature rasters from Drive**
```bash
python3 src/inference/download_feature_rasters.py
```

**7. Run pixel-level inference**
```bash
python3 src/inference/local_pixel_inference.py
```

**8. Aggregate to buildings and classify**
```bash
python3 src/postprocessing/pixel_postprocessing.py
python3 src/postprocessing/classify_building_damage.py
```

**9. Calibrate threshold**
```bash
python3 src/classification/ablation/threshold_sweep.py
```

### B. Cross-conflict transfer (Mosul, Raqqa, Aleppo)

**1. Export feature rasters**
```bash
python3 src/data/transfer_cities/pixel_inference/export_feature_rasters_transfer.py --city MOS
```

**2. Download rasters and run inference**
```bash
python3 src/data/transfer_cities/pixel_inference/download_feature_rasters_transfer.py --city MOS
python3 src/data/transfer_cities/pixel_inference/pixel_inference_transfer.py --city MOS
```

**3. Evaluate**
```bash
python3 src/data/transfer_cities/pixel_inference/evaluate_pixel_transfer.py --city MOS
```

Replace `MOS` with `RAQ` or `ALP` for Raqqa and Aleppo.

### C. Mosul local retraining

**1. Create east-bank test labels**
```bash
python3 src/data/transfer_cities/retrain/create_mosul_east_bank_labels.py
```

**2. Train Mosul-specific model**
```bash
python3 src/data/transfer_cities/retrain/main_local_mosul_retrain.py
```

**3. Run inference with retrained model**
```bash
python3 src/data/transfer_cities/retrain/mosul_retrain_pixel_inference.py
```

**4. Find optimal threshold**
```bash
python3 src/data/transfer_cities/retrain/mosul_optimal_threshold.py
```

### D. Ablation study

Run after Step 4 of the Gaza pipeline.

**1. Pixel-level ablation (n_trees, polarisation, SAR statistics)**
```bash
python3 src/classification/ablation/ablation_pixel_level.py
```

**2. mtry sweep**
```bash
python3 src/classification/ablation/ablation_mtry.py
```

**3. Generate figures**
```bash
python3 src/visualisation/plot/plot_ablation_summary.py
python3 src/visualisation/plot/plot_oob_mtry.py
python3 src/visualisation/plot/plot_oob_ntrees.py
python3 src/visualisation/plot/plot_threshold_sweeps.py
```

---

## Citation

If you use this code, please cite the original Ukraine methodology:

```bibtex
@article{Dietrich2025,
  author={Dietrich, Olivier and Peters, Torben and Sainte Fare Garnot, Vivien
          and Sticher, Valerie and Ton-That Whelan, Thao and Schindler, Konrad
          and Wegner, Jan Dirk},
  title={An open-source tool for mapping war destruction at scale in Ukraine
         using Sentinel-1 time series},
  journal={Communications Earth & Environment},
  year={2025},
  doi={10.1038/s43247-025-02183-7}
}
```


---

## Licence

MIT — see [LICENSE](LICENSE) for details.
