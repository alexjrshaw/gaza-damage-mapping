# Cross-Conflict Building Damage Monitoring From Space

**Alex Shaw, MSc GIS, University of Edinburgh**

This repository adapts the open-source war damage mapping pipeline developed by [Dietrich et al. (2025)](https://www.nature.com/articles/s43247-025-02183-7) for Ukraine to the Gaza Strip, and tests whether the Gaza-trained model generalises to three further conflicts (Mosul, Raqqa, Aleppo) without retraining.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## Key findings

- The Gaza-trained model detected **85.8% of held-out UNOSAT damage** (balanced accuracy 88.9%) at a calibrated threshold of t=0.67
- **151,368 buildings (68.5% of all studied)** were classified as damaged across Gaza
- Applied without retraining to Mosul, Raqqa and Aleppo, balanced accuracy held at **64.0–64.8%** despite differing geography and conflict history
- Local retraining on Mosul data did not improve on zero-shot transfer once thresholds were properly calibrated

---

## Study area and data

| Component | Source | Details |
|---|---|---|
| SAR imagery | Sentinel-1 (Copernicus / ESA) | GRD Product, VV+VH, IW mode, via Google Earth Engine |
| Damage labels | UNOSAT Gaza 'Comprehensive Damage Assessments' | 14 releases, Oct 2023–Oct 2025; 198,308 assessed structures |
| Building footprints | HOTOSM Gaza Buildings | 330,079 outlines; 220,820 after 50m² filter |
| Admin boundaries | OCHA COD-AB Palestine | Governorate level (admin2) |
| Transfer cities | UNOSAT: products 1188 (Mosul), 1192 (Raqqa), 1118 (Aleppo) | Single release per city |

**Training areas (Gaza):** North Gaza, Gaza City
**Test areas (Gaza):** Deir al-Balah, Khan Younis, Rafah

**Training areas (Mosul retrain):** West bank (lon < 43.1262°E), 6,184 points
**Test areas (Mosul retrain):** East bank (lon ≥ 43.1262°E), 7,250 points

---

## Key methodological adaptations from Dietrich et al. (2025)

| Adaptation | Rationale |
|---|---|
| HOTOSM footprints instead of Overture Maps | Largest available Gaza building inventory |
| Sentinel-2 excluded | No performance improvement (Dietrich et al., 2025, Supplementary Note 6) |
| Two-month assessment windows instead of three-month | Matched UNOSAT Gaza release cadence |
| 14 assessment epochs (Oct 2023–Oct 2025) | Full two-year conflict coverage |
| Feature computation, training and inference moved to local HPC | Gaza's point density exceeded GEE's computational limits |
| scikit-learn Random Forest instead of GEE SMILE | Required by local feature computation; same hyperparameters retained |
| Cross-conflict transfer evaluation (Mosul, Raqqa, Aleppo) | Dietrich et al. (2025) argue their model "will adapt well to new areas." This study provides empirical proof |
| Mosul local retraining comparison | Tests whether training on local data improves on zero-shot transfer, and under what conditions |

---

## Repository structure
```
gaza-damage-mapping/
├── check_environment.py
├── requirements.txt
├── reauth_gdrive.py
├── setup.py / setup.cfg / pyproject.toml
├── LICENSE / README.md / .gitignore
├── secrets/.gitkeep
├── src/
│   ├── classification/    (8 scripts)
│   ├── data/
│   │   ├── hotosm/        (3 scripts)
│   │   ├── sentinel1/     (5 scripts)
│   │   ├── transfer_cities/
│   │   │   ├── pixel_inference/  (4 scripts)
│   │   │   └── retrain/          (6 scripts)
│   │   ├── unosat.py, quadkeys.py, utils.py
│   ├── inference/         (3 scripts)
│   ├── postprocessing/    (8 scripts)
│   ├── utils/             (4 scripts)
│   └── visualisation/     (5 scripts)
└── test_sites/
    ├── processed/ (Mosul, Raqqa, Aleppo)
    └── raw/       (Mosul, Raqqa, Aleppo + excluded: Fallujah, Myanmar)
```
---

## Setup

Developed on the University of Edinburgh Forth HPC cluster (Python 3.10.12, Ubuntu).

### 1. Clone the repository

```bash
git clone https://github.com/alexjrshaw/gaza-damage-mapping.git
cd gaza-damage-mapping
```

### 2. Python environment

```bash
python3 -m venv alex
source alex/bin/activate        # Linux/Mac
pip install -r requirements.txt
```

### 3. Verify your environment

```bash
python3 check_environment.py
```

This checks all required packages are installed and reports any that are missing.

### 4. Google Earth Engine

You need a GEE account with access to the `gaza-damage-mapping` project, or your own registered cloud project.

```bash
earthengine authenticate
earthengine set_project gaza-damage-mapping
```

Update `ASSETS_PATH` in `src/constants.py` to point to your GEE project if using your own.

### 5. Google Drive credentials

Required for downloading feature rasters exported by GEE. GEE exports to a shared Drive folder; the download scripts poll Drive and delete files after download to manage quota.

1. Go to [Google Cloud Console](https://console.cloud.google.com) → Enable **Google Drive API**
2. Create an OAuth client ID (Desktop app) → download `client_secrets.json`
3. Place `client_secrets.json` in the `secrets/` folder
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

Note: Drive authentication must be completed interactively on the Forth login node before running any download scripts. Run `python3 src/utils/gdrive.py` once to trigger the OAuth flow and cache credentials.

### 6. Data

All data is either downloaded automatically or publicly available:

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

### Gaza (main pipeline)

Steps 1–4 require internet access and should run interactively on the Forth login node in persistent `screen` sessions.

**1. Upload UNOSAT labels to GEE**
```bash
python3 src/data/unosat.py
```

**2. Extract Sentinel-1 intermediate time series (GEE)**
```bash
python3 src/data/sentinel1/intermediate_data.py
```

**3. Download intermediate assets to Forth**
```bash
python3 src/data/sentinel1/download_intermediate_assets.py
```

**4. Compute features locally**
```bash
python3 src/data/sentinel1/extract_features_local.py
```

**5. Train and evaluate classifier**
```bash
python3 src/classification/main_local.py
```

**6. Export feature rasters from GEE**
```bash
python3 src/inference/export_feature_rasters.py
```

**7. Download feature rasters from Drive**
```bash
python3 src/inference/download_feature_rasters.py
```

**8. Run local pixel inference**
```bash
python3 src/inference/local_pixel_inference.py
```

**9. Postprocess: aggregate to buildings and classify**
```bash
python3 src/postprocessing/pixel_postprocessing.py
python3 src/postprocessing/classify_building_damage.py
```

### Cross-conflict transfer (Mosul, Raqqa, Aleppo)

Steps 1–3 require internet access and must run interactively on the Forth login node.

**1. Upload transfer city UNOSAT labels to GEE**
```bash
python3 src/data/transfer_cities/upload_unosat_to_gee.py
```

**2. Export feature rasters**
```bash
python3 src/data/transfer_cities/pixel_inference/export_feature_rasters_transfer.py
```

**3. Download and run inference**
```bash
python3 src/data/transfer_cities/pixel_inference/download_feature_rasters_transfer.py
python3 src/data/transfer_cities/pixel_inference/pixel_inference_transfer.py
```
*Inference applies the Gaza-trained model unchanged (zero-shot transfer).*

**4. Evaluate**
```bash
python3 src/data/transfer_cities/pixel_inference/evaluate_pixel_transfer.py
```
*Evaluation is run at two thresholds: t=0.5 (default) and t=0.670 (Gaza-calibrated, 90% precision target).*

### Mosul local retraining

Tests whether training on Mosul's own data improves on zero-shot transfer.

**1. Create east-bank test labels**
```bash
python3 src/data/transfer_cities/retrain/create_mosul_east_bank_labels.py
```
*Splits Mosul UNOSAT points along the Tigris River (lon=43.1262°E): west bank for training, east bank for testing.*

**2. Train Mosul-specific model**
```bash
python3 src/data/transfer_cities/retrain/main_local_mosul_retrain.py
```

**3. Run pixel inference with retrained model**
```bash
python3 src/data/transfer_cities/retrain/mosul_retrain_pixel_inference.py
```
*Sweeps t=0.0–1.0 to find the threshold achieving 90% precision on Mosul's own data, mirroring how Gaza's t=0.670 was derived. The retrained model's optimal threshold: t=0.44*

**4. Find best threshold for retrained model**
```bash
python3 src/data/transfer_cities/retrain/mosul_optimal_threshold.py
```

**5. Compare zero-shot vs retrained**
```bash
python3 src/data/transfer_cities/retrain/verify_mosul_retrain_comparison.py
```

### Ablation study

Run after Step 5 of the Gaza pipeline (trained model and feature rasters must exist).

**1. Pixel-level ablation**
```bash
python3 src/classification/ablation_pixel_level.py
```
*Runs all ablation variants at t=0.670 (Gaza-calibrated threshold)*

**2. Mtry ablation**
```bash
python3 src/classification/ablation_mtry.py
```
*Tests mtry values 1–25 via OOB error during training.*

**3. Make figures**
```bash
python3 src/visualisation/plot_ablation_figures.py
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
