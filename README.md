# An Open-Source Tool for Mapping War Destruction in Gaza using Sentinel-1 Time Series

**Alex Shaw**, MSc GIS, University of Edinburgh

This project adapts the open-source war damage mapping tool developed by Dietrich et al. (2025) for Ukraine to the Gaza Strip, using Sentinel-1 SAR time series and UNOSAT damage assessments.

[![Original Paper](https://img.shields.io/badge/arXiv-PDF-b31b1b)](https://arxiv.org/abs/2406.02506)
[![Original Repo](https://img.shields.io/badge/Original_Repo-link-blue)](https://github.com/prs-eth/ukraine-damage-mapping-tool)
[![License: MIT](https://img.shields.io/badge/License-MIT-929292.svg)](LICENSE)

---

## Study Area

The Gaza Strip, Occupied Palestinian Territory. Five governorates:

| AOI  | Governorate   | Role     |
|------|---------------|----------|
| GAZ1 | North Gaza    | Training |
| GAZ2 | Gaza          | Training |
| GAZ3 | Deir Al-Balah | Test     |
| GAZ4 | Khan Yunis    | Test     |
| GAZ5 | Rafah         | Test     |

---

## Data Sources

SAR imagery: Sentinel-1 (Copernicus). VV+VH polarisation, IW mode, via Google Earth Engine
Damage labels: [UNOSAT Gaza CDA, 11 October 2025](https://unosat.org/products/4213). 198,308 assessed structures.
Building footprints: [HOTOSM Gaza Buildings](https://data.humdata.org/dataset/hotosm_pse_buildings). 330,079 manually delineated pre-conflict buildings.
Admin boundaries: [OCHA COD-AB Palestine](https://data.humdata.org/dataset/cod-ab-pse). Governorate level (admin2).

**Key methodological adaptations from Dietrich et al. (2025):**
- HOTOSM building footprints used instead of Overture Maps (Scher & Van Den Hoek 2025)
- Sentinel-2 excluded (no performance improvement per Dietrich et al. 2025, Supplementary Note 6)
- 2-month post-conflict windows instead of 3-month (aligned with UNOSAT revisit cadence)
- 14 assessment epochs (Oct 2023 – Oct 2025) instead of Ukraine's 2 epochs
- Damage classes 1 (Destroyed) and 2 (Severely Damaged) used for training

---

## Repository Structure

notebooks/                     # Jupyter notebooks
├── classification.ipynb
├── country_stats.ipynb
└── evaluation.ipynb
src/                           # Source code
├── classification/          # Model training and evaluation
│     ├── dataset.py
│     ├── main.py
│     ├── metrics.py
│     ├── models.py
│     ├── reducers.py
│     └── utils.py
│
├── data/                    # Data processing
│     ├── hotosm/            # HOTOSM building footprints (replaces Overture)
│     │     ├── download.py
│     │     └── preprocessing.py
│     ├── overture/          # Retained for reference (Ukraine only)
│     ├── sentinel1/         # Sentinel-1 SAR processing
│     ├── sentinel2/         # Retained for reference (not used for Gaza)
│     ├── quadkeys.py
│     ├── unosat.py
│     └── utils.py
│
├── inference/               # Full Gaza inference
│     ├── dense_inference.py
│     └── full_gaza.py
│
├── postprocessing/          # Results processing
│     ├── drive_to_results.py
│     └── utils.py
│
├── utils/                   # Utility functions
│     ├── gdrive.py
│     ├── gee.py
│     ├── geo.py
│     └── time.py
│
├── constants.py
└── init.py

---

## Setup

*Developed on University of Edinburgh Linux (Python 3.10.12) and Windows 11*

### 1. Clone the repository

```bash
git clone https://github.com/alexjrshaw/gaza-damage-mapping.git
cd gaza-damage-mapping
```

### 2. Python environment

```bash
python -m venv alex
source alex/bin/activate        # Linux/Mac
alex\Scripts\Activate.ps1       # Windows
pip install -r requirements.txt
```

### 3. Google Earth Engine

You need a GEE account with a registered cloud project.

```bash
python -c "import ee; ee.Authenticate()"
earthengine set_project YOUR-PROJECT-ID
```

Update `ASSETS_PATH` in `src/constants.py` to point to your GEE project.

### 4. Google Drive credentials

Required for downloading inference results. Follow these steps:

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

### 5. Data

All data is either downloaded automatically or publicly available:

**UNOSAT labels** — download from [unosat.org/products/4213](https://unosat.org/products/4213) and place the GDB in `data/raw/`. Then run:

```bash
python src/data/unosat.py
```

**HOTOSM buildings** — downloaded automatically when running:

```bash
python src/data/hotosm/preprocessing.py
```

**Admin boundaries** — downloaded automatically from OCHA HDX.

**Sentinel-1** — processed in the cloud via Google Earth Engine.

---

## ▶️ Running the Pipeline

### Step 1 — Upload UNOSAT labels to GEE

```bash
python src/data/unosat.py
```

### Step 2 — Extract Sentinel-1 intermediate data

```bash
python src/data/sentinel1/intermediate_data.py
```

### Step 3 — Extract features

```bash
python src/data/sentinel1/extract_features.py
```

### Step 4 — Train and evaluate classifier

```bash
python src/classification/main.py
```

### Step 5 — Run full Gaza inference

```bash
python src/inference/full_gaza.py
```

### Step 6 — Postprocess results

```bash
python src/postprocessing/drive_to_results.py
```

---

## CItation

If you use this code, please cite the original Ukraine methodology:

```bibtex
@article{Dietrich2025,
  author={Dietrich, Olivier and Peters, Torben and Sainte Fare Garnot, Vivien
          and Sticher, Valerie and Ton-That Whelan, Thao and Schindler, Konrad
          and Wegner, Jan Dirk},
  title={An open-source tool for mapping war destruction at scale in Ukraine
         using Sentinel-1 time series},
  journal={Communications Earth \& Environment},
  year={2025},
  doi={10.1038/s43247-025-02183-7}
}
```

And the HOTOSM building footprint methodology:

```bibtex
@article{ScherVanDenHoek2025,
  author={Scher, Corey and Van Den Hoek, Jamon},
  title={Active InSAR monitoring of building damage in Gaza during the
         Israel-Hamas war},
  year={2025},
  note={Preprint}
}
```

---

## Licence

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.
