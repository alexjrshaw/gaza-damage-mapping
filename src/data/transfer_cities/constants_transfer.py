"""
Constants for zero-shot transfer city evaluation.

Mirrors src/constants.py structure for Gaza, adapted for three
conflict cities from the Ballinger et al. PWTT dataset.

Cities:
    ALP - Aleppo, Syria       assessed 2016-09-18 (UNOSAT product 1118)
    RAQ - Raqqa, Syria        assessed 2017-10-21 (UNOSAT product 1192)
    MOS - Mosul, Iraq         assessed 2017-08-04 (UNOSAT product 1188)

Sentinel-1 coverage confirmed:
    ALP: first image 2014-10-06, 189 images (2014-2016)
    RAQ: first image 2014-10-13, 184 images (2014-2017)
    MOS: first image 2014-10-03, 292 images (2014-2017)

Valid orbits (>1 image only):
    ALP: 116, 14, 21
    RAQ: 116, 123
    MOS: 72, 152, 145

Time period design (mirrors Gaza/Dietrich et al.):
    - PRE_PERIOD: fixed one-year baseline (T0 equivalent), used for SAR
      feature computation AND included as first POST_PERIOD entry to
      generate label=0 examples where pre==post (mirrors Gaza exactly)
    - POST_PERIODS: 2-month windows from ~pre-period start → assessment date
      Windows ending before conflict_start → label=0
      Windows ending after conflict_start  → label=1 (if date_first_severe <= end_post)

Conflict start dates and rationale:
    ALP: 2016-02-05 - Syrian government forces cut rebel supply lines,
         beginning the final siege of Aleppo. Note: actual conflict began
         July 2012, prior to Sentinel-1 coverage. Windows before 2016-02-05
         represent active but lower-intensity conflict, not true peacetime.
    RAQ: 2016-11-06 - Launch of Operation Wrath of Euphrates (SDF campaign
         to isolate and capture Raqqa from ISIS).
    MOS: 2016-10-16 - Start of the Battle of Mosul (Iraqi forces to retake
         city from ISIS, the largest urban battle since 2003).

label=0 / label=1 window counts:
    ALP: 9 label=0 (incl. pre-period), 4 label=1
    RAQ: 7 label=0 (incl. pre-period), 6 label=1
    MOS: 8 label=0 (incl. pre-period), 5 label=1
"""

from pathlib import Path

# Project paths
_THIS_FILE = Path(__file__)
SRC_PATH = _THIS_FILE.parent.parent.parent  # src/
PROJECT_PATH = SRC_PATH.parent  # repo root

TRANSFER_RAW_DIR = PROJECT_PATH / "test_sites" / "raw"
TRANSFER_DATA_DIR = PROJECT_PATH / "test_sites" / "processed"
TRANSFER_CACHE_DIR = PROJECT_PATH / "data" / "transfer_cities" / "intermediate_features_cache"
TRANSFER_FEATURES_DIR = PROJECT_PATH / "data" / "transfer_cities" / "features_ready"
TRANSFER_RUNS_DIR = PROJECT_PATH / "data" / "transfer_cities" / "runs"

# GEE asset folder for transfer city intermediate features
TRANSFER_GEE_FOLDER = "projects/gaza-damage-mapping/assets/transfer-cities/"

# Shared S1 bands (unchanged from Gaza)
S1_BANDS = ["VV", "VH"]

# Aleppo (ALP) - Syria, assessed 2016-09-18

ALP_CONFLICT_START = "2016-02-05"
ALP_ASSESSMENT_DATE = "2016-09-18"
ALP_ORBITS = [116, 14, 21]

# Fixed one-year pre-conflict baseline — mirrors Gaza PRE_PERIOD
# Starts at first available Sentinel-1 image for Aleppo (2014-10-06)
ALP_PRE_PERIOD = ("2014-10-06", "2015-10-05")

# 2-month post-windows: pre-period entry first (T0 equivalent, label=0),
# then 2-month steps from 2014-10-06 → 2016-09-18
# Windows ending before ALP_CONFLICT_START (2016-02-05) → label=0
# Windows ending after  ALP_CONFLICT_START              → label=1
ALP_POST_PERIODS = [
    ("2014-10-06", "2015-10-05"),  # Pre-period as post-window (T0) → label=0
    ("2014-10-06", "2014-12-05"),  # label=0
    ("2014-12-06", "2015-02-04"),  # label=0
    ("2015-02-05", "2015-04-06"),  # label=0
    ("2015-04-07", "2015-06-06"),  # label=0
    ("2015-06-07", "2015-08-06"),  # label=0
    ("2015-08-07", "2015-10-06"),  # label=0
    ("2015-10-07", "2015-12-06"),  # label=0
    ("2015-12-07", "2016-02-04"),  # label=0
    ("2016-02-05", "2016-04-05"),  # label=1 - post-escalation
    ("2016-04-06", "2016-06-05"),  # label=1
    ("2016-06-06", "2016-08-05"),  # label=1
    ("2016-08-06", "2016-09-18"),  # label=1 - final window (assessment date)
]

# Raqqa (RAQ) - Syria, assessed 2017-10-21

RAQ_CONFLICT_START = "2016-11-06"
RAQ_ASSESSMENT_DATE = "2017-10-21"
RAQ_ORBITS = [116, 123]

# Fixed one-year pre-conflict baseline
RAQ_PRE_PERIOD = ("2015-10-21", "2016-10-20")

# 2-month post-windows: pre-period entry first (T0), then 2-month steps
# Windows ending before RAQ_CONFLICT_START (2016-11-06) → label=0
# Windows ending after  RAQ_CONFLICT_START              → label=1
RAQ_POST_PERIODS = [
    ("2015-10-21", "2016-10-20"),  # Pre-period as post-window (T0) → label=0
    ("2015-10-21", "2015-12-20"),  # label=0
    ("2015-12-21", "2016-02-19"),  # label=0
    ("2016-02-20", "2016-04-20"),  # label=0
    ("2016-04-21", "2016-06-20"),  # label=0
    ("2016-06-21", "2016-08-20"),  # label=0
    ("2016-08-21", "2016-10-20"),  # label=0
    ("2016-10-21", "2016-12-20"),  # label=1 - Operation Wrath of Euphrates launched Nov 2016
    ("2016-12-21", "2017-02-19"),  # label=1
    ("2017-02-20", "2017-04-21"),  # label=1
    ("2017-04-22", "2017-06-21"),  # label=1
    ("2017-06-22", "2017-08-21"),  # label=1
    ("2017-08-22", "2017-10-21"),  # label=1 - final window (assessment date)
]

# Mosul (MOS) - Iraq, assessed 2017-08-04

MOS_CONFLICT_START = "2016-10-16"
MOS_ASSESSMENT_DATE = "2017-08-04"
MOS_ORBITS = [72, 152, 145]

# Fixed one-year pre-conflict baseline
MOS_PRE_PERIOD = ("2015-08-04", "2016-08-03")

# 2-month post-windows: pre-period entry first (T0), then 2-month steps
# Windows ending before MOS_CONFLICT_START (2016-10-16) → label=0
# Windows ending after  MOS_CONFLICT_START              → label=1
MOS_POST_PERIODS = [
    ("2015-08-04", "2016-08-03"),  # Pre-period as post-window (T0) → label=0
    ("2015-08-04", "2015-10-03"),  # label=0
    ("2015-10-04", "2015-12-03"),  # label=0
    ("2015-12-04", "2016-02-02"),  # label=0
    ("2016-02-03", "2016-04-03"),  # label=0
    ("2016-04-04", "2016-06-03"),  # label=0
    ("2016-06-04", "2016-08-03"),  # label=0
    ("2016-08-04", "2016-10-03"),  # label=0
    ("2016-10-04", "2016-12-03"),  # label=1 - Battle of Mosul begins Oct 16
    ("2016-12-04", "2017-02-02"),  # label=1
    ("2017-02-03", "2017-04-04"),  # label=1
    ("2017-04-05", "2017-06-04"),  # label=1
    ("2017-06-05", "2017-08-04"),  # label=1 - final window (assessment date)
]

# Yei (YEI) - South Sudan, assessed 2017-03-05

YEI_CONFLICT_START = "2016-07-01"
YEI_ASSESSMENT_DATE = "2017-03-05"
YEI_ORBITS = [21, 102]

# Pre-period starts at first available S1 image (2015-01-11)
# Only 8 images available in pre-period — noted as limitation
YEI_PRE_PERIOD = ("2015-01-11", "2016-06-30")

# Windows ending before YEI_CONFLICT_START (2016-07-01) → label=0
# Windows ending after  YEI_CONFLICT_START              → label=1
YEI_POST_PERIODS = [
    ("2015-01-11", "2016-06-30"),  # Pre-period as post-window (T0) → label=0
    ("2015-01-11", "2015-03-10"),  # label=0
    ("2015-03-11", "2015-05-10"),  # label=0
    ("2015-05-11", "2015-07-10"),  # label=0
    ("2015-07-11", "2015-09-10"),  # label=0
    ("2015-09-11", "2015-11-10"),  # label=0
    ("2015-11-11", "2016-01-10"),  # label=0
    ("2016-01-11", "2016-03-10"),  # label=0
    ("2016-03-11", "2016-06-30"),  # label=0
    ("2016-07-01", "2016-09-01"),  # label=1
    ("2016-09-02", "2016-11-02"),  # label=1
    ("2016-11-03", "2017-01-03"),  # label=1
    ("2017-01-04", "2017-03-05"),  # label=1 - final window (assessment date)
]

# Lookup by city ID

TRANSFER_CITIES = {
    "ALP": {
        "city_name": "Aleppo",
        "country": "Syria",
        "conflict_start": ALP_CONFLICT_START,
        "assessment_date": ALP_ASSESSMENT_DATE,
        "orbits": ALP_ORBITS,
        "pre_period": ALP_PRE_PERIOD,
        "post_periods": ALP_POST_PERIODS,
        "unosat_labels": TRANSFER_DATA_DIR / "alp" / "unosat_labels.geojson",
        "unosat_aoi": TRANSFER_DATA_DIR / "alp" / "unosat_aoi.geojson",
        "gee_start": "2014-10-01",
        "gee_end": "2017-01-01",
        "label_0_windows": 9,
        "label_1_windows": 4,
    },
    "RAQ": {
        "city_name": "Raqqa",
        "country": "Syria",
        "conflict_start": RAQ_CONFLICT_START,
        "assessment_date": RAQ_ASSESSMENT_DATE,
        "orbits": RAQ_ORBITS,
        "pre_period": RAQ_PRE_PERIOD,
        "post_periods": RAQ_POST_PERIODS,
        "unosat_labels": TRANSFER_DATA_DIR / "raq" / "unosat_labels.geojson",
        "unosat_aoi": TRANSFER_DATA_DIR / "raq" / "unosat_aoi.geojson",
        "gee_start": "2015-01-01",
        "gee_end": "2018-01-01",
        "label_0_windows": 7,
        "label_1_windows": 6,
    },
    "MOS": {
        "city_name": "Mosul",
        "country": "Iraq",
        "conflict_start": MOS_CONFLICT_START,
        "assessment_date": MOS_ASSESSMENT_DATE,
        "orbits": MOS_ORBITS,
        "pre_period": MOS_PRE_PERIOD,
        "post_periods": MOS_POST_PERIODS,
        "unosat_labels": TRANSFER_DATA_DIR / "mos" / "unosat_labels.geojson",
        "unosat_aoi": TRANSFER_DATA_DIR / "mos" / "unosat_aoi.geojson",
        "gee_start": "2015-01-01",
        "gee_end": "2018-01-01",
        "label_0_windows": 8,
        "label_1_windows": 5,
    },
    "YEI": {
        "city_name": "Yei",
        "country": "South Sudan",
        "conflict_start": YEI_CONFLICT_START,
        "assessment_date": YEI_ASSESSMENT_DATE,
        "orbits": YEI_ORBITS,
        "pre_period": YEI_PRE_PERIOD,
        "post_periods": YEI_POST_PERIODS,
        "unosat_labels": TRANSFER_DATA_DIR / "yei" / "unosat_labels.geojson",
        "unosat_aoi": TRANSFER_DATA_DIR / "yei" / "unosat_aoi.geojson",
        "gee_start": "2015-01-01",
        "gee_end": "2017-06-01",
        "label_0_windows": 9,
        "label_1_windows": 4,
    },
}

# Retrained Mosul comparison (east-bank test points only)
# Added for the local-retraining comparison: evaluates the Mosul-retrained
# model (trained on west-bank points, see main_local_mosul_retrain.py)
# against the east-bank points only, since the west-bank points were used
# for training and would inflate apparent performance if included here.
TRANSFER_CITIES["MOS_RETRAINED_EAST_ONLY"] = {
    **TRANSFER_CITIES["MOS"],
    "unosat_labels": PROJECT_PATH
    / "test_sites"
    / "processed"
    / "mos"
    / "unosat_labels_east_bank_only.geojson",
}
