from pathlib import Path

# ------------------- PROJECT CONSTANTS -------------------

# AOIs correspond to Gaza governorates:
# GAZ1=North Gaza, GAZ2=Gaza, GAZ3=Deir Al-Balah, GAZ4=Khan Yunis, GAZ5=Rafah
AOIS_TRAIN = ["GAZ1", "GAZ2"]
AOIS_TEST  = ["GAZ3", "GAZ4", "GAZ5"]
AOIS = AOIS_TRAIN + AOIS_TEST

S1_BANDS = ["VV", "VH"]
S2_BANDS = ["B1", "B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B9", "B11", "B12"]

GAZA_WAR_START = "2023-10-07"

# ------------------- TIME PERIODS -------------------
# One year pre-conflict baseline (Oct 2022 - Oct 2023), consistent with Dietrich et al. (2025)
PRE_PERIOD = ("2022-10-07", "2023-10-06")

# 12 x 2-month post-conflict windows covering full conflict period
POST_PERIODS = [
    ("2023-10-07", "2023-12-06"),   # Window 1
    ("2023-12-07", "2024-02-06"),   # Window 2
    ("2024-02-07", "2024-04-06"),   # Window 3
    ("2024-04-07", "2024-06-06"),   # Window 4
    ("2024-06-07", "2024-08-06"),   # Window 5
    ("2024-08-07", "2024-10-06"),   # Window 6
    ("2024-10-07", "2024-12-06"),   # Window 7
    ("2024-12-07", "2025-02-06"),   # Window 8 — no UNOSAT assessment
    ("2025-02-07", "2025-04-06"),   # Window 9
    ("2025-04-07", "2025-06-06"),   # Window 10 — no UNOSAT assessment
    ("2025-06-07", "2025-08-06"),   # Window 11
    ("2025-08-07", "2025-10-06"),   # Window 12
    ("2025-10-07", "2025-12-06"),   # Window 13 — captures final UNOSAT assessment
]

# ------------------- LOCAL PATH CONSTANTS -------------------
constants_path = Path(__file__)
SRC_PATH = constants_path.parent
PROJECT_PATH = SRC_PATH.parent

SECRETS_PATH  = PROJECT_PATH / "secrets"
DATA_PATH     = PROJECT_PATH / "data"
OVERTURE_PATH = DATA_PATH / "overture_buildings"

# ------------------- GEE PATH CONSTANTS -------------------
ASSETS_PATH     = "projects/gaza-damage-mapping/assets/gaza-mapping-tool/"
OLD_ASSETS_PATH = "projects/gaza-damage-mapping/assets/"