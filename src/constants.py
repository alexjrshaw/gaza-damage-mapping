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
# One year pre-conflict baseline (Oct 2021 - Oct 2022), consistent with Dietrich et al. (2025)
PRE_PERIOD = ("2021-10-07", "2022-10-06")  # shifted back one year

# XXXX x 2-month post-conflict windows covering full conflict period, plus some pre-war for backscatter comparison
POST_PERIODS = [
    ("2022-10-07", "2022-12-06"),   # Pre-war window 1 → label=0
    ("2022-12-07", "2023-02-06"),   # Pre-war window 2 → label=0
    ("2023-02-07", "2023-04-06"),   # Pre-war window 3 → label=0
    ("2023-04-07", "2023-06-06"),   # Pre-war window 4 → label=0
    ("2023-06-07", "2023-08-06"),   # Pre-war window 5 → label=0
    ("2023-08-07", "2023-10-06"),   # Pre-war window 6 → label=0
    ("2023-10-07", "2023-12-06"),   # Post-war window 1 → label=1
    ("2023-12-07", "2024-02-06"),   # Post-war window 2 → label=1
    ("2024-02-07", "2024-04-06"),   # Post-war window 3 → label=1
    ("2024-04-07", "2024-06-06"),   # Post-war window 4 → label=1
    ("2024-06-07", "2024-08-06"),   # Post-war window 5 → label=1
    ("2024-08-07", "2024-10-06"),   # Post-war window 6 → label=1
    ("2024-10-07", "2024-12-06"),   # Post-war window 7 → label=1
    ("2024-12-07", "2025-02-06"),   # Post-war window 8 → label=1
    ("2025-02-07", "2025-04-06"),   # Post-war window 9 → label=1
    ("2025-04-07", "2025-06-06"),   # Post-war window 10 → label=1
    ("2025-06-07", "2025-08-06"),   # Post-war window 11 → label=1
    ("2025-08-07", "2025-10-06"),   # Post-war window 12 → label=1
    ("2025-10-07", "2025-12-06"),   # Post-war window 13 → label=1
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