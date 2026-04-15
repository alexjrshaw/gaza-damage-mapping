"""
Download HOTOSM building footprints for Gaza from HDX.

Source: Humanitarian OpenStreetMap Team (HOTOSM)
Reference: Scher & Van Den Hoek (2025) identify HOTOSM as the most
comprehensive pre-conflict building footprint dataset for Gaza.
URL: https://data.humdata.org/dataset/hotosm_pse_buildings
"""

import io
import zipfile

import requests

from src.constants import DATA_PATH
from src.utils.time import timeit

HOTOSM_RAW_FP = DATA_PATH / "raw/hotosm_pse_buildings_polygons_geojson.geojson"


@timeit
def download_hotosm_buildings() -> None:
    """
    Download HOTOSM building footprints for Palestine from HDX.

    Saves raw GeoJSON to data/raw/. Skips download if file already exists.
    Covers all of Palestine (West Bank + Gaza) — filtering to Gaza
    happens in preprocessing.py.
    """
    if HOTOSM_RAW_FP.exists():
        print(f"HOTOSM buildings already downloaded at {HOTOSM_RAW_FP}")
        return

    print("Downloading HOTOSM building footprints from HDX...")
    HOTOSM_RAW_FP.parent.mkdir(exist_ok=True, parents=True)

    url = (
        "https://data.humdata.org/dataset/hotosm_pse_buildings/resource/"
        "3382a75a-91e3-413d-af09-ab04191725df/download/"
        "hotosm_pse_buildings_polygons_geojson.zip"
    )
    r = requests.get(url, stream=True)
    z = zipfile.ZipFile(io.BytesIO(r.content))
    z.extractall(HOTOSM_RAW_FP.parent)
    print(f"Saved to {HOTOSM_RAW_FP}")


if __name__ == "__main__":
    download_hotosm_buildings()
