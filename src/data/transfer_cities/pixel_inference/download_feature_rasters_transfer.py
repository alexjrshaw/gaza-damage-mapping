"""
Download transfer city feature rasters from Google Drive to Forth.

Adapted from src/inference/download_feature_rasters.py for transfer cities.
Downloads single-tile GeoTIFFs per orbit per window.

Output:
    data/transfer_cities/feature_rasters/{city_id}/{window_str}/orbit{orbit}/
        {city_id}_{window_str}_orbit{orbit}.tif

Usage:
    python3 src/data/transfer_cities/download_feature_rasters_transfer.py --city RAQ
"""

import time
import argparse
from pathlib import Path

from src.constants import DATA_PATH
from src.utils.gdrive import drive, get_folder_id, get_files_in_folder
from src.data.transfer_cities.constants_transfer import TRANSFER_CITIES

DRIVE_BASE   = "transfer_feature_rasters"
LOCAL_BASE   = DATA_PATH / "transfer_cities" / "feature_rasters"


def download_city(city_id: str) -> None:
    cfg = TRANSFER_CITIES[city_id]
    pre_period   = cfg["pre_period"]
    post_periods = cfg["post_periods"]
    orbits       = cfg["orbits"]
    all_periods  = [pre_period] + list(post_periods)

    print(f"Downloading feature rasters for {city_id}...")

    for i, post_period in enumerate(all_periods):
        window_str = f"w{i+1:02d}_{post_period[0]}_{post_period[1]}"

        for orbit in orbits:
            drive_folder = f"{DRIVE_BASE}/{city_id}/{window_str}/orbit{orbit}"
            local_dir = LOCAL_BASE / city_id / window_str / f"orbit{orbit}"

            # Check what's on Drive
            try:
                folder_id = get_folder_id(drive_folder)
            except Exception:
                continue  # folder doesn't exist yet

            files = drive.ListFile({
                "q": f"'{folder_id}' in parents and trashed=false"
            }).GetList()
            tif_files = [(f["title"], f["id"]) for f in files if f["title"].endswith(".tif")]

            if not tif_files:
                continue

            local_dir.mkdir(parents=True, exist_ok=True)

            for filename, file_id in tif_files:
                fp_local = local_dir / filename
                if fp_local.exists():
                    print(f"  {window_str}/orbit{orbit}/{filename}: already downloaded")
                    continue

                print(f"  Downloading {window_str}/orbit{orbit}/{filename}...")
                f = drive.CreateFile({"id": file_id})
                f.GetContentFile(str(fp_local))
                print(f"  Saved -> {fp_local}")

    print(f"\nDownload complete for {city_id}.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--city", type=str, required=True)
    args = parser.parse_args()
    download_city(args.city.upper())
