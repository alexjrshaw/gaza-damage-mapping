"""
Download transfer city feature rasters from Google Drive to Forth.

Downloads quadkey-tiled GeoTIFFs, deleting from Drive after each download
to manage Drive space — mirrors Gaza's download_feature_rasters.py exactly.

Output:
    data/transfer_cities/feature_rasters/{city_id}/{window_str}/orbit{orbit}/
        qk_{qk_id}.tif  (one per quadkey tile)

Usage:
    python3 src/data/transfer_cities/pixel_inference/download_feature_rasters_transfer.py --city MOS
    python3 src/data/transfer_cities/pixel_inference/download_feature_rasters_transfer.py --city ALP --keep
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, "/scratch/s1214882/gaza-damage-mapping")

from src.constants import DATA_PATH
from src.data.transfer_cities.constants_transfer import TRANSFER_CITIES
from src.utils.gdrive import drive, get_folder_id

LOCAL_BASE = DATA_PATH / "transfer_cities" / "feature_rasters"
DELETE_AFTER_DOWNLOAD = True


def download_city(city_id: str, delete: bool = DELETE_AFTER_DOWNLOAD) -> None:
    cfg = TRANSFER_CITIES[city_id]
    pre_period = cfg["pre_period"]
    post_periods = cfg["post_periods"]
    orbits = cfg["orbits"]
    all_periods = [pre_period] + list(post_periods)

    print(f"Downloading feature rasters for {city_id}...")
    print(f"Delete from Drive after download: {delete}")

    n_downloaded = 0
    n_skipped = 0

    for i, post_period in enumerate(all_periods):
        window_str = f"w{i+1:02d}_{post_period[0]}_{post_period[1]}"

        for orbit in orbits:
            drive_folder = f"transfer_feature_rasters/{city_id}/{window_str}/orbit{orbit}"
            local_dir = LOCAL_BASE / city_id / window_str / f"orbit{orbit}"

            # Find Drive folder
            try:
                folder_id = get_folder_id(drive_folder)
            except Exception:
                continue

            # List all TIF files in folder
            files = drive.ListFile({"q": f"'{folder_id}' in parents and trashed=false"}).GetList()
            tif_files = [(f["title"], f["id"]) for f in files if f["title"].endswith(".tif")]

            if not tif_files:
                continue

            local_dir.mkdir(parents=True, exist_ok=True)
            print(f"  {window_str}/orbit{orbit}: {len(tif_files)} tiles")

            for fname, file_id in tif_files:
                fp_out = local_dir / fname
                if fp_out.exists():
                    if delete:
                        drive.CreateFile({"id": file_id}).Delete()
                    n_skipped += 1
                    continue

                f = drive.CreateFile({"id": file_id})
                f.GetContentFile(str(fp_out))
                n_downloaded += 1

                if delete:
                    f.Delete()

            print(f"    Downloaded {len(tif_files)} tiles")

    print(f"\nDone: {n_downloaded} downloaded, {n_skipped} skipped.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--city", type=str, required=True)
    parser.add_argument("--keep", action="store_true", help="Keep files on Drive after download")
    args = parser.parse_args()
    download_city(args.city.upper(), delete=not args.keep)
