"""
Export Sentinel-1 feature rasters from GEE for transfer city pixel-level inference.

Adapted from src/inference/export_feature_rasters.py for transfer cities.
Key differences from Gaza:
    - Single AOI export (no quadkey tiling — cities are small enough)
    - City-specific orbits, time periods from constants_transfer.py
    - Exports to Drive folder: transfer_feature_rasters/{city_id}/{window_str}/orbit{orbit}/

Currently configured for Raqqa as proof-of-concept pixel-level transfer.

Usage:
    cd /scratch/s1214882/gaza-damage-mapping
    python3 src/data/transfer_cities/export_feature_rasters_transfer.py --city RAQ
"""

import ee
import argparse
from pathlib import Path

from src.utils.gee import init_gee
from src.utils.gdrive import create_drive_folder
from src.inference.dense_inference import col_to_features
from src.data.sentinel1.collection import get_s1_collection
from src.data.transfer_cities.constants_transfer import TRANSFER_CITIES, TRANSFER_GEE_FOLDER

init_gee()

SCALE = 10
REDUCER_NAMES = ["mean", "stdDev", "median", "min", "max", "skew", "kurtosis"]
EXTRACT_WINDOW = "1x1"
DRIVE_BASE = "transfer_feature_rasters"


def export_feature_rasters_city(city_id: str, force: bool = False) -> None:
    cfg = TRANSFER_CITIES[city_id]
    city_name = cfg["city_name"]
    orbits = cfg["orbits"]
    pre_period = cfg["pre_period"]
    post_periods = cfg["post_periods"]

    # Load AOI geometry from GEE
    aoi_asset = TRANSFER_GEE_FOLDER + f"AOIs/{city_id}"
    geo = ee.FeatureCollection(aoi_asset).geometry()

    all_periods = [pre_period] + list(post_periods)
    print(f"{city_id} ({city_name}): {len(all_periods)} windows × {len(orbits)} orbits")

    try:
        create_drive_folder(DRIVE_BASE)
    except Exception:
        pass

    for i, post_period in enumerate(all_periods):
        window_str = f"w{i+1:02d}_{post_period[0]}_{post_period[1]}"
        time_periods = dict(pre=pre_period, post=post_period)

        print(f"\nWindow {i+1:02d}/{len(all_periods)}: {window_str}")

        for orbit in orbits:
            drive_folder = f"{DRIVE_BASE}/{city_id}/{window_str}/orbit{orbit}"

            try:
                create_drive_folder(f"{DRIVE_BASE}/{city_id}")
            except Exception:
                pass
            try:
                create_drive_folder(f"{DRIVE_BASE}/{city_id}/{window_str}")
            except Exception:
                pass
            try:
                create_drive_folder(drive_folder)
            except Exception:
                pass

            # Check if already exported
            from src.utils.gdrive import get_files_in_folder
            try:
                existing = get_files_in_folder(drive_folder, return_names=True)
                if existing and not force:
                    print(f"  orbit{orbit}: already exported ({len(existing)} files) — skipping")
                    continue
            except Exception:
                pass

            # Get S1 collection filtered to AOI and orbit
            s1 = get_s1_collection(geo, cfg["gee_start"], cfg["gee_end"])
            s1_orbit = s1.filter(ee.Filter.eq("relativeOrbitNumber_start", orbit))

            # Compute 28-band feature image
            feature_img = col_to_features(s1_orbit, REDUCER_NAMES, time_periods, EXTRACT_WINDOW)

            description = f"{city_id}_{window_str}_orbit{orbit}"[:100]

            task = ee.batch.Export.image.toDrive(
                image=feature_img.toFloat(),
                description=description,
                folder=drive_folder,
                fileNamePrefix=f"{city_id}_{window_str}_orbit{orbit}",
                region=geo,
                scale=SCALE,
                maxPixels=1e13,
                fileFormat="GeoTIFF",
            )
            task.start()
            print(f"  orbit{orbit}: export started -> {drive_folder}")

    print(f"\nAll export tasks started for {city_id}.")
    print("Monitor at https://code.earthengine.google.com/tasks")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--city", type=str, required=True)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    export_feature_rasters_city(args.city.upper(), force=args.force)
