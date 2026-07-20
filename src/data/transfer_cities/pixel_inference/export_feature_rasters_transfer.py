"""
Export Sentinel-1 feature rasters from GEE for transfer city pixel-level inference.

Uses quadkey tiling at zoom 12 — identical to Gaza pipeline — to keep
individual GeoTIFF sizes small (~5-20 MB) and avoid Google Drive storage limits.

Mirrors src/inference/export_feature_rasters.py exactly, adapted for:
    - Transfer city AOIs (from test_sites/processed/{city}/unosat_aoi.geojson)
    - City-specific orbits and time periods from constants_transfer.py
    - Generic quadkey grid (not Gaza-specific)

Output (Google Drive):
    transfer_feature_rasters/{city_id}/{window_str}/orbit{orbit}/qk_{qk_id}.tif
        - 28 bands: VV/VH x pre/post x 7 reducers
        - Float32, 10m resolution
        - One file per quadkey tile per orbit per window

Usage:
    # Export all windows
    python3 src/data/transfer_cities/pixel_inference/export_feature_rasters_transfer.py --city MOS

    # Export specific windows only (e.g. after Drive space freed)
    python3 src/data/transfer_cities/pixel_inference/export_feature_rasters_transfer.py --city ALP --windows w03 w04 w05

    # Force re-export even if already exists
    python3 src/data/transfer_cities/pixel_inference/export_feature_rasters_transfer.py --city MOS --force
"""

import argparse
import sys
from pathlib import Path

import ee
import geemap
import geopandas as gpd

sys.path.insert(0, "/scratch/s1214882/gaza-damage-mapping")

from src.data.quadkeys import get_intersecting_quadkeys
from src.data.sentinel1.collection import get_s1_collection
from src.data.transfer_cities.constants_transfer import TRANSFER_CITIES, TRANSFER_GEE_FOLDER
from src.utils.gee import col_to_features
from src.utils.gdrive import create_drive_folder, get_files_in_folder
from src.utils.gee import asset_exists, create_folders_recursively, init_gee

init_gee()

SCALE = 10
REDUCER_NAMES = ["mean", "stdDev", "median", "min", "max", "skew", "kurtosis"]
EXTRACT_WINDOW = "1x1"
QUADKEY_ZOOM = 12  # Same as Gaza pipeline
DRIVE_BASE = "transfer_feature_rasters"
LOCAL_BASE = Path("/scratch/s1214882/gaza-damage-mapping/data/transfer_cities/feature_rasters")


def get_city_quadkeys_gee(city_id: str, aoi_fp: Path) -> tuple:
    """
    Get quadkey grid for a transfer city AOI.
    Computes locally and uploads to GEE if not already there.

    Returns:
        (ee.FeatureCollection of quadkey grid, list of quadkey IDs)
    """
    asset_id = TRANSFER_GEE_FOLDER + f"quadkeys/{city_id}_zoom{QUADKEY_ZOOM}"

    if not asset_exists(asset_id):
        print(f"  Creating quadkey grid for {city_id} zoom {QUADKEY_ZOOM}...")
        gdf_aoi = gpd.read_file(aoi_fp)
        aoi_geom = gdf_aoi.geometry.iloc[0]
        qk_grid = get_intersecting_quadkeys(aoi_geom, QUADKEY_ZOOM)
        print(f"  {len(qk_grid)} quadkey tiles")

        # Upload to GEE
        create_folders_recursively(asset_id, last_one_is_asset=True)
        fc = geemap.geopandas_to_ee(qk_grid)
        task = ee.batch.Export.table.toAsset(
            collection=fc,
            description=f"{city_id}_quadkeys_zoom{QUADKEY_ZOOM}",
            assetId=asset_id,
        )
        task.start()
        print(f"  Uploading quadkey grid to GEE... waiting...")
        import time

        while not asset_exists(asset_id):
            time.sleep(5)
        print(f"  Quadkey grid ready: {asset_id}")

    grids = ee.FeatureCollection(asset_id)
    ids = grids.aggregate_array("qk").getInfo()
    return grids, ids


def already_exported(
    city_id: str, window_str: str, orbit: int, qk_id: str, drive_folder: str
) -> bool:
    """Check if tile already exists locally or on Drive."""
    fp_local = LOCAL_BASE / city_id / window_str / f"orbit{orbit}" / f"qk_{qk_id}.tif"
    if fp_local.exists():
        return True
    try:
        files = get_files_in_folder(drive_folder, return_names=True)
        return f"qk_{qk_id}.tif" in files
    except Exception:
        return False


def export_feature_rasters_city(
    city_id: str,
    force: bool = False,
    window_filter: list = None,
) -> None:
    cfg = TRANSFER_CITIES[city_id]
    orbits = cfg["orbits"]
    pre_period = cfg["pre_period"]
    post_periods = cfg["post_periods"]
    all_periods = [pre_period] + list(post_periods)

    print(f"\n{city_id} ({cfg['city_name']}): {len(all_periods)} windows × {len(orbits)} orbits")
    if window_filter:
        print(f"  Window filter: {window_filter}")

    # Get quadkey grid
    print(f"\nLoading quadkey grid (zoom {QUADKEY_ZOOM})...")
    grids, ids = get_city_quadkeys_gee(city_id, cfg["unosat_aoi"])
    print(f"  {len(ids)} quadkey tiles")

    # Ensure Drive base folders exist
    for folder in [DRIVE_BASE, f"{DRIVE_BASE}/{city_id}"]:
        try:
            create_drive_folder(folder)
        except Exception:
            pass

    n_started = 0
    n_skipped = 0

    for i, post_period in enumerate(all_periods):
        window_str = f"w{i+1:02d}_{post_period[0]}_{post_period[1]}"
        time_periods = dict(pre=pre_period, post=post_period)

        # Apply window filter
        if window_filter and not any(window_str.startswith(w) for w in window_filter):
            continue

        print(f"\n  Window {i+1:02d}/{len(all_periods)}: {window_str}")

        for orbit in orbits:
            drive_folder = f"{DRIVE_BASE}/{city_id}/{window_str}/orbit{orbit}"

            # Create Drive folder
            for folder in [
                f"{DRIVE_BASE}/{city_id}/{window_str}",
                drive_folder,
            ]:
                try:
                    create_drive_folder(folder)
                except Exception:
                    pass

            # Filter already-exported tiles
            ids_to_export = (
                ids
                if force
                else [
                    qk_id
                    for qk_id in ids
                    if not already_exported(city_id, window_str, orbit, qk_id, drive_folder)
                ]
            )

            if not ids_to_export:
                print(f"    orbit{orbit}: all {len(ids)} tiles already exported — skipping")
                n_skipped += len(ids)
                continue

            print(f"    orbit{orbit}: exporting {len(ids_to_export)}/{len(ids)} tiles...")

            # Get S1 collection for this orbit
            s1 = get_s1_collection(
                grids.geometry(),
                cfg["gee_start"],
                cfg["gee_end"],
            ).filter(ee.Filter.eq("relativeOrbitNumber_start", orbit))

            for qk_id in ids_to_export:
                geo = grids.filter(ee.Filter.eq("qk", qk_id)).geometry()

                # Compute 28-band feature image for this tile
                feature_img = col_to_features(s1, REDUCER_NAMES, time_periods, EXTRACT_WINDOW)

                description = f"{city_id}_{window_str}_orbit{orbit}_qk{qk_id}"[:100]

                task = ee.batch.Export.image.toDrive(
                    image=feature_img.toFloat(),
                    description=description,
                    folder=drive_folder,
                    fileNamePrefix=f"qk_{qk_id}",
                    region=geo,
                    scale=SCALE,
                    maxPixels=1e13,
                    fileFormat="GeoTIFF",
                )
                task.start()
                n_started += 1

    print(f"\n{city_id}: {n_started} tasks started, {n_skipped} tiles skipped.")
    if n_started > 0:
        print("Monitor at https://code.earthengine.google.com/tasks")
        print("Run download script while tasks complete to free Drive space.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--city", type=str, required=True)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--windows", nargs="+", default=None)
    args = parser.parse_args()

    export_feature_rasters_city(
        city_id=args.city.upper(),
        force=args.force,
        window_filter=args.windows,
    )
