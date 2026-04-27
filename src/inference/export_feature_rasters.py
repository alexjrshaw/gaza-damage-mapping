"""
Export Sentinel-1 feature rasters from GEE for local pixel-level inference.

Option D implementation — exports 28-band GeoTIFF feature rasters from GEE
covering Gaza at 10m resolution, one per time window per orbit. These are
then classified locally using the trained sklearn Random Forest.

This avoids the GEE computation graph scaling problem that prevented
extract_features.py from working at Gaza's point density. GEE raster
operations scale reliably to Gaza's full extent.

Mirrors Dietrich et al.'s dense_inference.py / full_gaza.py approach:
    - Same 7 statistical reducers: mean, stdDev, median, min, max, skew, kurtosis
    - Same feature naming: VV_pre_1x1_mean, VH_post_1x1_stdDev etc.
    - Same 10m resolution
    - Same quadkey tile grid (zoom 12) to avoid GEE export size limits
    - Same orbit aggregation (mean over 3 orbits)

The only deviation from Dietrich et al.:
    - Classification uses sklearn RF locally instead of GEE SMILE RF
    - Feature export replaces in-GEE classification

Output:
    Google Drive: gaza_feature_rasters/{window_str}/{orbit}/qk_{qk_id}.tif
        - 28 bands: VV/VH × pre/post × 7 reducers
        - Float32, 10m resolution
        - One file per quadkey tile per orbit per time window

Usage:
    python3 src/inference/export_feature_rasters.py
"""

import ee
from omegaconf import OmegaConf
from tqdm import tqdm

from src.constants import PRE_PERIOD, POST_PERIODS, AOIS
from src.data.quadkeys import load_gaza_quadkeys_gee
from src.inference.dense_inference import col_to_features
from src.data.sentinel1.collection import get_s1_collection
from src.utils.gdrive import create_drive_folder, get_files_in_folder
from src.utils.gee import init_gee

init_gee(project="gaza-damage-mapping")

# ==================== CONSTANTS ====================

RUN_NAME = "gaza_feature_rasters"
QUADKEY_ZOOM = 12       # Same as full_gaza.py — ~2.4km² tiles
SCALE = 10              # 10m resolution — same as Dietrich et al.
ORBITS = [87, 94, 160]  # Gaza S1 orbits
REDUCER_NAMES = ["mean", "stdDev", "median", "min", "max", "skew", "kurtosis"]
EXTRACT_WINDOW = "1x1"


# ==================== EXPORT ====================

def export_feature_rasters_for_window(
    post_period: tuple[str, str],
    window_str: str,
    drive_folder_base: str = RUN_NAME,
) -> None:
    """
    Export 28-band feature rasters for one time window, for each orbit.

    One GeoTIFF per quadkey tile per orbit — mirrors full_gaza.py tile structure.
    Bands: VV_pre_1x1_mean ... VH_post_1x1_kurtosis (28 bands total)

    Args:
        post_period: (start, end) date strings for post-conflict window
        window_str: String identifier for this window e.g. 'w07_2023-10-07_2023-12-06'
        drive_folder_base: Base Drive folder name
    """
    time_periods = dict(pre=PRE_PERIOD, post=post_period)

    # Load quadkey grid covering Gaza
    grids = load_gaza_quadkeys_gee(zoom=QUADKEY_ZOOM)
    ids = grids.aggregate_array("qk").getInfo()
    print(f"  {len(ids)} quadkey tiles")

    for orbit in ORBITS:
        drive_folder = f"{drive_folder_base}/{window_str}/orbit{orbit}"

        # Filter IDs already exported for this orbit/window
        ids_to_export = _filter_existing(ids, drive_folder)
        if not ids_to_export:
            print(f"  orbit{orbit}: all tiles already exported, skipping")
            continue

        print(f"  orbit{orbit}: exporting {len(ids_to_export)} tiles...")

        for qk_id in tqdm(ids_to_export, desc=f"    orbit{orbit}"):
            geo = grids.filter(ee.Filter.eq("qk", qk_id)).geometry()

            # Get S1 collection for this geometry filtered to this orbit
            s1 = get_s1_collection(geo)
            s1_orbit = s1.filter(ee.Filter.eq("relativeOrbitNumber_start", orbit))

            # Compute 28-band feature image — reuses dense_inference.col_to_features exactly
            feature_img = col_to_features(s1_orbit, REDUCER_NAMES, time_periods, EXTRACT_WINDOW)

            # Export to Drive as Float32 GeoTIFF
            description = f"{window_str}_orbit{orbit}_qk{qk_id}"
            if len(description) > 100:
                description = description[:100]

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


def _filter_existing(ids: list[str], drive_folder: str) -> list[str]:
    """Filter out quadkey IDs already exported to Drive."""
    try:
        files = get_files_in_folder(drive_folder, return_names=True)
        existing = {f.split("qk_")[-1].split(".")[0] for f in files if f.startswith("qk_")}
        ids = [i for i in ids if i not in existing]
    except Exception:
        pass  # folder doesn't exist yet — export all
    return ids


# ==================== MAIN ====================

if __name__ == "__main__":
    # All 19 windows: 1 pre-period window + 18 post windows
    # Window 1 (w01) = PRE_PERIOD post window — label=0 reference
    # Windows 7-19 = post-war windows — label=1
    all_periods = [PRE_PERIOD] + list(POST_PERIODS)

    print(f"Exporting feature rasters for {len(all_periods)} windows × {len(ORBITS)} orbits")
    print(f"Drive folder: {RUN_NAME}/")
    print()

    # Create base Drive folder
    try:
        create_drive_folder(RUN_NAME)
    except Exception:
        pass

    for i, post_period in enumerate(all_periods):
        window_str = f"w{i+1:02d}_{post_period[0]}_{post_period[1]}"
        print(f"Window {i+1:02d}/{len(all_periods)}: {window_str}")

        try:
            create_drive_folder(f"{RUN_NAME}/{window_str}")
        except Exception:
            pass

        export_feature_rasters_for_window(post_period, window_str)
        print()

    print("All export tasks submitted.")
    print("Monitor progress in GEE task manager.")
    print(f"Downloads will appear in Google Drive under '{RUN_NAME}/'")
