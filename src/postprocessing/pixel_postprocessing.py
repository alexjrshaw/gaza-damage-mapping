"""
Pixel-level postprocessing for Gaza damage mapping.

Local equivalent of drive_to_results.py from Dietrich et al. (2025).
Assigns pixel-level damage probability rasters to HOTOSM building
footprints, producing building-level damage estimates per time window.

Mirrors drive_to_results.py exactly in three steps:
    1. Merge quadkey tiles → single GeoTIFF per time window
       (equivalent to download_and_merge_all_dates)
    2. Assign pixel predictions to buildings per admin unit
       (equivalent to create_all_gdf_overture_with_preds)
    3. Aggregate all admin results → buildings_preds.parquet
       (equivalent to aggregate_all_preds)

Key adaptations from Dietrich et al.:
    - Reads from local probability rasters instead of Drive
    - Uses HOTOSM buildings (gaza_buildings.parquet) instead of Overture
    - Admin units are governorates (adm2_name) instead of adm3
    - Window naming follows our w{i}_{start}_{end} convention

Usage:
    python3 src/postprocessing/pixel_postprocessing.py
"""

import multiprocessing as mp
import warnings
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rioxarray
import xarray as xr
from osgeo import gdal
from shapely.wkb import loads as wkb_loads
from tqdm.auto import tqdm

from src.constants import DATA_PATH, GAZA_WAR_START, POST_PERIODS, PRE_PERIOD
from src.postprocessing.utils import vectorize_xarray_3d
from src.utils.time import timeit

# ==================== CONSTANTS ====================

PROBABILITY_RASTERS_DIR = DATA_PATH / "probability_rasters"
BUILDINGS_FP = DATA_PATH / "overture_buildings/gaza_buildings.parquet"
MERGED_RASTERS_DIR = DATA_PATH / "merged_probability_rasters"
OUTPUT_DIR = DATA_PATH / "pixel_postprocessing"
UTM_CRS = "EPSG:32636"


# ==================== STEP 1: MERGE TILES ====================

def merge_tiles_for_window(window_str: str, force_recreate: bool = False) -> Path | None:
    """
    Merge all quadkey tiles for one window into a single GeoTIFF.

    Mirrors download_and_merge() in drive_to_results.py.
    Uses gdal.Warp to mosaic tiles — identical approach to Dietrich et al.

    Args:
        window_str: Window identifier e.g. 'w07_2023-10-07_2023-12-06'

    Returns:
        Path to merged GeoTIFF, or None if no tiles found.
    """
    MERGED_RASTERS_DIR.mkdir(exist_ok=True, parents=True)
    fp_out = MERGED_RASTERS_DIR / f"gaza_{window_str}.tif"

    if fp_out.exists() and not force_recreate:
        print(f"  {window_str}: already merged")
        return fp_out

    tile_dir = PROBABILITY_RASTERS_DIR / window_str
    if not tile_dir.exists():
        print(f"  {window_str}: no tiles found — skipping")
        return None

    tif_files = sorted(str(fp) for fp in tile_dir.glob("qk_*.tif"))
    if not tif_files:
        print(f"  {window_str}: no .tif files found — skipping")
        return None

    print(f"  {window_str}: merging {len(tif_files)} tiles...")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        gdal.Warp(str(fp_out), tif_files, format="GTiff")
    print(f"  {window_str}: merged → {fp_out.name}")
    return fp_out


def merge_all_windows(force_recreate: bool = False) -> list[Path]:
    """Merge tiles for all available windows."""
    print("=" * 60)
    print("Step 1: Merging quadkey tiles per window")
    print("=" * 60)

    windows = sorted(d.name for d in PROBABILITY_RASTERS_DIR.iterdir() if d.is_dir())
    merged = []
    for window_str in windows:
        fp = merge_tiles_for_window(window_str, force_recreate)
        if fp is not None:
            merged.append(fp)

    print(f"\nMerged {len(merged)} windows")
    return merged


# ==================== STEP 2: ASSIGN TO BUILDINGS ====================

def load_buildings() -> gpd.GeoDataFrame:
    """Load HOTOSM building footprints."""
    df = pd.read_parquet(BUILDINGS_FP)
    gdf = gpd.GeoDataFrame(df, geometry="geometry", crs="EPSG:4326")
    gdf = gdf[gdf["area_m2"] >= 50].copy()
    return gdf


def get_post_date_from_window(window_str: str) -> str:
    """Extract start date from window string e.g. 'w07_2023-10-07_2023-12-06' → '2023-10-07'."""
    parts = window_str.split("_")
    return f"{parts[1]}"


def create_buildings_with_preds_for_admin(
    adm2_name: str,
    merged_fps: list[Path],
    folder_preds: Path,
    verbose: int = 0,
) -> None:
    """
    Assign pixel predictions to buildings for one admin unit.

    Mirrors create_gdf_overture_with_preds() in drive_to_results.py exactly:
        1. Load buildings for admin unit
        2. Read and stack prediction rasters
        3. Vectorize pixels → polygons
        4. Intersect with building polygons
        5. Compute area-weighted mean and max per building per date
        6. Save as GeoJSON

    Args:
        adm2_name: Governorate name e.g. 'North Gaza'
        merged_fps: List of merged GeoTIFF paths
        folder_preds: Output folder for per-admin GeoJSONs
        verbose: Verbosity level
    """
    fp_out = folder_preds / f"{adm2_name.replace(' ', '_')}.geojson"
    if fp_out.exists():
        if verbose:
            print(f"  {adm2_name}: already processed")
        return

    # Load buildings for this admin unit
    gdf_buildings = load_buildings()
    gdf_buildings = gdf_buildings[gdf_buildings["adm2_name"] == adm2_name].copy()
    gdf_buildings = gdf_buildings.set_index("building_id")

    if len(gdf_buildings) == 0:
        print(f"  {adm2_name}: no buildings found")
        return

    if verbose:
        print(f"  {adm2_name}: {len(gdf_buildings):,} buildings")

    from shapely.geometry import box
    total_bounds = box(*gdf_buildings.total_bounds)

    # Get post dates from window strings
    post_dates = [get_post_date_from_window(fp.stem.replace("gaza_", "")) for fp in merged_fps]

    # Read and stack prediction rasters — mirrors Dietrich et al. exactly
    from src.data.utils import read_fp_within_geo
    dates_var = xr.Variable("date", pd.to_datetime(post_dates))
    preds = xr.concat(
        [read_fp_within_geo(fp, total_bounds) for fp in merged_fps],
        dim=dates_var
    ).squeeze(dim="band")

    if verbose:
        print(f"  {adm2_name}: rasters stacked {preds.shape}")

    # Vectorize pixels — reuses Dietrich et al.'s vectorize_xarray_3d unchanged
    gdf_pixels = vectorize_xarray_3d(preds, post_dates)
    if verbose:
        print(f"  {adm2_name}: {len(gdf_pixels):,} pixels vectorized")

    # Intersect buildings with pixels — identical to Dietrich et al.
    overlap = gpd.overlay(
        gdf_buildings.reset_index(),
        gdf_pixels,
        how="intersection"
    ).set_index("building_id")

    if len(overlap) == 0:
        print(f"  {adm2_name}: no overlaps found")
        return

    if verbose:
        print(f"  {adm2_name}: {len(overlap):,} overlaps")

    # Compute intersection area — identical to Dietrich et al.
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning)
        overlap["polygon_area"] = overlap.area

    # Compute area-weighted mean — identical to Dietrich et al.
    cols_weighted = [f"{d}_weighted_value" for d in post_dates]
    overlap[cols_weighted] = overlap[post_dates].multiply(overlap["polygon_area"], axis=0)
    grps = overlap.groupby("building_id")
    gdf_weighted_mean = grps[cols_weighted].sum().divide(grps["polygon_area"].sum(), axis=0)
    gdf_weighted_mean = gdf_weighted_mean.stack().reset_index(level=1)
    gdf_weighted_mean.columns = ["post_date", "weighted_mean"]
    gdf_weighted_mean["post_date"] = gdf_weighted_mean["post_date"].apply(lambda x: x.split("_")[0])
    gdf_weighted_mean.set_index("post_date", append=True, inplace=True)

    # Compute max — identical to Dietrich et al.
    gdf_max = overlap.groupby("building_id")[post_dates].max().stack().to_frame(name="max")
    gdf_max.index.names = ["building_id", "post_date"]

    # Merge with buildings — identical to Dietrich et al.
    gdf_buildings_with_preds = gdf_buildings.join(gdf_weighted_mean).join(gdf_max).sort_index()

    # Save
    gdf_buildings_with_preds.reset_index().to_file(fp_out, driver="GeoJSON")
    print(f"  {adm2_name}: saved {len(gdf_buildings_with_preds):,} building-date rows → {fp_out.name}")


def create_all_buildings_with_preds(
    merged_fps: list[Path],
    cpu: int = 3,
) -> None:
    """
    Assign pixel predictions to buildings for all admin units.

    Mirrors create_all_gdf_overture_with_preds_mp() in drive_to_results.py.
    Processes each governorate separately to manage memory.
    """
    print("\n" + "=" * 60)
    print("Step 2: Assigning predictions to buildings per admin unit")
    print("=" * 60)

    folder_preds = OUTPUT_DIR / "admin_preds"
    folder_preds.mkdir(exist_ok=True, parents=True)

    # Get all admin units from buildings
    df = pd.read_parquet(BUILDINGS_FP)
    adm2_names = sorted(df["adm2_name"].dropna().unique())
    print(f"Processing {len(adm2_names)} admin units: {adm2_names}")

    args = [
        (adm2_name, merged_fps, folder_preds, 1)
        for adm2_name in adm2_names
        if not (folder_preds / f"{adm2_name.replace(' ', '_')}.geojson").exists()
    ]

    if not args:
        print("All admin units already processed.")
        return

    # Process sequentially to avoid memory issues
    # (multiprocessing version available but memory-intensive)
    for adm2_name, fps, folder, verbose in args:
        create_buildings_with_preds_for_admin(adm2_name, fps, folder, verbose)


# ==================== STEP 3: AGGREGATE ====================

def process_admin_file(adm2_name: str) -> pd.DataFrame | None:
    """Load one admin GeoJSON and pivot to wide format."""
    folder_preds = OUTPUT_DIR / "admin_preds"
    fp = folder_preds / f"{adm2_name.replace(' ', '_')}.geojson"
    try:
        gdf = gpd.read_file(fp)
        return gdf.pivot_table(
            index="building_id",
            columns="post_date",
            values="weighted_mean",
        )
    except Exception as e:
        print(f"Error processing {adm2_name}: {e}")
        return None


def aggregate_all_preds() -> pd.DataFrame:
    """
    Aggregate all admin predictions into buildings_preds.parquet.

    Mirrors aggregate_all_preds() in drive_to_results.py exactly.
    """
    print("\n" + "=" * 60)
    print("Step 3: Aggregating all admin predictions")
    print("=" * 60)

    df = pd.read_parquet(BUILDINGS_FP)
    adm2_names = sorted(df["adm2_name"].dropna().unique())

    folder_preds = OUTPUT_DIR / "admin_preds"
    available = [a for a in adm2_names
                 if (folder_preds / f"{a.replace(' ', '_')}.geojson").exists()]
    print(f"Aggregating {len(available)} admin units...")

    df_preds = []
    for adm2_name in available:
        result = process_admin_file(adm2_name)
        if result is not None:
            df_preds.append(result)

    if not df_preds:
        print("No admin predictions found.")
        return pd.DataFrame()

    df_preds = pd.concat(df_preds, axis=0)

    # Merge with building metadata — mirrors Dietrich et al.
    df_buildings = pd.read_parquet(BUILDINGS_FP).set_index("building_id")
    df_buildings_with_preds = df_buildings.join(df_preds, how="left")

    # Save
    fp_out = OUTPUT_DIR / "buildings_preds.parquet"
    df_buildings_with_preds.to_parquet(fp_out)
    print(f"Saved to {fp_out}")
    print(f"Shape: {df_buildings_with_preds.shape}")

    # Summary
    post_war_cols = [c for c in df_preds.columns if c >= GAZA_WAR_START]
    if post_war_cols:
        print("\n=== Per-governorate damage summary ===")
        df_buildings_with_preds["max_post_war"] = df_buildings_with_preds[post_war_cols].max(axis=1)
        summary = df_buildings_with_preds.groupby("adm2_name").agg(
            n_buildings=("max_post_war", "count"),
            mean_damage=("max_post_war", "mean"),
            pct_over_50=("max_post_war", lambda x: (x > 0.5).mean() * 100),
        ).round(3)
        print(summary.to_string())

    return df_buildings_with_preds


# ==================== MAIN ====================

@timeit
def pixel_postprocessing(force_recreate: bool = False) -> pd.DataFrame:
    """
    Full pixel-level postprocessing pipeline.

    Local equivalent of drive_to_result() in drive_to_results.py.
    """
    OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

    # Step 1: Merge tiles
    merged_fps = merge_all_windows(force_recreate)
    if not merged_fps:
        print("No merged rasters available. Run local_pixel_inference.py first.")
        return pd.DataFrame()

    # Step 2: Assign to buildings
    create_all_buildings_with_preds(merged_fps)

    # Step 3: Aggregate
    return aggregate_all_preds()


if __name__ == "__main__":
    pixel_postprocessing()
