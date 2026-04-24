"""
Local postprocessing for Gaza damage mapping.

Assigns UNOSAT point-level damage predictions to HOTOSM building footprints,
following Dietrich et al. (2025) methodology adapted for point predictions
rather than pixel rasters.

Dietrich et al. approach:
    1. Vectorize raster pixels → pixel polygons
    2. Intersect pixel polygons with building footprints
    3. Compute area-weighted mean prediction per building per time window
    4. Compute max prediction per building per time window
    5. Save buildings_preds.parquet

Local adaptation (point predictions instead of raster pixels):
    1. Buffer each UNOSAT prediction point by BUFFER_M metres → point polygons
    2. Intersect buffered points with building footprints
    3. Compute area-weighted mean prediction per building per time window
    4. Compute max prediction per building per time window
    5. Save buildings_preds.parquet

Predictions are scaled 0-255 (as in Dietrich et al.) and converted to
probabilities [0, 1] by dividing by 255.

Output:
    data/postprocessing/buildings_preds.parquet
        - index: building_id
        - columns: pred_{date} (weighted mean probability 0-1) per window
        - columns: max_{date} (max probability 0-1) per window
        - metadata: area, lon, lat, governorate, aoi from HOTOSM buildings
"""

import warnings
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.wkb import loads as wkb_loads
from tqdm.auto import tqdm

from src.constants import DATA_PATH, GAZA_WAR_START

# ==================== CONSTANTS ====================

# Buffer around each UNOSAT point in metres
# 10m matches Dietrich et al. building buffer in preprocessing.py
BUFFER_M = 10

# UTM zone 36N — appropriate for Gaza, used for accurate metre-based buffering
UTM_CRS = "EPSG:32636"

# Input paths
RUN_NAME = "rf_s1_2months_50trees_1x1_all7reducers_baseline"
PREDS_FP = DATA_PATH / f"runs/{RUN_NAME}/{RUN_NAME}.geojson"
BUILDINGS_FP = DATA_PATH / "overture_buildings/gaza_buildings.parquet"

# Output path
OUT_DIR = DATA_PATH / "postprocessing"
OUT_FP = OUT_DIR / "buildings_preds.parquet"


# ==================== LOADING ====================

def load_predictions(fp: Path = PREDS_FP) -> gpd.GeoDataFrame:
    """
    Load UNOSAT point predictions from GeoJSON.

    Returns GeoDataFrame with:
        - geometry: point geometry (EPSG:4326)
        - pred_{date}: damage probability scaled 0-255
        - unosat_id, aoi, date columns
    """
    print(f"Loading predictions from {fp}...")
    gdf = gpd.read_file(fp)
    print(f"  Loaded {len(gdf):,} prediction points")

    # Get pred columns
    pred_cols = [c for c in gdf.columns if c.startswith("pred_")]
    print(f"  {len(pred_cols)} time windows: {pred_cols[0]} to {pred_cols[-1]}")

    return gdf, pred_cols


def load_buildings(fp: Path = BUILDINGS_FP) -> gpd.GeoDataFrame:
    """
    Load HOTOSM building footprints from parquet.

    Returns GeoDataFrame with building polygons (EPSG:4326).
    """
    print(f"\nLoading buildings from {fp}...")
    gdf = gpd.read_parquet(fp)

    # Filter to buildings >= 50m² (matches Dietrich et al.)
    gdf = gdf[gdf["area_m2"] >= 50].copy()
    print(f"  Loaded {len(gdf):,} buildings >= 50m²")

    return gdf


# ==================== SPATIAL JOIN ====================

def buffer_predictions(gdf_preds: gpd.GeoDataFrame, buffer_m: int = BUFFER_M) -> gpd.GeoDataFrame:
    """
    Buffer prediction points by buffer_m metres.

    Converts to UTM for accurate metre-based buffering, then back to WGS84.
    """
    print(f"\nBuffering prediction points by {buffer_m}m...")
    gdf_utm = gdf_preds.to_crs(UTM_CRS)
    gdf_utm["geometry"] = gdf_utm.geometry.buffer(buffer_m)
    gdf_buffered = gdf_utm.to_crs("EPSG:4326")
    print(f"  Created {len(gdf_buffered):,} buffered point polygons")
    return gdf_buffered


def intersect_with_buildings(
    gdf_preds_buffered: gpd.GeoDataFrame,
    gdf_buildings: gpd.GeoDataFrame,
    pred_cols: list[str],
) -> gpd.GeoDataFrame:
    """
    Intersect buffered prediction points with building footprints.

    For each building-point overlap, records the intersection area
    and prediction values — used for area-weighted mean calculation.

    Processes by AOI to manage memory.
    """
    print("\nIntersecting predictions with buildings...")

    aois = gdf_preds_buffered["aoi"].unique()
    all_overlaps = []

    for aoi in tqdm(aois, desc="AOIs"):
        preds_aoi = gdf_preds_buffered[gdf_preds_buffered["aoi"] == aoi]

        # Filter buildings to AOI bounding box for speed
        bounds = preds_aoi.total_bounds
        buildings_aoi = gdf_buildings.cx[bounds[0]:bounds[2], bounds[1]:bounds[3]].copy()

        if len(buildings_aoi) == 0:
            print(f"  {aoi}: no buildings found in bounds")
            continue

        # Compute intersection
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning)
            overlap = gpd.overlay(
                buildings_aoi[["building_id", "geometry"]].reset_index(drop=True),
                preds_aoi[["unosat_id", "aoi", "geometry"] + pred_cols].reset_index(drop=True),
                how="intersection",
            )

        if len(overlap) == 0:
            print(f"  {aoi}: no overlaps found")
            continue

        # Compute intersection area in UTM metres
        overlap_utm = overlap.to_crs(UTM_CRS)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning)
            overlap["intersection_area"] = overlap_utm.geometry.area

        all_overlaps.append(overlap)
        print(f"  {aoi}: {len(overlap):,} building-point overlaps")

    if not all_overlaps:
        raise ValueError("No overlaps found between predictions and buildings!")

    result = pd.concat(all_overlaps, ignore_index=True)
    print(f"\nTotal overlaps: {len(result):,}")
    return result


# ==================== AGGREGATION ====================

def compute_building_predictions(
    overlap: gpd.GeoDataFrame,
    gdf_buildings: gpd.GeoDataFrame,
    pred_cols: list[str],
) -> pd.DataFrame:
    """
    Compute weighted mean and max damage probability per building per window.

    Follows Dietrich et al. postprocessing exactly:
        - weighted_mean = sum(pred * area) / sum(area) for each building
        - max = max(pred) for each building
        - probabilities scaled back to [0, 1] by dividing by 255

    Returns DataFrame indexed by building_id.
    """
    print("\nComputing per-building damage predictions...")

    # Compute weighted values
    for col in pred_cols:
        overlap[f"{col}_weighted"] = overlap[col] * overlap["intersection_area"]

    # Group by building
    grp = overlap.groupby("building_id")

    # Weighted mean per window
    weighted_cols = [f"{c}_weighted" for c in pred_cols]
    weighted_sum = grp[weighted_cols].sum()
    area_sum = grp["intersection_area"].sum()

    weighted_mean = weighted_sum.divide(area_sum, axis=0)
    weighted_mean.columns = [c.replace("_weighted", "") for c in weighted_mean.columns]

    # Max per window
    max_vals = grp[pred_cols].max()
    max_vals.columns = [c.replace("pred_", "max_") for c in max_vals.columns]

    # Scale to [0, 1]
    weighted_mean = weighted_mean / 255.0
    max_vals = max_vals / 255.0

    # Combine
    result = weighted_mean.join(max_vals)
    print(f"  Computed predictions for {len(result):,} buildings")

    # Merge with building metadata
    building_meta = gdf_buildings.set_index("building_id")[
        ["area_m2", "lon", "lat", "adm2_name"]
    ]

    result = building_meta.join(result, how="right")

    # Print summary
    post_war_cols = [c for c in weighted_mean.columns if c >= f"pred_{GAZA_WAR_START}"]
    if post_war_cols:
        mean_damage = result[post_war_cols].max(axis=1).mean()
        pct_damaged = (result[post_war_cols].max(axis=1) > 0.5).mean() * 100
        print(f"\n  Mean max post-war damage probability: {mean_damage:.3f}")
        print(f"  Buildings with >50% damage probability: {pct_damaged:.1f}%")

    return result


# ==================== MAIN ====================

def local_postprocessing(
    run_name: str = RUN_NAME,
    buffer_m: int = BUFFER_M,
) -> pd.DataFrame:
    """
    Full local postprocessing pipeline.

    Assigns UNOSAT point predictions to HOTOSM building footprints
    following Dietrich et al. (2025) methodology.

    Args:
        run_name: Name of the run directory containing predictions GeoJSON.
        buffer_m: Buffer radius in metres around each UNOSAT point.

    Returns:
        DataFrame with damage predictions per building.
    """
    OUT_DIR.mkdir(exist_ok=True, parents=True)

    preds_fp = DATA_PATH / f"runs/{run_name}/{run_name}.geojson"

    # Load data
    gdf_preds, pred_cols = load_predictions(preds_fp)
    gdf_buildings = load_buildings()

    # Buffer prediction points
    gdf_preds_buffered = buffer_predictions(gdf_preds, buffer_m)

    # Intersect with buildings
    overlap = intersect_with_buildings(gdf_preds_buffered, gdf_buildings, pred_cols)

    # Compute per-building predictions
    buildings_preds = compute_building_predictions(overlap, gdf_buildings, pred_cols)

    # Save
    buildings_preds.to_parquet(OUT_FP)
    print(f"\nSaved to {OUT_FP}")
    print(f"Shape: {buildings_preds.shape}")

    return buildings_preds


if __name__ == "__main__":
    result = local_postprocessing()

    # Print per-governorate summary
    print("\n=== Per-governorate damage summary ===")
    post_war_pred_cols = [c for c in result.columns
                          if c.startswith("pred_") and c >= f"pred_{GAZA_WAR_START}"]
    if "adm2_name" in result.columns and post_war_pred_cols:
        result["max_post_war_damage"] = result[post_war_pred_cols].max(axis=1)
        summary = result.groupby("adm2_name").agg(
            n_buildings=("max_post_war_damage", "count"),
            mean_damage=("max_post_war_damage", "mean"),
            pct_over_50=("max_post_war_damage", lambda x: (x > 0.5).mean() * 100),
        ).round(3)
        print(summary.to_string())
