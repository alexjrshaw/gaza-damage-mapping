"""
Preprocess HOTOSM building footprints for Gaza.

Replaces the Overture Maps pipeline used for Ukraine, following
Scher & Van Den Hoek (2025) who identify HOTOSM as the most
comprehensive pre-conflict building footprint dataset for Gaza.

Pipeline:
1. Download raw data if not present (via download.py)
2. Filter to Gaza Strip only using official OCHA boundary
3. Filter buildings >= 50m2 (matching Dietrich et al. 2025 threshold)
4. Add area, centroid lon/lat
5. Add UNOSAT damage info (5m buffer overlay)
6. Add admin (governorate) info
7. Save as parquet for use in postprocessing
"""

import geopandas as gpd

from src.constants import DATA_PATH, OVERTURE_PATH
from src.data.hotosm.download import HOTOSM_RAW_FP, download_hotosm_buildings
from src.data.unosat import load_unosat_labels
from src.utils.geo import (
    get_best_utm_crs_from_gdf,
    load_gaza_admin_polygons,
    load_gaza_strip_boundary,
)
from src.utils.time import timeit

# Output path — named for pipeline compatibility with drive_to_results.py
HOTOSM_PROCESSED_FP = OVERTURE_PATH / "gaza_buildings.parquet"

# Minimum building area — matches Dietrich et al. (2025)
MIN_BUILDING_AREA_M2 = 50


@timeit
def process_hotosm() -> None:
    """
    Full preprocessing pipeline for HOTOSM Gaza buildings.

    Produces gaza_buildings.parquet with one row per building,
    including area, centroid, UNOSAT damage info and governorate.
    """

    # Step 1 — Download if needed
    download_hotosm_buildings()

    # Step 2 — Load and filter to Gaza Strip
    print("Loading HOTOSM buildings...")
    gdf = gpd.read_file(HOTOSM_RAW_FP)
    print(f"  Total buildings (Palestine): {len(gdf):,}")

    gaza_boundary = load_gaza_strip_boundary()
    gdf_proj = gdf.to_crs("EPSG:32636")
    gdf = gdf[gdf_proj.geometry.centroid.to_crs("EPSG:4326").within(gaza_boundary)].copy()
    print(f"  Buildings within Gaza Strip: {len(gdf):,}")

    # Step 3 — Compute area and filter small buildings
    gdf_proj = gdf.to_crs("EPSG:32636")  # UTM zone 36N covers Gaza
    gdf["area_m2"] = gdf_proj.geometry.area
    centroids = gdf.to_crs("EPSG:32636").geometry.centroid.to_crs("EPSG:4326")
    gdf["lon"] = centroids.x
    gdf["lat"] = centroids.y
    gdf = gdf[gdf["area_m2"] >= MIN_BUILDING_AREA_M2].copy()
    print(f"  Buildings >= {MIN_BUILDING_AREA_M2}m2: {len(gdf):,}")

    # Step 4 — Rename osm_id to building_id for pipeline compatibility
    gdf = gdf.rename(columns={"osm_id": "building_id"})

    # Step 5 — Add UNOSAT damage info
    print("Adding UNOSAT damage info...")
    gdf = add_unosat_info(gdf)

    # Step 6 — Add admin info
    print("Adding admin (governorate) info...")
    gdf = add_admin_info(gdf)

    # Step 7 — Save
    HOTOSM_PROCESSED_FP.parent.mkdir(exist_ok=True, parents=True)
    gdf.to_parquet(HOTOSM_PROCESSED_FP)
    print(f"Saved {len(gdf):,} buildings to {HOTOSM_PROCESSED_FP}")


@timeit
@timeit
def add_unosat_info(gdf: gpd.GeoDataFrame, buffer_m: int = 5) -> gpd.GeoDataFrame:
    """
    Add UNOSAT damage info to buildings using a 5m buffer overlay.
    Processes one AOI at a time to keep memory manageable,
    following Dietrich et al. (2025) approach.
    """
    from src.data.unosat import load_unosat_labels, load_unosat_geo
    from src.utils.geo import get_best_utm_crs_from_gdf
    from src.constants import AOIS

    gdf_buildings_with_unosat = []

    for aoi in AOIS:
        print(f"  Processing AOI {aoi}...")

        # Load UNOSAT labels for this AOI (all damage classes)
        points = load_unosat_labels(aoi=aoi, labels_to_keep=None, combine_epoch="last").reset_index()

        # Get AOI geometry and subset buildings
        geo = load_unosat_geo(aoi)
        gdf_aoi = gdf[gdf.geometry.intersects(geo)].copy()

        if len(gdf_aoi) == 0:
            print(f"    No buildings found for {aoi}")
            continue

        # Project to UTM for accurate buffering
        crs_proj = get_best_utm_crs_from_gdf(points)
        gdf_aoi.geometry = gdf_aoi.to_crs(crs_proj).buffer(buffer_m).to_crs("EPSG:4326")

        # Overlay buildings with UNOSAT points
        pts_in = gpd.overlay(
            points[["unosat_id", "damage", "date", "aoi", "geometry"]],
            gdf_aoi[["building_id", "geometry"]],
            how="intersection",
        )

        if len(pts_in) == 0:
            continue

        # Keep most severe damage per building
        pts_sorted = pts_in.sort_values("damage")
        buildings_with_unosat = pts_sorted.groupby("building_id").agg(
            {"damage": "first", "unosat_id": "first", "date": "first", "aoi": "first"}
        )
        buildings_with_unosat.rename(
            columns={"damage": "unosat_damage", "date": "unosat_date", "aoi": "unosat_aoi"},
            inplace=True,
        )
        gdf_buildings_with_unosat.append(buildings_with_unosat)
        print(f"    Found {len(buildings_with_unosat)} buildings with UNOSAT damage in {aoi}")

    if gdf_buildings_with_unosat:
        all_unosat = pd.concat(gdf_buildings_with_unosat)
        gdf = gdf.merge(all_unosat, on="building_id", how="left")
    else:
        gdf["unosat_damage"] = None
        gdf["unosat_date"] = None
        gdf["unosat_aoi"] = None

    print(f"  Buildings with UNOSAT damage label: {gdf['unosat_damage'].notna().sum():,}")
    return gdf


@timeit
def add_admin_info(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    Add governorate (admin level 2) to each building via spatial join on centroid.

    Args:
        gdf (gpd.GeoDataFrame): Buildings GeoDataFrame with lon/lat columns.

    Returns:
        gpd.GeoDataFrame: Buildings with adm2_name and adm2_id columns added.
    """
    adm2 = load_gaza_admin_polygons(adm_level=2)[
        ["adm2_name", "admin_id", "geometry"]
    ].rename(columns={"admin_id": "adm2_id"})

    # Create point GeoDataFrame from centroids
    gdf_centroids = gpd.GeoDataFrame(
        gdf[["building_id", "lon", "lat"]],
        geometry=gpd.points_from_xy(gdf["lon"], gdf["lat"]),
        crs="EPSG:4326",
    )

    # Spatial join
    joined = gpd.sjoin(gdf_centroids, adm2, how="left", predicate="within")
    gdf = gdf.merge(
        joined[["building_id", "adm2_name", "adm2_id"]],
        on="building_id",
        how="left",
    )
    print(f"  Buildings with governorate assigned: {gdf['adm2_name'].notna().sum():,}")
    return gdf


if __name__ == "__main__":
    process_hotosm()
