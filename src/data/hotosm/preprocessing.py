"""
Preprocess HOTOSM building footprints for Gaza.

Replaces the Overture Maps pipeline used for Ukraine (overture/preprocessing.py),
following Scher & Van Den Hoek (2025) who identify HOTOSM as the most
comprehensive pre-conflict building footprint dataset for Gaza.

The structure and logic follows Dietrich et al. (2025) exactly, with three
adaptations for Gaza:
    1. HOTOSM GeoJSON instead of Overture Maps parquet
    2. OCHA admin boundaries instead of Ukraine admin shapefiles
    3. gpd.sjoin instead of gpd.overlay in add_unosat_info — same result
       but far more memory efficient for Gaza's dense urban geography
"""

import geopandas as gpd
import pandas as pd
from tqdm.auto import tqdm

from shapely.wkb import loads as wkb_loads
from src.constants import DATA_PATH, OVERTURE_PATH
from src.data.hotosm.download import HOTOSM_RAW_FP, download_hotosm_buildings
from src.data.unosat import load_unosat_geo, load_unosat_labels
from src.data.utils import get_all_aois
from src.utils.geo import get_best_utm_crs_from_gdf, load_gaza_admin_polygons, load_gaza_strip_boundary
from src.utils.time import timeit

# Output path — mirrors OVERTURE_PROCESSED_FP in Ukraine pipeline
HOTOSM_PROCESSED_FP = OVERTURE_PATH / "gaza_buildings.parquet"

# Minimum building area — matches Dietrich et al. (2025)
MIN_BUILDING_AREA_M2 = 50


@timeit
def process_hotosm() -> None:
    """
    Process HOTOSM building footprints for Gaza.

    Mirrors process_overture() in Ukraine pipeline:
        1. Keep only buildings within Gaza Strip and relevant properties
        2. Add UNOSAT info
        3. Add admin info
    """

    # Step 1 — Download if needed
    download_hotosm_buildings()

    # Keep only buildings within Gaza Strip and relevant properties
    print("Keeping only buildings within Gaza Strip and relevant properties...")
    only_in_gaza_and_relevant_properties()

    # Add UNOSAT info
    print("Adding UNOSAT info...")
    add_unosat_info()

    # Add admin info
    print("Adding admin info...")
    add_admin_info()


@timeit
def only_in_gaza_and_relevant_properties() -> None:
    """
    Filter to buildings within Gaza Strip and compute relevant properties.

    Mirrors only_in_ukraine_and_relevant_properties() but uses GeoPandas
    instead of DuckDB since HOTOSM data is GeoJSON not parquet.
    """

    print("Loading HOTOSM buildings...")
    gdf = gpd.read_file(HOTOSM_RAW_FP)
    print(f"  Total buildings (Palestine): {len(gdf):,}")

    # Filter to Gaza Strip using OCHA boundary
    gaza_boundary = load_gaza_strip_boundary()
    gdf_proj = gdf.to_crs("EPSG:32636")
    gdf = gdf[gdf_proj.geometry.centroid.to_crs("EPSG:4326").within(gaza_boundary)].copy()
    print(f"  Buildings within Gaza Strip: {len(gdf):,}")

    # Compute area and centroid
    gdf_proj = gdf.to_crs("EPSG:32636")
    gdf["area_m2"] = gdf_proj.geometry.area
    centroids = gdf_proj.geometry.centroid.to_crs("EPSG:4326")
    gdf["lon"] = centroids.x
    gdf["lat"] = centroids.y

    # Filter small buildings — matches Dietrich et al. (2025)
    gdf = gdf[gdf["area_m2"] >= MIN_BUILDING_AREA_M2].copy()
    print(f"  Buildings >= {MIN_BUILDING_AREA_M2}m2: {len(gdf):,}")

    # Rename osm_id to building_id for pipeline compatibility
    gdf = gdf.rename(columns={"osm_id": "building_id"}).reset_index(drop=True)

    # Keep relevant columns — mirrors Ukraine pipeline
    gdf = gdf[["building_id", "geometry", "area_m2", "lon", "lat"]]

    # Save geometry as WKB — mirrors Ukraine pipeline (geometry_wkb column)
    gdf["geometry_wkb"] = gdf.geometry.apply(lambda g: g.wkb)
    gdf = gdf.drop(columns=["geometry"])

    # Save to parquet
    HOTOSM_PROCESSED_FP.parent.mkdir(exist_ok=True, parents=True)
    gdf.to_parquet(HOTOSM_PROCESSED_FP)
    print("Done")


@timeit
def add_unosat_info(buffer: int = 5) -> None:
    """
    Add UNOSAT damage info to buildings using a 5m buffer spatial join.

    Mirrors add_unosat_info() in Ukraine pipeline. Gaza-specific adaptation:
    uses gpd.sjoin instead of gpd.overlay — identical output but more memory
    efficient for Gaza's dense urban geography where gpd.overlay causes OOM.

    Args:
        buffer (int): Buffer around buildings in metres. Defaults to 5.
    """

    # Read processed buildings — mirrors Ukraine pipeline
    df = pd.read_parquet(HOTOSM_PROCESSED_FP)
    df["geometry"] = df.geometry_wkb.apply(lambda x: wkb_loads(bytes(x)))
    gdf_all = gpd.GeoDataFrame(df, geometry="geometry", crs="EPSG:4326")
    del df
    print("Buildings loaded")

    # Loop over AOIs — mirrors Ukraine pipeline
    gdf_buildings_with_unosat = []

    for aoi in tqdm(get_all_aois()):
        # Labels for the given AOI (all damage classes)
        points = load_unosat_labels(aoi, labels_to_keep=None, combine_epoch="first_severe").reset_index()

        # Keep only buildings within the AOI
        geo = load_unosat_geo(aoi)
        gdf = gdf_all[gdf_all.geometry.intersects(geo)].copy()

        if len(gdf) == 0:
            print(f"  No buildings found for {aoi}")
            continue

        # Buffer around buildings
        crs_proj = get_best_utm_crs_from_gdf(points)
        gdf = gdf.copy()
        gdf.geometry = gdf.to_crs(crs_proj).buffer(buffer).to_crs("EPSG:4326")

        # Spatial join — Gaza adaptation: sjoin instead of overlay
        # gpd.overlay creates intersection geometries (OOM for dense Gaza data)
        # gpd.sjoin just matches IDs — identical output, fraction of memory
        pts_in = gpd.sjoin(
            points[["unosat_id", "damage", "date", "aoi", "geometry"]],
            gdf[["building_id", "geometry"]],
            how="inner",
            predicate="within",
        )

        if len(pts_in) == 0:
            continue

        # Keep lowest damage level (most severe) — mirrors Ukraine pipeline
        pts_sorted = pts_in.sort_values(by=["damage"])
        buildings_with_unosat = pts_sorted.groupby("building_id").agg(
            {"damage": "first", "unosat_id": "first", "date": "first"}
        )
        buildings_with_unosat["aoi"] = aoi
        gdf_buildings_with_unosat.append(buildings_with_unosat)

    gdf_buildings_with_unosat = pd.concat(gdf_buildings_with_unosat)
    gdf_buildings_with_unosat.rename(
        columns={"damage": "unosat_damage", "date": "unosat_date", "aoi": "unosat_aoi"},
        inplace=True,
    )

    gdf_all = gdf_all.merge(gdf_buildings_with_unosat, on="building_id", how="left")

    # Save
    gdf_all.to_parquet(HOTOSM_PROCESSED_FP)
    print("Done")
    print(f"  Buildings with UNOSAT damage label: {gdf_all['unosat_damage'].notna().sum():,}")


@timeit
def add_admin_info() -> None:
    """
    Add governorate (admin level 2) to each building.

    Mirrors add_admin_info() in Ukraine pipeline but uses OCHA
    boundaries instead of Ukraine admin shapefiles, and uses
    gpd.sjoin instead of DuckDB for simplicity.
    """

    # Load buildings — mirrors Ukraine pipeline
    df = pd.read_parquet(HOTOSM_PROCESSED_FP)
    df["geometry"] = df.geometry_wkb.apply(lambda x: wkb_loads(bytes(x)))
    gdf_all = gpd.GeoDataFrame(df, geometry="geometry", crs="EPSG:4326")
    del df

    # Load admin boundaries
    adm2 = load_gaza_admin_polygons(adm_level=2)[
        ["adm2_name", "admin_id", "geometry"]
    ].rename(columns={"admin_id": "adm2_id"})

    # Spatial join on centroid
    gdf_centroids = gpd.GeoDataFrame(
        gdf_all[["building_id"]],
        geometry=gpd.points_from_xy(gdf_all["lon"], gdf_all["lat"]),
        crs="EPSG:4326",
    )
    joined = gpd.sjoin(gdf_centroids, adm2, how="left", predicate="within")
    gdf_all = gdf_all.merge(
        joined[["building_id", "adm2_name", "adm2_id"]],
        on="building_id",
        how="left",
    )

    # Save
    gdf_all.to_parquet(HOTOSM_PROCESSED_FP)
    print("Done")
    print(f"  Buildings with governorate assigned: {gdf_all['adm2_name'].notna().sum():,}")

    # Summary
    print("\n── Buildings per governorate ──")
    print(gdf_all["adm2_name"].value_counts().to_string())


if __name__ == "__main__":
    process_hotosm()
