"""
Preprocess UNOSAT damage assessment shapefiles for three transfer cities.

Converts raw shapefiles to the schema expected by load_unosat_labels()
in src/data/unosat.py, filtering to damage classes 1 (Destroyed) and
2 (Severe Damage) only - consistent with Gaza pipeline's labels_to_keep=[1,2].

Cities:
    Aleppo, Syria       - product 1118, assessed 2016-09-18
    Raqqa, Syria        - product 1192, assessed 2017-10-21
    Mosul, Iraq         - product 1188, assessed 2017-08-04

Output schema (matches load_unosat_labels()):
    unosat_id           - unique ID: {city_id}_{row_index}_1
    site_id             - original row index
    aoi                 - city identifier (ALP, RAQ, MOS)
    damage              - 1 or 2
    ep                  - always 1 (single epoch per point)
    date                - assessment date string
    geometry            - point geometry (EPSG:4326)
    date_first          - same as date (single epoch)
    date_first_severe   - same as date (all points are class 1 or 2)
    date_first_destroyed - same as date if damage==1, else None
    damage_max          - same as damage (single epoch)

Output files:
    test_sites/processed/{city_id}/unosat_labels.geojson
    test_sites/processed/{city_id}/unosat_aoi.geojson

Usage:
    cd /scratch/s1214882/gaza-damage-mapping
    python3 src/data/transfer_cities/preprocess_transfer_unosat.py
"""

from pathlib import Path

import geopandas as gpd
from shapely.ops import unary_union

# Paths
RAW_BASE = Path("test_sites/raw")
OUT_BASE = Path("test_sites/processed")

# Damage string to standard class mapping
DAMAGE_MAP = {
    "Destroyed": 1,
    "Severe Damage": 2,
    # All others dropped
}

# City configurations
# damage_field: column containing the most recent damage assessment
# date: assessment date (hardcoded - verified from shapefile inspection)
# shp: path to shapefile relative to RAW_BASE
CITIES = [
    {
        "city_id": "ALP",
        "city_name": "Aleppo",
        "country": "Syria",
        "assessment_date": "2016-09-18",
        "conflict_start": "2012-01-01",
        "shp": (
            "_static_unosat_filesystem_1118_UNOSAT_CE20130604SYR_Syria_Damage_Assessment_2016_shp"
            "/6_Damage_Sites_Aleppo_SDA.shp"
        ),
        "damage_field": "DmgCls_4",  # most recent epoch column
    },
    {
        "city_id": "RAQ",
        "city_name": "Raqqa",
        "country": "Syria",
        "assessment_date": "2017-10-21",
        "conflict_start": "2014-01-01",
        "shp": (
            "_static_unosat_filesystem_1192_CE20130604SYR_Raqqa_Deir_shp"
            "/Damage_Sites_Raqqa_CDA.shp"
        ),
        "damage_field": "DaSitCl5",  # most recent epoch column
    },
    {
        "city_id": "MOS",
        "city_name": "Mosul",
        "country": "Iraq",
        "assessment_date": "2017-08-04",
        "conflict_start": "2014-06-01",
        "shp": (
            "_static_unosat_filesystem_1188_Damage_assessment_Mosul_20170804_shp"
            "/Damage_assessment_Mosul_20170804_shp/Mosul_Damage_Sites_20170804.shp"
        ),
        "damage_field": "Main_Damag",  # single epoch
    },
]


def preprocess_city(city: dict) -> None:
    city_id = city["city_id"]
    city_name = city["city_name"]
    shp_fp = RAW_BASE / city["shp"]
    out_dir = OUT_BASE / city_id.lower()
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"{city_name} ({city['country']}, {city['assessment_date']})")
    print(f"{'='*60}")

    # Load shapefile
    print(f"  Reading: {shp_fp.name}")
    gdf = gpd.read_file(shp_fp)
    print(f"  Raw features: {len(gdf):,}")

    # Reproject to WGS84 if needed
    if gdf.crs is None:
        print("  WARNING: No CRS - assuming EPSG:4326")
        gdf = gdf.set_crs("EPSG:4326")
    elif gdf.crs.to_epsg() != 4326:
        gdf = gdf.to_crs("EPSG:4326")

    # Map damage values
    dmg_field = city["damage_field"]
    gdf["damage_std"] = gdf[dmg_field].map(DAMAGE_MAP)

    # Print what we're keeping and dropping
    print(f"\n  Raw damage value counts ({dmg_field}):")
    for val, count in gdf[dmg_field].value_counts(dropna=False).items():
        mapped = DAMAGE_MAP.get(val, "DROP")
        print(f"    {str(val):<45} {count:>6,}  -> {mapped}")

    # Filter to classes 1 and 2 only
    gdf = gdf[gdf["damage_std"].isin([1, 2])].copy()
    print(f"\n  After filtering to classes 1+2: {len(gdf):,}")
    print(f"  Class distribution: {gdf['damage_std'].value_counts().sort_index().to_dict()}")

    # Build output schema matching load_unosat_labels()
    assessment_date = city["assessment_date"]
    records = []
    for idx, row in gdf.iterrows():
        dmg = int(row["damage_std"])
        records.append(
            {
                "unosat_id": f"{city_id}_{idx}_1",
                "site_id": idx,
                "aoi": city_id,
                "damage": dmg,
                "ep": 1,
                "date": assessment_date,
                "geometry": row["geometry"],
                "date_first": assessment_date,
                "date_first_severe": assessment_date,
                "date_first_destroyed": assessment_date if dmg == 1 else None,
                "damage_max": dmg,
            }
        )

    gdf_out = gpd.GeoDataFrame(records, geometry="geometry", crs="EPSG:4326")
    gdf_out = gdf_out.set_index("unosat_id")

    # Save labels
    fp_labels = out_dir / "unosat_labels.geojson"
    gdf_out.to_file(fp_labels, driver="GeoJSON")
    print(f"\n  Saved {len(gdf_out):,} labels -> {fp_labels}")

    # Save AOI boundary (convex hull of all damage points)
    aoi_geom = unary_union(gdf_out.geometry).convex_hull
    gdf_aoi = gpd.GeoDataFrame(
        [
            {
                "aoi": city_id,
                "city": city_name,
                "country": city["country"],
                "assessment_date": assessment_date,
                "conflict_start": city["conflict_start"],
                "geometry": aoi_geom,
            }
        ],
        geometry="geometry",
        crs="EPSG:4326",
    )
    fp_aoi = out_dir / "unosat_aoi.geojson"
    gdf_aoi.to_file(fp_aoi, driver="GeoJSON")
    print(f"  Saved AOI boundary -> {fp_aoi}")


def main() -> None:
    OUT_BASE.mkdir(parents=True, exist_ok=True)

    for city in CITIES:
        preprocess_city(city)

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"  {'City':<12} {'Total':>8} {'Destroyed':>10} {'Severe':>8}")
    print("  " + "-" * 42)
    for city in CITIES:
        fp = OUT_BASE / city["city_id"].lower() / "unosat_labels.geojson"
        if fp.exists():
            gdf = gpd.read_file(fp)
            n1 = (gdf["damage"] == 1).sum()
            n2 = (gdf["damage"] == 2).sum()
            print(f"  {city['city_name']:<12} {len(gdf):>8,} {n1:>10,} {n2:>8,}")
        else:
            print(f"  {city['city_name']:<12} MISSING")


if __name__ == "__main__":
    main()
