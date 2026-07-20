"""
Damage rate by OSM building type, mirroring Dietrich et al.'s (2025) Table 3.

Joins this study's building-level damage classification (buildings_damage.parquet,
from Equation 3) against OSM 'building' type tags from the raw HOTOSM export, via
building_id == osm_id (confirmed 100% match for all 220,820 Gaza buildings).

Following Dietrich et al. (2025), only buildings with a specific (non-generic
"yes") OSM building tag are included in the breakdown; the overall % of buildings
carrying any usable type metadata is reported alongside, mirroring their own
"For a subset of buildings... we were able to retrieve meta-data" framing.

Usage:
    python3 alex/tmp/damage_by_building_type.py
"""

import geopandas as gpd
import pandas as pd

from src.constants import DATA_PATH

DAMAGE_FP = DATA_PATH / "pixel_postprocessing/buildings_damage.parquet"
RAW_HOTOSM_FP = DATA_PATH / "raw/hotosm_pse_buildings_polygons_geojson.geojson"
MIN_BUILDINGS_PER_TYPE = 20  # exclude very small categories from the headline table


def main():
    print("Loading building damage classifications...")
    damage = pd.read_parquet(DAMAGE_FP)
    print(f"  {len(damage):,} buildings")

    print("Loading raw HOTOSM OSM building tags...")
    raw = gpd.read_file(RAW_HOTOSM_FP, columns=["osm_id", "building"])
    raw = raw.rename(columns={"osm_id": "building_id"}).set_index("building_id")
    print(f"  {len(raw):,} buildings (all of Palestine)")

    joined = damage.join(raw[["building"]], how="left")
    print(f"\nJoined: {len(joined):,} rows")

    n_total = len(joined)
    n_any_tag = joined["building"].notna().sum()
    n_meaningful = (joined["building"].notna() & (joined["building"] != "yes")).sum()
    print(f"  Buildings with any 'building' tag: {n_any_tag:,} ({n_any_tag/n_total*100:.1f}%)")
    print(f"  Buildings with a specific (non-generic) type: {n_meaningful:,} " f"({n_meaningful/n_total*100:.1f}%)")

    # Restrict to the meaningful subset for the headline breakdown
    subset = joined[joined["building"].notna() & (joined["building"] != "yes")].copy()

    summary = (
        subset.groupby("building")
        .agg(n_buildings=("damaged", "count"), n_damaged=("damaged", "sum"))
        .assign(pct_damaged=lambda d: d["n_damaged"] / d["n_buildings"] * 100)
        .sort_values("n_buildings", ascending=False)
    )
    summary = summary[summary["n_buildings"] >= MIN_BUILDINGS_PER_TYPE]

    print("\n=== Damage rate by OSM building type ===")
    print(summary.to_string())

    overall_pct = joined["damaged"].mean() * 100
    print(f"\nOverall damage rate (all buildings): {overall_pct:.1f}%")

    out_fp = DATA_PATH / "ablation_runs/figures/damage_by_building_type.csv"
    summary.to_csv(out_fp)
    print(f"\nSaved to {out_fp}")


if __name__ == "__main__":
    main()
