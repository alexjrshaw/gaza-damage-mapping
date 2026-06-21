"""
Create GEE intermediate assets for transfer city zero-shot evaluation.

Local equivalent of src/data/sentinel1/intermediate_data.py, adapted
for the three transfer cities (Aleppo, Raqqa, Mosul).

For each city and each valid orbit, creates a GEE FeatureCollection
asset containing one row per (UNOSAT point x S1 image), with VV and
VH backscatter values extracted at each point location.

These assets are then downloaded locally by download_intermediate_assets.py
and used by extract_features_local.py to compute the 28 SAR features.

GEE asset paths:
    projects/gaza-damage-mapping/assets/transfer-cities/
        intermediate_features/ts_s1_1x1/{city_id}_orbit{orbit}

Usage:
    Run interactively (requires internet/GEE access):
    cd /scratch/s1214882/gaza-damage-mapping
    python3 src/data/transfer_cities/create_intermediate_assets.py
"""

import sys

import ee

sys.path.insert(0, "/scratch/s1214882/gaza-damage-mapping")

from src.data.sentinel1.collection import get_s1_collection
from src.data.transfer_cities.constants_transfer import TRANSFER_CITIES, TRANSFER_GEE_FOLDER
from src.utils.gee import asset_exists, create_folder, fill_nan_with_mean, init_gee

init_gee()

SCALE = 10  # 10m — matches Gaza pipeline
EXTRACT = "1x1"
INTERMEDIATE_FOLDER = TRANSFER_GEE_FOLDER + f"intermediate_features/ts_s1_{EXTRACT}"


def ensure_intermediate_folder() -> None:
    """Create intermediate features folder structure if needed."""
    folders = [
        TRANSFER_GEE_FOLDER + "intermediate_features",
        INTERMEDIATE_FOLDER,
    ]
    for folder in folders:
        if not asset_exists(folder):
            create_folder(folder)
            print(f"  Created: {folder.split('assets/')[-1]}")
        else:
            print(f"  Exists:  {folder.split('assets/')[-1]}")


def create_fc_city_orbit(
    city_id: str,
    orbit: int,
    cfg: dict,
    force: bool = False,
) -> None:
    """
    Create GEE intermediate asset for one city/orbit combination.

    Mirrors create_fc_aoi_orbit() in intermediate_data.py exactly:
        1. Load UNOSAT labels FeatureCollection from GEE
        2. Load AOI geometry from GEE
        3. Filter S1 collection to AOI, date range, and orbit
        4. Extract VV/VH at all UNOSAT points for each S1 image
        5. Export as GEE asset

    Args:
        city_id: City identifier e.g. 'ALP'
        orbit:   Sentinel-1 relative orbit number
        cfg:     City config dict from TRANSFER_CITIES
        force:   Re-export even if asset already exists
    """
    asset_id = INTERMEDIATE_FOLDER + f"/{city_id}_orbit{orbit}"

    if asset_exists(asset_id) and not force:
        print(f"  {city_id}_orbit{orbit}: already exists — skipping")
        return

    print(f"  {city_id}_orbit{orbit}: creating...")

    # Load UNOSAT labels from GEE
    labels_asset = TRANSFER_GEE_FOLDER + f"UNOSAT_labels/{city_id}"
    labels = ee.FeatureCollection(labels_asset)

    # Load AOI geometry from GEE
    aoi_asset = TRANSFER_GEE_FOLDER + f"AOIs/{city_id}"
    geo = ee.FeatureCollection(aoi_asset).geometry()

    # Load S1 collection filtered to city date range and orbit
    s1 = get_s1_collection(
        geo=geo,
        start=cfg["gee_start"],
        end=cfg["gee_end"],
    ).filterMetadata("relativeOrbitNumber_start", "equals", orbit)

    # Fill NaN values with column mean — mirrors Gaza pipeline
    s1 = fill_nan_with_mean(s1)

    print(f"    S1 images available: checking...")

    def extract_image(img):
        """Extract VV and VH at all UNOSAT points for one S1 image."""
        return (
            img.select(["VV", "VH"])
            .reduceRegions(
                collection=labels,
                reducer=ee.Reducer.mean(),
                scale=SCALE,
            )
            .map(lambda f: f.set("system:time_start", img.get("system:time_start")))
        )

    # Apply to all images and flatten — mirrors Gaza pipeline exactly
    fc_extracted = s1.map(extract_image).flatten()

    # Export to GEE asset
    task = ee.batch.Export.table.toAsset(
        collection=fc_extracted,
        description=f"{city_id}_orbit{orbit}_{SCALE}m"[:100],
        assetId=asset_id,
    )
    task.start()
    print(f"  {city_id}_orbit{orbit}: export started -> {asset_id.split('assets/')[-1]}")


def create_all_intermediate_assets(force: bool = False) -> None:
    """Create intermediate assets for all cities and orbits."""

    print("Ensuring intermediate folder structure...")
    ensure_intermediate_folder()

    total = sum(len(cfg["orbits"]) for cfg in TRANSFER_CITIES.values())
    done = 0

    for city_id, cfg in TRANSFER_CITIES.items():
        print(f"\n{'='*60}")
        print(f"{city_id} — {cfg['city_name']} ({cfg['country']})")
        print(f"  Orbits: {cfg['orbits']}")
        print(f"  Date range: {cfg['gee_start']} → {cfg['gee_end']}")
        print(f"  UNOSAT points: loading from GEE")
        print(f"{'='*60}")

        for orbit in cfg["orbits"]:
            create_fc_city_orbit(city_id, orbit, cfg, force=force)
            done += 1

    print(f"\n{'='*60}")
    print(f"All {total} export tasks started.")
    print("These run asynchronously in GEE — check progress at:")
    print("  https://code.earthengine.google.com/tasks")
    print(f"{'='*60}")

    # Print expected asset paths
    print("\nExpected asset paths when complete:")
    for city_id, cfg in TRANSFER_CITIES.items():
        for orbit in cfg["orbits"]:
            asset_id = INTERMEDIATE_FOLDER + f"/{city_id}_orbit{orbit}"
            status = "EXISTS" if asset_exists(asset_id) else "PENDING"
            print(f"  {status}: {asset_id.split('assets/')[-1]}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true", help="Re-export even if assets already exist")
    parser.add_argument("--city", type=str, default=None, help="Run for a single city only (e.g. ALP)")
    args = parser.parse_args()

    if args.city:
        city_id = args.city.upper()
        assert city_id in TRANSFER_CITIES, f"Unknown city: {city_id}"
        cfg = TRANSFER_CITIES[city_id]
        ensure_intermediate_folder()
        print(f"Running for {city_id} only...")
        for orbit in cfg["orbits"]:
            create_fc_city_orbit(city_id, orbit, cfg, force=args.force)
    else:
        create_all_intermediate_assets(force=args.force)
