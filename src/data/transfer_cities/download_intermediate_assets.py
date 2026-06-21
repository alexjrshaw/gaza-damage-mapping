"""
Download GEE intermediate assets for transfer cities to local parquet cache.

Local equivalent of src/data/sentinel1/download_intermediate_assets.py,
adapted for the three transfer cities (Aleppo, Raqqa, Mosul).

Must be run interactively (not as Slurm batch job) since Forth compute
nodes don't have internet access. Login node and interactive sessions do.

Downloads each city/orbit GEE asset to:
    data/transfer_cities/intermediate_features_cache/{city_id}_orbit{orbit}.parquet

Usage:
    cd /scratch/s1214882/gaza-damage-mapping
    python3 src/data/transfer_cities/download_intermediate_assets.py
"""

import sys
import time

import ee
import pandas as pd

sys.path.insert(0, "/scratch/s1214882/gaza-damage-mapping")

from src.data.transfer_cities.constants_transfer import TRANSFER_CACHE_DIR, TRANSFER_CITIES, TRANSFER_GEE_FOLDER
from src.utils.gdrive import drive_to_local
from src.utils.gee import asset_exists, init_gee

init_gee()

EXTRACT = "1x1"
INTERMEDIATE_FOLDER = TRANSFER_GEE_FOLDER + f"intermediate_features/ts_s1_{EXTRACT}"
DRIVE_FOLDER = "transfer_cities_intermediate_features"


def download_intermediate_asset(
    city_id: str,
    orbit: int,
    force: bool = False,
) -> None:
    """
    Download one GEE intermediate asset to local parquet cache.

    Mirrors download_intermediate_asset() in download_intermediate_assets.py:
        1. Check if cache file already exists
        2. Export GEE asset to Google Drive as CSV
        3. Download CSV from Drive to local tmp directory
        4. Convert CSV to parquet and save to cache

    Args:
        city_id: City identifier e.g. 'ALP'
        orbit:   Sentinel-1 relative orbit number
        force:   Re-download even if cache exists
    """
    TRANSFER_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    fp = TRANSFER_CACHE_DIR / f"{city_id}_orbit{orbit}.parquet"

    if fp.exists() and not force:
        print(f"  {city_id}_orbit{orbit}: already cached")
        return

    asset_id = INTERMEDIATE_FOLDER + f"/{city_id}_orbit{orbit}"
    if not asset_exists(asset_id):
        print(f"  {city_id}_orbit{orbit}: GEE asset not found — run create_intermediate_assets.py first")
        return

    print(f"  {city_id}_orbit{orbit}: exporting to Drive...")
    fc = ee.FeatureCollection(asset_id)
    description = f"{city_id}_orbit{orbit}_features"

    task = ee.batch.Export.table.toDrive(
        collection=fc,
        description=description,
        folder=DRIVE_FOLDER,
        fileFormat="CSV",
    )
    task.start()

    # Wait for export to complete
    print(f"  {city_id}_orbit{orbit}: waiting for Drive export...")
    while True:
        status = task.status()
        state = status["state"]
        if state == "COMPLETED":
            break
        elif state in ["FAILED", "CANCELLED"]:
            raise RuntimeError(
                f"Drive export failed for {city_id}_orbit{orbit}: " f"{status.get('error_message', 'unknown error')}"
            )
        time.sleep(30)
    print(f"  {city_id}_orbit{orbit}: export complete, downloading from Drive...")

    # Download CSV from Drive
    tmp_dir = TRANSFER_CACHE_DIR / "tmp"
    tmp_dir.mkdir(exist_ok=True)
    drive_to_local(DRIVE_FOLDER, tmp_dir, delete_in_drive=True, verbose=0)

    # Convert CSV to parquet
    csv_fp = tmp_dir / f"{description}.csv"
    if not csv_fp.exists():
        # GEE sometimes splits large files — check for numbered parts
        parts = sorted(tmp_dir.glob(f"{description}*.csv"))
        if parts:
            print(f"  {city_id}_orbit{orbit}: merging {len(parts)} CSV parts...")
            df = pd.concat([pd.read_csv(p) for p in parts], ignore_index=True)
        else:
            raise FileNotFoundError(f"CSV not found after Drive download: {csv_fp}")
    else:
        df = pd.read_csv(csv_fp)

    df.to_parquet(fp)
    print(f"  {city_id}_orbit{orbit}: {len(df):,} rows saved -> {fp.name}")


def download_all(force: bool = False) -> None:
    """Download all transfer city intermediate assets."""

    total = sum(len(cfg["orbits"]) for cfg in TRANSFER_CITIES.values())
    print(f"Downloading {total} intermediate assets to {TRANSFER_CACHE_DIR}")

    for city_id, cfg in TRANSFER_CITIES.items():
        print(f"\n{'='*60}")
        print(f"{city_id} — {cfg['city_name']} ({cfg['country']})")
        print(f"  Orbits: {cfg['orbits']}")
        print(f"{'='*60}")

        for orbit in cfg["orbits"]:
            download_intermediate_asset(city_id, orbit, force=force)

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    for city_id, cfg in TRANSFER_CITIES.items():
        for orbit in cfg["orbits"]:
            fp = TRANSFER_CACHE_DIR / f"{city_id}_orbit{orbit}.parquet"
            if fp.exists():
                df = pd.read_parquet(fp)
                print(f"  {city_id}_orbit{orbit}: {len(df):,} rows")
            else:
                print(f"  {city_id}_orbit{orbit}: MISSING")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true", help="Re-download even if cache exists")
    parser.add_argument("--city", type=str, default=None, help="Download single city only (e.g. ALP)")
    args = parser.parse_args()

    if args.city:
        city_id = args.city.upper()
        assert city_id in TRANSFER_CITIES, f"Unknown city: {city_id}"
        cfg = TRANSFER_CITIES[city_id]
        print(f"Downloading {city_id} only...")
        for orbit in cfg["orbits"]:
            download_intermediate_asset(city_id, orbit, force=args.force)
    else:
        download_all(force=args.force)
