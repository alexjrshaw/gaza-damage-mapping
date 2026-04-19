"""
Download GEE intermediate assets to local parquet cache.

Run this interactively (not as a Slurm batch job) since Forth compute
nodes don't have internet access. Login node and interactive sessions do.

Usage:
    python src/data/sentinel1/download_intermediate_assets.py
"""

import ee
import pandas as pd
from pathlib import Path
from src.constants import DATA_PATH, ASSETS_PATH, AOIS
from src.utils.gee import init_gee

init_gee()

CACHE_DIR = DATA_PATH / "intermediate_features_cache"
ORBITS = [87, 94, 160]
EXTRACT_WINDOW = "1x1"


def download_intermediate_asset(aoi: str, orbit: int, force: bool = False) -> Path:
    """
    Download a GEE intermediate asset to local parquet cache.

    Args:
        aoi: AOI name e.g. 'GAZ1'
        orbit: Orbit number e.g. 87
        force: Re-download even if cache exists

    Returns:
        Path to local parquet file
    """
    CACHE_DIR.mkdir(exist_ok=True, parents=True)
    fp = CACHE_DIR / f"{aoi}_orbit{orbit}.parquet"

    if fp.exists() and not force:
        print(f"  {aoi}_orbit{orbit}: already cached ✓")
        return fp

    print(f"  {aoi}_orbit{orbit}: downloading from GEE...")
    asset_id = ASSETS_PATH + f"intermediate_features/ts_s1_{EXTRACT_WINDOW}/{aoi}_orbit{orbit}"
    fc = ee.FeatureCollection(asset_id)
    total = fc.size().getInfo()
    print(f"    {total:,} rows to download...")

    # Download in batches to avoid memory issues
    batch_size = 50000
    records = []
    for start in range(0, total, batch_size):
        batch = fc.toList(batch_size, start).getInfo()
        for feat in batch:
            records.append(feat["properties"])
        print(f"    Downloaded {min(start + batch_size, total):,} / {total:,} rows")

    df = pd.DataFrame(records)
    df.to_parquet(fp)
    print(f"  {aoi}_orbit{orbit}: saved {len(df):,} rows to {fp.name} ✓")
    return fp


if __name__ == "__main__":
    print("Downloading all intermediate assets to local cache...")
    print(f"Cache directory: {CACHE_DIR}")
    print()

    for aoi in AOIS:
        print(f"\n{aoi}:")
        for orbit in ORBITS:
            download_intermediate_asset(aoi, orbit)

    print("\nAll assets downloaded. Ready to run extract_features_local.py as batch job.")
