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
    Download a GEE intermediate asset to local parquet cache via Drive export.
    Uses Export.table.toDrive() — proven reliable for large collections.

    Args:
        aoi: AOI name e.g. 'GAZ1'
        orbit: Orbit number e.g. 87
        force: Re-download even if cache exists

    Returns:
        Path to local parquet file
    """
    import time
    from src.utils.gdrive import drive_to_local

    CACHE_DIR.mkdir(exist_ok=True, parents=True)
    fp = CACHE_DIR / f"{aoi}_orbit{orbit}.parquet"

    if fp.exists() and not force:
        print(f"  {aoi}_orbit{orbit}: already cached ✓")
        return fp

    print(f"  {aoi}_orbit{orbit}: exporting to Drive...")
    asset_id = ASSETS_PATH + f"intermediate_features/ts_s1_{EXTRACT_WINDOW}/{aoi}_orbit{orbit}"
    fc = ee.FeatureCollection(asset_id)
    description = f"{aoi}_orbit{orbit}_features"
    drive_folder = "gaza_intermediate_features"

    task = ee.batch.Export.table.toDrive(
        collection=fc,
        description=description,
        folder=drive_folder,
        fileFormat="CSV",
    )
    task.start()

    # Wait for export to complete
    print(f"  {aoi}_orbit{orbit}: waiting for export...")
    while True:
        status = task.status()
        state = status["state"]
        if state == "COMPLETED":
            break
        elif state in ["FAILED", "CANCELLED"]:
            raise RuntimeError(f"Export failed: {status}")
        time.sleep(30)
    print(f"  {aoi}_orbit{orbit}: export complete, downloading from Drive...")

    # Download CSV from Drive
    tmp_dir = CACHE_DIR / "tmp"
    tmp_dir.mkdir(exist_ok=True)
    drive_to_local(drive_folder, tmp_dir, delete_in_drive=False, verbose=0)

    # Convert CSV to parquet
    csv_fp = tmp_dir / f"{description}.csv"
    df = pd.read_csv(csv_fp)
    df.to_parquet(fp)
    print(f"  {aoi}_orbit{orbit}: {len(df):,} rows saved ✓")
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
