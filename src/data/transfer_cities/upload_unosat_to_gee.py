"""
Upload transfer city UNOSAT labels to GEE as FeatureCollections.

Mirrors upload_gaza_unosat_to_gee() in src/data/unosat.py, adapted
for the three transfer cities. Uses chunked upload for large datasets
(>5000 features) to stay within GEE's 10MB payload limit.

GEE asset paths:
    projects/gaza-damage-mapping/assets/transfer-cities/UNOSAT_labels/{city_id}
    projects/gaza-damage-mapping/assets/transfer-cities/AOIs/{city_id}

Usage:
    Run interactively (not as Slurm batch job - requires internet access):
    cd /scratch/s1214882/gaza-damage-mapping
    python3 src/data/transfer_cities/upload_unosat_to_gee.py
"""

import time
import ee
import geemap
import geopandas as gpd
import sys
sys.path.insert(0, '/scratch/s1214882/gaza-damage-mapping')

from src.utils.gee import init_gee, asset_exists, create_folder
from src.data.transfer_cities.constants_transfer import TRANSFER_CITIES, TRANSFER_GEE_FOLDER

init_gee()

CHUNK_SIZE = 5000

# Folders to create in order (parent before child)
GEE_FOLDERS = [
    "projects/gaza-damage-mapping/assets/transfer-cities",
    "projects/gaza-damage-mapping/assets/transfer-cities/UNOSAT_labels",
    "projects/gaza-damage-mapping/assets/transfer-cities/AOIs",
]


def wait_for_task(task, asset_id: str) -> None:
    """Wait for a GEE export task to complete."""
    while True:
        status = task.status()
        state = status["state"]
        if state == "COMPLETED":
            print(f"  Done: {asset_id.split('/')[-1]}")
            return
        elif state in ["FAILED", "CANCELLED"]:
            raise RuntimeError(
                f"Export failed for {asset_id}: "
                f"{status.get('error_message', 'unknown error')}"
            )
        time.sleep(10)


def ensure_folders() -> None:
    """Create GEE folder structure if it doesn't exist."""
    for folder in GEE_FOLDERS:
        if not asset_exists(folder):
            create_folder(folder)
        else:
            print(f"  Folder exists: {folder.split('assets/')[-1]}")


def upload_direct(gdf: gpd.GeoDataFrame, asset_id: str, description: str) -> None:
    """Upload a small GeoDataFrame directly (< CHUNK_SIZE features)."""
    fc = geemap.geopandas_to_ee(gdf)
    task = ee.batch.Export.table.toAsset(
        collection=fc,
        description=description[:100],
        assetId=asset_id,
    )
    task.start()
    wait_for_task(task, asset_id)


def upload_chunked(gdf: gpd.GeoDataFrame, asset_id: str, description: str) -> None:
    """
    Upload a large GeoDataFrame in chunks, then merge into one asset.
    Mirrors upload_chunked() in src/data/unosat.py exactly.
    """
    chunks = [gdf.iloc[i:i+CHUNK_SIZE] for i in range(0, len(gdf), CHUNK_SIZE)]
    print(f"  Uploading {len(gdf):,} features in {len(chunks)} chunks...")

    # Upload chunks
    chunk_ids = []
    for i, chunk in enumerate(chunks):
        chunk_id = asset_id + f"_tmp_chunk{i}"
        if not asset_exists(chunk_id):
            fc = geemap.geopandas_to_ee(chunk.copy())
            task = ee.batch.Export.table.toAsset(
                collection=fc,
                description=f"{description[:90]}_chunk{i}",
                assetId=chunk_id,
            )
            task.start()
            print(f"  Chunk {i+1}/{len(chunks)} started...")
        else:
            print(f"  Chunk {i+1}/{len(chunks)} already exists, skipping...")
        chunk_ids.append(chunk_id)

    # Wait for all chunks
    print(f"  Waiting for {len(chunks)} chunks to complete...")
    for chunk_id in chunk_ids:
        while not asset_exists(chunk_id):
            time.sleep(10)
    print(f"  All chunks uploaded, merging...")

    # Merge chunks into single asset
    merged = ee.FeatureCollection(
        [ee.FeatureCollection(cid) for cid in chunk_ids]
    ).flatten()
    merge_task = ee.batch.Export.table.toAsset(
        collection=merged,
        description=description[:100],
        assetId=asset_id,
    )
    merge_task.start()
    wait_for_task(merge_task, asset_id)

    # Clean up temporary chunks
    for chunk_id in chunk_ids:
        ee.data.deleteAsset(chunk_id)
        print(f"  Deleted temp chunk: {chunk_id.split('/')[-1]}")


def upload_gdf(gdf: gpd.GeoDataFrame, asset_id: str, description: str) -> None:
    """Upload a GeoDataFrame to GEE, using chunking if needed."""
    if asset_exists(asset_id):
        print(f"  Already exists - skipping: {asset_id.split('/')[-1]}")
        return

    print(f"  Uploading {len(gdf):,} features -> {asset_id.split('/')[-1]}...")
    if len(gdf) > CHUNK_SIZE:
        upload_chunked(gdf, asset_id, description)
    else:
        upload_direct(gdf, asset_id, description)


def upload_transfer_cities() -> None:
    """Upload UNOSAT labels and AOI boundaries for all three transfer cities."""

    print("Creating GEE folder structure...")
    ensure_folders()

    for city_id, cfg in TRANSFER_CITIES.items():
        print(f"\n{'='*60}")
        print(f"{city_id} - {cfg['city_name']} ({cfg['country']})")
        print(f"{'='*60}")

        gdf_labels = gpd.read_file(cfg["unosat_labels"])
        gdf_aoi    = gpd.read_file(cfg["unosat_aoi"])

        print(f"  Labels: {len(gdf_labels):,} points (classes 1+2)")

        # Upload labels
        upload_gdf(
            gdf_labels.reset_index(),
            TRANSFER_GEE_FOLDER + f"UNOSAT_labels/{city_id}",
            f"UNOSAT_labels_{city_id}",
        )

        # Upload AOI boundary
        upload_gdf(
            gdf_aoi,
            TRANSFER_GEE_FOLDER + f"AOIs/{city_id}",
            f"AOI_{city_id}",
        )

    print(f"\n{'='*60}")
    print("All uploads complete.")
    print(f"{'='*60}")

    # Verify
    print("\nVerifying assets...")
    for city_id in TRANSFER_CITIES:
        for asset_type in ["UNOSAT_labels", "AOIs"]:
            asset_id = TRANSFER_GEE_FOLDER + f"{asset_type}/{city_id}"
            status = "EXISTS" if asset_exists(asset_id) else "MISSING"
            print(f"  {status}: {asset_id.split('assets/')[-1]}")


if __name__ == "__main__":
    upload_transfer_cities()
