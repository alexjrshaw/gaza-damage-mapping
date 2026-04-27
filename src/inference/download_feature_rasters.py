"""
Download feature raster tiles from Google Drive to Forth scratch,
deleting from Drive after each download to manage Drive space.

Run this script in a separate screen session while GEE export tasks
are running. It polls Drive every POLL_INTERVAL seconds and downloads
completed tiles, freeing Drive space as it goes.

This allows full inference to proceed even with limited Drive storage —
tiles are downloaded and deleted as soon as they complete, so Drive
usage stays low regardless of total export size.

Usage:
    screen -S download
    python3 src/inference/download_feature_rasters.py
    Ctrl+A D  # detach

Output:
    /scratch/s1214882/gaza-damage-mapping/data/feature_rasters/
        {window_str}/orbit{orbit}/qk_{qk_id}.tif
"""

import time
from pathlib import Path

from tqdm.auto import tqdm

from src.constants import DATA_PATH
from src.utils.gdrive import drive_to_local, get_files_in_folder

# ==================== CONSTANTS ====================

DRIVE_BASE = "gaza_feature_rasters"
LOCAL_BASE = DATA_PATH / "feature_rasters"
POLL_INTERVAL = 120   # seconds between Drive checks
DELETE_AFTER_DOWNLOAD = True  # free Drive space after download


# ==================== DOWNLOAD ====================

def get_drive_windows() -> list[str]:
    """List all window folders in Drive base folder."""
    try:
        items = get_files_in_folder(DRIVE_BASE, return_names=True)
        return [i for i in items if i.startswith("w")]
    except Exception:
        return []


def get_drive_orbits(window_str: str) -> list[str]:
    """List all orbit folders within a window folder."""
    try:
        items = get_files_in_folder(f"{DRIVE_BASE}/{window_str}", return_names=True)
        return [i for i in items if i.startswith("orbit")]
    except Exception:
        return []


def get_drive_tiles(window_str: str, orbit_str: str) -> list[str]:
    """List all tile filenames in a Drive orbit folder."""
    try:
        items = get_files_in_folder(
            f"{DRIVE_BASE}/{window_str}/{orbit_str}",
            return_names=True,
        )
        return [i for i in items if i.startswith("qk_") and i.endswith(".tif")]
    except Exception:
        return []


def already_downloaded(window_str: str, orbit_str: str, filename: str) -> bool:
    """Check if tile already exists in local scratch."""
    fp = LOCAL_BASE / window_str / orbit_str / filename
    return fp.exists() and fp.stat().st_size > 0


def download_orbit_folder(
    window_str: str,
    orbit_str: str,
) -> int:
    """
    Download all completed tiles from one Drive orbit folder to scratch.
    Deletes from Drive after download.
    Returns number of files downloaded.
    """
    local_dir = LOCAL_BASE / window_str / orbit_str
    local_dir.mkdir(exist_ok=True, parents=True)

    drive_folder = f"{DRIVE_BASE}/{window_str}/{orbit_str}"

    # Get tiles in Drive not yet downloaded locally
    drive_tiles = get_drive_tiles(window_str, orbit_str)
    new_tiles = [t for t in drive_tiles
                 if not already_downloaded(window_str, orbit_str, t)]

    if not new_tiles:
        return 0

    try:
        drive_to_local(
            folder_name=drive_folder,
            local_folder=local_dir,
            delete_in_drive=DELETE_AFTER_DOWNLOAD,
            verbose=0,
        )
        return len(new_tiles)
    except Exception as e:
        print(f"    ERROR downloading {drive_folder}: {e}")
        return 0


# ==================== MAIN LOOP ====================

def run_download_loop() -> None:
    """
    Continuously poll Drive and download completed tiles.
    Runs until interrupted with Ctrl+C.
    """
    LOCAL_BASE.mkdir(exist_ok=True, parents=True)

    print(f"Monitoring Drive folder: {DRIVE_BASE}/")
    print(f"Downloading to: {LOCAL_BASE}")
    print(f"Poll interval: {POLL_INTERVAL}s")
    print(f"Delete from Drive after download: {DELETE_AFTER_DOWNLOAD}")
    print("\nPress Ctrl+C to stop.\n")

    total_downloaded = 0
    total_skipped = 0

    while True:
        windows = get_drive_windows()
        if not windows:
            print(f"No windows found in Drive yet. Waiting {POLL_INTERVAL}s...")
            time.sleep(POLL_INTERVAL)
            continue

        n_new = 0
        for window_str in sorted(windows):
            orbits = get_drive_orbits(window_str)
            for orbit_str in sorted(orbits):
                n = download_orbit_folder(window_str, orbit_str)
                n_new += n

        if n_new > 0:
            total_downloaded += n_new
            print(f"Downloaded {n_new} new tiles "
                  f"(total: {total_downloaded})")
            local_size_gb = sum(
                f.stat().st_size for f in LOCAL_BASE.rglob("*.tif")
            ) / 1e9
            print(f"Local storage used: {local_size_gb:.2f} GB")
        else:
            print(f"No new tiles. Waiting {POLL_INTERVAL}s... "
                  f"(total downloaded: {total_downloaded})")

        time.sleep(POLL_INTERVAL)


if __name__ == "__main__":
    try:
        run_download_loop()
    except KeyboardInterrupt:
        print("\nDownload loop stopped.")