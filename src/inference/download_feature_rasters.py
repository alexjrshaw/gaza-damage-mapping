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

from src.constants import DATA_PATH
from src.utils.gdrive import drive_to_local, get_files_in_folder, get_folder_id, drive

# ==================== CONSTANTS ====================

DRIVE_BASE = "gaza_feature_rasters"
LOCAL_BASE = DATA_PATH / "feature_rasters"
POLL_INTERVAL = 120   # seconds between Drive checks
DELETE_AFTER_DOWNLOAD = True  # free Drive space after download


# ==================== DOWNLOAD ====================

def get_drive_windows() -> list[str]:
    """Returns list of window name strings from flat Drive folder names."""
    try:
        all_folders = drive.ListFile({
            "q": "mimeType='application/vnd.google-apps.folder' and trashed=false",
            "maxResults": 1000,
        }).GetList()
        windows = set()
        for f in all_folders:
            title = f["title"]
            if title.startswith(DRIVE_BASE + "/"):
                parts = title.split("/")
                if len(parts) == 3:  # DRIVE_BASE/window/orbit
                    windows.add(parts[1])
        return sorted(windows)
    except Exception as e:
        print(f"Error getting windows: {e}")
        return []


def get_orbit_folder_id(window_str: str, orbit_str: str) -> str | None:
    """Get folder ID for a flat-named orbit folder."""
    folder_name = f"{DRIVE_BASE}/{window_str}/{orbit_str}"
    try:
        return get_folder_id(folder_name)
    except Exception:
        return None


def get_drive_tiles_by_id(orbit_id: str) -> list[tuple[str, str]]:
    """Returns list of (filename, file_id) tuples within an orbit folder."""
    try:
        items = drive.ListFile({
            "q": f"'{orbit_id}' in parents and trashed=false"
        }).GetList()
        return [(i["title"], i["id"]) for i in items
                if i["title"].startswith("qk_") and i["title"].endswith(".tif")]
    except Exception:
        return []


def already_downloaded(window_str: str, orbit_str: str, filename: str) -> bool:
    """Check if tile already exists in local scratch."""
    fp = LOCAL_BASE / window_str / orbit_str / filename
    return fp.exists() and fp.stat().st_size > 0


def download_orbit_folder(
    window_str: str,
    orbit_str: str,
    orbit_id: str,
) -> int:
    local_dir = LOCAL_BASE / window_str / orbit_str
    local_dir.mkdir(exist_ok=True, parents=True)

    tiles = get_drive_tiles_by_id(orbit_id)
    new_tiles = [(name, fid) for name, fid in tiles
                 if not already_downloaded(window_str, orbit_str, name)]

    if not new_tiles:
        return 0

    for filename, file_id in new_tiles:
        try:
            f = drive.CreateFile({"id": file_id})
            f.GetContentFile(str(local_dir / filename))
            if DELETE_AFTER_DOWNLOAD:
                f.Delete()
        except Exception as e:
            print(f"    ERROR: {filename}: {e}")

    return len(new_tiles)


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
        
        n_new = 0
        for window_str in sorted(get_drive_windows()):
            for orbit_str in ["orbit87", "orbit94", "orbit160"]:
                orbit_id = get_orbit_folder_id(window_str, orbit_str)
                if orbit_id is None:
                    continue
                n = download_orbit_folder(window_str, orbit_str, orbit_id)
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