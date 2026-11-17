#!/usr/bin/env python3
"""
Sanity-check finished production tiles: download a few from Drive and look at them.

For every tile it checks the file against the plan (data/syria/ee_run/plan.csv):
  * one Uint8 band per window, in the plan's order
  * the raster covers the task's bounding box (to within 2 pixels)
  * a window that has orbits in the plan holds data; a window without orbits is all 0
  * how much of each band is nodata (0) and how many pixels are above the damage threshold (>= 170);
    the share above the threshold is only printed, because most pixels are not buildings
and prints one line per tile and window plus a list of anything that looks wrong.

  python3 src/syria/check_tiles.py                 # the first 5 finished tasks in the ledger
  python3 src/syria/check_tiles.py --n 20          # more of them
  python3 src/syria/check_tiles.py --tiles 1221121003120 ...    # named z13 tiles
  python3 src/syria/check_tiles.py --no-download   # only look at what is already in data/syria/ee_run/tiles/

Files stay in Drive unless you add --delete-drive. Downloads go to data/syria/ee_run/tiles/.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

OUT_DIR = Path("data/syria/ee_run")
DRIVE_FOLDER = "syria_run"
THRESHOLD_U8 = 170
PIXEL_DEG = 10 / 111320.0  # about 8.98e-5 degrees, the 10 m export scale in EPSG:4326
MIN_DATA_SHARE = 0.5  # a window with orbits should have data on at least this share of pixels


def read_plan():
    plan = pd.read_csv(OUT_DIR / "plan.csv", dtype=str).fillna("")
    for c in ("west", "south", "east", "north"):
        plan[c] = plan[c].astype(float)
    return plan.set_index("z13")


def window_names(plan):
    """Band names in order: from windows.json, else from the plan's orbits_ columns (an older plan)."""
    import json

    f = OUT_DIR / "windows.json"
    if f.exists():
        return [w["name"] for w in json.loads(f.read_text())["windows"]]
    return sorted(c[len("orbits_"):] for c in plan.columns if c.startswith("orbits_"))


def check_tile(path, row, names_expected):
    """Returns (list of per-window dicts, list of problem strings)."""
    import rasterio

    problems = []
    with rasterio.open(path) as src:
        arr = src.read()
        bounds = src.bounds
        dtype = src.dtypes[0]
        names = [d for d in (src.descriptions or [])]
    if dtype != "uint8":
        problems.append(f"dtype is {dtype}, expected uint8")
    if arr.shape[0] != len(names_expected):
        problems.append(f"{arr.shape[0]} bands, expected {len(names_expected)} (one per window)")
    tol = 2 * PIXEL_DEG
    want = (row.west, row.south, row.east, row.north)
    got = (bounds.left, bounds.bottom, bounds.right, bounds.top)
    if any(abs(a - b) > tol for a, b in zip(want, got)):
        problems.append(f"extent differs from the plan by more than 2 pixels: plan {tuple(round(v, 5) for v in want)}, file {tuple(round(v, 5) for v in got)}")
    stats = []
    for i, wname in enumerate(names_expected[: arr.shape[0]]):
        band = arr[i]
        has_orbits = str(getattr(row, f"orbits_{wname}")) != ""
        data = band > 0
        share = float(data.mean())
        hi = float((band[data] >= THRESHOLD_U8).mean()) if data.any() else 0.0
        mean = float(band[data].mean()) if data.any() else 0.0
        stats.append(dict(window=wname, orbits=getattr(row, f"orbits_{wname}"), data_share=share, mean=mean, above=hi))
        if names and i < len(names) and names[i] and names[i] != wname:
            problems.append(f"band {i + 1} is named {names[i]}, expected {wname}")
        if has_orbits and share == 0:
            problems.append(f"{wname}: orbits {getattr(row, f'orbits_{wname}')} are in the plan but the band is all 0")
        elif has_orbits and share < MIN_DATA_SHARE:
            problems.append(f"{wname}: only {share:.0%} of pixels hold data")
        if not has_orbits and share > 0:
            problems.append(f"{wname}: no orbit in the plan but the band holds data")
    return arr.shape, stats, problems


def download_tiles(z13s, local_dir, delete):
    from src.utils.gdrive import drive, get_folder_id

    folder_id = get_folder_id(DRIVE_FOLDER)
    local_dir.mkdir(parents=True, exist_ok=True)
    missing = []
    for z in z13s:
        name = f"qk_{z}.tif"
        files = drive.ListFile({"q": f"'{folder_id}' in parents and title='{name}' and trashed=false"}).GetList()
        if not files:
            missing.append(z)
            continue
        f = drive.CreateFile({"id": files[0]["id"]})
        f.GetContentFile(str(local_dir / name))
        if delete:
            f.Delete()
    return missing


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.strip().split("\n\n")[0])
    p.add_argument("--n", type=int, default=5)
    p.add_argument("--tiles", nargs="+")
    p.add_argument("--no-download", action="store_true")
    p.add_argument("--delete-drive", action="store_true")
    p.add_argument("--local-dir")
    p.add_argument("--tag", default="", help="a tagged run (default: the main run)")
    args = p.parse_args(argv)
    global OUT_DIR, DRIVE_FOLDER
    tag = args.tag.strip().lower()
    OUT_DIR = Path("data/syria/ee_run" + (f"_{tag}" if tag else ""))
    DRIVE_FOLDER = "syria_run" + (f"_{tag}" if tag else "")

    plan = read_plan()
    names = window_names(plan)
    local_dir = Path(args.local_dir) if args.local_dir else OUT_DIR / "tiles"
    if args.tiles:
        z13s = args.tiles
    elif args.no_download:
        z13s = sorted(f.stem[3:] for f in local_dir.glob("qk_*.tif"))[: args.n]
    else:
        led = pd.read_csv(OUT_DIR / "ledger.csv", dtype={"z13": str})
        z13s = led[led.state == "SUCCEEDED"].z13.tolist()[: args.n]
    if not z13s:
        sys.exit("No tiles to check yet (no SUCCEEDED tasks in the ledger, or no files in the local folder).")
    unknown = [z for z in z13s if z not in plan.index]
    if unknown:
        sys.exit(f"Not in the plan: {unknown}")

    if not args.no_download:
        missing = download_tiles(z13s, local_dir, args.delete_drive)
        if missing:
            print(f"Not found in Drive folder {DRIVE_FOLDER}: {missing}")
        z13s = [z for z in z13s if z not in missing]

    all_problems = {}
    for z in z13s:
        path = local_dir / f"qk_{z}.tif"
        if not path.exists():
            all_problems[z] = ["file missing"]
            continue
        row = next(plan.loc[[z]].reset_index().itertuples())
        shape, stats, problems = check_tile(path, row, names)
        n_z14 = int(row.n_z14)
        print(f"\n{z}  {shape[2]} x {shape[1]} pixels, {shape[0]} bands, {n_z14} z14 tile(s), baseline orbits {row.baseline_orbits}")
        for s in stats:
            print(f"  {s['window']}  orbits {s['orbits'] or '-':<12} data {s['data_share']:6.1%}   mean {s['mean']:6.1f}   >= {THRESHOLD_U8}: {s['above']:6.2%}")
        if problems:
            all_problems[z] = problems
    print()
    if all_problems:
        print(f"PROBLEMS in {len(all_problems)} of {len(z13s)} tiles:")
        for z, probs in all_problems.items():
            for msg in probs:
                print(f"  {z}: {msg}")
        return 1
    print(f"All {len(z13s)} tiles passed the checks.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
