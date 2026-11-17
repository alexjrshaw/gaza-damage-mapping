#!/usr/bin/env python3
"""
Move finished tiles from Google Drive to Forth, check each copy, and only then delete it from Drive.

Earth Engine writes one GeoTIFF per task into the Drive folder syria_run_<tag>. Drive is small (15 GB shared
with Gmail and Photos), so the tiles are pulled to data/syria/ee_run_<tag>/tiles/ as they finish, the same
way the Gaza and transfer-city features were. A Drive copy is deleted (permanently, it does not go to the
bin) only when the local copy
  * has exactly the size Drive reports,
  * opens as a GeoTIFF with one Uint8 band per window in the plan.
A tile that fails either test is kept in Drive and reported. Tiles already local are checked the same way
and their Drive copy is removed once they pass.

  python3 src/syria/fetch_tiles.py --tag aleppo               # one pass: fetch what is finished, delete from Drive
  python3 src/syria/fetch_tiles.py --tag aleppo --loop        # repeat every 10 min until the run is finished and every tile is local
  python3 src/syria/fetch_tiles.py --tag aleppo --keep        # fetch and check, leave the Drive copies
  python3 src/syria/fetch_tiles.py --tag aleppo --limit 50    # at most 50 tiles in this pass

Run it in its own tmux window next to `ee_production.py watch`. It reads the ledger, so run it where the
watch is writing it (the same data/syria/ee_run_<tag>/ folder).
"""
import argparse
import sys
import time
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))


def verify_local(path, n_bands, expected_size=None):
    """Returns '' when the file is sound, else a reason."""
    import rasterio

    if not path.exists():
        return "missing"
    if expected_size is not None and path.stat().st_size != expected_size:
        return f"size {path.stat().st_size} differs from Drive's {expected_size}"
    try:
        with rasterio.open(path) as src:
            if src.count != n_bands:
                return f"{src.count} bands, expected {n_bands}"
            if src.dtypes[0] != "uint8":
                return f"dtype {src.dtypes[0]}, expected uint8"
            src.read(1, window=((0, 1), (0, 1)))
    except Exception as e:  # unreadable or truncated
        return f"cannot be read: {e}"
    return ""


def drive_listing(drive, folder_id):
    """{title: (id, size)} for the live files of the folder."""
    files = drive.ListFile({"q": f"'{folder_id}' in parents and trashed=false", "maxResults": 1000}).GetList()
    return {f["title"]: (f["id"], int(f.get("fileSize") or 0)) for f in files}


def one_pass(drive, folder_id, wanted, tiles_dir, n_bands, keep=False, limit=None):
    """wanted: z13 ids whose task SUCCEEDED. Returns dict of counts and a list of problems."""
    tiles_dir.mkdir(parents=True, exist_ok=True)
    listing = drive_listing(drive, folder_id)
    res = dict(downloaded=0, deleted=0, already_local=0, not_in_drive=0)
    problems = []
    n = 0
    for z in wanted:
        name = f"qk_{z}.tif"
        local = tiles_dir / name
        if name not in listing:
            if local.exists():
                res["already_local"] += 1
            else:
                res["not_in_drive"] += 1
            continue
        if limit is not None and n >= limit:
            break
        fid, size = listing[name]
        if not local.exists() or verify_local(local, n_bands, size):
            part = tiles_dir / (name + ".part")
            drive.CreateFile({"id": fid}).GetContentFile(str(part))
            part.replace(local)
            res["downloaded"] += 1
        n += 1
        why = verify_local(local, n_bands, size)
        if why:
            problems.append((z, why))
            continue
        if not keep:
            drive.CreateFile({"id": fid}).Delete()
            res["deleted"] += 1
    return res, problems


def run_state(out_dir):
    led = pd.read_csv(out_dir / "ledger.csv", dtype={"z13": str})
    wanted = led[led.state == "SUCCEEDED"].z13.tolist()
    open_ = int(led.state.isin(["PENDING", "RUNNING"]).sum())
    return wanted, open_, len(led)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.strip().split("\n\n")[0])
    p.add_argument("--tag", default="", help="a tagged run (default: the main run)")
    p.add_argument("--keep", action="store_true", help="do not delete the Drive copies")
    p.add_argument("--loop", action="store_true", help="repeat until every planned task has been submitted and finished, and every tile is local")
    p.add_argument("--interval", type=int, default=600, help="seconds between passes with --loop")
    p.add_argument("--limit", type=int, default=None, help="at most this many tiles per pass")
    args = p.parse_args(argv)

    from src.syria import check_tiles
    from src.utils.gdrive import drive, get_folder_id

    tag = args.tag.strip().lower()
    out_dir = Path("data/syria/ee_run" + (f"_{tag}" if tag else ""))
    check_tiles.OUT_DIR = out_dir
    plan = check_tiles.read_plan()
    n_bands = len(check_tiles.window_names(plan))
    tiles_dir = out_dir / "tiles"
    folder = "syria_run" + (f"_{tag}" if tag else "")
    folder_id = get_folder_id(folder)
    while True:
        wanted, open_, total = run_state(out_dir)
        res, problems = one_pass(drive, folder_id, wanted, tiles_dir, n_bands, args.keep, args.limit)
        local = sum((tiles_dir / f"qk_{z}.tif").exists() for z in wanted)
        print(f"{time.strftime('%H:%M:%S')}  succeeded {len(wanted):,} of {len(plan):,} planned (submitted {total:,}, pending/running {open_:,});  local {local:,};  "
              f"downloaded {res['downloaded']}, deleted from Drive {res['deleted']}")
        for z, why in problems:
            print(f"  PROBLEM {z}: {why} (left in Drive)")
        if not args.loop:
            return 1 if problems else 0
        if open_ == 0 and total >= len(plan) and local == len(wanted) and not problems:
            print(f"Run finished: all {len(wanted):,} succeeded tiles are local" + ("" if args.keep else " and removed from Drive") + ".")
            return 0
        time.sleep(args.interval)


if __name__ == "__main__":
    sys.exit(main())
