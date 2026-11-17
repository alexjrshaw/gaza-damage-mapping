#!/usr/bin/env python3
"""
Does the 50 m2 building filter change which tiles the plan runs? (No Earth Engine cost, buildings file only.)

The plan runs every z14 tile with at least --min-buildings (10) Overture buildings, counting ALL buildings.
classify_buildings.py now leaves out buildings under --min-area-m2 (50). This script counts, per z14 tile, the
buildings in total and the buildings of at least that area (bounding-box area, an upper bound on footprint area), and reports

  * how many of the tiles that run would hold fewer than 10 buildings of at least 50 m2, and what share of the
    50 m2+ buildings those tiles hold (what is lost if they were skipped);
  * the same for the Aleppo plan (data/syria/ee_run_aleppo/plan.csv) when it is there;
  * how many tiles and tasks a plan would need if it counted only the 50 m2+ buildings (a possible saving for the country run).

  python3 src/syria/check_area_filter.py
  python3 src/syria/check_area_filter.py --plan data/syria/ee_run_aleppo/plan.csv --min-area-m2 50
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

ZOOM = 14


def box_area_m2(xmin, ymin, xmax, ymax):
    lat = (np.asarray(ymin) + np.asarray(ymax)) / 2
    return (np.asarray(xmax) - np.asarray(xmin)) * 111320.0 * np.cos(np.radians(lat)) * (np.asarray(ymax) - np.asarray(ymin)) * 110574.0


def tile_counts(ft, buildings, window, min_area):
    """DataFrame quadkey, n_all, n_big for every z14 tile that holds a building."""
    parts = list(ft.iter_bounds(Path(buildings), window=window))
    xmin, ymin, xmax, ymax = (np.concatenate([p[i] for p in parts]) for i in range(4))
    cx, cy = (xmin + xmax) / 2, (ymin + ymax) / 2
    big = box_area_m2(xmin, ymin, xmax, ymax) >= min_area
    x, y = ft.lonlat_to_tile(cx, cy, ZOOM)
    x, y = np.asarray(x, np.int64), np.asarray(y, np.int64)
    keys, inv = np.unique(x * (1 << ZOOM) + y, return_inverse=True)
    n_all = np.bincount(inv)
    n_big = np.bincount(inv, weights=big.astype(float)).astype(int)
    qk = np.array(list(map(str, ft.quadkeys(keys // (1 << ZOOM), keys % (1 << ZOOM), ZOOM))))
    return pd.DataFrame({"quadkey": qk, "n_all": n_all, "n_big": n_big}), int(len(cx)), int(big.sum())


def summarise(label, t, min_b):
    """t: tiles that run now (n_all >= min_b) with n_all and n_big."""
    run = t[t.n_all >= min_b]
    kept = run[run.n_big >= min_b]
    dropped = run[run.n_big < min_b]
    big_all = int(run.n_big.sum())
    tasks_now, tasks_new = run.quadkey.str[:-1].nunique(), kept.quadkey.str[:-1].nunique()
    print(f"\n{label}")
    print(f"  tiles that run now (>= {min_b} buildings of any size): {len(run):,}  ({tasks_now:,} z13 tasks)")
    print(f"  of those, tiles with fewer than {min_b} buildings of >= the minimum area: {len(dropped):,} ({len(dropped) / max(len(run), 1):.1%})")
    print(f"  buildings of >= the minimum area in those tiles: {int(dropped.n_big.sum()):,} of {big_all:,} ({dropped.n_big.sum() / max(big_all, 1):.2%})")
    print(f"  a plan counting only those buildings would need {len(kept):,} tiles ({len(kept) / max(len(run), 1):.1%}) and {tasks_new:,} tasks ({tasks_new / max(tasks_now, 1):.1%})")
    return dict(run=len(run), dropped=len(dropped), tasks_now=tasks_now, tasks_new=tasks_new)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.strip().split("\n\n")[0])
    p.add_argument("--buildings", default="data/syria/footprints/overture_buildings_bbox.parquet")
    p.add_argument("--tiles-csv", default="data/syria/footprints/tiles_z14.csv")
    p.add_argument("--plan", default="data/syria/ee_run_aleppo/plan.csv", help="a run's plan.csv for the regional figures")
    p.add_argument("--min-area-m2", type=float, default=50.0)
    p.add_argument("--min-buildings", type=int, default=10)
    p.add_argument("--window", nargs=4, type=float, default=[35.5, 32.0, 42.5, 37.5], metavar=("W", "S", "E", "N"))
    args = p.parse_args(argv)

    from src.syria import footprint_tiles as ft

    counts, n, nbig = tile_counts(ft, args.buildings, tuple(args.window), args.min_area_m2)
    print(f"{n:,} buildings in the window, {nbig:,} ({nbig / n:.1%}) with box area >= {args.min_area_m2:g} m2, in {len(counts):,} z14 tiles")
    z = pd.read_csv(args.tiles_csv, dtype={"quadkey": str})
    z = z[z.n_buildings >= args.min_buildings]
    t = z[["quadkey", "n_buildings"]].merge(counts, on="quadkey", how="left").fillna({"n_all": 0, "n_big": 0})
    miss = int((t.n_all == 0).sum())
    print(f"The plan's {len(z):,} tiles: {len(t) - miss:,} found in the buildings file" + (f", {miss:,} not found (counted as 0 big buildings)" if miss else ""))
    off = (t.n_all - t.n_buildings).abs()
    print(f"  tile counts agree with tiles_z14.csv in {(off == 0).mean():.1%} of tiles (median difference {off.median():.0f})")
    t["n_all"] = t.n_buildings  # use the plan's own count for 'runs now'
    summarise("WHOLE COUNTRY", t, args.min_buildings)
    plan = Path(args.plan)
    if plan.exists():
        pl = pd.read_csv(plan, dtype=str)
        reg = t[t.quadkey.str[:-1].isin(set(pl.z13))]
        print(f"\nThe plan {plan} has {len(pl):,} tasks and {pd.to_numeric(pl.n_z14).sum():,.0f} z14 tiles; {len(reg):,} tiles found in its z13 squares")
        summarise(f"PLAN {plan.parent.name}", reg, args.min_buildings)
    return 0


if __name__ == "__main__":
    sys.exit(main())
