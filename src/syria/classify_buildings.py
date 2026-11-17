#!/usr/bin/env python3
"""
Building-level damage classes from the finished tiles of a run (the Syria version of Equation 3).

Gaza (classify_building_damage.py) calls a building damaged when its highest pixel probability after the
war start reaches t and before it stays below t. The Syria run has no pre-war windows (the baseline is
Oct 2014 - Oct 2015, the windows are 2016 ... 2025), so the same threshold t is applied to every window and
the building is classed by HOW MANY windows reach it:

    confirmed    : the building reaches t in 2 or more windows
    possible     : the building reaches t in exactly 1 window
    not_damaged  : no window with data reaches t
    not_assessed : the building could not be assessed (see "reason")

For every building and window the value is the maximum probability over the pixels that the building's
bounding box covers (at least the pixel under its centre). Overture footprints are held here as boxes
only, so this is a slightly generous stand-in for "inside the footprint"; use --footprint 3x3 for the
3 x 3 pixels around the centre, the stand-in used at the UNOSAT points. A pixel with value 0 is no data.

Reasons for not_assessed:
    tile_below_threshold : the building lies in a z14 tile with fewer than --min-buildings buildings, which
                           the run does not classify by design (the tiles are inside the task rectangles)
    no_valid_window      : every window is no data at the building (no usable orbit, or off the raster)

Buildings are taken from every task rectangle of the plan whose tile is finished and local. Buildings outside
all task rectangles (other parts of the country, z13 tiles without a populated z14 tile) are not in the output.

  python3 src/syria/classify_buildings.py --tag aleppo                  # tiles already in data/syria/ee_run_aleppo/tiles/
  python3 src/syria/classify_buildings.py --tag aleppo --download       # fetch them from Drive first (copies stay in Drive)
  python3 src/syria/classify_buildings.py --tag aleppo --threshold 0.670
  python3 src/syria/classify_buildings.py --tag aleppo --footprint 3x3

Writes <run folder>/buildings_classified.parquet (one row per building: lon, lat, tile, class, reason, n_valid,
n_above, and the maximum for every window), summary_by_class.csv, and with --points-gpkg a GeoPackage of the
confirmed and possible buildings (centre points) for QGIS.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

DEFAULT_THRESHOLD = 0.670  # Gaza's calibrated value (170 of 255)
CONFIRM_WINDOWS = 2
TASK_ZOOM = 13
SPARSE_ZOOM = 14


def classify_values(values, t_u8, confirm=CONFIRM_WINDOWS):
    """
    values: (n_buildings, n_windows) array of 0-255 maxima, NaN = no data.
    Returns (n_valid, n_above, cls, reason) with cls an array of strings.
    """
    v = np.asarray(values, float)
    valid = ~np.isnan(v)
    n_valid = valid.sum(axis=1)
    n_above = np.where(valid, v >= t_u8, False).sum(axis=1)
    cls = np.full(len(v), "not_damaged", dtype=object)
    cls[n_above >= 1] = "possible"
    cls[n_above >= confirm] = "confirmed"
    reason = np.full(len(v), "", dtype=object)
    none = n_valid == 0
    cls[none] = "not_assessed"
    reason[none] = "no_valid_window"
    return n_valid, n_above, cls, reason


def building_window_max(path, n_bands, cx, cy, xmin, ymin, xmax, ymax, mode="bbox"):
    """
    (n_buildings, n_bands) maxima over each building's pixels. mode 'bbox': pixels the box covers, at least the
    centre pixel; '3x3': the 3 x 3 pixels around the centre. NaN where the building is off the raster or every
    pixel is 0 (no data).
    """
    import rasterio

    out = np.full((len(cx), n_bands), np.nan)
    with rasterio.open(path) as src:
        if src.count < n_bands:
            raise ValueError(f"{path} has {src.count} bands, expected {n_bands}")
        arr = src.read()[:n_bands]
        tr = src.transform
        rc, cc = rasterio.transform.rowcol(tr, cx, cy)
        if mode == "bbox":
            r0, c0 = rasterio.transform.rowcol(tr, xmin, ymax)  # top-left
            r1, c1 = rasterio.transform.rowcol(tr, xmax, ymin)  # bottom-right
    rc, cc = np.asarray(rc), np.asarray(cc)
    h, w = arr.shape[1:]
    if mode == "bbox":
        r0, r1, c0, c1 = (np.asarray(a) for a in (r0, r1, c0, c1))
        r0, r1 = np.minimum(r0, rc), np.maximum(r1, rc)  # the centre pixel is always included
        c0, c1 = np.minimum(c0, cc), np.maximum(c1, cc)
    else:
        r0, r1, c0, c1 = rc - 1, rc + 1, cc - 1, cc + 1
    for i in range(len(cx)):
        if rc[i] < 0 or cc[i] < 0 or rc[i] >= h or cc[i] >= w:
            continue
        a, b, c, d = max(r0[i], 0), min(r1[i], h - 1), max(c0[i], 0), min(c1[i], w - 1)
        sl = arr[:, a:b + 1, c:d + 1].reshape(n_bands, -1)
        mx = sl.max(axis=1)
        out[i] = np.where(mx > 0, mx, np.nan)
    return out


def read_buildings(ft, path, bbox):
    parts = list(ft.iter_bounds(Path(path), window=bbox))
    if not parts:
        return (np.zeros(0),) * 4
    return tuple(np.concatenate([p[i] for p in parts]) for i in range(4))


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.strip().split("\n\n")[0])
    p.add_argument("--tag", default="", help="a tagged run (default: the main run)")
    p.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD, help="probability threshold, default Gaza's 0.670")
    p.add_argument("--confirm-windows", type=int, default=CONFIRM_WINDOWS, help="windows needed for 'confirmed'")
    p.add_argument("--footprint", choices=["bbox", "3x3"], default="bbox")
    p.add_argument("--min-buildings", type=int, default=10, help="the z14 tile threshold used in the plan")
    p.add_argument("--tiles-csv", default="data/syria/footprints/tiles_z14.csv")
    p.add_argument("--buildings", default="data/syria/footprints/overture_buildings_bbox.parquet")
    p.add_argument("--download", action="store_true", help="fetch finished tiles that are not local yet (copies stay in Drive)")
    p.add_argument("--allow-missing", action="store_true", help="go on when some planned tiles are not finished or not local")
    p.add_argument("--points-gpkg", action="store_true", help="also write confirmed and possible buildings as points to a GeoPackage")
    args = p.parse_args(argv)

    from src.syria import check_tiles, footprint_tiles as ft

    tag = args.tag.strip().lower()
    out_dir = Path("data/syria/ee_run" + (f"_{tag}" if tag else ""))
    check_tiles.OUT_DIR = out_dir
    check_tiles.DRIVE_FOLDER = "syria_run" + (f"_{tag}" if tag else "")
    plan = check_tiles.read_plan()
    names = check_tiles.window_names(plan)
    led = pd.read_csv(out_dir / "ledger.csv", dtype={"z13": str})
    done = set(led[led.state == "SUCCEEDED"].z13)
    tiles_dir = out_dir / "tiles"
    todo = [z for z in plan.index if z in done]
    if args.download:
        miss = check_tiles.download_tiles([z for z in todo if not (tiles_dir / f"qk_{z}.tif").exists()], tiles_dir, delete=False)
        if miss:
            print(f"Not found in Drive: {len(miss)} tiles")
    have = [z for z in todo if (tiles_dir / f"qk_{z}.tif").exists()]
    n_missing = len(plan) - len(have)
    print(f"{len(plan):,} planned tasks, {len(have):,} finished and local, {n_missing:,} not available; windows: {', '.join(names)}")
    if not have:
        sys.exit("No finished tiles are local yet. Run ee_production.py status, then add --download.")
    if n_missing and not args.allow_missing:
        sys.exit(f"{n_missing:,} planned tiles are not finished or not local. Wait for the run to finish, or add --allow-missing to classify what is there.")

    t_u8 = args.threshold * 255
    z14 = pd.read_csv(args.tiles_csv, dtype={"quadkey": str})
    good14 = z14[z14.n_buildings >= args.min_buildings].quadkey.to_numpy()
    print(f"Threshold {args.threshold:g} ({t_u8:.0f} of 255); 'confirmed' needs {args.confirm_windows} windows; footprint mode {args.footprint}")

    frames = []
    for k, z in enumerate(have, 1):
        row = plan.loc[z]
        bbox = (row.west, row.south, row.east, row.north)
        xmin, ymin, xmax, ymax = read_buildings(ft, args.buildings, bbox)
        cx, cy = (xmin + xmax) / 2, (ymin + ymax) / 2
        keep = (cx >= bbox[0]) & (cx <= bbox[2]) & (cy >= bbox[1]) & (cy <= bbox[3])
        xmin, ymin, xmax, ymax, cx, cy = (a[keep] for a in (xmin, ymin, xmax, ymax, cx, cy))
        if len(cx) == 0:
            continue
        x14, y14 = ft.lonlat_to_tile(cx, cy, SPARSE_ZOOM)
        q14 = np.array(list(map(str, ft.quadkeys(x14, y14, SPARSE_ZOOM))))
        in_good = np.isin(q14, good14)
        vals = np.full((len(cx), len(names)), np.nan)
        if in_good.any():
            sel = np.where(in_good)[0]
            vals[sel] = building_window_max(tiles_dir / f"qk_{z}.tif", len(names), cx[sel], cy[sel], xmin[sel], ymin[sel], xmax[sel], ymax[sel], args.footprint)
        n_valid, n_above, cls, reason = classify_values(vals, t_u8, args.confirm_windows)
        cls = np.where(in_good, cls, "not_assessed").astype(object)
        reason = np.where(in_good, reason, "tile_below_threshold").astype(object)
        df = pd.DataFrame({"lon": cx, "lat": cy, "z13": z, "z14": q14, "class": cls, "reason": reason,
                           "n_valid": n_valid, "n_above": n_above})
        for i, w in enumerate(names):
            df[w] = vals[:, i]
        frames.append(df)
        if k % 100 == 0:
            print(f"  {k:,} of {len(have):,} tiles done")
    if not frames:
        sys.exit("No buildings found inside the task rectangles. Check --buildings.")
    res = pd.concat(frames, ignore_index=True)
    out = out_dir / "buildings_classified.parquet"
    res.to_parquet(out, index=False)

    summ = res.groupby(["class", "reason"]).size().rename("buildings").reset_index()
    summ["share"] = summ.buildings / len(res)
    summ.to_csv(out_dir / "summary_by_class.csv", index=False)
    print(f"\n{len(res):,} buildings in {res.z13.nunique():,} tiles")
    for r in summ.to_dict("records"):
        label = r["class"] + (f" ({r['reason']})" if r["reason"] else "")
        print(f"  {label:<46}{r['buildings']:>12,}{r['share']:>9.1%}")
    assessed = res[res["class"] != "not_assessed"]
    if len(assessed):
        print(f"\nOf {len(assessed):,} assessed buildings: confirmed {(assessed['class'] == 'confirmed').mean():.1%}, possible {(assessed['class'] == 'possible').mean():.1%}")
        print("Share of assessed buildings at or above the threshold, by window:")
        for w in names:
            v = assessed[w].dropna()
            print(f"  {w:<8}{len(v):>12,} with data{(v >= t_u8).mean() if len(v) else float('nan'):>9.1%}")
    print(f"\nSaved {out} and summary_by_class.csv")
    if args.points_gpkg:
        import geopandas as gpd

        pts = res[res["class"].isin(["confirmed", "possible"])]
        g = gpd.GeoDataFrame(pts.drop(columns=["lon", "lat"]), geometry=gpd.points_from_xy(pts.lon, pts.lat), crs=4326)
        g.to_file(out_dir / "buildings_damaged_points.gpkg", driver="GPKG")
        print(f"Saved {out_dir / 'buildings_damaged_points.gpkg'} ({len(pts):,} points)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
