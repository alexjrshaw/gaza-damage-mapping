#!/usr/bin/env python3
"""
Binary building damage from the finished tiles of a run: Equation 3 of Dietrich et al. (2025a), exactly as the Gaza
pipeline applies it (src/postprocessing/classify_building_damage.py):

    damaged = 1  if  max(post windows) >= t  AND  max(pre windows) < t,     else 0        t = int(0.670 x 255) = 170

and the window values are taken the way the Gaza pipeline takes them (create_buildings_with_preds_for_admin, then
process_admin_file, which uses "weighted_mean"): for every building polygon and window, the AREA-WEIGHTED MEAN of the pixel
probabilities, each pixel weighted by the area of its overlap with the polygon. The maximum in Equation 3 is the maximum
over the windows, not over the pixels. A pixel value of 0 is no data and counts as 0, as in Gaza. Buildings under --min-area-m2 (default 50, as in
Gaza) are left out, using the true polygon area in an equal-area projection.

Pre windows: the windows whose role in windows.json is "pre" (set by `ee_production.py plan --annual N --n-pre K`), or
the ones you name with --pre-windows. Every other window is a post window. There is no other rule: no counting of
windows, no extra classes.

Missing data is handled exactly as in Gaza. A pixel with no data (0) counts as 0 in the weighted mean and keeps its area in
the weights. So a building with no data in every pre window has max_pre = 0 and is damaged if max_post >= t, and one with
no data in every post window has max_post = 0 and is not damaged. Only a building that overlaps no pixel of its tile at
all is NaN, which makes the comparison False (not damaged). The `reason` column flags no_valid_pre / no_valid_post (no
pixel above 0 in that group) so you can count these cases; it never changes the result.

Buildings are left out of the assessment (class not_assessed, damaged missing) only when
    tile_below_threshold : it lies in a z14 tile with fewer than --min-buildings buildings, which the run does not classify
                           (a Syria cost-saving rule; Gaza ran every tile)
    below_min_area       : its polygon is smaller than --min-area-m2 (Gaza drops these buildings too)

The buildings file must hold polygons (a `geometry` column, WKB). The box-only file made by `footprint_tiles.py download`
without --extra-columns geometry is refused.

  python3 src/syria/classify_buildings.py --tag annual                      # pre windows from windows.json
  python3 src/syria/classify_buildings.py --tag aleppo --pre-windows w2016  # a run planned with --years: name them
  python3 src/syria/classify_buildings.py --tag annual --download           # fetch finished tiles from Drive first (copies stay in Drive)
  python3 src/syria/classify_buildings.py --tag annual --threshold 0.670 --points-gpkg

Writes <run folder>/buildings_classified.parquet (one row per building: lon, lat, z13, z14, area_m2, class, reason,
damaged 1/0, max_pre, max_post (maxima over windows) and the weighted mean of every window), summary_by_class.csv, and with --points-gpkg a GeoPackage
of the damaged buildings (centre points) for QGIS. Buildings outside every task rectangle are not in the output.
"""
import argparse
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

DEFAULT_THRESHOLD = 0.670  # Gaza's calibrated value
MIN_AREA_M2 = 50.0  # Gaza's minimum building area
TASK_ZOOM = 13
SPARSE_ZOOM = 14
NO_GEOMETRY = ("The buildings file {path} has no polygon geometry (columns: {cols}), so the area-weighted mean inside the building polygon, "
               "the method Gaza uses, cannot be computed. Download polygons:\n"
               "  python3 -m src.syria.footprint_tiles download --extra-columns geometry --out data/syria/footprints/overture_buildings_geom.parquet\n"
               "and pass it with --buildings.")


def threshold_u8(t):
    """Gaza: THRESHOLD = int(0.670 * 255) = 170 (truncated, not rounded)."""
    return int(t * 255)


def window_extremes(values, pre):
    """max over the pre windows and over the post windows of an (n_buildings, n_windows) array (NaN = no data). NaN when a group has no data."""
    v = np.asarray(values, float)
    pre = list(pre)
    post = [i for i in range(v.shape[1]) if i not in pre]
    if not pre or not post:
        raise ValueError("need at least one pre window and one post window")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nanmax(v[:, pre], axis=1), np.nanmax(v[:, post], axis=1)


def equation3(max_pre, max_post, t, valid_pre=None, valid_post=None):
    """
    Gaza's rule, unchanged: damaged = (max_post >= t) & (max_pre < t), as 0/1. NaN compares False, so missing data gives 0,
    as in Gaza. Returns (damaged, flag) where flag marks no_valid_pre / no_valid_post (no window of that group has a pixel above 0 at the building) for information only.
    """
    max_pre, max_post = np.asarray(max_pre, float), np.asarray(max_post, float)
    with np.errstate(invalid="ignore"):
        damaged = ((max_post >= t) & (max_pre < t)).astype(int)
    flag = np.full(len(max_pre), "", dtype=object)
    no_pre = np.isnan(max_pre) if valid_pre is None else ~np.asarray(valid_pre, bool)
    no_post = np.isnan(max_post) if valid_post is None else ~np.asarray(valid_post, bool)
    flag[no_pre] = "no_valid_pre"
    flag[no_post] = "no_valid_post"  # takes precedence when both are missing
    return damaged, flag


def polygon_area_m2(geoms):
    """Polygon areas in square metres (EPSG:6933, equal-area)."""
    import shapely
    from pyproj import Transformer

    tr = Transformer.from_crs(4326, 6933, always_xy=True)
    proj = shapely.transform(np.asarray(geoms, dtype=object), lambda c: np.column_stack(tr.transform(c[:, 0], c[:, 1])))
    return shapely.area(proj)


def valid_polygons(geoms):
    """Repair invalid polygons (Overture has a few) so intersections do not fail."""
    import shapely

    geoms = np.asarray(geoms, dtype=object)
    bad = ~shapely.is_valid(geoms)
    if bad.any():
        geoms = geoms.copy()
        geoms[bad] = shapely.make_valid(geoms[bad])
    return geoms


def polygon_window_wmean(path, n_bands, geoms, block=20000):
    """
    Returns (values, has_data), both (n_buildings, n_bands).
    values: for each polygon the AREA-WEIGHTED MEAN of each band over the pixels it overlaps, with the overlap area of
    polygon and pixel square as the weight. This is what the Gaza pipeline puts in buildings_preds.parquet
    (create_buildings_with_preds_for_admin: gpd.overlay of buildings and pixel squares, weights = intersection area,
    process_admin_file pivots "weighted_mean"). Areas are in the raster's own units (degrees squared), as there.
    A pixel value of 0 (no data) contributes 0 to the sum but keeps its area in the weights, exactly as in Gaza, where
    NaN pixels are skipped in the numerator of the groupby sum but their area stays in the denominator. NaN only where
    the polygon overlaps no pixel of the tile (Gaza: the building drops out of the overlay and is NaN).
    has_data: True where at least one overlapped pixel is above 0. It is information only; it never enters the result.
    """
    import rasterio
    import shapely

    with rasterio.open(path) as src:
        if src.count < n_bands:
            raise ValueError(f"{path} has {src.count} bands, expected {n_bands}")
        arr = src.read()[:n_bands]
        tr = src.transform
    H, W = arr.shape[1:]
    geoms = valid_polygons(geoms)
    out = np.full((len(geoms), n_bands), np.nan)
    has = np.zeros((len(geoms), n_bands), dtype=bool)
    for k in range(0, len(geoms), block):
        g = geoms[k:k + block]
        n = len(g)
        bb = shapely.bounds(g)
        c0 = np.clip(np.floor((bb[:, 0] - tr.c) / tr.a).astype(int), 0, W)
        c1 = np.clip(np.floor((bb[:, 2] - tr.c) / tr.a).astype(int), -1, W - 1)
        r0 = np.clip(np.floor((bb[:, 3] - tr.f) / tr.e).astype(int), 0, H)  # tr.e < 0: the northern edge is the smaller row
        r1 = np.clip(np.floor((bb[:, 1] - tr.f) / tr.e).astype(int), -1, H - 1)
        nc = np.maximum(c1 - c0 + 1, 0)
        nr = np.maximum(r1 - r0 + 1, 0)
        cnt = nr * nc
        total = int(cnt.sum())
        if total == 0:
            continue
        idx = np.repeat(np.arange(n), cnt)
        start = np.cumsum(cnt) - cnt
        j = np.arange(total) - np.repeat(start, cnt)
        rr = r0[idx] + j // nc[idx]
        cc = c0[idx] + j % nc[idx]
        x0 = tr.c + cc * tr.a
        ytop = tr.f + rr * tr.e
        squares = shapely.box(x0, ytop + tr.e, x0 + tr.a, ytop)
        w = shapely.area(shapely.intersection(g[idx], squares))
        v = arr[:, rr, cc].T.astype(float)  # (pairs, bands)
        ok = v > 0
        den = np.bincount(idx, weights=w, minlength=n)
        for band in range(n_bands):
            num = np.bincount(idx, weights=w * v[:, band], minlength=n)
            with np.errstate(invalid="ignore", divide="ignore"):
                out[k:k + n, band] = np.where(den > 0, num / den, np.nan)
            has[k:k + n, band] = np.bincount(idx, weights=w * ok[:, band], minlength=n) > 0
    return out, has


def iter_polygon_chunks(path, tile_boxes, ft, chunk=100):
    """
    Yields (z13, geometries) for every task tile: the building polygons whose bounding-box centre lies in the task
    rectangle. Reads the file once per chunk of tiles, using the box columns to skip what is far away.
    tile_boxes: {z13: (west, south, east, north)}.
    """
    import pyarrow.compute as pc
    import pyarrow.dataset as ds
    import shapely

    dataset = ds.dataset(str(path), format="parquet")
    names = dataset.schema.names
    if "geometry" not in names:
        raise SystemExit(NO_GEOMETRY.format(path=path, cols=", ".join(names)))
    flat = all(f in names for f in ("xmin", "ymin", "xmax", "ymax"))
    struct = "bbox" in names
    tiles = sorted(tile_boxes)
    for k in range(0, len(tiles), chunk):
        ids = tiles[k:k + chunk]
        W = min(tile_boxes[z][0] for z in ids); S = min(tile_boxes[z][1] for z in ids)
        E = max(tile_boxes[z][2] for z in ids); N = max(tile_boxes[z][3] for z in ids)
        if flat:
            expr = (pc.field("xmax") >= W) & (pc.field("xmin") <= E) & (pc.field("ymax") >= S) & (pc.field("ymin") <= N)
        elif struct:
            expr = ((pc.field("bbox", "xmax") >= W) & (pc.field("bbox", "xmin") <= E)
                    & (pc.field("bbox", "ymax") >= S) & (pc.field("bbox", "ymin") <= N))
        else:
            expr = None
        found = {z: [] for z in ids}
        for batch in dataset.to_batches(columns=["geometry"], filter=expr, batch_size=200_000):
            if batch.num_rows == 0:
                continue
            geoms = shapely.from_wkb(batch.column("geometry").to_numpy(zero_copy_only=False))
            geoms = geoms[~shapely.is_missing(geoms) & ~shapely.is_empty(geoms)]
            if len(geoms) == 0:
                continue
            bb = shapely.bounds(geoms)
            cx, cy = (bb[:, 0] + bb[:, 2]) / 2, (bb[:, 1] + bb[:, 3]) / 2
            x, y = ft.lonlat_to_tile(cx, cy, TASK_ZOOM)
            q = np.array(list(map(str, ft.quadkeys(x, y, TASK_ZOOM))))
            for z in np.unique(q):
                if z not in found:
                    continue
                w, s, e, n = tile_boxes[z]
                sel = (q == z) & (cx >= w) & (cx <= e) & (cy >= s) & (cy <= n)
                if sel.any():
                    found[z].append(geoms[sel])
        for z in ids:
            yield z, (np.concatenate(found[z]) if found[z] else np.array([], dtype=object))


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.strip().split("\n\n")[0])
    p.add_argument("--tag", default="", help="a tagged run (default: the main run)")
    p.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD, help="probability threshold; the value used is int(threshold x 255), as in Gaza (default 0.670 -> 170)")
    p.add_argument("--pre-windows", nargs="+", metavar="WINDOW", default=["auto"],
                   help="window names to treat as pre; default 'auto' = the windows with role 'pre' in windows.json")
    p.add_argument("--min-area-m2", type=float, default=MIN_AREA_M2, help="smaller buildings are not assessed (default 50, as in Gaza; 0 keeps all)")
    p.add_argument("--min-buildings", type=int, default=10, help="the z14 tile threshold used in the plan")
    p.add_argument("--tiles-csv", default="data/syria/footprints/tiles_z14.csv")
    p.add_argument("--buildings", default="data/syria/footprints/overture_buildings_geom.parquet", help="Overture buildings WITH polygon geometry")
    p.add_argument("--chunk-tiles", type=int, default=100, help="task tiles per pass over the buildings file")
    p.add_argument("--download", action="store_true", help="fetch finished tiles that are not local yet (copies stay in Drive)")
    p.add_argument("--allow-missing", action="store_true", help="go on when some planned tiles are not finished or not local")
    p.add_argument("--points-gpkg", action="store_true", help="also write the damaged buildings as points to a GeoPackage")
    args = p.parse_args(argv)

    from src.syria import check_tiles, footprint_tiles as ft

    tag = args.tag.strip().lower()
    out_dir = Path("data/syria/ee_run" + (f"_{tag}" if tag else ""))
    check_tiles.OUT_DIR = out_dir
    check_tiles.DRIVE_FOLDER = "syria_run" + (f"_{tag}" if tag else "")
    plan = check_tiles.read_plan()
    names = check_tiles.window_names(plan)

    if args.pre_windows == ["auto"]:
        roles = [w.get("role") for w in json.loads((out_dir / "windows.json").read_text())["windows"]]
        pre_names = [n for n, r in zip(names, roles) if r == "pre"]
        if not pre_names:
            sys.exit("No window in windows.json has role 'pre' (the run was not planned with --annual). Name the pre windows, e.g. --pre-windows w2016.")
    else:
        unknown = [w for w in args.pre_windows if w not in names]
        if unknown:
            sys.exit(f"Unknown window(s) {unknown}; the run's windows are {names}.")
        pre_names = list(args.pre_windows)
    if len(pre_names) == len(names):
        sys.exit("Every window is a pre window: nothing is left for the post windows.")
    pre_idx = [names.index(w) for w in pre_names]
    post_names = [n for n in names if n not in pre_names]

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
    print(f"{len(plan):,} planned tasks, {len(have):,} finished and local, {n_missing:,} not available")
    print(f"Windows: pre {', '.join(pre_names)}; post {', '.join(post_names)}")
    if not have:
        sys.exit("No finished tiles are local yet. Run ee_production.py status, then add --download.")
    if n_missing and not args.allow_missing:
        sys.exit(f"{n_missing:,} planned tiles are not finished or not local. Wait for the run to finish, or add --allow-missing to classify what is there.")

    t = threshold_u8(args.threshold)
    z14 = pd.read_csv(args.tiles_csv, dtype={"quadkey": str})
    good14 = z14[z14.n_buildings >= args.min_buildings].quadkey.to_numpy()
    print(f"Threshold {args.threshold:g} -> {t} of 255 (Gaza: int(0.670 x 255) = 170); damaged = max(post) >= {t} and max(pre) < {t}; "
          f"minimum building area {args.min_area_m2:g} m2; building value = area-weighted mean of the pixels the polygon overlaps (as Gaza)")

    boxes = {z: tuple(plan.loc[z, ["west", "south", "east", "north"]].astype(float)) for z in have}
    frames = []
    for k, (z, geoms) in enumerate(iter_polygon_chunks(args.buildings, boxes, ft, args.chunk_tiles), 1):
        if len(geoms) == 0:
            continue
        import shapely

        bb = shapely.bounds(geoms)
        cx, cy = (bb[:, 0] + bb[:, 2]) / 2, (bb[:, 1] + bb[:, 3]) / 2
        x14, y14 = ft.lonlat_to_tile(cx, cy, SPARSE_ZOOM)
        q14 = np.array(list(map(str, ft.quadkeys(x14, y14, SPARSE_ZOOM))))
        in_good = np.isin(q14, good14)
        area = polygon_area_m2(geoms)
        small = area < args.min_area_m2
        assess = in_good & ~small
        vals = np.full((len(geoms), len(names)), np.nan)
        has = np.zeros((len(geoms), len(names)), dtype=bool)
        if assess.any():
            sel = np.where(assess)[0]
            vals[sel], has[sel] = polygon_window_wmean(tiles_dir / f"qk_{z}.tif", len(names), geoms[sel])
        post_idx = [i for i in range(len(names)) if i not in pre_idx]
        max_pre, max_post = window_extremes(vals, pre_idx)
        damaged, reason = equation3(max_pre, max_post, t, has[:, pre_idx].any(axis=1), has[:, post_idx].any(axis=1))
        cls = np.where(damaged == 1, "damaged", "not_damaged").astype(object)
        reason = np.where(~in_good, "tile_below_threshold", np.where(small, "below_min_area", reason)).astype(object)
        cls[~assess] = "not_assessed"
        damaged = np.where(assess, damaged, -1)
        df = pd.DataFrame({"lon": cx, "lat": cy, "z13": z, "z14": q14, "area_m2": area, "class": cls, "reason": reason,
                           "damaged": pd.array(np.where(damaged >= 0, damaged, pd.NA), dtype="Int8"),
                           "max_pre": max_pre, "max_post": max_post})
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
        print(f"\nOf {len(assessed):,} assessed buildings, {(assessed['class'] == 'damaged').mean():.1%} are damaged (Equation 3).")
        for fl in ("no_valid_pre", "no_valid_post"):
            n_fl = int((assessed["reason"] == fl).sum())
            if n_fl:
                print(f"  {n_fl:,} assessed buildings ({n_fl / len(assessed):.1%}) have {fl.replace('_', ' ')} (window value 0, as in Gaza)")
        print(f"Share of assessed buildings with the window value (weighted mean) at or above {t}, by window:")
        for w in names:
            v = assessed[w].dropna()
            tag_ = "pre " if w in pre_names else "post"
            print(f"  {w:<8}{tag_}{len(v):>12,} with data{(v >= t).mean() if len(v) else float('nan'):>9.1%}")
    print(f"\nSaved {out} and summary_by_class.csv")
    if args.points_gpkg:
        import geopandas as gpd

        pts = res[res["class"] == "damaged"]
        g = gpd.GeoDataFrame(pts.drop(columns=["lon", "lat", "damaged"]), geometry=gpd.points_from_xy(pts.lon, pts.lat), crs=4326)
        g.to_file(out_dir / "buildings_damaged_points.gpkg", driver="GPKG")
        print(f"Saved {out_dir / 'buildings_damaged_points.gpkg'} ({len(pts):,} points)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
