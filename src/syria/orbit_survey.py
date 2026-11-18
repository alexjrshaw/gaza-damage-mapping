#!/usr/bin/env python3
"""
Which Sentinel-1 orbits cover each Syria tile, and are there enough images?

Costs no Earth Engine compute: it only reads image footprints (a metadata query), then
does the geometry on the login node. For every (one or two candidate baselines, and the 2-month assessment windows) it counts
the Sentinel-1A IW VV+VH images per relative orbit.

Same rule as the Gaza pipeline (src/data/sentinel1/orbits.py): an orbit is valid for a
tile when it has at least 10 images over the tile in the one-year baseline. Windows are
checked separately, because a 2-month window holds only about 5 images per orbit.

Run from the repository root:

  python3 src/syria/orbit_survey.py
  python3 src/syria/orbit_survey.py --zoom 13 --windows 2016 2017 2018 2020 2023 2025
  python3 src/syria/orbit_survey.py --tiles-csv data/syria/footprints/tiles_z14.csv --min-buildings 10

Writes data/syria/orbit_survey/images_per_tile_orbit.csv (one row per tile, period and orbit),
periods.json (the dates of every period, checked by the plan) and tile_orbits.csv (one row per tile with its valid orbits), and prints a summary.
"""
import argparse
import sys
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

SYRIA_BBOX = (35.5, 32.2, 42.5, 37.4)  # west, south, east, north (same box as footprint_tiles.py)
MIN_BASELINE_IMAGES = 10  # the Gaza rule
MIN_WINDOW_IMAGES = 2  # a 2-month window with fewer images per orbit is too thin for the features
FULL_COVER = 0.95  # share of the tile an image footprint must cover to count as "full"
# One-year baselines to compare. A is Aleppo's (the first Sentinel-1 year), B is Raqqa's.
BASELINES = {"baseA": ("2014-10-06", "2015-10-05"), "baseB": ("2015-10-21", "2016-10-20")}


def window_for_year(year):
    """The 2-month window 7 Aug - 6 Oct of that year (the same calendar slot as a Gaza window)."""
    return f"win{year}", (f"{year}-08-07", f"{year}-10-06")


def month_chunks(start, end):
    """Split [start, end] (ISO dates) into pieces of at most about 31 days."""
    a, b = date.fromisoformat(start), date.fromisoformat(end)
    out = []
    while a <= b:
        z = min(a + timedelta(days=30), b)
        out.append((a.isoformat(), (z + timedelta(days=1)).isoformat()))
        a = z + timedelta(days=1)
    return out


def quadkey_bounds(qk):
    import math

    z, x, y = len(qk), 0, 0
    for i, ch in enumerate(qk):
        bit = z - 1 - i
        d = int(ch)
        x |= (d & 1) << bit
        y |= ((d >> 1) & 1) << bit
    n = 2**z
    lat = lambda yy: math.degrees(math.atan(math.sinh(math.pi * (1 - 2 * yy / n))))
    return x / n * 360 - 180, lat(y + 1), (x + 1) / n * 360 - 180, lat(y)


def fetch_footprints(ee, start, end, region):
    """[(orbit, polygon coordinates)] of every S1A IW VV+VH image in [start, end] over the region."""
    from src.data.sentinel1.collection import get_s1_collection

    rows = []
    for a, b in month_chunks(start, end):
        col = get_s1_collection(region, a, b)
        feats = col.map(
            lambda im: ee.Feature(im.geometry(), {"orbit": im.get("relativeOrbitNumber_start")})
        ).getInfo()["features"]
        for f in feats:
            rows.append((int(f["properties"]["orbit"]), f["geometry"]))
    return rows


def count_per_tile(footprints, tile_polys, tile_ids):
    """DataFrame tile, orbit, n_any, n_full: images whose footprint touches / covers >= 95% of the tile."""
    import shapely
    from shapely.geometry import shape

    tiles = np.array(tile_polys, dtype=object)
    areas = shapely.area(tiles)
    tree = shapely.STRtree(tiles)
    acc = {}
    for orbit, geom in footprints:
        poly = shape(geom)
        if not poly.is_valid:
            poly = poly.buffer(0)
        if poly.is_empty:
            continue
        hit = tree.query(poly, predicate="intersects")
        if len(hit) == 0:
            continue
        frac = shapely.area(shapely.intersection(tiles[hit], poly)) / areas[hit]
        for i, f in zip(hit, frac):
            if f <= 0:
                continue
            key = (tile_ids[i], orbit)
            n_any, n_full = acc.get(key, (0, 0))
            acc[key] = (n_any + 1, n_full + (1 if f >= FULL_COVER else 0))
    return pd.DataFrame(
        [(t, o, a, f) for (t, o), (a, f) in acc.items()], columns=["tile", "orbit", "n_any", "n_full"]
    )


def valid_orbits_table(long, tiles, baseline, windows):
    """One row per tile: valid orbits (>=10 baseline images), and per window the orbits with enough images."""
    base = long[(long.period == baseline) & (long.n_full >= MIN_BASELINE_IMAGES)]
    rows = []
    for t in tiles:
        orbits = sorted(base[base.tile == t].orbit.tolist())
        row = {"tile": t, "baseline": baseline, "orbits": ",".join(map(str, orbits)), "n_orbits": len(orbits)}
        for w in windows:
            sub = long[(long.period == w) & (long.tile == t) & (long.orbit.isin(orbits))]
            ok = sub[sub.n_full >= MIN_WINDOW_IMAGES].orbit.tolist()
            row[f"{w}_orbits_ok"] = len(ok)
        rows.append(row)
    return pd.DataFrame(rows)


def summarise(table, baselines, windows, long, tiles):
    n = len(tiles)
    print(f"\n{n:,} tiles surveyed\n")
    for b in baselines:
        sub = valid_orbits_table(long, tiles, b, windows)
        print(f"Baseline {b} {BASELINES[b]}: tiles with at least one valid orbit (>= {MIN_BASELINE_IMAGES} images):"
              f" {(sub.n_orbits > 0).sum():,} of {n:,}")
        print("  valid orbits per tile: " + ", ".join(f"{k} orbits: {v:,}" for k, v in sub.n_orbits.value_counts().sort_index().items()))
        for w in windows:
            col = f"{w}_orbits_ok"
            print(f"  {w}: tiles where every valid orbit has >= {MIN_WINDOW_IMAGES} images: "
                  f"{(sub[col] == sub.n_orbits)[sub.n_orbits > 0].sum():,};  with none: {(sub[col] == 0)[sub.n_orbits > 0].sum():,}")
        print()


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.strip().split("\n\n")[0])
    p.add_argument("--tiles-csv", default="data/syria/footprints/tiles_z14.csv")
    p.add_argument("--min-buildings", type=int, default=10)
    p.add_argument("--windows", nargs="+", type=int, default=[2016, 2017, 2018, 2020, 2023, 2025])
    p.add_argument("--city", help="survey the window list of a transfer city (ALP or RAQ) instead of --windows")
    p.add_argument("--annual", type=int, metavar="N", help="survey N consecutive one-year windows after baseline A (yr01 ... yrNN), as in ee_production.py plan --annual N")
    p.add_argument("--bbox", nargs=4, type=float, metavar=("W", "S", "E", "N"),
                   help="survey only tiles inside this box (a region trial); the image query is limited to it too")
    p.add_argument("--boundary", help="survey only tiles touching this boundary file (an administrative layer)")
    p.add_argument("--where", nargs="+", metavar="FIELD=VALUE", help="pick features of the boundary file, e.g. admin1Name_en=Aleppo")
    p.add_argument("--zoom", type=int, default=12, choices=[12, 13, 14],
                   help="size of the tiles to survey: 12 (default) or 13 (the unit of the production tasks)")
    p.add_argument("--out-dir", default="data/syria/orbit_survey")
    args = p.parse_args(argv)

    import ee
    from shapely.geometry import box

    from src.utils.gee import init_gee

    init_gee()
    df = pd.read_csv(args.tiles_csv, dtype={"quadkey": str, "parent_z12": str})
    df = df[df.n_buildings >= args.min_buildings]
    region = None
    if args.boundary or args.bbox:
        from src.syria.ee_production import load_region, tiles_in_region

        region = load_region(args.boundary, args.where) if args.boundary else None
        df = df[tiles_in_region(df.quadkey.tolist(), tuple(args.bbox) if args.bbox else None, region)]
        if df.empty:
            sys.exit("No z14 tile with enough buildings inside that area.")
    tiles = sorted(df.quadkey.str[: args.zoom].unique())
    polys = [box(*quadkey_bounds(t)) for t in tiles]
    print(f"{len(tiles):,} z{args.zoom} tiles hold z14 tiles with >= {args.min_buildings} buildings.")

    periods = dict(BASELINES)
    if args.annual:
        from src.syria.ee_production import annual_windows

        wl = annual_windows(args.annual)
        for wdict in wl:
            periods[wdict["period"]] = (wdict["start"], wdict["end"])
        windows = [wdict["period"] for wdict in wl]
    elif args.city:
        from src.syria.ee_production import city_windows

        for wdict in city_windows(args.city)[0]:
            periods[wdict["period"]] = (wdict["start"], wdict["end"])
        windows = [wdict["period"] for wdict in city_windows(args.city)[0]]
    else:
        for y in args.windows:
            name, rng = window_for_year(y)
            periods[name] = rng
        windows = [window_for_year(y)[0] for y in args.windows]
    if args.bbox or region is not None:
        pad = 0.3  # image footprints are large; a margin keeps every image that touches the area
        if region is not None:
            w, s_, e, n = region.bounds
            if args.bbox:
                w, s_, e, n = max(w, args.bbox[0]), max(s_, args.bbox[1]), min(e, args.bbox[2]), min(n, args.bbox[3])
        else:
            w, s_, e, n = args.bbox
        region = ee.Geometry.Rectangle([w - pad, s_ - pad, e + pad, n + pad], proj="EPSG:4326", geodesic=False)
    else:
        region = ee.Geometry.Rectangle(list(SYRIA_BBOX), proj="EPSG:4326", geodesic=False)

    parts = []
    for name, (a, b) in periods.items():
        fp = fetch_footprints(ee, a, b, region)
        print(f"  {name} {a} .. {b}: {len(fp):,} image footprints")
        part = count_per_tile(fp, polys, tiles)
        part["period"] = name
        parts.append(part)
    long = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=["tile", "orbit", "n_any", "n_full", "period"])
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    long.to_csv(out / "images_per_tile_orbit.csv", index=False)
    import json

    (out / "periods.json").write_text(json.dumps({k: list(v) for k, v in periods.items()}, indent=2))  # the plan checks its windows against this

    for b in BASELINES:
        valid_orbits_table(long, tiles, b, windows).to_csv(out / f"tile_orbits_{b}.csv", index=False)
    summarise(None, list(BASELINES), windows, long, tiles)
    print(f"Wrote {out / 'images_per_tile_orbit.csv'} and tile_orbits_baseA.csv / tile_orbits_baseB.csv")
    return 0


if __name__ == "__main__":
    sys.exit(main())
