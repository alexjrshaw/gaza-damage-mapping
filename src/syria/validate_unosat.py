#!/usr/bin/env python3
"""
Does the Syria run separate damaged from undamaged buildings? Test on the two UNOSAT cities.

For Aleppo (UNOSAT 2016-09-18) and Raqqa (UNOSAT 2017-10-21) it takes the production tiles of the
matching window (w2016 for Aleppo, w2017 for Raqqa) and reads the probability at
  * every UNOSAT damaged building (positives), and
  * Overture buildings inside the assessed area that have no UNOSAT point within 60 m (assumed
    undamaged; UNOSAT lists damaged buildings only, so this is an assumption, not a label),
as the MAXIMUM probability in the 3 x 3 pixels around the building centre (about 30 m), a stand-in for
the maximum inside the footprint that the building step will use.

It prints, per city: how well the probability ranks damaged above undamaged (AUC), and for a range of
thresholds the share of damaged buildings found (TPR) and of undamaged buildings wrongly flagged (FPR),
with the best threshold (Youden's J) next to the Gaza value (0.670 = 170 of 255).

  python3 src/syria/validate_unosat.py --download      # fetch the needed tiles from Drive first (kept there)
  python3 src/syria/validate_unosat.py                 # use tiles already in data/syria/ee_run/tiles/
  python3 src/syria/validate_unosat.py --cities ALP --window-year 2016

Writes data/syria/ee_run/validation_<city>.csv (one row per sampled building).
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
TASK_ZOOM = 13
GAZA_U8 = 170
CITY_YEAR = {"ALP": 2016, "RAQ": 2017}
NEG_RADIUS_M = 60.0
RADIUS_PX = 1  # 3 x 3 pixels
THRESHOLDS_U8 = list(range(100, 240, 10)) + [GAZA_U8]


def auc(pos, neg):
    """Probability that a random damaged building scores above a random undamaged one (ties count half)."""
    pos, neg = np.asarray(pos, float), np.asarray(neg, float)
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    allv = np.concatenate([pos, neg])
    ranks = pd.Series(allv).rank(method="average").to_numpy()
    return float((ranks[: len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def roc_table(pos, neg, thresholds=THRESHOLDS_U8):
    rows = []
    for t in sorted(set(thresholds)):
        tpr, fpr = float((np.asarray(pos) >= t).mean()), float((np.asarray(neg) >= t).mean())
        rows.append(dict(threshold_u8=t, threshold=round(t / 255, 3), tpr=tpr, fpr=fpr, youden=tpr - fpr))
    return pd.DataFrame(rows)


def sample_max(path, band_index, lon, lat, radius=RADIUS_PX):
    """Max of the band in a (2r+1)^2 window around each lon/lat; NaN when outside the raster or no data."""
    import rasterio

    out = np.full(len(lon), np.nan)
    with rasterio.open(path) as src:
        band = src.read(band_index + 1)
        rows, cols = rasterio.transform.rowcol(src.transform, lon, lat)
    rows, cols = np.asarray(rows), np.asarray(cols)
    h, w = band.shape
    for i, (r, c) in enumerate(zip(rows, cols)):
        if r < 0 or c < 0 or r >= h or c >= w:
            continue
        win = band[max(r - radius, 0): r + radius + 1, max(c - radius, 0): c + radius + 1]
        if win.size and win.max() > 0:
            out[i] = win.max()
    return out


def read_city(ft, city, plan_years, args):
    base = ft.labels_dir() / city.lower()
    pos_lon, pos_lat = ft.read_points(base / "unosat_labels.geojson", "damage", ["1", "2"])
    import geopandas as gpd
    import shapely

    aoi_gdf = gpd.read_file(base / "unosat_aoi.geojson").to_crs(4326)
    aoi = shapely.union_all(np.asarray(aoi_gdf.geometry.values))  # works on older geopandas too (no GeoDataFrame.union_all)
    m = 0.01
    window = (pos_lon.min() - m, pos_lat.min() - m, pos_lon.max() + m, pos_lat.max() + m)
    parts = list(ft.iter_bounds(Path(args.buildings), window=window))
    if not parts:
        return pos_lon, pos_lat, np.zeros(0), np.zeros(0)
    xmin, ymin, xmax, ymax = (np.concatenate([p[i] for p in parts]) for i in range(4))
    cx, cy = (xmin + xmax) / 2, (ymin + ymax) / 2
    inside = shapely.contains_xy(aoi, cx, cy)
    cx, cy = cx[inside], cy[inside]
    near = ft.nearest_footprint_m(cx, cy, (pos_lon, pos_lat, pos_lon, pos_lat))  # distance to the nearest damaged point
    keep = near > NEG_RADIUS_M
    return pos_lon, pos_lat, cx[keep], cy[keep]


def tile_ids(ft, lon, lat):
    x, y = ft.lonlat_to_tile(lon, lat, TASK_ZOOM)
    return np.array(list(map(str, ft.quadkeys(x, y, TASK_ZOOM))))


def values_for(ft, lon, lat, year, years, tiles_dir):
    ids = tile_ids(ft, lon, lat)
    out = np.full(len(lon), np.nan)
    band_index = years.index(year)
    for z in np.unique(ids):
        f = tiles_dir / f"qk_{z}.tif"
        if not f.exists():
            continue
        sel = ids == z
        out[sel] = sample_max(f, band_index, lon[sel], lat[sel])
    return out, ids


def report(city, year, pos, neg, n_pos_all, n_neg_all):
    print(f"\n=== {city}  window w{year} ===")
    print(f"  damaged buildings with a value: {len(pos):,} of {n_pos_all:,};  undamaged (assumed): {len(neg):,} of {n_neg_all:,}")
    if len(pos) == 0 or len(neg) == 0:
        print("  not enough data to score (are the tiles downloaded and the tasks finished?)")
        return None
    print(f"  median probability: damaged {np.median(pos) / 255:.2f}, undamaged {np.median(neg) / 255:.2f}")
    print(f"  AUC (damaged ranked above undamaged): {auc(pos, neg):.3f}   (0.5 = no skill, 1.0 = perfect)")
    t = roc_table(pos, neg)
    best = t.loc[t.youden.idxmax()]
    print("  threshold   found (TPR)   wrongly flagged (FPR)")
    for r in t.itertuples():
        mark = "  <- Gaza 0.670" if r.threshold_u8 == GAZA_U8 else ("  <- best" if r.threshold_u8 == best.threshold_u8 else "")
        print(f"    {r.threshold:5.3f}     {r.tpr:7.1%}       {r.fpr:7.1%}{mark}")
    return best


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.strip().split("\n\n")[0])
    p.add_argument("--cities", nargs="+", default=["ALP", "RAQ"])
    p.add_argument("--window-year", type=int, help="override the window (default: ALP 2016, RAQ 2017)")
    p.add_argument("--tiles-dir", default=str(OUT_DIR / "tiles"))
    p.add_argument("--buildings", default="data/syria/footprints/overture_buildings_bbox.parquet")
    p.add_argument("--download", action="store_true", help="fetch the needed tiles from Drive first (copies stay in Drive)")
    p.add_argument("--max-negatives", type=int, default=200000)
    args = p.parse_args(argv)

    from src.syria import footprint_tiles as ft

    plan = pd.read_csv(OUT_DIR / "plan.csv", dtype=str).fillna("")
    years = sorted(int(c.split("_w")[1]) for c in plan.columns if c.startswith("orbits_w"))
    tiles_dir = Path(args.tiles_dir)
    for city in args.cities:
        year = args.window_year or CITY_YEAR[city]
        if year not in years:
            sys.exit(f"Window {year} is not in the plan ({years}).")
        pl, pb, nl, nb = read_city(ft, city, years, args)
        if len(nl) > args.max_negatives:
            idx = np.random.default_rng(0).choice(len(nl), args.max_negatives, replace=False)
            nl, nb = nl[idx], nb[idx]
        if args.download:
            from src.syria.check_tiles import download_tiles

            need = sorted(set(tile_ids(ft, pl, pb)) | set(tile_ids(ft, nl, nb)))
            need = [z for z in need if z in set(plan.z13)]
            missing = download_tiles(need, tiles_dir, delete=False)
            print(f"{city}: {len(need) - len(missing)} tiles downloaded, {len(missing)} not in Drive yet")
        pv, _ = values_for(ft, pl, pb, year, years, tiles_dir)
        nv, _ = values_for(ft, nl, nb, year, years, tiles_dir)
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({"lon": np.r_[pl, nl], "lat": np.r_[pb, nb], "damaged": np.r_[np.ones(len(pl)), np.zeros(len(nl))], "p_u8": np.r_[pv, nv]}).to_csv(
            OUT_DIR / f"validation_{city.lower()}.csv", index=False)
        report(city, year, pv[~np.isnan(pv)], nv[~np.isnan(nv)], len(pl), len(nl))
    return 0


if __name__ == "__main__":
    sys.exit(main())
