#!/usr/bin/env python3
"""
Independent re-computation of the Gaza-protocol accuracy figures from the raw tiles.

It shares NO code with evaluate_transfer.py / validate_unosat.py / footprint_tiles.py: its own quadkey maths,
its own 3 x 3 maximum, its own precision / recall / F1 / balanced accuracy / AUC (plain numpy, no sklearn).
If its numbers match evaluate_transfer.py's, the figures do not depend on a bug in that code.

  python3 src/syria/independent_check.py --run data/syria/ee_run_pilot --labels PATH/unosat_labels.geojson \
      --neg t01 --pos t03
"""
import argparse
import json
import math
from pathlib import Path

import numpy as np
import rasterio
from rasterio.warp import transform as warp_transform

ZOOM = 13


def quadkey(lon, lat, zoom=ZOOM):
    n = 2 ** zoom
    x = int(math.floor((lon + 180.0) / 360.0 * n))
    lat_r = math.radians(lat)
    y = int(math.floor((1.0 - math.asinh(math.tan(lat_r)) / math.pi) / 2.0 * n))
    x, y = min(max(x, 0), n - 1), min(max(y, 0), n - 1)
    qk = ""
    for i in range(zoom, 0, -1):
        d = 0
        mask = 1 << (i - 1)
        if x & mask:
            d += 1
        if y & mask:
            d += 2
        qk += str(d)
    return qk


def sample_window_max(lons, lats, tiles_dir, band_index):
    """Max of the 3 x 3 pixels around each point in one band; NaN if the tile is missing, the point is outside, or all 9 pixels are 0 (no data)."""
    out = np.full(len(lons), np.nan)
    keys = [quadkey(a, b) for a, b in zip(lons, lats)]
    for qk in sorted(set(keys)):
        f = Path(tiles_dir) / f"qk_{qk}.tif"
        if not f.exists():
            continue
        idx = [i for i, k in enumerate(keys) if k == qk]
        with rasterio.open(f) as src:
            band = src.read(band_index + 1)
            xs, ys = [lons[i] for i in idx], [lats[i] for i in idx]
            if src.crs is not None and src.crs.to_epsg() != 4326:
                xs, ys = warp_transform("EPSG:4326", src.crs, xs, ys)
            h, w = band.shape
            for j, i in enumerate(idx):
                r, c = src.index(xs[j], ys[j])
                if r < 0 or c < 0 or r >= h or c >= w:
                    continue
                win = band[max(r - 1, 0): r + 2, max(c - 1, 0): c + 2]
                if win.size and win.max() > 0:
                    out[i] = float(win.max())
    return out


def auc_rank(pos, neg):
    """P(random positive scores above random negative), ties count half."""
    allv = np.concatenate([pos, neg])
    order = allv.argsort(kind="mergesort")
    ranks = np.empty(len(allv))
    sv = allv[order]
    i = 0
    while i < len(sv):
        j = i
        while j + 1 < len(sv) and sv[j + 1] == sv[i]:
            j += 1
        ranks[order[i: j + 1]] = (i + j) / 2.0 + 1
        i = j + 1
    return (ranks[: len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))


def at_threshold(pos, neg, t):
    cut = t * 255.0
    tp, fn = int((pos >= cut).sum()), int((pos < cut).sum())
    fp, tn = int((neg >= cut).sum()), int((neg < cut).sum())
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    bal = (rec + tn / (tn + fp)) / 2 if tn + fp else float("nan")
    return dict(t=t, precision=prec, recall=rec, f1=f1, balanced_acc=bal, tp=tp, fp=fp, fn=fn, tn=tn)


def read_points(path):
    """Longitude / latitude of every Point in the labels file, reprojected to EPSG:4326 if the file says it is in another CRS."""
    path = Path(path)
    if not path.is_file():
        raise SystemExit(f"Labels file not found: '{path}'. Find it with the command in the instructions.")
    try:
        import geopandas as gpd

        g = gpd.read_file(path)
        if g.crs is not None:
            g = g.to_crs(4326)
        g = g[g.geometry.geom_type == "Point"]
        return g.geometry.x.tolist(), g.geometry.y.tolist()
    except ImportError:
        gj = json.loads(path.read_text())
        if "crs" in gj and "4326" not in json.dumps(gj["crs"]) and "CRS84" not in json.dumps(gj["crs"]):
            raise SystemExit("The labels file declares a CRS other than EPSG:4326 and geopandas is not installed.")
        lons, lats = [], []
        for ft in gj["features"]:
            g = ft["geometry"]
            if g and g["type"] == "Point":
                lons.append(g["coordinates"][0])
                lats.append(g["coordinates"][1])
        return lons, lats


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--run", required=True, help="run folder, e.g. data/syria/ee_run_pilot")
    p.add_argument("--labels", required=True, help="the city's UNOSAT labels (GeoJSON of points, EPSG:4326)")
    p.add_argument("--neg", required=True, help="label-0 window name, e.g. t01")
    p.add_argument("--pos", required=True, help="label-1 window name, e.g. t03")
    a = p.parse_args(argv)

    run = Path(a.run)
    names = [w["name"] for w in json.loads((run / "windows.json").read_text())["windows"]]
    lons, lats = read_points(a.labels)
    print(f"{len(lons):,} UNOSAT points; window order in the tiles: {names}")
    print(f"band index of {a.neg} = {names.index(a.neg) + 1}, of {a.pos} = {names.index(a.pos) + 1} (1-based)")
    nv = sample_window_max(lons, lats, run / "tiles", names.index(a.neg))
    pv = sample_window_max(lons, lats, run / "tiles", names.index(a.pos))
    print(f"valid values: {a.neg} {np.isfinite(nv).sum():,}, {a.pos} {np.isfinite(pv).sum():,}; mean {a.neg} {np.nanmean(nv):.1f}, {a.pos} {np.nanmean(pv):.1f}")
    nv, pv = nv[np.isfinite(nv)], pv[np.isfinite(pv)]
    print(f"AUC (continuous): {auc_rank(pv, nv):.3f}   n_pos {len(pv):,}  n_neg {len(nv):,}\n")
    print(f"{'t':>7}{'cut/255':>9}{'F1':>8}{'Prec':>8}{'Rec':>8}{'BalAcc':>8}")
    for t in (0.5, 0.670):
        r = at_threshold(pv, nv, t)
        print(f"{t:>7.3f}{t * 255:>9.2f}{r['f1']:>8.3f}{r['precision']:>8.3f}{r['recall']:>8.3f}{r['balanced_acc']:>8.3f}")
    for t in np.arange(0.0, 1.0005, 0.005):
        r = at_threshold(pv, nv, float(t))
        if r["precision"] >= 0.90 and (pv >= t * 255).any():
            print(f"{t:>7.3f}{t * 255:>9.2f}{r['f1']:>8.3f}{r['precision']:>8.3f}{r['recall']:>8.3f}{r['balanced_acc']:>8.3f}   <- lowest threshold with precision >= 90% (step 0.005)")
            break
    print("\n(precision/recall/F1 use value >= t x 255, the same cut as evaluate_transfer.py: 0.5 -> 127.5, 0.670 -> 170.85)")


if __name__ == "__main__":
    main()
