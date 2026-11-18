#!/usr/bin/env python3
"""
Pixel-level evaluation of a region run, done exactly like the local transfer workflow
(src/data/transfer_cities/pixel_inference/evaluate_pixel_transfer.py):

  * the UNOSAT labelled points of the city are sampled in every window (3 x 3 pixels, maximum);
  * a window that ends on or before the conflict start counts as "undamaged" (label 0), a window
    after it as "damaged" (label 1), so the same points serve as their own negatives;
  * no-data pixels are left out, and windows with under 50% valid points are excluded;
  * precision, recall, F1 and AUC are reported at 0.5 and 0.670 (Gaza's threshold).

The difference is where the probabilities come from: the Earth Engine region run (per-tile orbits,
all tiles of the region) instead of the local rasters. If the numbers agree with the earlier local
result (data/transfer_cities/runs/<city>/metrics_pixel.json), running in Earth Engine over a region
changes nothing. It also shows the threshold chosen the way Gaza's 0.670 was (lowest threshold with 90% precision, sweep step 0.005), and the best by F1 and by balanced accuracy (AUCbin). F1 favours flagging everything when
about half the samples are positive, so the balanced-accuracy row is the one to read for a threshold.

  python3 src/syria/evaluate_transfer.py --city ALP --tag alp --download
  python3 src/syria/evaluate_transfer.py --city ALP --tag alp           # tiles already downloaded

Writes data/syria/ee_run_<tag>/metrics_pixel_<city>.json.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

THRESHOLDS = [0.5, 0.670]
USABLE_PCT = 50.0
TASK_ZOOM = 13


def metrics_at(y_true, y_score, t_u8):
    from sklearn import metrics as sk

    y_pred = y_score >= t_u8
    out = dict(
        f1=float(sk.f1_score(y_true, y_pred, zero_division=0)),
        precision=float(sk.precision_score(y_true, y_pred, zero_division=0)),
        recall=float(sk.recall_score(y_true, y_pred, zero_division=0)),
        accuracy=float(sk.accuracy_score(y_true, y_pred)),
    )
    two = len(set(y_true)) > 1
    out["auc_binary"] = float(sk.roc_auc_score(y_true, y_pred)) if two else float("nan")  # what the local script calls AUC
    out["auc_continuous"] = float(sk.roc_auc_score(y_true, y_score)) if two else float("nan")
    return out


def best_threshold(y_true, y_score, key="f1", grid=np.arange(0.30, 0.91, 0.01)):
    """Threshold on the grid that maximises key: 'f1', or 'auc_binary' (= balanced accuracy at that threshold)."""
    best = None
    for t in grid:
        m = metrics_at(y_true, y_score, t * 255)
        if best is None or m[key] > best[1][key]:
            best = (float(t), m)
    return best


def threshold_for_precision(y_true, y_score, target=0.90, step=0.005):
    """
    The way Gaza's t = 0.670 and Mosul's own threshold were chosen (threshold_sweep.py, mosul_optimal_threshold.py):
    sweep 0 to 1 in steps of 0.005 and take the LOWEST threshold at which precision reaches the target.
    Returns (threshold, metrics) or None if no threshold on the sweep reaches it.
    """
    for t in np.arange(0.0, 1.0 + step / 2, step):
        m = metrics_at(y_true, y_score, t * 255)
        if m["precision"] >= target and (y_score >= t * 255).any():
            return float(t), m
    return None


def select_windows(names, ends, until):
    """Windows that end on or before `until` (YYYY-MM-DD); all of them when until is empty."""
    return [n for n in names if not until or ends[n] <= until]


def split_windows(names, ends, conflict_start):
    neg = [n for n in names if ends[n] <= conflict_start]
    pos = [n for n in names if ends[n] > conflict_start]
    return neg, pos


def evaluate(df, ends, conflict_start):
    """df: one column per window (0-255, NaN = no data), one row per UNOSAT point. Returns (results, notes)."""
    valid = {w: float(df[w].notna().mean() * 100) for w in df.columns}
    usable = [w for w in df.columns if valid[w] >= USABLE_PCT]
    neg, pos = split_windows(list(df.columns), ends, conflict_start)
    neg_u, pos_u = [w for w in neg if w in usable], [w for w in pos if w in usable]
    notes = dict(valid=valid, usable=usable, excluded=[w for w in df.columns if w not in usable], label0=neg_u, label1=pos_u)
    if not neg_u or not pos_u:
        return {}, notes
    pv, nv = df[pos_u].to_numpy().ravel(), df[neg_u].to_numpy().ravel()
    pv, nv = pv[~np.isnan(pv)], nv[~np.isnan(nv)]
    y_true = np.r_[np.ones(len(pv)), np.zeros(len(nv))]
    y_score = np.r_[pv, nv]
    res = {}
    for t in THRESHOLDS:
        res[f"t{t}"] = dict(threshold=t, n_pos=int(len(pv)), n_neg=int(len(nv)), **metrics_at(y_true, y_score, t * 255))
    bt, bm = best_threshold(y_true, y_score)
    res["best_f1"] = dict(threshold=round(bt, 2), n_pos=int(len(pv)), n_neg=int(len(nv)), **bm)
    hit = threshold_for_precision(y_true, y_score)
    if hit is not None:
        res["prec90"] = dict(threshold=round(hit[0], 3), n_pos=int(len(pv)), n_neg=int(len(nv)), **hit[1])
    bt, bm = best_threshold(y_true, y_score, key="auc_binary")
    res["best_balacc"] = dict(threshold=round(bt, 2), n_pos=int(len(pv)), n_neg=int(len(nv)), **bm)
    return res, notes


def load_points(cfg):
    import geopandas as gpd

    gdf = gpd.read_file(cfg["unosat_labels"])
    if gdf.crs is not None:
        gdf = gdf.to_crs(4326)
    return gdf.geometry.x.to_numpy(), gdf.geometry.y.to_numpy()


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.strip().split("\n\n")[0])
    p.add_argument("--city", required=True)
    p.add_argument("--tag", default="", help="the tagged run to evaluate")
    p.add_argument("--download", action="store_true", help="fetch the needed tiles from Drive first (copies stay in Drive)")
    p.add_argument("--until", default="", help="leave out windows ending after this date (YYYY-MM-DD), or 'assessment' for the city's UNOSAT date. "
                   "The transfer-city runs stopped at the assessment date; annual windows run for years beyond it.")
    p.add_argument("--windows", nargs="+", metavar="WINDOW", help="score only these windows, e.g. t01 t03 (t01 = label 0, t03 = the first window wholly after the damage); applied after --until")
    args = p.parse_args(argv)

    from src.data.transfer_cities.constants_transfer import TRANSFER_CITIES
    from src.syria import check_tiles, footprint_tiles as ft
    from src.syria.validate_unosat import values_for, tile_ids

    city = args.city.upper()
    tag = args.tag.strip().lower()
    out_dir = Path("data/syria/ee_run" + (f"_{tag}" if tag else ""))
    meta = json.loads((out_dir / "windows.json").read_text())
    names = [w["name"] for w in meta["windows"]]
    ends = {w["name"]: w["end"] for w in meta["windows"]}
    cfg = TRANSFER_CITIES[city]
    conflict_start = str(cfg["conflict_start"])
    until = str(cfg["assessment_date"]) if args.until == "assessment" else args.until
    names = select_windows(names, ends, until)
    if args.windows:
        unknown = [w for w in args.windows if w not in names]
        if unknown:
            sys.exit(f"Window(s) {unknown} are not available (after --until the windows are {names}).")
        names = [w for w in names if w in args.windows]
    if not names:
        sys.exit(f"No window ends on or before {until}.")
    plan = pd.read_csv(out_dir / "plan.csv", dtype=str)

    lon, lat = load_points(cfg)
    print(f"{city}: {len(lon):,} UNOSAT points, {len(names)} windows, conflict start {conflict_start}" + (f", windows ending after {until} left out" if until else ""))
    ids = tile_ids(ft, lon, lat)
    need = sorted(set(ids) & set(plan.z13))
    outside = int((~np.isin(ids, plan.z13)).sum())
    if outside:
        print(f"  {outside:,} points lie in tiles that are not in this run's plan and are left out")
    tiles_dir = out_dir / "tiles"
    if args.download:
        check_tiles.DRIVE_FOLDER = "syria_run" + (f"_{tag}" if tag else "")
        missing = check_tiles.download_tiles(need, tiles_dir, delete=False)
        print(f"  {len(need) - len(missing)} tiles downloaded, {len(missing)} not in Drive yet")

    cols = {}
    for w in names:
        vals, _ = values_for(ft, lon, lat, w, names, tiles_dir)
        cols[w] = vals
    df = pd.DataFrame(cols)
    res, notes = evaluate(df, ends, conflict_start)
    print(f"\n  {'window':<8}{'ends':<12}{'valid':>8}{'mean':>8}  label")
    for w in names:
        label = "0" if ends[w] <= conflict_start else "1"
        mean = float(np.nanmean(df[w])) if df[w].notna().any() else float("nan")
        print(f"  {w:<8}{ends[w]:<12}{notes['valid'][w]:>7.1f}%{mean:>8.1f}  {label}{'   (excluded)' if w in notes['excluded'] else ''}")
    if not res:
        print("\n  Missing usable windows of one class (are all the tasks finished and downloaded?).")
        return 1

    local_f = Path("data/transfer_cities/runs") / city / "metrics_pixel.json"
    local = json.loads(local_f.read_text()) if local_f.exists() else {}
    print(f"\n  {'':<14}{'t':>7}{'F1':>8}{'Prec':>8}{'Rec':>8}{'AUCbin':>8}{'AUCcont':>9}{'n_pos':>9}{'n_neg':>9}")
    for key, r in res.items():
        tag_ = {"best_f1": "best F1", "best_balacc": "best bal.acc", "prec90": "90% precision"}.get(key, "Syria run")
        print(f"  {tag_:<14}{r['threshold']:>7.3f}{r['f1']:>8.3f}{r['precision']:>8.3f}{r['recall']:>8.3f}{r['auc_binary']:>8.3f}{r['auc_continuous']:>9.3f}{r['n_pos']:>9,}{r['n_neg']:>9,}")
        if key in local:
            lr = local[key]
            print(f"  {'local run':<14}{lr['threshold']:>7.3f}{lr['f1']:>8.3f}{lr['precision']:>8.3f}{lr['recall']:>8.3f}{(lr.get('roc_auc') or float('nan')):>8.3f}{'':>9}{lr['n_pos']:>9,}{lr['n_neg']:>9,}")
    out = out_dir / f"metrics_pixel_{city}.json"
    out.write_text(json.dumps(dict(results=res, notes=notes), indent=2))
    print(f"\n  Saved {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
