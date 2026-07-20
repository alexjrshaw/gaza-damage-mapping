"""
Pixel-level evaluation for transfer cities.

Merges quadkey probability raster tiles per window, then samples at
UNOSAT point locations with 3x3 pixel window (max aggregation) —
mirrors Dietrich et al. evaluation methodology exactly.

Fixes applied (vs original):
1. Column collision: windows were originally keyed by end-date string,
   so two windows sharing the same end-date (e.g. w02/w08, or w02/w10)
   silently overwrote one another in the predictions dict, discarding
   one window's data entirely. Now keyed by unique window_str instead.
2. NaN coercion: all-NaN patches (e.g. from insufficient Sentinel-1
   temporal density to compute skew/kurtosis reducers) were previously
   coerced to 0.0, which falsely counts as a confident "undamaged"
   prediction and corrupts recall. Now kept as NaN and excluded from
   metric calculation entirely. Windows with <50% valid coverage are
   flagged and reported as "excluded" in the output.

Input:
    data/transfer_cities/probability_rasters/{city_id}/{window_str}/
        qk_{qk_id}.tif  (multiple tiles per window)

Output:
    data/transfer_cities/runs/{city_id}/metrics_pixel.json

Usage:
    python3 src/data/transfer_cities/pixel_inference/evaluate_pixel_transfer.py --city MOS
"""

import argparse
import json
import sys

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio.merge import merge
from sklearn import metrics as sk_metrics

sys.path.insert(0, "/scratch/s1214882/gaza-damage-mapping")

from src.constants import DATA_PATH
from src.data.transfer_cities.constants_transfer import TRANSFER_CITIES

TRANSFER_PROB_BASE = DATA_PATH / "transfer_cities" / "probability_rasters"
TRANSFER_RUNS_DIR = DATA_PATH / "transfer_cities" / "runs"
WINDOW_SIZE = 3
THRESHOLDS = [0.5, 0.655, 0.670]
USABLE_THRESHOLD_PCT = 50.0


def sample_merged_raster(tiles: list, gdf: gpd.GeoDataFrame) -> np.ndarray:
    """
    Merge quadkey tiles and sample at UNOSAT points with 3x3 window.
    Mirrors Dietrich et al. 3x3 max aggregation exactly.
    """
    # Open all tiles
    srcs = [rasterio.open(fp) for fp in tiles]
    try:
        merged, transform = merge(srcs)
        merged = merged[0].astype(np.float32)
        merged[merged == 0] = np.nan

        # Get CRS and transform from first tile
        crs = srcs[0].crs
        half = WINDOW_SIZE // 2

        results = []
        for geom in gdf.geometry:
            # Convert point to pixel coordinates
            col, row = ~transform * (geom.x, geom.y)
            col, row = int(col), int(row)

            # Extract 3x3 patch
            r_start = max(0, row - half)
            r_end = min(merged.shape[0], row + half + 1)
            c_start = max(0, col - half)
            c_end = min(merged.shape[1], col + half + 1)

            patch = merged[r_start:r_end, c_start:c_end]
            val = np.nanmax(patch) if patch.size > 0 else np.nan
            results.append(val)  # keep NaN as missing; do not coerce to 0.0

        return np.array(results, dtype=np.float32)
    finally:
        for src in srcs:
            src.close()


def evaluate_pixel_city(city_id: str) -> dict:
    cfg = TRANSFER_CITIES[city_id]
    conflict_start = cfg["conflict_start"]
    prob_base = TRANSFER_PROB_BASE / city_id

    print(f"\n{'='*60}")
    print(f"{city_id} — {cfg['city_name']} — PIXEL-LEVEL (3x3 max)")
    print(f"{'='*60}")

    if not prob_base.exists():
        print(f"  No probability rasters at {prob_base}")
        return {}

    gdf = gpd.read_file(cfg["unosat_labels"])
    print(f"  UNOSAT points: {len(gdf):,}")

    pre_period = cfg["pre_period"]
    post_periods = cfg["post_periods"]
    all_periods = [pre_period] + list(post_periods)

    pred_cols = {}  # keyed by UNIQUE window_str -- avoids collisions when
    # two windows share the same end-date (e.g. w02/w08)
    window_meta = {}  # window_str -> (end_post, valid_pct)

    for i, post_period in enumerate(all_periods):
        window_str = f"w{i+1:02d}_{post_period[0]}_{post_period[1]}"
        end_post = post_period[1]
        window_dir = prob_base / window_str

        if not window_dir.exists():
            continue

        tiles = sorted(window_dir.glob("qk_*.tif"))
        if not tiles:
            continue

        vals = sample_merged_raster(tiles, gdf)
        pred_cols[window_str] = vals
        valid_pct = (~np.isnan(vals)).sum() / len(vals) * 100
        window_meta[window_str] = (end_post, valid_pct)
        print(
            f"  {window_str}: {len(tiles)} tiles, mean={np.nanmean(vals):.1f}, valid={valid_pct:.1f}%"
        )

    if not pred_cols:
        print("  No probability rasters found.")
        return {}

    usable_windows = {w for w, (_, pct) in window_meta.items() if pct >= USABLE_THRESHOLD_PCT}
    unusable_windows = {w for w, (_, pct) in window_meta.items() if pct < USABLE_THRESHOLD_PCT}

    col_neg = [w for w in pred_cols if window_meta[w][0] <= conflict_start]
    col_pos = [w for w in pred_cols if window_meta[w][0] > conflict_start]

    print(f"\n  label=0 windows: {len(col_neg)} -> {sorted(col_neg)}")
    print(f"  label=1 windows: {len(col_pos)} -> {sorted(col_pos)}")
    print(f"  Usable windows (>={USABLE_THRESHOLD_PCT}% valid): {sorted(usable_windows)}")
    print(
        f"  Excluded windows (<{USABLE_THRESHOLD_PCT}% valid -- insufficient SAR "
        f"temporal density for skew/kurtosis): {sorted(unusable_windows)}"
    )

    if not col_pos or not col_neg:
        print("  WARNING: Missing positive or negative windows")
        return {}

    df_preds = pd.DataFrame({w: pred_cols[w] for w in pred_cols})

    results = {}
    print(f"\n  {'t':>7} {'F1':>7} {'Prec':>7} {'Rec':>7} {'AUC':>7} {'n_pos':>8} {'n_neg':>8}")
    print(f"  {'-'*55}")

    for t in THRESHOLDS:
        t_scaled = t * 255

        pos_vals = df_preds[col_pos].values.flatten()
        neg_vals = df_preds[col_neg].values.flatten()

        pos_valid = ~np.isnan(pos_vals)
        neg_valid = ~np.isnan(neg_vals)

        y_pos = pos_vals[pos_valid] >= t_scaled
        y_neg = neg_vals[neg_valid] >= t_scaled
        y_preds = np.concatenate([y_pos, y_neg])
        y_trues = np.concatenate([np.ones(y_pos.size), np.zeros(y_neg.size)])
        n_excl = (pos_vals.size - y_pos.size) + (neg_vals.size - y_neg.size)

        f1 = sk_metrics.f1_score(y_trues, y_preds, zero_division=0)
        prec = sk_metrics.precision_score(y_trues, y_preds, zero_division=0)
        rec = sk_metrics.recall_score(y_trues, y_preds, zero_division=0)
        auc = sk_metrics.roc_auc_score(y_trues, y_preds) if len(set(y_trues)) > 1 else float("nan")
        acc = sk_metrics.accuracy_score(y_trues, y_preds)

        results[f"t{t}"] = {
            "f1": round(f1, 4),
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "roc_auc": round(auc, 4) if not np.isnan(auc) else None,
            "accuracy": round(acc, 4),
            "threshold": t,
            "n_pos": int(y_pos.size),
            "n_neg": int(y_neg.size),
            "n_excluded_nan": int(n_excl),
            "window_size": WINDOW_SIZE,
            "window_agg": "max",
            "usable_windows": sorted(usable_windows),
            "excluded_windows": sorted(unusable_windows),
        }
        auc_str = f"{auc:>7.3f}" if not np.isnan(auc) else f"{'nan':>7}"
        print(
            f"  {t:>7.3f} {f1:>7.3f} {prec:>7.3f} {rec:>7.3f} "
            f"{auc_str} {y_pos.size:>8,} {y_neg.size:>8,}  (excluded={n_excl:,})"
        )

    # Save
    run_dir = TRANSFER_RUNS_DIR / city_id
    run_dir.mkdir(parents=True, exist_ok=True)
    fp_out = run_dir / "metrics_pixel.json"
    with open(fp_out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Saved -> {fp_out}")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--city", type=str, default="RAQ")
    args = parser.parse_args()
    evaluate_pixel_city(args.city.upper())
