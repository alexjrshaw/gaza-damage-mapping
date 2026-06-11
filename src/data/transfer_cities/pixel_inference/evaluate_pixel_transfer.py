"""
Pixel-level evaluation for transfer city proof-of-concept.

Samples probability rasters at UNOSAT point locations using a 3x3
pixel window (max aggregation) — mirrors Dietrich et al. evaluation.ipynb
and Gaza's pixel_postprocessing.py exactly.

Uses direct rasterio sampling rather than xarray for robustness.

Input:
    data/transfer_cities/probability_rasters/{city_id}/{window_str}/
        {city_id}_{window_str}.tif

Output:
    data/transfer_cities/runs/{city_id}/metrics_pixel.json

Usage:
    python3 src/data/transfer_cities/pixel_inference/evaluate_pixel_transfer.py --city RAQ
"""

import json
import argparse
import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from rasterio.windows import Window
from pathlib import Path
from sklearn import metrics as sk_metrics

import sys
sys.path.insert(0, '/scratch/s1214882/gaza-damage-mapping')
from src.data.transfer_cities.constants_transfer import TRANSFER_CITIES
from src.constants import DATA_PATH

TRANSFER_PROB_RASTERS = DATA_PATH / "transfer_cities" / "probability_rasters"
TRANSFER_RUNS_DIR     = DATA_PATH / "transfer_cities" / "runs"
WINDOW_SIZE = 3
WINDOW_AGG  = "max"
THRESHOLDS  = [0.5, 0.650, 0.655]


def sample_raster_3x3(fp: Path, gdf: gpd.GeoDataFrame) -> np.ndarray:
    """
    Sample raster at each UNOSAT point with 3x3 pixel window (max aggregation).
    Mirrors Dietrich et al. evaluation methodology exactly.
    """
    half = WINDOW_SIZE // 2
    results = []
    with rasterio.open(fp) as src:
        for geom in gdf.geometry:
            row, col = src.index(geom.x, geom.y)
            win = Window(
                col_off=max(0, col - half),
                row_off=max(0, row - half),
                width=min(WINDOW_SIZE, src.width - max(0, col - half)),
                height=min(WINDOW_SIZE, src.height - max(0, row - half)),
            )
            patch = src.read(1, window=win).astype(np.float32)
            patch[patch == 0] = np.nan
            val = np.nanmax(patch) if WINDOW_AGG == "max" else np.nanmean(patch)
            results.append(val if not np.isnan(val) else 0.0)
    return np.array(results, dtype=np.float32)


def evaluate_pixel_city(city_id: str) -> dict:
    cfg = TRANSFER_CITIES[city_id]
    conflict_start = cfg["conflict_start"]
    prob_base = TRANSFER_PROB_RASTERS / city_id

    print(f"\n{'='*60}")
    print(f"{city_id} — {cfg['city_name']} ({cfg['country']}) — PIXEL-LEVEL")
    print(f"  3x3 window max aggregation, mirrors Dietrich et al.")
    print(f"{'='*60}")

    if not prob_base.exists():
        print(f"  No probability rasters found at {prob_base}")
        print(f"  Run pixel_inference_transfer.py --city {city_id} first.")
        return {}

    gdf = gpd.read_file(cfg["unosat_labels"])
    print(f"  UNOSAT points: {len(gdf):,}")

    pre_period   = cfg["pre_period"]
    post_periods = cfg["post_periods"]
    all_periods  = [pre_period] + list(post_periods)

    # Sample all rasters
    pred_cols = {}
    for i, post_period in enumerate(all_periods):
        window_str = f"w{i+1:02d}_{post_period[0]}_{post_period[1]}"
        fp = prob_base / window_str / f"{city_id}_{window_str}.tif"
        end_post = post_period[1]
        col = f"pred_{end_post}"

        if not fp.exists():
            print(f"  MISSING: {window_str}")
            continue

        vals = sample_raster_3x3(fp, gdf)
        pred_cols[col] = vals

    if not pred_cols:
        print("  No rasters sampled successfully.")
        return {}

    df_preds = pd.DataFrame(pred_cols)
    col_neg = [c for c in df_preds.columns if c.split("pred_")[1] <= conflict_start]
    col_pos = [c for c in df_preds.columns if c.split("pred_")[1] > conflict_start]

    print(f"  label=0 windows: {len(col_neg)}")
    print(f"  label=1 windows: {len(col_pos)}")

    if not col_pos or not col_neg:
        print("  WARNING: Missing positive or negative windows")
        return {}

    results = {}
    print(f"\n  {'t':>7} {'F1':>7} {'Prec':>7} {'Rec':>7} {'AUC':>7} {'n_pos':>8} {'n_neg':>8}")
    print(f"  {'-'*55}")

    for t in THRESHOLDS:
        t_scaled = t * 255
        y_pos   = (df_preds[col_pos] >= t_scaled).values.flatten()
        y_neg   = (df_preds[col_neg] >= t_scaled).values.flatten()
        y_preds = np.concatenate([y_pos, y_neg])
        y_trues = np.concatenate([np.ones(y_pos.size), np.zeros(y_neg.size)])

        f1   = sk_metrics.f1_score(y_trues, y_preds, zero_division=0)
        prec = sk_metrics.precision_score(y_trues, y_preds, zero_division=0)
        rec  = sk_metrics.recall_score(y_trues, y_preds, zero_division=0)
        auc  = sk_metrics.roc_auc_score(y_trues, y_preds)
        acc  = sk_metrics.accuracy_score(y_trues, y_preds)

        results[f"t{t}"] = {
            "f1": round(f1, 4), "precision": round(prec, 4),
            "recall": round(rec, 4), "roc_auc": round(auc, 4),
            "accuracy": round(acc, 4), "threshold": t,
            "n_pos": int(y_pos.size), "n_neg": int(y_neg.size),
            "window_size": WINDOW_SIZE, "window_agg": WINDOW_AGG,
        }
        print(f"  {t:>7.3f} {f1:>7.3f} {prec:>7.3f} {rec:>7.3f} {auc:>7.3f} {y_pos.size:>8,} {y_neg.size:>8,}")

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
