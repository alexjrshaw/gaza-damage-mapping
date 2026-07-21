"""
Mosul zero-shot performance at t=0.44 (retrained's own
optimal threshold, applied here just out of curiosity).
"""

import shutil
import sys

sys.path.insert(0, "/scratch/s1214882/gaza-damage-mapping")

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from pathlib import Path
from rasterio.merge import merge
from sklearn import metrics as sk_metrics
from src.data.transfer_cities.constants_transfer import TRANSFER_CITIES

DATA_PATH = Path("/scratch/s1214882/gaza-damage-mapping/data")
TEMP_DIR = Path("/tmp/zeroshot_at_044_check")
if TEMP_DIR.exists():
    shutil.rmtree(TEMP_DIR)
shutil.copytree(DATA_PATH / "transfer_cities/probability_rasters/MOS", TEMP_DIR)

cfg = TRANSFER_CITIES["MOS_RETRAINED_EAST_ONLY"]
conflict_start = cfg["conflict_start"]
gdf = gpd.read_file(cfg["unosat_labels"])


def sample_merged_raster(tiles, gdf, window=3):
    srcs = [rasterio.open(fp) for fp in tiles]
    try:
        merged, transform = merge(srcs)
        merged = merged[0].astype(np.float32)
        merged[merged == 0] = np.nan
        half = window // 2
        results = []
        for geom in gdf.geometry:
            col, row = ~transform * (geom.x, geom.y)
            col, row = int(col), int(row)
            r0, r1 = max(0, row - half), min(merged.shape[0], row + half + 1)
            c0, c1 = max(0, col - half), min(merged.shape[1], col + half + 1)
            patch = merged[r0:r1, c0:c1]
            results.append(np.nanmax(patch) if patch.size > 0 else np.nan)
        return np.array(results, dtype=np.float32)
    finally:
        for src in srcs:
            src.close()


all_periods = [cfg["pre_period"]] + list(cfg["post_periods"])
pred_cols, window_meta = {}, {}
for i, post_period in enumerate(all_periods):
    window_str = f"w{i+1:02d}_{post_period[0]}_{post_period[1]}"
    window_dir = TEMP_DIR / window_str
    if not window_dir.exists():
        continue
    tiles = sorted(window_dir.glob("qk_*.tif"))
    if not tiles:
        continue
    pred_cols[window_str] = sample_merged_raster(tiles, gdf)
    window_meta[window_str] = post_period[1]

df_preds = pd.DataFrame(pred_cols)
col_neg = [w for w in pred_cols if window_meta[w] <= conflict_start]
col_pos = [w for w in pred_cols if window_meta[w] > conflict_start]
pos_vals = df_preds[col_pos].values.flatten()
neg_vals = df_preds[col_neg].values.flatten()

t_scaled = 0.44 * 255
y_pos = pos_vals[~np.isnan(pos_vals)] >= t_scaled
y_neg = neg_vals[~np.isnan(neg_vals)] >= t_scaled
y_preds = np.concatenate([y_pos, y_neg])
y_trues = np.concatenate([np.ones(y_pos.size), np.zeros(y_neg.size)])

print(f"Zero-shot at t=0.44 (east-bank points):")
print(f"  F1:        {sk_metrics.f1_score(y_trues, y_preds, zero_division=0):.3f}")
print(f"  Precision: {sk_metrics.precision_score(y_trues, y_preds, zero_division=0):.3f}")
print(f"  Recall:    {sk_metrics.recall_score(y_trues, y_preds, zero_division=0):.3f}")
print(f"  'AUC':     {sk_metrics.roc_auc_score(y_trues, y_preds):.3f}")
print(f"  Accuracy:  {sk_metrics.accuracy_score(y_trues, y_preds):.3f}")

shutil.rmtree(TEMP_DIR)
