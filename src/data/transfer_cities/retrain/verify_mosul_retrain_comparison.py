"""
Full re-verification of the Mosul zero-shot vs retrained comparison.
Reruns all three pieces fresh and explicitly checks each against the
previously saved/reported figures.
"""

import json
import shutil
import subprocess
import sys
from pathlib import Path

DATA_PATH = Path("/scratch/s1214882/gaza-damage-mapping/data")
sys.path.insert(0, "/scratch/s1214882/gaza-damage-mapping")

EXPECTED = {
    "retrained_t0.5": {
        "f1": 0.363,
        "precision": 0.931,
        "recall": 0.226,
        "roc_auc": 0.608,
        "accuracy": 0.718,
    },
    "retrained_t0.67": {
        "f1": 0.181,
        "precision": 0.973,
        "recall": 0.100,
        "roc_auc": 0.549,
        "accuracy": 0.678,
    },
}


def check(label, actual, expected, tol=0.005):
    ok = all(abs(actual[k] - expected[k]) < tol for k in expected)
    status = "MATCH" if ok else "MISMATCH"
    print(f"  [{status}] {label}: {actual}")
    if not ok:
        print(f"    Expected: {expected}")
    return ok


print("=" * 70)
print("STEP 1: Rerun evaluate_pixel_transfer.py fresh on the REAL retrained rasters")
print("=" * 70)
result = subprocess.run(
    [
        "python3",
        "src/data/transfer_cities/pixel_inference/evaluate_pixel_transfer.py",
        "--city",
        "MOS_RETRAINED_EAST_ONLY",
    ],
    cwd="/scratch/s1214882/gaza-damage-mapping",
    capture_output=True,
    text=True,
)
print(result.stdout[-1500:])

with open(DATA_PATH / "transfer_cities/runs/MOS_RETRAINED_EAST_ONLY/metrics_pixel.json") as f:
    m_fresh = json.load(f)

all_ok = True
all_ok &= check("Retrained t=0.5 (fresh)", m_fresh["t0.5"], EXPECTED["retrained_t0.5"])
all_ok &= check("Retrained t=0.67 (fresh)", m_fresh["t0.67"], EXPECTED["retrained_t0.67"])

print("\n" + "=" * 70)
print("STEP 2: Rerun zero-shot-on-east-bank fresh, via safe temp copy")
print("=" * 70)
import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio.merge import merge
from sklearn import metrics as sk_metrics
from src.data.transfer_cities.constants_transfer import TRANSFER_CITIES

TEMP_DIR = Path("/tmp/verify_zeroshot_eastbank")
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

EXPECTED_ZS = {
    0.5: {"f1": 0.646, "precision": 0.526, "recall": 0.835, "roc_auc": 0.709, "accuracy": 0.673},
    0.67: {"f1": 0.479, "precision": 0.766, "recall": 0.349, "roc_auc": 0.645, "accuracy": 0.729},
}
for t in [0.5, 0.67]:
    t_scaled = t * 255
    y_pos = pos_vals[~np.isnan(pos_vals)] >= t_scaled
    y_neg = neg_vals[~np.isnan(neg_vals)] >= t_scaled
    y_preds = np.concatenate([y_pos, y_neg])
    y_trues = np.concatenate([np.ones(y_pos.size), np.zeros(y_neg.size)])
    actual = {
        "f1": sk_metrics.f1_score(y_trues, y_preds, zero_division=0),
        "precision": sk_metrics.precision_score(y_trues, y_preds, zero_division=0),
        "recall": sk_metrics.recall_score(y_trues, y_preds, zero_division=0),
        "roc_auc": sk_metrics.roc_auc_score(y_trues, y_preds),
        "accuracy": sk_metrics.accuracy_score(y_trues, y_preds),
    }
    all_ok &= check(f"Zero-shot t={t} (fresh)", actual, EXPECTED_ZS[t])

shutil.rmtree(TEMP_DIR)

print("\n" + "=" * 70)
print("STEP 3: Re-verify retrained's own optimal threshold (0.44)")
print("=" * 70)
prob_base = DATA_PATH / "transfer_cities/probability_rasters/MOS_RETRAINED_EAST_ONLY"
pred_cols2, window_meta2 = {}, {}
for i, post_period in enumerate(all_periods):
    window_str = f"w{i+1:02d}_{post_period[0]}_{post_period[1]}"
    window_dir = prob_base / window_str
    if not window_dir.exists():
        continue
    tiles = sorted(window_dir.glob("qk_*.tif"))
    if not tiles:
        continue
    pred_cols2[window_str] = sample_merged_raster(tiles, gdf)
    window_meta2[window_str] = post_period[1]

df_preds2 = pd.DataFrame(pred_cols2)
col_neg2 = [w for w in pred_cols2 if window_meta2[w] <= conflict_start]
col_pos2 = [w for w in pred_cols2 if window_meta2[w] > conflict_start]
pos_vals2 = df_preds2[col_pos2].values.flatten()
neg_vals2 = df_preds2[col_neg2].values.flatten()
y_scores = np.concatenate([pos_vals2, neg_vals2])
y_true = np.concatenate(
    [np.ones(pos_vals2[~np.isnan(pos_vals2)].size), np.zeros(neg_vals2[~np.isnan(neg_vals2)].size)]
)
y_scores_valid = y_scores[~np.isnan(y_scores)]

optimal_t = None
for t in np.arange(0.0, 1.001, 0.005) * 255:
    y_pred = (y_scores_valid >= t).astype(int)
    if y_pred.sum() == 0:
        continue
    precision = sk_metrics.precision_score(y_true, y_pred, zero_division=0)
    if precision >= 0.90:
        optimal_t = t
        break

if optimal_t is not None:
    y_pred = (y_scores_valid >= optimal_t).astype(int)
    actual_opt = {
        "f1": sk_metrics.f1_score(y_true, y_pred, zero_division=0),
        "precision": sk_metrics.precision_score(y_true, y_pred, zero_division=0),
        "recall": sk_metrics.recall_score(y_true, y_pred, zero_division=0),
        "roc_auc": sk_metrics.roc_auc_score(y_true, y_pred),
        "accuracy": sk_metrics.accuracy_score(y_true, y_pred),
    }
    print(f"  Own optimal threshold found: {optimal_t/255:.4f}")
    all_ok &= check(
        "Retrained own-optimal (fresh)",
        actual_opt,
        {"f1": 0.451, "precision": 0.901, "recall": 0.301, "roc_auc": 0.641, "accuracy": 0.7386},
    )

print("\n" + "=" * 70)
print(
    f"OVERALL: {'ALL FIGURES VERIFIED CORRECT' if all_ok else 'SOME FIGURES DO NOT MATCH - investigate'}"
)
print("=" * 70)
