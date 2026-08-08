"""
Find Mosul model's optimal threshold,
mirroring exactly how Gaza's t=0.67 was selected (fine-grained sweep,
step=0.005, first threshold where precision crosses 90%).

Does not modify any production script. Reuses the same tile-merging and
point-sampling logic as evaluate_pixel_transfer.py (3x3 max window, NaN
preserved), for both:
    - Zero-shot Mosul (Gaza-trained model, evaluated on all of Mosul)
    - Retrained Mosul (trained on west-bank points, evaluated on
      east-bank-only points)

For each, reports:
    - AUC (threshold-independent, already computed)
    - Mosul's own optimal threshold (90% precision target)
    - Precision/recall/F1/balanced accuracy at that optimal threshold
    - The same four metrics at Gaza's borrowed threshold (0.670), for
      direct comparison

Usage:
    python3 src/data/transfer_cities/retrain/mosul_optimal_threshold.py
"""

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
WINDOW_SIZE = 3
GAZA_THRESHOLD = 0.67 * 255
SWEEP_STEP = 0.005
TARGET_PRECISION = 0.90


# Merge probability raster tiles and sample at UNOSAT point locations
def sample_merged_raster(tiles: list, gdf: gpd.GeoDataFrame) -> np.ndarray:
    """Identical to evaluate_pixel_transfer.py's current (NaN-preserving) version."""
    srcs = [rasterio.open(fp) for fp in tiles]
    try:
        merged, transform = merge(srcs)
        merged = merged[0].astype(np.float32)
        merged[merged == 0] = np.nan
        half = WINDOW_SIZE // 2

# Threshold sweep
        results = []
        for geom in gdf.geometry:
            col, row = ~transform * (geom.x, geom.y)
            col, row = int(col), int(row)
            r_start = max(0, row - half)
            r_end = min(merged.shape[0], row + half + 1)
            c_start = max(0, col - half)
            c_end = min(merged.shape[1], col + half + 1)
            patch = merged[r_start:r_end, c_start:c_end]
            val = np.nanmax(patch) if patch.size > 0 else np.nan
            results.append(val)

        return np.array(results, dtype=np.float32)
    finally:
        for src in srcs:
            src.close()


# Load sampled probability scores and ground truth labels
def load_scores(city_id: str) -> tuple[np.ndarray, np.ndarray]:
    """Load continuous scores and true labels for one city config."""
    cfg = TRANSFER_CITIES[city_id]
    conflict_start = cfg["conflict_start"]
    prob_base = TRANSFER_PROB_BASE / city_id

    gdf = gpd.read_file(cfg["unosat_labels"])
    pre_period = cfg["pre_period"]
    post_periods = cfg["post_periods"]
    all_periods = [pre_period] + list(post_periods)

    pred_cols = {}
    window_meta = {}
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
        window_meta[window_str] = end_post

    df_preds = pd.DataFrame(pred_cols)
    col_neg = [w for w in pred_cols if window_meta[w] <= conflict_start]
    col_pos = [w for w in pred_cols if window_meta[w] > conflict_start]

    pos_vals = df_preds[col_pos].values.flatten()
    neg_vals = df_preds[col_neg].values.flatten()
    y_scores = np.concatenate([pos_vals, neg_vals])
    y_true = np.concatenate([np.ones(pos_vals.size), np.zeros(neg_vals.size)])

    mask = ~np.isnan(y_scores)
    return y_scores[mask], y_true[mask]


# Compute precision, recall, F1 and balanced accuracy at a given threshold
def metrics_at_threshold(y_true: np.ndarray, y_scores: np.ndarray, t_scaled: float) -> dict:
    y_pred = (y_scores >= t_scaled).astype(int)
    return {
        "threshold_0_255": round(t_scaled, 2),
        "threshold_0_1": round(t_scaled / 255, 4),
        "precision": round(sk_metrics.precision_score(y_true, y_pred, zero_division=0), 4),
        "recall": round(sk_metrics.recall_score(y_true, y_pred, zero_division=0), 4),
        "f1": round(sk_metrics.f1_score(y_true, y_pred, zero_division=0), 4),
        "balanced_accuracy_AUC": round(
            sk_metrics.roc_auc_score(y_true, y_pred), 4
        ),  # matches Gaza/Dietrich "AUC" convention
        "balanced_accuracy": round(sk_metrics.balanced_accuracy_score(y_true, y_pred), 4),
        "accuracy": round(sk_metrics.accuracy_score(y_true, y_pred), 4),
    }


# Find lowest threshold achieving 90% precision (mirrors Gaza calibration)
def find_optimal_threshold(y_true: np.ndarray, y_scores: np.ndarray) -> float:
    """Fine-grained sweep, step=0.005, first threshold where precision >= 90%."""
    thresholds = np.arange(0.0, 1.001, SWEEP_STEP) * 255
    for t in thresholds:
        y_pred = (y_scores >= t).astype(int)
        if y_pred.sum() == 0:
            continue
        precision = sk_metrics.precision_score(y_true, y_pred, zero_division=0)
        if precision >= TARGET_PRECISION:
            return round(float(t), 2)
    return None


# Print metrics at Mosul optimal and Gaza borrowed thresholds for comparison
def report(label: str, city_id: str):
    print(f"\n{'='*65}")
    print(f"{label}")
    print(f"{'='*65}")

    y_scores, y_true = load_scores(city_id)
    print(f"  Valid samples: {len(y_scores):,}")

    true_auc = sk_metrics.roc_auc_score(y_true, y_scores)
    print(f"  True AUC (threshold-independent): {true_auc:.4f}")

    optimal_t = find_optimal_threshold(y_true, y_scores)
    if optimal_t is None:
        print(f"  Could not reach {TARGET_PRECISION*100:.0f}% precision at any threshold.")
        return

    print(f"\n  Own optimal threshold (90% precision target): {optimal_t/255:.4f}")
    own_metrics = metrics_at_threshold(y_true, y_scores, optimal_t)
    for k, v in own_metrics.items():
        print(f"    {k}: {v}")

    print(f"\n  At Gaza's borrowed threshold (0.67), for comparison:")
    gaza_t_metrics = metrics_at_threshold(y_true, y_scores, GAZA_THRESHOLD)
    for k, v in gaza_t_metrics.items():
        print(f"    {k}: {v}")


# Entry point
if __name__ == "__main__":
    report(
        "MOSUL - RETRAINED (trained on west bank, evaluated on east bank)",
        "MOS_RETRAINED_EAST_ONLY",
    )
