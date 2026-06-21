"""
Zero-shot transfer evaluation for Aleppo, Raqqa, and Mosul.

Evaluates the Gaza-trained model's predictions against UNOSAT labels
for each transfer city.

Evaluation approach for single-epoch assessments:
    Due to single assessment dates per city, only one label=1 window
    exists per city (the window ending on/after the assessment date).
    Evaluation therefore compares:
        Positive: predictions at the label=1 window (post-conflict)
        Negative: predictions at all label=0 windows (pre-conflict)
    This mirrors the Dietrich et al. approach but adapted for single-epoch data.

Thresholds: t=0.5 (primary) and t=0.655 (Dietrich et al. optimal)

Output:
    data/transfer_cities/runs/{city_id}/metrics.json
    data/transfer_cities/runs/transfer_results_summary.csv

Usage:
    cd /scratch/s1214882/gaza-damage-mapping
    python3 src/data/transfer_cities/evaluate_transfer.py
"""

import json
import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from sklearn import metrics as sk_metrics

sys.path.insert(0, "/scratch/s1214882/gaza-damage-mapping")

from src.data.transfer_cities.constants_transfer import TRANSFER_CITIES, TRANSFER_RUNS_DIR

THRESHOLDS = [0.5, 0.655, 0.675]

# Baselines for comparison table
GAZA_RESULTS = {
    "f1": 0.826,
    "precision": 0.787,
    "recall": 0.870,
    "roc_auc": 0.834,
    "accuracy": 0.831,
}
UKRAINE_RESULTS = {
    "f1": 0.749,
    "precision": 0.671,
    "recall": 0.846,
    "roc_auc": 0.813,
    "accuracy": 0.803,
}


def evaluate_city(city_id: str, cfg: dict, threshold: float = 0.5) -> dict:
    """
    Evaluate transfer city predictions against UNOSAT labels.

    For single-epoch cities: uses label=1 window (post-conflict) vs
    all label=0 windows (pre-conflict) for positive/negative examples.

    Each UNOSAT point contributes:
        - One positive prediction from the label=1 window
        - One negative prediction per label=0 window
    """
    fp = TRANSFER_RUNS_DIR / city_id / f"{city_id}_predictions.geojson"
    assert fp.exists(), f"Predictions not found: {fp}"

    gdf = gpd.read_file(fp)
    conflict_start = cfg["conflict_start"]
    threshold_scaled = threshold * 255

    pred_cols = [c for c in gdf.columns if c.startswith("pred_")]
    col_neg = [c for c in pred_cols if c.split("pred_")[1] <= conflict_start]
    col_pos = [c for c in pred_cols if c.split("pred_")[1] > conflict_start]

    print(f"  label=0 windows: {len(col_neg)}")
    print(f"  label=1 windows: {len(col_pos)}")

    if not col_pos:
        print(f"  WARNING: No label=1 prediction columns found for {city_id}")
        return {}
    if not col_neg:
        print(f"  WARNING: No label=0 prediction columns found for {city_id}")
        return {}

    # Positive examples: predictions at label=1 windows
    y_pred_pos = (gdf[col_pos] >= threshold_scaled).astype(int).values.flatten()
    y_true_pos = np.ones(y_pred_pos.size)

    # Negative examples: predictions at label=0 windows
    y_pred_neg = (gdf[col_neg] >= threshold_scaled).astype(int).values.flatten()
    y_true_neg = np.zeros(y_pred_neg.size)

    y_preds = np.concatenate([y_pred_pos, y_pred_neg])
    y_trues = np.concatenate([y_true_pos, y_true_neg])

    precision = sk_metrics.precision_score(y_trues, y_preds, zero_division=0)
    recall = sk_metrics.recall_score(y_trues, y_preds, zero_division=0)
    f1 = sk_metrics.f1_score(y_trues, y_preds, zero_division=0)
    accuracy = sk_metrics.accuracy_score(y_trues, y_preds)
    roc_auc = sk_metrics.roc_auc_score(y_trues, y_preds)

    return {
        "f1": round(f1, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "roc_auc": round(roc_auc, 4),
        "accuracy": round(accuracy, 4),
        "threshold": threshold,
        "n_pos": int(y_true_pos.size),
        "n_neg": int(y_true_neg.size),
    }


def evaluate_all() -> None:
    all_results = {}

    for city_id, cfg in TRANSFER_CITIES.items():
        print(f"\n{'='*60}")
        print(f"{city_id} — {cfg['city_name']} ({cfg['country']})")
        print(f"{'='*60}")

        city_results = {}
        for t in THRESHOLDS:
            m = evaluate_city(city_id, cfg, threshold=t)
            if m:
                city_results[f"t{t}"] = m
                print(
                    f"  t={t:.3f}: F1={m['f1']:.3f}  P={m['precision']:.3f}"
                    f"  R={m['recall']:.3f}  AUC={m['roc_auc']:.3f}"
                    f"  (n_pos={m['n_pos']:,}  n_neg={m['n_neg']:,})"
                )

        all_results[city_id] = city_results

        run_dir = TRANSFER_RUNS_DIR / city_id
        run_dir.mkdir(parents=True, exist_ok=True)
        with open(run_dir / "metrics.json", "w") as f:
            json.dump(city_results, f, indent=2)
        print(f"  Saved -> {run_dir / 'metrics.json'}")

    # Comparison table at t=0.5
    print(f"\n{'='*70}")
    print("ZERO-SHOT TRANSFER RESULTS (t=0.5)")
    print(f"{'='*70}")
    print(f"{'Model/City':<28} {'F1':>7} {'Prec':>7} {'Recall':>7} {'AUC':>7}")
    print("-" * 60)
    print(
        f"{'Ukraine (Dietrich 2025)':<28} "
        f"{UKRAINE_RESULTS['f1']:>7.3f} {UKRAINE_RESULTS['precision']:>7.3f} "
        f"{UKRAINE_RESULTS['recall']:>7.3f} {UKRAINE_RESULTS['roc_auc']:>7.3f}"
    )
    print(
        f"{'Gaza (AOI split, trained)':<28} "
        f"{GAZA_RESULTS['f1']:>7.3f} {GAZA_RESULTS['precision']:>7.3f} "
        f"{GAZA_RESULTS['recall']:>7.3f} {GAZA_RESULTS['roc_auc']:>7.3f}"
    )
    print("-" * 60)
    for city_id, cfg in TRANSFER_CITIES.items():
        if city_id in all_results and "t0.5" in all_results[city_id]:
            m = all_results[city_id]["t0.5"]
            label = f"{cfg['city_name']} (zero-shot)"
            print(f"  {label:<26} {m['f1']:>7.3f} {m['precision']:>7.3f} " f"{m['recall']:>7.3f} {m['roc_auc']:>7.3f}")
    print(f"{'='*70}")

    # Save summary CSV
    rows = []
    for city_id, city_results in all_results.items():
        cfg = TRANSFER_CITIES[city_id]
        for t_key, m in city_results.items():
            rows.append(
                {
                    "city": city_id,
                    "city_name": cfg["city_name"],
                    "country": cfg["country"],
                    **m,
                }
            )
    df_summary = pd.DataFrame(rows)
    fp_summary = TRANSFER_RUNS_DIR / "transfer_results_summary.csv"
    df_summary.to_csv(fp_summary, index=False)
    print(f"\nSummary saved -> {fp_summary}")


if __name__ == "__main__":
    evaluate_all()
