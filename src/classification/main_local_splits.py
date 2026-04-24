"""
Classification pipeline for alternative train/test split strategies.

Runs full_pipeline_local for both random split strategies and prints
a comparison table against the baseline AOI-based split.

Usage:
    python3 src/classification/main_local_splits.py
"""

import json
from pathlib import Path

import pandas as pd
from omegaconf import OmegaConf

from src.classification.dataset_local import get_dataset_ready_local
from src.classification.main_local import full_pipeline_local
from src.constants import DATA_PATH, PRE_PERIOD, POST_PERIODS, AOIS_TEST

FEATURES_DIR = DATA_PATH / "features_ready"
RUNS_DIR = DATA_PATH / "runs"


def run_split(strategy: str) -> dict:
    """Run full pipeline for a given split strategy."""

    cfg = OmegaConf.create(
        dict(
            aggregation_method="mean",
            run_suffix=strategy, # Makes name unique - added for 20:80 train/test AOI split test
            model_name="random_forest",
            model_kwargs=dict(numberOfTrees=50, minLeafPopulation=3, maxNodes=1e4),
            data=dict(
                s1=dict(subset_bands=None),
                s2=None,
                aois_test=AOIS_TEST,
                damages_to_keep=[1, 2],
                extract_winds="1x1",
                time_periods=dict(pre=PRE_PERIOD, post="2months"),
                split_strategy=strategy,
            ),
            reducer_names=["mean", "stdDev", "median", "min", "max", "skew", "kurtosis"],
            seed=0,
            local_folder=DATA_PATH / "runs",
            train_on_all_data=False,
        )
    )

    print(f"\n{'='*60}")
    print(f"Running split strategy: {strategy}")
    print(f"{'='*60}")
    return full_pipeline_local(cfg, force_recreate=True) #, split_strategy=strategy) commented out for 20:80 train/test AOI split test


if __name__ == "__main__":

    results = {}

    # Load baseline results
    baseline_fp = RUNS_DIR / "rf_s1_2months_50trees_1x1_all7reducers_baseline" / "metrics.json"
    if baseline_fp.exists():
        with open(baseline_fp) as f:
            results["aoi_based"] = json.load(f)
        print("Loaded baseline (aoi_based) results from file.")

    # Run random_all
    for strategy in ["random_all", "random_per_aoi"]:
        train_fp = FEATURES_DIR / f"s1_1x1_2months_train_{strategy}.parquet"
        test_fp  = FEATURES_DIR / f"s1_1x1_2months_test_{strategy}.parquet"
        if not train_fp.exists() or not test_fp.exists():
            print(f"\nFeatures for {strategy} not found — run run_features_splits.sh first.")
            continue
        results[strategy] = run_split(strategy)

    # Print comparison table
    print("\n" + "=" * 70)
    print("COMPARISON TABLE")
    print("=" * 70)
    print(f"{'Strategy':<25} {'F1':>8} {'Precision':>10} {'Recall':>8} {'AUC':>8}")
    print("-" * 70)

    ukraine = {"f1": 0.749, "precision": 0.671, "recall": 0.846, "roc_auc": 0.813}
    print(f"{'Ukraine (Dietrich 2025)':<25} {ukraine['f1']:>8.3f} {ukraine['precision']:>10.3f} {ukraine['recall']:>8.3f} {ukraine['roc_auc']:>8.3f}")
    print("-" * 70)

    for strategy, m in results.items():
        f1   = m.get("f1", float("nan"))
        prec = m.get("precision", float("nan"))
        rec  = m.get("recall", float("nan"))
        auc  = m.get("roc_auc", float("nan"))
        print(f"{strategy:<25} {f1:>8.3f} {prec:>10.3f} {rec:>8.3f} {auc:>8.3f}")

    print("=" * 70)
