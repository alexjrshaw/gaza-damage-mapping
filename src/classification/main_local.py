"""
Local training and evaluation pipeline for Gaza damage mapping.

Local equivalent of main.py. Replaces the GEE-based pipeline with
scikit-learn since features are stored locally on Forth.

Mirrors full_pipeline() in main.py exactly:
    1. Train Random Forest on train split (GAZ1+GAZ2)
    2. Predict probabilities on test split (GAZ3+GAZ4+GAZ5)
    3. Format predictions to match get_metrics() input format
    4. Compute metrics using existing metrics.py (unchanged)

The output GeoDataFrame matches the format produced by main.py:
    - index: (unosat_id, aoi)
    - columns: pred_{start_post} (probabilities scaled 0-255)
    - date: date_first_severe for each point

This allows metrics.py to be used without modification, computing
the same date-wise evaluation as Dietrich et al. (2025).

Gaza adaptation: local pandas/sklearn instead of GEE SMILE RF.
"""

import json
import pickle
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from omegaconf import DictConfig, OmegaConf

from src.classification.dataset_local import get_dataset_ready_local
from src.classification.metrics import get_metrics
from src.classification.models_local import (
    classifier_factory_local,
    load_classifier_local,
    save_classifier_local,
)
from src.classification.utils import get_features_names, get_run_name
from src.constants import DATA_PATH, PRE_PERIOD, POST_PERIODS
from src.data.unosat import load_unosat_labels
from src.utils.time import timeit

RUNS_DIR = DATA_PATH / "runs"
RUNS_DIR.mkdir(exist_ok=True, parents=True)


@timeit
def full_pipeline_local(cfg: DictConfig, force_recreate: bool = False) -> dict:
    """
    Full local pipeline from config to metrics.

    Local equivalent of full_pipeline() in main.py.

    1. Classifier is loaded if it exists, otherwise trained and saved.
    2. Test set is classified.
    3. Predictions are formatted to match get_metrics() input.
    4. Metrics are computed using existing metrics.py.

    Args:
        cfg (DictConfig): Config dict (same structure as main.py).
        force_recreate (bool): Retrain even if model already exists.

    Returns:
        dict: Metrics (f1, precision, recall, roc_auc, accuracy).
    """
    run_name = get_run_name(cfg)
    print(f"Running local pipeline for {run_name}")

    run_dir        = RUNS_DIR / run_name
    fp_preds_local = run_dir / f"{run_name}.geojson"
    run_dir.mkdir(exist_ok=True, parents=True)

    if not fp_preds_local.exists() or force_recreate:

        # --- Train or load classifier ---
        clf = load_or_create_classifier_local(cfg, force_recreate)

        # --- Load test features ---
        print("\nLoading test features...")
        df_test = get_dataset_ready_local(
            sat=get_sat_from_cfg_local(cfg),
            split="test",
            post_dates=cfg.data.time_periods["post"],
            extract_wind=cfg.data.extract_winds,
            split_strategy=cfg.data.get("split_strategy", "aoi")
        )

        # --- Get feature names ---
        feature_cols = get_features_names(cfg)
        print(f"Test set: {len(df_test):,} rows, {len(feature_cols)} features")

        # Drop NaN rows
        df_test = df_test.dropna(subset=feature_cols)

        # --- Predict probabilities ---
        print("Classifying test set...")
        X_test = df_test[feature_cols].values
        y_prob = clf.predict_proba(X_test)[:, 1]
        df_test = df_test.copy()
        df_test["prob"] = y_prob

        # --- Format predictions to match main.py output ---
        print("Formatting predictions...")
        gdf = _format_predictions(df_test, cfg)

        # --- Save predictions ---
        fp_preds_local.parent.mkdir(exist_ok=True, parents=True)
        gdf.to_file(fp_preds_local, driver="GeoJSON")
        print(f"Predictions saved to {fp_preds_local}")

    else:
        print(f"Predictions for {run_name} already exist, loading...")
        gdf = gpd.read_file(fp_preds_local).set_index(["unosat_id", "aoi"])

    # --- Compute metrics using existing metrics.py ---
    print("\nComputing metrics...")
    result_metrics = get_metrics(
        gdf,
        threshold=0.5,
        method="date-wise",
        print_classification_report=True,
        only_2022_for_pos=False,
        digits=3,
        return_preds=False,
    )

    # Print comparison with Ukraine baselines
    print("\n── Comparison with Ukraine baselines ──")
    print(f"  F1:        {result_metrics['f1']:.3f}  (Ukraine: 0.749)")
    print(f"  Precision: {result_metrics['precision']:.3f}  (Ukraine: 0.671)")
    print(f"  Recall:    {result_metrics['recall']:.3f}  (Ukraine: 0.846)")
    print(f"  AUC:       {result_metrics['roc_auc']:.3f}  (Ukraine: 0.813)")

    # Save metrics
    fp_metrics = run_dir / "metrics.json"
    with open(fp_metrics, "w") as f:
        json.dump({k: float(v) for k, v in result_metrics.items()}, f, indent=2)
    print(f"Metrics saved to {fp_metrics}")

    return result_metrics


def load_or_create_classifier_local(
    cfg: DictConfig,
    force_recreate: bool = False,
) -> object:
    """
    Load classifier if it exists, otherwise train and save.

    Local equivalent of load_or_create_classifier() in main.py.
    """
    run_name = get_run_name(cfg)
    fp_model = RUNS_DIR / run_name / "model.pkl"

    if fp_model.exists() and not force_recreate:
        print(f"Loading existing classifier from {fp_model}")
        return load_classifier_local(fp_model)

    print("Training classifier...")
    clf = get_classifier_trained_local(cfg, verbose=1)
    save_classifier_local(clf, fp_model)
    return clf


def get_classifier_trained_local(cfg: DictConfig, verbose: int = 1) -> object:
    """
    Train classifier from config.

    Local equivalent of get_classifier_trained() in main.py.
    """
    clf = classifier_factory_local(
        cfg.model_name,
        seed=cfg.seed,
        verbose=verbose,
        **cfg.model_kwargs,
    )

    print("Loading train features...")
    df_train = get_dataset_ready_local(
        sat=get_sat_from_cfg_local(cfg),
        split="train",
        post_dates=cfg.data.time_periods["post"],
        extract_wind=cfg.data.extract_winds,
        split_strategy=cfg.data.get("split_strategy", "aoi")
    )

    feature_cols = get_features_names(cfg)
    if verbose:
        print(f"Train set: {len(df_train):,} rows, {len(feature_cols)} features")
        print(f"Label distribution: {df_train['label'].value_counts().sort_index().to_dict()}")

    df_train = df_train.dropna(subset=feature_cols)
    X_train = df_train[feature_cols].values
    y_train = df_train["label"].values

    print("Fitting Random Forest...")
    clf.fit(X_train, y_train)
    print("Training complete.")
    return clf


def _format_predictions(df_test: pd.DataFrame, cfg: DictConfig) -> gpd.GeoDataFrame:
    """
    Format predictions to match get_metrics() input format.

    Produces a GeoDataFrame matching exactly what main.py produces:
        - index: (unosat_id, aoi)
        - columns: pred_{start_post} (probabilities scaled 0-255)
        - date: date_first_severe (tunosat in Dietrich et al. eq. 1)
        - geometry: point geometry
    """
    # Average over orbits for each (unosat_id, aoi, start_post)
    preds = (
        df_test.groupby(["unosat_id", "aoi", "start_post"])["prob"]
        .mean()
        .mul(255)
        .astype(int)
        .reset_index()
        .rename(columns={"prob": "classification"})
    )

    # Pivot to wide format
    preds_wide = (
        preds.pivot(
            index=["unosat_id", "aoi"],
            columns="start_post",
            values="classification",
        )
        .sort_values(["aoi", "unosat_id"])
    )
    preds_wide.columns = [f"pred_{c}" for c in preds_wide.columns]

    # Join with UNOSAT labels to get date and geometry
    all_labels = load_unosat_labels(
        combine_epoch="first_severe",
        labels_to_keep=[1, 2],
    ).reset_index()

    gdf = preds_wide.join(
        all_labels[["unosat_id", "aoi", "date", "geometry"]]
        .set_index(["unosat_id", "aoi"]),
        on=["unosat_id", "aoi"],
    )
    gdf["date"] = pd.to_datetime(gdf["date"])

    return gpd.GeoDataFrame(gdf, geometry="geometry")


def get_sat_from_cfg_local(cfg: DictConfig) -> str:
    """Get satellite name from config. Mirrors get_sat_from_cfg() in utils.py."""
    s1 = "s1" if cfg.data.s1 else None
    s2 = "s2" if cfg.data.s2 else None
    return "s1_s2" if s1 and s2 else s1 or s2


if __name__ == "__main__":
    from src.constants import AOIS_TEST, DATA_PATH

    cfg = OmegaConf.create(
        dict(
            aggregation_method="mean",
            model_name="random_forest",
            model_kwargs=dict(numberOfTrees=50, minLeafPopulation=3, maxNodes=1e4), # class_weight='balanced') added after baseline results; commented out for 20:80 train/test AOI split test
            data=dict(
                s1=dict(subset_bands=None),
                s2=None,
                aois_test=AOIS_TEST,
                damages_to_keep=[1, 2],
                extract_winds="1x1",
                time_periods=dict(pre=PRE_PERIOD, post="2months"),
                split_strategy="aoi" # Added after baseline results
            ),
            reducer_names=["mean", "stdDev", "median", "min", "max", "skew", "kurtosis"],
            seed=0,
            local_folder=DATA_PATH / "runs",
            train_on_all_data=False,
        )
    )

    result = full_pipeline_local(cfg, force_recreate=False) # force_recreate=False changed to =True after baseline results; changed back to =False for 20:80 train/test AOI split test
