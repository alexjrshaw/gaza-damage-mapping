"""
Predict on train AOIs (GAZ1, GAZ2) using the existing trained model,
then merge with test predictions to get full Gaza coverage.

This mirrors Dietrich et al.'s approach where full inference covers
all AOIs — not just the test split. Without this, North Gaza and
Gaza City buildings are missing from the postprocessing output.

Output:
    data/runs/rf_s1_2months_50trees_1x1_all7reducers_baseline/
        rf_s1_2months_50trees_1x1_all7reducers_baseline_all_aois.geojson
"""

import pickle

import geopandas as gpd
import pandas as pd
from omegaconf import OmegaConf

from src.classification.dataset_local import get_dataset_ready_local
from src.classification.main_local import _format_predictions
from src.classification.utils import get_features_names
from src.constants import DATA_PATH, PRE_PERIOD, AOIS_TRAIN

RUNS_DIR = DATA_PATH / "runs"
RUN_NAME = "rf_s1_2months_50trees_1x1_all7reducers_baseline"


def predict_train_aois() -> gpd.GeoDataFrame:
    """
    Load existing trained model and predict on GAZ1+GAZ2 train features.
    Returns GeoDataFrame in same format as test predictions.
    """
    cfg = OmegaConf.create(
        dict(
            aggregation_method="mean",
            model_name="random_forest",
            model_kwargs=dict(numberOfTrees=50, minLeafPopulation=3, maxNodes=1e4),
            data=dict(
                s1=dict(subset_bands=None),
                s2=None,
                aois_test=AOIS_TRAIN,
                damages_to_keep=[1, 2],
                extract_winds="1x1",
                time_periods=dict(pre=PRE_PERIOD, post="2months"),
                split_strategy="aoi",
            ),
            reducer_names=["mean", "stdDev", "median", "min", "max", "skew", "kurtosis"],
            seed=0,
            local_folder=DATA_PATH / "runs",
            train_on_all_data=False,
        )
    )

    # Load existing trained model
    fp_model = RUNS_DIR / RUN_NAME / "model.pkl"
    print(f"Loading model from {fp_model}...")
    with open(fp_model, "rb") as f:
        clf = pickle.load(f)
    print("Model loaded.")

    # Load train features
    print("\nLoading train features (GAZ1+GAZ2)...")
    df_train = get_dataset_ready_local(
        sat="s1",
        split="train",
        post_dates="2months",
        extract_wind="1x1",
        split_strategy="aoi",
    )
    print(f"  Loaded {len(df_train):,} rows")

    # Get feature columns
    feature_cols = get_features_names(cfg)
    df_train = df_train.dropna(subset=feature_cols)
    print(f"  After dropping NaN: {len(df_train):,} rows")

    # Predict
    print("\nClassifying train AOIs...")
    X = df_train[feature_cols].values
    y_prob = clf.predict_proba(X)[:, 1]
    df_train = df_train.copy()
    df_train["prob"] = y_prob

    # Format to match test predictions
    print("Formatting predictions...")
    gdf_train = _format_predictions(df_train, cfg)
    print(f"  Train predictions: {len(gdf_train):,} points")

    return gdf_train


def merge_and_save(gdf_train: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    Load existing test predictions, merge with train predictions,
    save combined GeoJSON.
    """
    fp_test = RUNS_DIR / RUN_NAME / f"{RUN_NAME}.geojson"
    print(f"\nLoading test predictions from {fp_test}...")
    gdf_test = gpd.read_file(fp_test).set_index(["unosat_id", "aoi"])
    print(f"  Test predictions: {len(gdf_test):,} points")

    # Merge
    gdf_all = pd.concat([gdf_train, gdf_test])
    gdf_all = gpd.GeoDataFrame(gdf_all, geometry="geometry", crs="EPSG:4326")
    print(f"\nCombined predictions: {len(gdf_all):,} points")
    print(f"AOIs covered: {sorted(gdf_all.index.get_level_values('aoi').unique())}")

    # Save
    fp_out = RUNS_DIR / RUN_NAME / f"{RUN_NAME}_all_aois.geojson"
    gdf_all.reset_index().to_file(fp_out, driver="GeoJSON")
    print(f"Saved to {fp_out}")

    return gdf_all


if __name__ == "__main__":
    gdf_train = predict_train_aois()
    gdf_all = merge_and_save(gdf_train)
    print("\nNow rerun local_postprocessing.py with RUN_NAME ending in '_all_aois'")
