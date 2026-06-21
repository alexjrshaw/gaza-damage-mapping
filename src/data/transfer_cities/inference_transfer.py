"""
Zero-shot transfer inference for Aleppo, Raqqa, and Mosul.

Loads the Gaza-trained Random Forest model and applies it to the
transfer city features without any retraining.

Mirrors the inference step in main_local.py exactly:
    1. Load Gaza model (rf_s1_2months_50trees_1x1_all7reducers_baseline)
    2. Load transfer city features
    3. Predict damage probabilities
    4. Format predictions to match get_metrics() input format
    5. Save predictions GeoJSON per city

Output:
    data/transfer_cities/runs/{city_id}/{city_id}_predictions.geojson

Usage:
    cd /scratch/s1214882/gaza-damage-mapping
    python3 src/data/transfer_cities/inference_transfer.py
"""

import json
import pickle
import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from omegaconf import OmegaConf

sys.path.insert(0, "/scratch/s1214882/gaza-damage-mapping")

from src.classification.utils import get_features_names
from src.constants import DATA_PATH, PRE_PERIOD
from src.data.transfer_cities.constants_transfer import TRANSFER_CITIES, TRANSFER_FEATURES_DIR, TRANSFER_RUNS_DIR

# Gaza baseline model
GAZA_MODEL_RUN = "rf_s1_2months_50trees_1x1_all7reducers_baseline"
GAZA_MODEL_FP = DATA_PATH / f"runs/{GAZA_MODEL_RUN}/model.pkl"

# Feature config — must match Gaza training exactly
GAZA_CFG = OmegaConf.create(
    dict(
        data=dict(
            s1=dict(subset_bands=None),
            s2=None,
            extract_winds="1x1",
            time_periods=dict(pre=PRE_PERIOD, post="2months"),
        ),
        reducer_names=["mean", "stdDev", "median", "min", "max", "skew", "kurtosis"],
    )
)
FEATURE_COLS = get_features_names(GAZA_CFG)


def load_gaza_model():
    """Load the Gaza-trained Random Forest model."""
    assert GAZA_MODEL_FP.exists(), f"Gaza model not found: {GAZA_MODEL_FP}"
    with open(GAZA_MODEL_FP, "rb") as f:
        clf = pickle.load(f)
    print(f"Loaded Gaza model: {GAZA_MODEL_RUN}")
    print(f"  Features expected: {len(FEATURE_COLS)}")
    return clf


def run_inference_city(city_id: str, cfg: dict, clf) -> gpd.GeoDataFrame:
    """
    Run inference for one transfer city.

    Mirrors full_pipeline_local() in main_local.py:
        1. Load features
        2. Predict probabilities
        3. Average over orbits per (unosat_id, start_post)
        4. Pivot to wide format
        5. Join with UNOSAT labels for geometry and date
    """
    fp = TRANSFER_FEATURES_DIR / f"{city_id}_features.parquet"
    assert fp.exists(), f"Features not found: {fp}"

    print(f"\n  Loading features: {fp.name}")
    df = pd.read_parquet(fp)
    print(f"  {len(df):,} rows, {len(FEATURE_COLS)} features")

    # Drop NaN rows
    df = df.dropna(subset=FEATURE_COLS)
    print(f"  {len(df):,} rows after dropping NaN")

    # Predict probabilities
    print(f"  Running inference...")
    X = df[FEATURE_COLS].values
    df = df.copy()
    df["prob"] = clf.predict_proba(X)[:, 1]

    # Average over orbits per (unosat_id, start_post) — mirrors _format_predictions()
    preds = (
        df.groupby(["unosat_id", "aoi", "start_post"])["prob"]
        .mean()
        .mul(255)
        .astype(int)
        .reset_index()
        .rename(columns={"prob": "classification"})
    )

    # Pivot to wide format
    preds_wide = preds.pivot(
        index=["unosat_id", "aoi"],
        columns="start_post",
        values="classification",
    ).sort_values(["aoi", "unosat_id"])
    preds_wide.columns = [f"pred_{c}" for c in preds_wide.columns]

    # Load UNOSAT labels for geometry and date
    gdf_labels = gpd.read_file(cfg["unosat_labels"])
    gdf_labels = gdf_labels.set_index("unosat_id")

    # Join predictions with geometry
    gdf = preds_wide.join(
        gdf_labels[["geometry", "date"]],
        on="unosat_id",
    )
    gdf["date"] = pd.to_datetime(gdf["date"])
    gdf = gpd.GeoDataFrame(gdf, geometry="geometry", crs="EPSG:4326")

    print(f"  Predictions: {len(gdf):,} points, {len(preds_wide.columns)} time windows")
    return gdf


def run_all_inference() -> None:
    """Run zero-shot inference for all three transfer cities."""
    TRANSFER_RUNS_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading Gaza model...")
    clf = load_gaza_model()

    for city_id, cfg in TRANSFER_CITIES.items():
        print(f"\n{'='*60}")
        print(f"{city_id} — {cfg['city_name']} ({cfg['country']})")
        print(f"{'='*60}")

        run_dir = TRANSFER_RUNS_DIR / city_id
        run_dir.mkdir(parents=True, exist_ok=True)
        fp_out = run_dir / f"{city_id}_predictions.geojson"

        if fp_out.exists():
            print(f"  Already exists — skipping: {fp_out.name}")
            continue

        gdf = run_inference_city(city_id, cfg, clf)
        gdf.reset_index().to_file(fp_out, driver="GeoJSON")
        print(f"  Saved -> {fp_out}")

    print(f"\n{'='*60}")
    print("Inference complete for all cities.")
    print(f"{'='*60}")


if __name__ == "__main__":
    run_all_inference()
