"""
Local pixel-level inference for transfer city proof-of-concept.

Adapted from src/inference/local_pixel_inference.py for transfer cities.
Applies the Gaza-trained Random Forest to feature rasters downloaded
from Google Drive, producing damage probability rasters.

Input:
    data/transfer_cities/feature_rasters/{city_id}/{window_str}/orbit{orbit}/
        {city_id}_{window_str}_orbit{orbit}.tif  (single tile per orbit)

Output:
    data/transfer_cities/probability_rasters/{city_id}/{window_str}/
        {city_id}_{window_str}.tif  (orbit-aggregated, Uint8 0-255)

Usage:
    python3 src/data/transfer_cities/pixel_inference_transfer.py --city RAQ
"""

import pickle
import warnings
import argparse
import numpy as np
import rasterio
from pathlib import Path
from omegaconf import OmegaConf

from src.constants import DATA_PATH, PRE_PERIOD
from src.classification.utils import get_features_names
from src.data.transfer_cities.constants_transfer import TRANSFER_CITIES

# Gaza baseline model
GAZA_MODEL_FP = DATA_PATH / "runs/rf_s1_2months_50trees_1x1_all7reducers_baseline/model.pkl"

# Feature names must match Gaza training exactly
CFG = OmegaConf.create(dict(
    data=dict(s1=dict(subset_bands=None), s2=None,
              extract_winds="1x1", time_periods=dict(pre=PRE_PERIOD, post="2months")),
    reducer_names=["mean", "stdDev", "median", "min", "max", "skew", "kurtosis"],
))
FEATURE_COLS = get_features_names(CFG)

TRANSFER_FEATURE_RASTERS = DATA_PATH / "transfer_cities" / "feature_rasters"
TRANSFER_PROB_RASTERS    = DATA_PATH / "transfer_cities" / "probability_rasters"


def load_model():
    with open(GAZA_MODEL_FP, "rb") as f:
        clf = pickle.load(f)
    print(f"Loaded Gaza model: {GAZA_MODEL_FP.parent.name}")
    return clf


def classify_raster(fp: Path, clf, feature_cols: list) -> tuple:
    """Classify a single feature raster tile."""
    with rasterio.open(fp) as src:
        data = src.read().astype(np.float32)
        band_names = list(src.descriptions)
        profile = src.profile.copy()

    n_bands, H, W = data.shape
    band_index = {name: i for i, name in enumerate(band_names)}

    available = [c for c in feature_cols if c in band_index]
    if not available:
        return np.full((H, W), np.nan, dtype=np.float32), profile

    order = [band_index[c] for c in available]
    data_sub = data[order]
    X = data_sub.reshape(len(order), -1).T
    valid = ~np.any(np.isnan(X), axis=1)

    prob_flat = np.full(H * W, np.nan, dtype=np.float32)
    if valid.any():
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            prob_flat[valid] = clf.predict_proba(X[valid])[:, 1]

    return prob_flat.reshape(H, W), profile


def run_pixel_inference_city(city_id: str, force: bool = False) -> None:
    cfg = TRANSFER_CITIES[city_id]
    orbits = cfg["orbits"]
    pre_period = cfg["pre_period"]
    post_periods = cfg["post_periods"]
    all_periods = [pre_period] + list(post_periods)

    feat_base = TRANSFER_FEATURE_RASTERS / city_id
    prob_base = TRANSFER_PROB_RASTERS / city_id
    prob_base.mkdir(parents=True, exist_ok=True)

    if not feat_base.exists():
        print(f"No feature rasters found at {feat_base}")
        print("Run export_feature_rasters_transfer.py and download from Drive first.")
        return

    clf = load_model()

    for i, post_period in enumerate(all_periods):
        window_str = f"w{i+1:02d}_{post_period[0]}_{post_period[1]}"
        fp_out = prob_base / window_str / f"{city_id}_{window_str}.tif"

        if fp_out.exists() and not force:
            print(f"  {window_str}: already exists — skipping")
            continue

        orbit_probs = []
        ref_profile = None

        for orbit in orbits:
            fp = feat_base / window_str / f"orbit{orbit}" / f"{city_id}_{window_str}_orbit{orbit}.tif"
            if not fp.exists():
                continue

            prob, profile = classify_raster(fp, clf, FEATURE_COLS)
            orbit_probs.append(prob)
            if ref_profile is None:
                ref_profile = profile

        if not orbit_probs:
            print(f"  {window_str}: no feature rasters found — skipping")
            continue

        # Aggregate orbits (mean — mirrors Dietrich et al.)
        stack = np.stack(orbit_probs, axis=0)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            prob_agg = np.nanmean(stack, axis=0)

        # Save as Uint8 0-255
        fp_out.parent.mkdir(parents=True, exist_ok=True)
        prob_uint8 = np.where(np.isnan(prob_agg), 0, prob_agg * 255).astype(np.uint8)
        out_profile = ref_profile.copy()
        out_profile.update(dtype=rasterio.uint8, count=1, nodata=0, compress="lzw")

        with rasterio.open(fp_out, "w", **out_profile) as dst:
            dst.write(prob_uint8[np.newaxis])

        print(f"  {window_str}: saved -> {fp_out.name}")

    print(f"\nPixel inference complete for {city_id}.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--city", type=str, required=True)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    run_pixel_inference_city(args.city.upper(), force=args.force)
