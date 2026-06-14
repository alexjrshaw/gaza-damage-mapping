"""
Local pixel-level inference for transfer city proof-of-concept.

Applies the Gaza-trained Random Forest to quadkey-tiled feature rasters,
producing damage probability rasters at 10m resolution.

Mirrors src/inference/local_pixel_inference.py exactly:
    - Loads feature rasters per quadkey tile per orbit
    - Classifies each tile
    - Aggregates across orbits (mean)
    - Saves probability rasters as Uint8 0-255

Input:
    data/transfer_cities/feature_rasters/{city_id}/{window_str}/orbit{orbit}/
        qk_{qk_id}.tif

Output:
    data/transfer_cities/probability_rasters/{city_id}/{window_str}/
        qk_{qk_id}.tif  (orbit-aggregated, Uint8 0-255)

Usage:
    python3 src/data/transfer_cities/pixel_inference/pixel_inference_transfer.py --city MOS
"""

import pickle
import warnings
import argparse
import numpy as np
import rasterio
from pathlib import Path
from omegaconf import OmegaConf
from tqdm.auto import tqdm

import sys
sys.path.insert(0, '/scratch/s1214882/gaza-damage-mapping')

from src.constants import DATA_PATH, PRE_PERIOD
from src.classification.utils import get_features_names
from src.data.transfer_cities.constants_transfer import TRANSFER_CITIES

GAZA_MODEL_FP        = DATA_PATH / "runs/rf_s1_2months_50trees_1x1_all7reducers_baseline/model.pkl"
TRANSFER_FEAT_BASE   = DATA_PATH / "transfer_cities" / "feature_rasters"
TRANSFER_PROB_BASE   = DATA_PATH / "transfer_cities" / "probability_rasters"

CFG = OmegaConf.create(dict(
    data=dict(s1=dict(subset_bands=None), s2=None,
              extract_winds="1x1", time_periods=dict(pre=PRE_PERIOD, post="2months")),
    reducer_names=["mean", "stdDev", "median", "min", "max", "skew", "kurtosis"],
))
FEATURE_COLS = get_features_names(CFG)


def load_model():
    with open(GAZA_MODEL_FP, "rb") as f:
        clf = pickle.load(f)
    print(f"Loaded Gaza model: {GAZA_MODEL_FP.parent.name}")
    return clf


def classify_tile(fp: Path, clf) -> tuple:
    """Classify a single feature raster tile — mirrors classify_window() in local_pixel_inference.py."""
    with rasterio.open(fp) as src:
        data = src.read().astype(np.float32)
        band_names = list(src.descriptions)
        profile = src.profile.copy()

    n_bands, H, W = data.shape
    band_index = {name: i for i, name in enumerate(band_names)}
    available = [c for c in FEATURE_COLS if c in band_index]

    if not available:
        return np.full((H, W), np.nan, dtype=np.float32), profile

    order = [band_index[c] for c in available]
    X = data[order].reshape(len(order), -1).T
    valid = ~np.any(np.isnan(X), axis=1)

    prob_flat = np.full(H * W, np.nan, dtype=np.float32)
    if valid.any():
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            prob_flat[valid] = clf.predict_proba(X[valid])[:, 1]

    return prob_flat.reshape(H, W), profile


def save_probability_tile(prob: np.ndarray, profile: dict, fp_out: Path) -> None:
    """Save probability raster as Uint8 0-255 — mirrors Gaza pipeline."""
    fp_out.parent.mkdir(parents=True, exist_ok=True)
    prob_uint8 = np.where(np.isnan(prob), 0, prob * 255).astype(np.uint8)
    out_profile = profile.copy()
    out_profile.update(dtype=rasterio.uint8, count=1, nodata=0, compress="lzw")
    with rasterio.open(fp_out, "w", **out_profile) as dst:
        dst.write(prob_uint8[np.newaxis])


def run_pixel_inference_city(city_id: str, force: bool = False) -> None:
    cfg    = TRANSFER_CITIES[city_id]
    orbits = cfg["orbits"]
    pre_period   = cfg["pre_period"]
    post_periods = cfg["post_periods"]
    all_periods  = [pre_period] + list(post_periods)

    feat_base = TRANSFER_FEAT_BASE / city_id
    prob_base = TRANSFER_PROB_BASE / city_id
    prob_base.mkdir(parents=True, exist_ok=True)

    if not feat_base.exists():
        print(f"No feature rasters at {feat_base} — run export first.")
        return

    clf = load_model()

    for i, post_period in enumerate(all_periods):
        window_str = f"w{i+1:02d}_{post_period[0]}_{post_period[1]}"
        out_dir    = prob_base / window_str

        # Find all tile IDs for this window
        tile_ids = set()
        for orbit in orbits:
            orbit_dir = feat_base / window_str / f"orbit{orbit}"
            if orbit_dir.exists():
                tile_ids.update(
                    fp.stem.replace("qk_", "")
                    for fp in orbit_dir.glob("qk_*.tif")
                )

        if not tile_ids:
            continue

        print(f"  {window_str}: {len(tile_ids)} tiles × {len(orbits)} orbits")

        n_skipped = 0
        for qk_id in tqdm(sorted(tile_ids), desc=f"    {window_str}"):
            fp_out = out_dir / f"qk_{qk_id}.tif"
            if fp_out.exists() and not force:
                n_skipped += 1
                continue

            orbit_probs    = []
            ref_profile    = None

            for orbit in orbits:
                fp = feat_base / window_str / f"orbit{orbit}" / f"qk_{qk_id}.tif"
                if not fp.exists():
                    continue
                prob, profile = classify_tile(fp, clf)
                orbit_probs.append(prob)
                if ref_profile is None:
                    ref_profile = profile

            if not orbit_probs:
                continue

            # Aggregate orbits (mean — mirrors Dietrich et al.)
            stack = np.stack(orbit_probs, axis=0)
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore")
                prob_agg = np.nanmean(stack, axis=0)

            save_probability_tile(prob_agg, ref_profile, fp_out)

        if n_skipped:
            print(f"    Skipped {n_skipped} already-classified tiles")

    print(f"\nPixel inference complete for {city_id}.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--city", type=str, required=True)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    run_pixel_inference_city(args.city.upper(), force=args.force)
