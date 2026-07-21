"""
Pixel-level inference for the Mosul retraining comparison.

Adaptation of src/inference/local_pixel_inference.py, with the model and
output path swapped to use the Mosul-retrained classifier instead of the
Gaza-trained one. Classifies Mosul's existing feature rasters (already
exported for the zero-shot transfer evaluation) so that the retrained
model's output is evaluated by the exact same downstream script
(evaluate_pixel_transfer.py) used for every other pixel-level result in
this dissertation.

Adaptations from local_pixel_inference.py (noted individually below):
    1. ORBITS: Mosul's three orbits (72, 145, 152) replace Gaza's (87, 94, 160).
    2. MODEL_FP: points to the Mosul-retrained model (trained on west-bank
       points only, see main_local_mosul_retrain.py) instead of Gaza's
       baseline model.
    3. FEATURE_RASTERS_DIR / PROBABILITY_RASTERS_DIR: point to Mosul's
       existing transfer-city raster folders instead of Gaza's.
    4. No other change. Classification, orbit aggregation (mean), and
       output scaling (Uint8, 0-255) are identical to local_pixel_inference.py.

Pipeline position:
    (existing) export_feature_rasters_transfer.py
        - main_local_mosul_retrain.py (trains model on west-bank points)
        - [this script] (classifies Mosul's feature rasters with the
           retrained model)
        - evaluate_pixel_transfer.py (unmodified; evaluated on east-bank
           test points only -- see note in that call below)

Usage:
    python3 alex/tmp/mosul_retrain_pixel_inference.py
"""

import pickle
import warnings
from pathlib import Path

import numpy as np
import rasterio
from omegaconf import OmegaConf
from tqdm.auto import tqdm

from src.classification.utils import get_features_names
from src.constants import DATA_PATH, PRE_PERIOD

# Constants
# Adaptation 1: Mosul's orbits, not Gaza's (87, 94, 160).
ORBITS = [72, 145, 152]

# Adaptation 3: Mosul's existing transfer-city raster folders.
FEATURE_RASTERS_DIR = DATA_PATH / "transfer_cities" / "feature_rasters" / "MOS"
PROBABILITY_RASTERS_DIR = (
    DATA_PATH / "transfer_cities" / "probability_rasters" / "MOS_RETRAINED_EAST_ONLY"
)

# Adaptation 2: the Mosul-retrained model, not Gaza's baseline model.
MODEL_FP = DATA_PATH / "transfer_cities" / "runs" / "MOS_retrained" / "model.pkl"

# Unchanged from local_pixel_inference.py: same feature config, same
# reducer set, same extraction window, so the 28 feature names and their
# order are identical to Gaza's.
CFG = OmegaConf.create(
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
FEATURE_COLS = get_features_names(CFG)


# Loading (unchanged)


def load_model(fp: Path = MODEL_FP):
    """Load trained sklearn RF model."""
    print(f"Loading model from {fp}...")
    with open(fp, "rb") as f:
        clf = pickle.load(f)
    print("  Model loaded.")
    return clf


def load_tile(fp: Path) -> tuple[np.ndarray, list[str], dict]:
    """Load a single feature GeoTIFF tile. Unchanged from local_pixel_inference.py."""
    with rasterio.open(fp) as src:
        data = src.read().astype(np.float32)
        band_names = list(src.descriptions)
        profile = src.profile.copy()
    return data, band_names, profile


# Classification (unchanged)


def classify_tile(
    data: np.ndarray,
    band_names: list[str],
    clf,
    feature_cols: list[str],
) -> np.ndarray:
    """Unchanged from local_pixel_inference.py."""
    n_bands, H, W = data.shape

    band_index = {name: i for i, name in enumerate(band_names)}
    try:
        order = [band_index[col] for col in feature_cols]
    except KeyError as e:
        raise ValueError(f"Band {e} not found in GeoTIFF. Available: {band_names}")
    data = data[order]

    X = data.reshape(n_bands, -1).T
    valid_mask = ~np.any(np.isnan(X), axis=1)
    prob_flat = np.full(H * W, np.nan, dtype=np.float32)
    if valid_mask.any():
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            prob_flat[valid_mask] = clf.predict_proba(X[valid_mask])[:, 1]
    return prob_flat.reshape(H, W)


def aggregate_orbits(probs: list[np.ndarray], method: str = "mean") -> np.ndarray:
    """Unchanged from local_pixel_inference.py."""
    stack = np.stack(probs, axis=0)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        if method == "mean":
            return np.nanmean(stack, axis=0)
        elif method == "max":
            return np.nanmax(stack, axis=0)
        elif method == "min":
            return np.nanmin(stack, axis=0)
        elif method == "median":
            return np.nanmedian(stack, axis=0)
        else:
            raise ValueError(f"Unknown aggregation method: {method}")


# Saving (unchanged)


def save_probability_tile(prob: np.ndarray, profile: dict, fp_out: Path) -> None:
    """Unchanged from local_pixel_inference.py."""
    fp_out.parent.mkdir(exist_ok=True, parents=True)
    prob_uint8 = np.where(np.isnan(prob), 0, prob * 255).astype(np.uint8)

    out_profile = profile.copy()
    out_profile.update(dtype=rasterio.uint8, count=1, nodata=0, compress="lzw")

    with rasterio.open(fp_out, "w", **out_profile) as dst:
        dst.write(prob_uint8[np.newaxis, :, :])


# Pipeline (unchanged except ORBITS source)


def classify_window(
    window_str: str,
    clf,
    feature_rasters_dir: Path = FEATURE_RASTERS_DIR,
    probability_rasters_dir: Path = PROBABILITY_RASTERS_DIR,
    aggregation_method: str = "mean",
    force_recreate: bool = False,
) -> None:
    """Unchanged from local_pixel_inference.py (uses module-level ORBITS)."""
    out_dir = probability_rasters_dir / window_str
    out_dir.mkdir(exist_ok=True, parents=True)

    tile_ids = set()
    for orbit in ORBITS:
        orbit_dir = feature_rasters_dir / window_str / f"orbit{orbit}"
        if orbit_dir.exists():
            tile_ids.update(fp.stem.replace("qk_", "") for fp in orbit_dir.glob("qk_*.tif"))

    if not tile_ids:
        print(f"  No tiles found for {window_str} -- skipping")
        return

    print(f"  {len(tile_ids)} tiles, aggregating {len(ORBITS)} orbits with {aggregation_method}")

    n_skipped = 0
    for qk_id in tqdm(sorted(tile_ids), desc=f"  {window_str}"):
        fp_out = out_dir / f"qk_{qk_id}.tif"

        if fp_out.exists() and not force_recreate:
            n_skipped += 1
            continue

        orbit_probs = []
        reference_profile = None

        for orbit in ORBITS:
            fp = feature_rasters_dir / window_str / f"orbit{orbit}" / f"qk_{qk_id}.tif"
            if not fp.exists():
                continue

            data, band_names, profile = load_tile(fp)
            if reference_profile is None:
                reference_profile = profile
            prob = classify_tile(data, band_names, clf, FEATURE_COLS)
            orbit_probs.append(prob)

        if not orbit_probs:
            continue

        prob_agg = aggregate_orbits(orbit_probs, method=aggregation_method)
        save_probability_tile(prob_agg, reference_profile, fp_out)

    if n_skipped:
        print(f"  Skipped {n_skipped} already-classified tiles")


def run_local_inference(
    feature_rasters_dir: Path = FEATURE_RASTERS_DIR,
    probability_rasters_dir: Path = PROBABILITY_RASTERS_DIR,
    aggregation_method: str = "mean",
    force_recreate: bool = False,
) -> None:
    """Unchanged from local_pixel_inference.py."""
    clf = load_model()

    if not feature_rasters_dir.exists():
        print(f"Feature rasters directory not found: {feature_rasters_dir}")
        return

    windows = sorted(d.name for d in feature_rasters_dir.iterdir() if d.is_dir())
    if not windows:
        print("No windows found.")
        return

    print(f"\nFound {len(windows)} windows to classify")
    print(f"Output: {probability_rasters_dir}")
    print()

    for window_str in windows:
        print(f"Window: {window_str}")
        classify_window(
            window_str=window_str,
            clf=clf,
            feature_rasters_dir=feature_rasters_dir,
            probability_rasters_dir=probability_rasters_dir,
            aggregation_method=aggregation_method,
            force_recreate=force_recreate,
        )
        print()

    print("Local inference complete.")
    print(f"Probability rasters saved to: {probability_rasters_dir}")


# Main

if __name__ == "__main__":
    run_local_inference(force_recreate=True)
