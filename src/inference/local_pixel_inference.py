"""
Local pixel-level inference for Gaza damage mapping.

Classifies exported feature rasters using the trained sklearn Random Forest,
producing damage probability rasters at 10m resolution across Gaza.

This is the local equivalent of GEE's dense_inference.py classify step.
Follows Dietrich et al. (2025) methodology exactly:
    - Loads 28-band feature GeoTIFFs exported by export_feature_rasters.py
    - Applies sklearn RF to every pixel
    - Aggregates predictions across 3 orbits (mean) — mirrors predict_geo()
    - Exports probability rasters scaled 0-255 (Uint8) — matches full_gaza.py output
    - Output format matches drive_to_results.py input exactly

Pipeline position:
    export_feature_rasters.py → [this script] → local_postprocessing_pixel.py

Input:
    data/feature_rasters/{window_str}/orbit{orbit}/qk_{qk_id}.tif
        - 28-band Float32 feature GeoTIFF per tile per orbit

Output:
    data/probability_rasters/{window_str}/qk_{qk_id}.tif
        - Single-band Uint8 probability raster (0-255) per tile
        - Orbit-aggregated mean (mirrors Dietrich et al. aggregation_method='mean')
"""

import pickle
import warnings
from pathlib import Path

import numpy as np
import rasterio
from rasterio.transform import from_bounds
from omegaconf import OmegaConf
from tqdm.auto import tqdm

from src.constants import DATA_PATH, PRE_PERIOD, POST_PERIODS
from src.classification.utils import get_features_names

# ==================== CONSTANTS ====================

ORBITS = [87, 94, 160]
FEATURE_RASTERS_DIR = DATA_PATH / "feature_rasters"
PROBABILITY_RASTERS_DIR = DATA_PATH / "probability_rasters"
RUN_NAME = "rf_s1_2months_50trees_1x1_all7reducers_baseline"
MODEL_FP = DATA_PATH / f"runs/{RUN_NAME}/model.pkl"

# Config needed to get feature names in correct order
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
FEATURE_COLS = get_features_names(CFG)  # 28 feature names in correct order


# ==================== LOADING ====================

def load_model(fp: Path = MODEL_FP):
    """Load trained sklearn RF model."""
    print(f"Loading model from {fp}...")
    with open(fp, "rb") as f:
        clf = pickle.load(f)
    print("  Model loaded.")
    return clf


def load_tile(fp: Path) -> tuple[np.ndarray, list[str], dict]:
    """
    Load a single feature GeoTIFF tile.

    Returns:
        data: (28, H, W) float32 array
        band_names: list of band names from GeoTIFF metadata
        profile: rasterio profile for output
    """
    with rasterio.open(fp) as src:
        data = src.read().astype(np.float32)
        band_names = list(src.descriptions)  # GEE exports band names here
        profile = src.profile.copy()
    return data, band_names, profile


# ==================== CLASSIFICATION ====================

def classify_tile(
    data: np.ndarray,
    band_names: list[str],
    clf,
    feature_cols: list[str],
) -> np.ndarray:
    n_bands, H, W = data.shape

    # Reorder bands to match model's expected feature order
    band_index = {name: i for i, name in enumerate(band_names)}
    try:
        order = [band_index[col] for col in feature_cols]
    except KeyError as e:
        raise ValueError(f"Band {e} not found in GeoTIFF. Available: {band_names}")
    data = data[order]  # reorder to match feature_cols

    X = data.reshape(n_bands, -1).T
    valid_mask = ~np.any(np.isnan(X), axis=1)
    prob_flat = np.full(H * W, np.nan, dtype=np.float32)
    if valid_mask.any():
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            prob_flat[valid_mask] = clf.predict_proba(X[valid_mask])[:, 1]
    return prob_flat.reshape(H, W)


def aggregate_orbits(
    probs: list[np.ndarray],
    method: str = "mean",
) -> np.ndarray:
    """
    Aggregate probability rasters across orbits.

    Mirrors Dietrich et al. aggregation_method='mean' in predict_geo().

    Args:
        probs: list of (H, W) probability arrays, one per orbit
        method: aggregation method ('mean', 'max', 'min', 'median')

    Returns:
        aggregated (H, W) probability array
    """
    stack = np.stack(probs, axis=0)  # (n_orbits, H, W)

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


# ==================== SAVING ====================

def save_probability_tile(
    prob: np.ndarray,
    profile: dict,
    fp_out: Path,
) -> None:
    """
    Save probability raster as Uint8 GeoTIFF scaled 0-255.

    Matches full_gaza.py output format:
        preds.multiply(2**8 - 1).toUint8()
    """
    fp_out.parent.mkdir(exist_ok=True, parents=True)

    # Scale to 0-255, set NaN to 0
    prob_uint8 = np.where(np.isnan(prob), 0, prob * 255).astype(np.uint8)

    out_profile = profile.copy()
    out_profile.update(
        dtype=rasterio.uint8,
        count=1,
        nodata=0,
        compress="lzw",
    )

    with rasterio.open(fp_out, "w", **out_profile) as dst:
        dst.write(prob_uint8[np.newaxis, :, :])


# ==================== PIPELINE ====================

def classify_window(
    window_str: str,
    clf,
    feature_rasters_dir: Path = FEATURE_RASTERS_DIR,
    probability_rasters_dir: Path = PROBABILITY_RASTERS_DIR,
    aggregation_method: str = "mean",
    force_recreate: bool = False,
) -> None:
    """
    Classify all tiles for one time window.

    For each quadkey tile:
        1. Load feature rasters for all 3 orbits
        2. Classify each orbit
        3. Aggregate across orbits (mean)
        4. Save probability raster

    Args:
        window_str: Window identifier e.g. 'w07_2023-10-07_2023-12-06'
        clf: Trained sklearn RF
        feature_rasters_dir: Base directory for feature rasters
        probability_rasters_dir: Output directory for probability rasters
        aggregation_method: Orbit aggregation method (default: 'mean')
        force_recreate: Overwrite existing probability rasters
    """
    out_dir = probability_rasters_dir / window_str
    out_dir.mkdir(exist_ok=True, parents=True)

    # Find all tile IDs for this window (from first available orbit)
    tile_ids = set()
    for orbit in ORBITS:
        orbit_dir = feature_rasters_dir / window_str / f"orbit{orbit}"
        if orbit_dir.exists():
            tile_ids.update(fp.stem.replace("qk_", "") for fp in orbit_dir.glob("qk_*.tif"))

    if not tile_ids:
        print(f"  No tiles found for {window_str} — skipping")
        return

    print(f"  {len(tile_ids)} tiles, aggregating {len(ORBITS)} orbits with {aggregation_method}")

    n_skipped = 0
    for qk_id in tqdm(sorted(tile_ids), desc=f"  {window_str}"):
        fp_out = out_dir / f"qk_{qk_id}.tif"

        if fp_out.exists() and not force_recreate:
            n_skipped += 1
            continue

        # Load and classify each orbit
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

        # Aggregate across orbits
        prob_agg = aggregate_orbits(orbit_probs, method=aggregation_method)

        # Save
        save_probability_tile(prob_agg, reference_profile, fp_out)

    if n_skipped:
        print(f"  Skipped {n_skipped} already-classified tiles")


def run_local_inference(
    feature_rasters_dir: Path = FEATURE_RASTERS_DIR,
    probability_rasters_dir: Path = PROBABILITY_RASTERS_DIR,
    aggregation_method: str = "mean",
    force_recreate: bool = False,
) -> None:
    """
    Run local pixel inference for all available time windows.

    Processes whatever windows are already downloaded — can be run
    incrementally as GEE exports complete.
    """
    # Load model
    clf = load_model()

    # Find all downloaded windows
    if not feature_rasters_dir.exists():
        print(f"Feature rasters directory not found: {feature_rasters_dir}")
        print("Run export_feature_rasters.py and download from Drive first.")
        return

    windows = sorted(d.name for d in feature_rasters_dir.iterdir() if d.is_dir())
    if not windows:
        print("No windows found. Download feature rasters from Drive first.")
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


# ==================== MAIN ====================

if __name__ == "__main__":
    run_local_inference()
