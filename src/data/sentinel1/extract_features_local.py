"""
Local feature extraction for Gaza damage mapping.

Replaces the GEE-based extract_features.py for the feature computation step.
Gaza's density (65,000+ points in 365km²) creates computation graphs too large
for GEE to handle. This script downloads the intermediate time series assets
from GEE and computes features locally using pandas — same result, far faster.

Follows Dietrich et al. (2025) methodology exactly:
    - Same 7 statistical features: mean, stdDev, median, min, max, skew, kurtosis
    - Same label assignment (eq. 1): y=0 pre-conflict, y=1 post-damage, y=-1 discard
    - Same feature naming convention: VV_pre_1x1_mean, VH_post_1x1_stdDev etc.
    - Same train/test split by AOI

Gaza-specific adaptation: computation moved from GEE to local pandas.
"""

import ee
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm.auto import tqdm

from src.constants import (
    AOIS_TRAIN, AOIS_TEST, AOIS,
    DATA_PATH, ASSETS_PATH,
    GAZA_WAR_START, PRE_PERIOD, POST_PERIODS,
)
from src.utils.gee import init_gee

# Local cache for downloaded intermediate assets
CACHE_DIR = DATA_PATH / "intermediate_features_cache"
FEATURES_DIR = DATA_PATH / "features_ready"

ORBITS = [87, 94, 160]
EXTRACT_WINDOW = "1x1"
REDUCER_NAMES = ["mean", "stdDev", "median", "min", "max", "skew", "kurtosis"]


# ==================== DOWNLOAD ====================

def download_intermediate_asset(aoi: str, orbit: int, force: bool = False) -> Path:
    """
    Download a GEE intermediate asset to local parquet cache.

    Args:
        aoi: AOI name e.g. 'GAZ1'
        orbit: Orbit number e.g. 87
        force: Re-download even if cache exists

    Returns:
        Path to local parquet file
    """
    CACHE_DIR.mkdir(exist_ok=True, parents=True)
    fp = CACHE_DIR / f"{aoi}_orbit{orbit}.parquet"

    if fp.exists() and not force:
        print(f"  {aoi}_orbit{orbit}: cached ✓")
        return fp

    print(f"  {aoi}_orbit{orbit}: downloading from GEE...")
    asset_id = ASSETS_PATH + f"intermediate_features/ts_s1_{EXTRACT_WINDOW}/{aoi}_orbit{orbit}"
    fc = ee.FeatureCollection(asset_id)

    # Download in batches to avoid memory issues
    total = fc.size().getInfo()
    batch_size = 50000
    records = []

    for start in range(0, total, batch_size):
        batch = fc.toList(batch_size, start).getInfo()
        for feat in batch:
            props = feat["properties"]
            records.append(props)

    df = pd.DataFrame(records)
    df.to_parquet(fp)
    print(f"  {aoi}_orbit{orbit}: {len(df):,} rows saved")
    return fp


def download_all_intermediate_assets(aois: list[str] = None) -> None:
    """Download all intermediate assets for given AOIs."""
    if aois is None:
        aois = AOIS
    print(f"Downloading intermediate assets for {aois}...")
    for aoi in aois:
        for orbit in ORBITS:
            download_intermediate_asset(aoi, orbit)


# ==================== FEATURE COMPUTATION ====================

def compute_stats(series: pd.Series) -> dict:
    """
    Compute 7 statistics for a series — mirrors GEE reducers.

    Matches Dietrich et al. reducer names exactly.
    """
    return {
        "mean":    series.mean(),
        "stdDev":  series.std(),
        "median":  series.median(),
        "min":     series.min(),
        "max":     series.max(),
        "skew":    series.skew(),
        "kurtosis": series.kurtosis(),
    }


def compute_features_for_asset(
    df: pd.DataFrame,
    pre_period: tuple[str, str],
    post_period: tuple[str, str],
) -> pd.DataFrame:
    """
    Compute pre and post features for one intermediate asset (one AOI, one orbit).

    For each UNOSAT point, computes 7 statistics for VV and VH across
    all images within the pre and post date windows.

    Args:
        df: Intermediate asset dataframe (rows = one image × one point)
        pre_period: (start, end) date strings for pre-conflict window
        post_period: (start, end) date strings for post-conflict window

    Returns:
        DataFrame with one row per point, feature columns + label
    """
    # Convert date column to string for comparison
    df["date"] = df["date"].astype(str)

    # --- Label assignment — Dietrich et al. eq. 1 ---
    end_post = post_period[1]
    if end_post <= GAZA_WAR_START:
        label = 0
    else:
        label = 1

    # --- Filter points for label=1 ---
    # Only keep points where damage was confirmed before end of post window
    # Uses date_first_severe — Gaza adaptation of tunosat in Dietrich et al.
    if label == 1:
        df = df[df["date_first_severe"] <= end_post].copy()

    if len(df) == 0:
        return pd.DataFrame()

    # --- Compute features for pre and post periods ---
    prefix_pre = f"pre_{EXTRACT_WINDOW}"
    prefix_post = f"post_{EXTRACT_WINDOW}"

    results = []
    for unosat_id, group in df.groupby("unosat_id"):
        row = {
            "unosat_id": unosat_id,
            "label": label,
            "aoi": group["aoi"].iloc[0],
            "damage": group["damage"].iloc[0],
            "date": group["date_first_severe"].iloc[0],
            "start_pre": pre_period[0],
            "end_pre": pre_period[1],
            "start_post": post_period[0],
            "end_post": post_period[1],
        }

        # Pre-period features
        pre = group[(group["date"] >= pre_period[0]) & (group["date"] <= pre_period[1])]
        for band in ["VV", "VH"]:
            if len(pre) > 0:
                stats = compute_stats(pre[band].astype(float))
            else:
                stats = {s: np.nan for s in REDUCER_NAMES}
            for stat, val in stats.items():
                row[f"{band}_{prefix_pre}_{stat}"] = val

        # Post-period features
        post = group[(group["date"] >= post_period[0]) & (group["date"] <= post_period[1])]
        for band in ["VV", "VH"]:
            if len(post) > 0:
                stats = compute_stats(post[band].astype(float))
            else:
                stats = {s: np.nan for s in REDUCER_NAMES}
            for stat, val in stats.items():
                row[f"{band}_{prefix_post}_{stat}"] = val

        results.append(row)

    return pd.DataFrame(results)


def extract_features_local(
    split: str,
    pre_period: tuple[str, str] = PRE_PERIOD,
    post_periods: list[tuple[str, str]] = POST_PERIODS,
) -> pd.DataFrame:
    """
    Extract features for all AOIs, orbits and time windows for a given split.

    Args:
        split: 'train' or 'test'
        pre_period: Pre-conflict date range
        post_periods: List of post-conflict date ranges

    Returns:
        DataFrame with all features ready for classification
    """
    aois = AOIS_TRAIN if split == "train" else AOIS_TEST
    all_features = []

    for aoi in aois:
        print(f"\nProcessing {aoi}...")
        for orbit in ORBITS:
            print(f"  Orbit {orbit}...")

            # Load cached asset
            fp = CACHE_DIR / f"{aoi}_orbit{orbit}.parquet"
            if not fp.exists():
                print(f"    Cache missing — downloading...")
                download_intermediate_asset(aoi, orbit)
            df = pd.read_parquet(fp)

            # Compute features for each post window
            for post_period in tqdm(post_periods, desc=f"    {aoi}_orbit{orbit}"):
                features = compute_features_for_asset(df, pre_period, post_period)
                if len(features) > 0:
                    features["orbit"] = orbit
                    all_features.append(features)

    if not all_features:
        print(f"WARNING: No features extracted for {split} split!")
        return pd.DataFrame()

    result = pd.concat(all_features, ignore_index=True)
    print(f"\n{split} split: {len(result):,} rows, {len(result.columns)} columns")
    return result


# ==================== MAIN ====================

if __name__ == "__main__":
    init_gee()
    FEATURES_DIR.mkdir(exist_ok=True, parents=True)

    # Step 1 — Download intermediate assets if not cached
    print("=" * 60)
    print("Step 1: Downloading intermediate assets...")
    print("=" * 60)
    download_all_intermediate_assets()

    # Step 2 — Extract features for train split
    print("\n" + "=" * 60)
    print("Step 2: Extracting train features...")
    print("=" * 60)
    train_features = extract_features_local("train")
    train_fp = FEATURES_DIR / "s1_1x1_2months_train.parquet"
    train_features.to_parquet(train_fp)
    print(f"Saved train features to {train_fp}")

    # Step 3 — Extract features for test split
    print("\n" + "=" * 60)
    print("Step 3: Extracting test features...")
    print("=" * 60)
    test_features = extract_features_local("test")
    test_fp = FEATURES_DIR / "s1_1x1_2months_test.parquet"
    test_features.to_parquet(test_fp)
    print(f"Saved test features to {test_fp}")

    print("\n" + "=" * 60)
    print("Feature extraction complete!")
    print(f"Train: {len(train_features):,} rows")
    print(f"Test:  {len(test_features):,} rows")
    print("=" * 60)
