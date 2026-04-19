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
Forth HPC compute nodes lack internet access, so pipeline is split:
    Step 1 (download.py): Run interactively — downloads GEE assets to local parquet
    Step 2 (this script): Run as Slurm batch job — computes features from local cache
"""

import numpy as np
import pandas as pd
from pathlib import Path
from tqdm.auto import tqdm

from src.constants import (
    AOIS_TRAIN, AOIS_TEST,
    DATA_PATH, ASSETS_PATH,
    GAZA_WAR_START, PRE_PERIOD, POST_PERIODS,
)

# Local cache for downloaded intermediate assets
CACHE_DIR = DATA_PATH / "intermediate_features_cache"
FEATURES_DIR = DATA_PATH / "features_ready"

ORBITS = [87, 94, 160]
EXTRACT_WINDOW = "1x1"
REDUCER_NAMES = ["mean", "stdDev", "median", "min", "max", "skew", "kurtosis"]


# ==================== FEATURE COMPUTATION ====================

def compute_stats(series: pd.Series) -> dict:
    """
    Compute 7 statistics for a series — mirrors GEE reducers.
    Matches Dietrich et al. reducer names exactly.
    """
    return {
        "mean":     series.mean(),
        "stdDev":   series.std(),
        "median":   series.median(),
        "min":      series.min(),
        "max":      series.max(),
        "skew":     series.skew(),
        "kurtosis": series.kurtosis(),
    }


def compute_features_for_window(
    df: pd.DataFrame,
    pre_period: tuple[str, str],
    post_period: tuple[str, str],
    orbit: int,
) -> pd.DataFrame:
    """
    Compute pre and post features for one intermediate asset and one time window.

    For each UNOSAT point, computes 7 statistics for VV and VH across
    all images within the pre and post date windows.

    Implements Dietrich et al. (2025) eq. 1 label assignment:
        y = 0  if end_post <= conflict_start
        y = 1  if end_post > date_first_severe (tunosat equivalent)
        y = -1 discard (points where damage confirmed after end_post)

    Args:
        df: Intermediate asset dataframe (rows = one image x one point)
        pre_period: (start, end) date strings for pre-conflict window
        post_period: (start, end) date strings for post-conflict window
        orbit: Orbit number

    Returns:
        DataFrame with one row per point and feature columns
    """

    df = df.copy()
    df["date"] = df["date"].astype(str)
    df["date_first_severe"] = df["date_first_severe"].astype(str)

    # --- Label assignment — Dietrich et al. eq. 1 ---
    end_post = post_period[1]
    if end_post <= GAZA_WAR_START:
        label = 0
    else:
        label = 1

    # --- Filter points ---
    # For label=1: only keep points where damage was confirmed
    # before end of post window (date_first_severe is tunosat)
    if label == 1:
        df = df[df["date_first_severe"] <= end_post].copy()

    if len(df) == 0:
        return pd.DataFrame()

    # --- Compute features per point ---
    prefix_pre  = f"pre_{EXTRACT_WINDOW}"
    prefix_post = f"post_{EXTRACT_WINDOW}"

    # Filter to pre and post date ranges
    pre_df  = df[(df["date"] >= pre_period[0])  & (df["date"] <= pre_period[1])]
    post_df = df[(df["date"] >= post_period[0]) & (df["date"] <= post_period[1])]

    # Get unique point metadata
    meta = df.groupby("unosat_id").first()[
        ["damage", "aoi", "date_first_severe", "site_id"]
    ].reset_index()
    meta = meta.rename(columns={"date_first_severe": "date"})

    results = meta.copy()
    results["label"] = label
    results["orbit"] = orbit
    results["start_pre"]  = pre_period[0]
    results["end_pre"]    = pre_period[1]
    results["start_post"] = post_period[0]
    results["end_post"]   = post_period[1]

    # Compute statistics for each band and period using vectorised groupby
    for band in ["VV", "VH"]:
        for period_df, prefix in [(pre_df, prefix_pre), (post_df, prefix_post)]:
            if len(period_df) > 0:
                period_df[band] = period_df[band].astype(float)
                stats = period_df.groupby("unosat_id")[band].agg(
                    mean="mean",
                    stdDev="std",
                    median="median",
                    min="min",
                    max="max",
                    skew=lambda x: x.skew(),
                    kurtosis=lambda x: x.kurtosis(),
                )
                stats.columns = [f"{band}_{prefix}_{s}" for s in REDUCER_NAMES]
                results = results.merge(stats, on="unosat_id", how="left")
            else:
                # No images in this window — fill with NaN
                for stat in REDUCER_NAMES:
                    results[f"{band}_{prefix}_{stat}"] = np.nan

    return results


def extract_features_local(
    split: str,
    pre_period: tuple[str, str] = PRE_PERIOD,
    post_periods: list[tuple[str, str]] = POST_PERIODS,
) -> pd.DataFrame:
    """
    Extract features for all AOIs, orbits and time windows for a given split.

    Reads from local parquet cache — no internet connection required.

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
            fp = CACHE_DIR / f"{aoi}_orbit{orbit}.parquet"
            assert fp.exists(), (
                f"Cache file {fp} not found. "
                f"Run download_intermediate_assets.py first."
            )

            print(f"  Loading {aoi}_orbit{orbit}...")
            df = pd.read_parquet(fp)

            for post_period in tqdm(post_periods, desc=f"    windows"):
                features = compute_features_for_window(
                    df, pre_period, post_period, orbit
                )
                if len(features) > 0:
                    all_features.append(features)

    if not all_features:
        print(f"WARNING: No features extracted for {split} split!")
        return pd.DataFrame()

    result = pd.concat(all_features, ignore_index=True)
    print(f"\n{split} split: {len(result):,} rows, {len(result.columns)} columns")
    return result


# ==================== MAIN ====================

if __name__ == "__main__":
    FEATURES_DIR.mkdir(exist_ok=True, parents=True)

    # Train split
    print("=" * 60)
    print("Extracting train features...")
    print("=" * 60)
    train_features = extract_features_local("train")
    train_fp = FEATURES_DIR / "s1_1x1_2months_train.parquet"
    train_features.to_parquet(train_fp)
    print(f"Saved train features to {train_fp}")

    # Test split
    print("\n" + "=" * 60)
    print("Extracting test features...")
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
