"""
Local feature extraction for Gaza damage mapping.

Replaces the GEE-based extract_features.py for the feature computation step.
Gaza's density (65,000+ points in 365km²) creates computation graphs too large
for GEE to handle. This script downloads the intermediate time series assets
from GEE and computes features locally using pandas.

Follows Dietrich et al. (2025) methodology:
    - Same 7 statistical features: mean, stdDev, median, min, max, skew, kurtosis
    - Same label assignment (eq. 1): y=0 pre-conflict, y=1 post-damage, y=-1 discard
    - Same feature naming convention: VV_pre_1x1_mean, VH_post_1x1_stdDev etc.

split_strategy support (added):
    - "aoi": train on AOIS_TRAIN, test on AOIS_TEST (governorate-level holdout)

Gaza-specific adaptation: computation moved from GEE to local pandas.
Forth HPC compute nodes lack internet access, so pipeline is split:
    Step 1 (download.py): Run interactively - downloads GEE assets to local parquet
    Step 2 (this script): Run as Slurm batch job - computes features from local cache
"""

import argparse

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm.auto import tqdm

from src.constants import AOIS_TEST, AOIS_TRAIN, DATA_PATH, GAZA_WAR_START, POST_PERIODS, PRE_PERIOD

CACHE_DIR = DATA_PATH / "intermediate_features_cache"
FEATURES_DIR = DATA_PATH / "features_ready"

ORBITS = [87, 94, 160]
EXTRACT_WINDOW = "1x1"
REDUCER_NAMES = ["mean", "stdDev", "median", "min", "max", "skew", "kurtosis"]
ALL_AOIS = list(AOIS_TRAIN) + list(AOIS_TEST)


def compute_stats(series: pd.Series) -> dict:
    return {
        "mean": series.mean(),
        "stdDev": series.std(),
        "median": series.median(),
        "min": series.min(),
        "max": series.max(),
        "skew": series.skew(),
        "kurtosis": series.kurtosis(),
    }


def compute_features_for_window(
    df: pd.DataFrame,
    pre_period: tuple[str, str],
    post_period: tuple[str, str],
    orbit: int,
) -> pd.DataFrame:
    df = df.copy()
    df["s1_date"] = pd.to_datetime(df["system:time_start"], unit="ms").dt.date.astype(str)
    df["date_first_severe"] = df["date_first_severe"].astype(str)

    end_post = post_period[1]
    if end_post <= GAZA_WAR_START:
        label = 0
    else:
        label = 1

    if label == 1:
        df = df[df["date_first_severe"] <= end_post].copy()

    if len(df) == 0:
        return pd.DataFrame()

    prefix_pre = f"pre_{EXTRACT_WINDOW}"
    prefix_post = f"post_{EXTRACT_WINDOW}"

    pre_df = df[(df["s1_date"] >= pre_period[0]) & (df["s1_date"] <= pre_period[1])]
    post_df = df[(df["s1_date"] >= post_period[0]) & (df["s1_date"] <= post_period[1])]

    meta = (
        df.groupby("unosat_id")
        .first()[["damage", "aoi", "date_first_severe", "site_id"]]
        .reset_index()
    )
    meta = meta.rename(columns={"date_first_severe": "date"})

    results = meta.copy()
    results["label"] = label
    results["orbit"] = orbit
    results["start_pre"] = pre_period[0]
    results["end_pre"] = pre_period[1]
    results["start_post"] = post_period[0]
    results["end_post"] = post_period[1]

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
                for stat in REDUCER_NAMES:
                    results[f"{band}_{prefix}_{stat}"] = np.nan

    return results


def get_aoi_point_assignment(
    split: str,
    split_strategy: str,
    seed: int = 0,
    test_frac: float = 0.2,
) -> dict:
    if split_strategy == "aoi":
        aois = AOIS_TRAIN if split == "train" else AOIS_TEST
        return {aoi: None for aoi in aois}

    elif split_strategy == "random_per_aoi":
        assignment = {}
        for aoi in ALL_AOIS:
            fp = CACHE_DIR / f"{aoi}_orbit{ORBITS[0]}.parquet"
            assert (
                fp.exists()
            ), f"Cache file {fp} not found. Run download_intermediate_assets.py first."
            ids = pd.read_parquet(fp, columns=["unosat_id"])["unosat_id"].unique()
            train_ids, test_ids = train_test_split(ids, test_size=test_frac, random_state=seed)
            assignment[aoi] = list(train_ids) if split == "train" else list(test_ids)
        return assignment

    else:
        raise ValueError(f"Unknown split_strategy: {split_strategy}")


def extract_features_local(
    split: str,
    split_strategy: str = "aoi",
    pre_period: tuple[str, str] = PRE_PERIOD,
    post_periods: list = POST_PERIODS,
    seed: int = 0,
) -> pd.DataFrame:
    aoi_point_assignment = get_aoi_point_assignment(split, split_strategy, seed=seed)
    all_features = []

    for aoi, point_ids in aoi_point_assignment.items():
        print(f"\nProcessing {aoi}...")
        for orbit in ORBITS:
            fp = CACHE_DIR / f"{aoi}_orbit{orbit}.parquet"
            assert (
                fp.exists()
            ), f"Cache file {fp} not found. Run download_intermediate_assets.py first."

            print(f"  Loading {aoi}_orbit{orbit}...")
            df = pd.read_parquet(fp)

            if point_ids is not None:
                df = df[df["unosat_id"].isin(point_ids)]

            for post_period in tqdm(post_periods, desc=f"    windows"):
                features = compute_features_for_window(df, pre_period, post_period, orbit)
                if len(features) > 0:
                    all_features.append(features)

    if not all_features:
        print(f"WARNING: No features extracted for {split} split!")
        return pd.DataFrame()

    result = pd.concat(all_features, ignore_index=True)
    print(f"\n{split} split: {len(result):,} rows, {len(result.columns)} columns")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--split_strategy", default="aoi", choices=["aoi", "random_per_aoi"])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--test_frac", type=float, default=0.2)
    args = parser.parse_args()

    FEATURES_DIR.mkdir(exist_ok=True, parents=True)
    suffix = "" if args.split_strategy == "aoi" else f"_{args.split_strategy}"
    all_periods = [PRE_PERIOD] + list(POST_PERIODS)

    print("=" * 60)
    print(f"Extracting train features (split_strategy={args.split_strategy})...")
    print("=" * 60)
    train_features = extract_features_local(
        "train", split_strategy=args.split_strategy, post_periods=all_periods, seed=args.seed
    )
    train_fp = FEATURES_DIR / f"s1_1x1_2months_train{suffix}.parquet"
    train_features.to_parquet(train_fp)
    print(f"Saved train features to {train_fp}")

    print("\n" + "=" * 60)
    print(f"Extracting test features (split_strategy={args.split_strategy})...")
    print("=" * 60)
    test_features = extract_features_local(
        "test", split_strategy=args.split_strategy, post_periods=all_periods, seed=args.seed
    )
    test_fp = FEATURES_DIR / f"s1_1x1_2months_test{suffix}.parquet"
    test_features.to_parquet(test_fp)
    print(f"Saved test features to {test_fp}")

    print("\n" + "=" * 60)
    print("Feature extraction complete!")
    print(f"Train: {len(train_features):,} rows")
    print(f"Test:  {len(test_features):,} rows")
    print("=" * 60)
