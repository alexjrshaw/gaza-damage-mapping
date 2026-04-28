"""
Feature extraction with alternative train/test split strategies.

Extends extract_features_local.py with two additional split strategies
for ablation studies:

    'random_all'     -- random 20:80 split across all Gaza points pooled
    'random_per_aoi' -- random 20:80 split within each governorate separately

The baseline AOI-based split (GAZ1+GAZ2 train, GAZ3+GAZ4+GAZ5 test) is
handled by the original extract_features_local.py and is not repeated here.

Splits are performed at the site_id level to avoid data leakage — all
rows belonging to the same physical point stay in the same split.

Outputs are saved to separate files to preserve all three baselines:
    s1_1x1_2months_{split}_random_all.parquet
    s1_1x1_2months_{split}_random_per_aoi.parquet
"""

import numpy as np
import pandas as pd
from pathlib import Path
from tqdm.auto import tqdm

from src.constants import (
    AOIS,
    DATA_PATH,
    GAZA_WAR_START, PRE_PERIOD, POST_PERIODS,
)
from src.data.sentinel1.extract_features_local import (
    compute_features_for_window,
    CACHE_DIR, ORBITS, EXTRACT_WINDOW, REDUCER_NAMES,
)

FEATURES_DIR = DATA_PATH / "features_ready"
RANDOM_SEED = 42
TRAIN_FRAC = 0.2


# ==================== SPLIT ASSIGNMENT ====================

def get_random_all_splits(seed: int = RANDOM_SEED) -> dict[str, set]:
    """
    Assign all unique site_ids across Gaza to train or test randomly.
    20% train, 80% test.

    Returns:
        dict with keys 'train' and 'test', values are sets of site_ids.
    """
    # Load all site_ids from all AOIs
    all_site_ids = set()
    for aoi in AOIS:
        fp = CACHE_DIR / f"{aoi}_orbit87.parquet"  # only need one orbit for IDs
        df = pd.read_parquet(fp, columns=["site_id"])
        all_site_ids.update(df["site_id"].unique())

    all_site_ids = sorted(all_site_ids)
    rng = np.random.default_rng(seed)
    rng.shuffle(all_site_ids)

    n_train = int(len(all_site_ids) * TRAIN_FRAC)
    train_ids = set(all_site_ids[:n_train])
    test_ids  = set(all_site_ids[n_train:])

    print(f"random_all split: {len(train_ids):,} train site_ids, {len(test_ids):,} test site_ids")
    return {"train": train_ids, "test": test_ids}


def get_random_per_aoi_splits(seed: int = RANDOM_SEED) -> dict[str, dict[str, set]]:
    """
    Assign site_ids to train or test randomly within each AOI.
    20% train, 80% test per governorate.

    Returns:
        dict[aoi] -> dict with keys 'train' and 'test', values are sets of site_ids.
    """
    splits = {}
    for aoi in AOIS:
        fp = CACHE_DIR / f"{aoi}_orbit87.parquet"
        df = pd.read_parquet(fp, columns=["site_id"])
        site_ids = sorted(df["site_id"].unique())

        rng = np.random.default_rng(seed)
        rng.shuffle(site_ids)

        n_train = int(len(site_ids) * TRAIN_FRAC)
        splits[aoi] = {
            "train": set(site_ids[:n_train]),
            "test":  set(site_ids[n_train:]),
        }
        print(f"  {aoi}: {len(splits[aoi]['train']):,} train, {len(splits[aoi]['test']):,} test")

    return splits


# ==================== FEATURE EXTRACTION ====================

def extract_features_random_all(
    split: str,
    site_ids: set,
    pre_period: tuple[str, str] = PRE_PERIOD,
    post_periods: list[tuple[str, str]] = None,
) -> pd.DataFrame:
    """
    Extract features for a random_all split — all AOIs, filtered by site_id.

    Args:
        split: 'train' or 'test' (used for logging only)
        site_ids: Set of site_ids belonging to this split
        pre_period: Pre-conflict date range
        post_periods: List of post-conflict date ranges

    Returns:
        DataFrame with all features ready for classification
    """
    if post_periods is None:
        post_periods = [PRE_PERIOD] + list(POST_PERIODS)

    all_features = []

    for aoi in AOIS:
        print(f"\nProcessing {aoi}...")
        for orbit in ORBITS:
            fp = CACHE_DIR / f"{aoi}_orbit{orbit}.parquet"
            assert fp.exists(), f"Cache file {fp} not found."

            print(f"  Loading {aoi}_orbit{orbit}...")
            df = pd.read_parquet(fp)

            # Filter to only site_ids belonging to this split
            df = df[df["site_id"].isin(site_ids)]
            if len(df) == 0:
                continue

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
    print(f"\n{split} split (random_all): {len(result):,} rows, {len(result.columns)} columns")
    return result


def extract_features_random_per_aoi(
    split: str,
    aoi_site_ids: dict[str, set],
    pre_period: tuple[str, str] = PRE_PERIOD,
    post_periods: list[tuple[str, str]] = None,
) -> pd.DataFrame:
    """
    Extract features for a random_per_aoi split — each AOI filtered by its site_ids.

    Args:
        split: 'train' or 'test' (used for logging only)
        aoi_site_ids: Dict mapping aoi -> set of site_ids for this split
        pre_period: Pre-conflict date range
        post_periods: List of post-conflict date ranges

    Returns:
        DataFrame with all features ready for classification
    """
    if post_periods is None:
        post_periods = [PRE_PERIOD] + list(POST_PERIODS)

    all_features = []

    for aoi in AOIS:
        site_ids = aoi_site_ids[aoi]
        print(f"\nProcessing {aoi} ({len(site_ids):,} {split} site_ids)...")

        for orbit in ORBITS:
            fp = CACHE_DIR / f"{aoi}_orbit{orbit}.parquet"
            assert fp.exists(), f"Cache file {fp} not found."

            print(f"  Loading {aoi}_orbit{orbit}...")
            df = pd.read_parquet(fp)

            # Filter to only site_ids belonging to this split
            df = df[df["site_id"].isin(site_ids)]
            if len(df) == 0:
                continue

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
    print(f"\n{split} split (random_per_aoi): {len(result):,} rows, {len(result.columns)} columns")
    return result


# ==================== MAIN ====================

if __name__ == "__main__":
    import sys
    FEATURES_DIR.mkdir(exist_ok=True, parents=True)

    # Which strategy to run — pass as argument: random_all or random_per_aoi
    strategy = sys.argv[1] if len(sys.argv) > 1 else "random_all"
    assert strategy in ("random_all", "random_per_aoi"), \
        f"Unknown strategy: {strategy}. Use 'random_all' or 'random_per_aoi'."

    all_periods = [PRE_PERIOD] + list(POST_PERIODS)

    print("=" * 60)
    print(f"Split strategy: {strategy}")
    print(f"Train fraction: {TRAIN_FRAC} ({int(TRAIN_FRAC*100)}% train, {int((1-TRAIN_FRAC)*100)}% test)")
    print(f"Random seed: {RANDOM_SEED}")
    print("=" * 60)

    if strategy == "random_all":
        print("\nAssigning site_ids to splits...")
        splits = get_random_all_splits()

        print("\n" + "=" * 60)
        print("Extracting train features...")
        print("=" * 60)
        train_features = extract_features_random_all("train", splits["train"], post_periods=all_periods)
        train_fp = FEATURES_DIR / "s1_1x1_2months_train_random_all.parquet"
        train_features.to_parquet(train_fp)
        print(f"Saved train features to {train_fp}")

        print("\n" + "=" * 60)
        print("Extracting test features...")
        print("=" * 60)
        test_features = extract_features_random_all("test", splits["test"], post_periods=all_periods)
        test_fp = FEATURES_DIR / "s1_1x1_2months_test_random_all.parquet"
        test_features.to_parquet(test_fp)
        print(f"Saved test features to {test_fp}")

    elif strategy == "random_per_aoi":
        print("\nAssigning site_ids to splits per AOI...")
        aoi_splits = get_random_per_aoi_splits()

        train_aoi_ids = {aoi: aoi_splits[aoi]["train"] for aoi in AOIS}
        test_aoi_ids  = {aoi: aoi_splits[aoi]["test"]  for aoi in AOIS}

        print("\n" + "=" * 60)
        print("Extracting train features...")
        print("=" * 60)
        train_features = extract_features_random_per_aoi("train", train_aoi_ids, post_periods=all_periods)
        train_fp = FEATURES_DIR / "s1_1x1_2months_train_random_per_aoi.parquet"
        train_features.to_parquet(train_fp)
        print(f"Saved train features to {train_fp}")

        print("\n" + "=" * 60)
        print("Extracting test features...")
        print("=" * 60)
        test_features = extract_features_random_per_aoi("test", test_aoi_ids, post_periods=all_periods)
        test_fp = FEATURES_DIR / "s1_1x1_2months_test_random_per_aoi.parquet"
        test_features.to_parquet(test_fp)
        print(f"Saved test features to {test_fp}")

    print("\n" + "=" * 60)
    print("Feature extraction complete!")
    print(f"Train: {len(train_features):,} rows")
    print(f"Test:  {len(test_features):,} rows")
    print("=" * 60)
