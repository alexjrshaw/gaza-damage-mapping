"""
Extract SAR features for transfer city zero-shot evaluation.

Local equivalent of src/data/sentinel1/extract_features_local.py,
adapted for the three transfer cities (Aleppo, Raqqa, Mosul).

Reads from local parquet cache downloaded by download_intermediate_assets.py
and computes 28 SAR features (7 statistics x VV+VH x pre+post) per
UNOSAT point per time window.

Mirrors compute_features_for_window() and extract_features_local() exactly,
with city-specific time periods and conflict start dates from constants_transfer.py.

Label assignment (Dietrich et al. eq. 1):
    label=0  if end_post <= conflict_start
    label=1  if end_post > conflict_start AND date_first_severe <= end_post
    discard  otherwise (no damage confirmed before end of window)

Output:
    data/transfer_cities/features_ready/{city_id}_features.parquet

Usage (as Slurm batch job):
    sbatch run_extract_features_transfer.sh

Or interactively:
    cd /scratch/s1214882/gaza-damage-mapping
    python3 src/data/transfer_cities/extract_features_transfer.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

sys.path.insert(0, "/scratch/s1214882/gaza-damage-mapping")

from src.data.transfer_cities.constants_transfer import (S1_BANDS, TRANSFER_CACHE_DIR, TRANSFER_CITIES,
                                                         TRANSFER_FEATURES_DIR)

EXTRACT_WINDOW = "1x1"
REDUCER_NAMES = ["mean", "stdDev", "median", "min", "max", "skew", "kurtosis"]


# ── Feature computation (identical to extract_features_local.py) ───────────────


def compute_features_for_window(
    df: pd.DataFrame,
    pre_period: tuple,
    post_period: tuple,
    orbit: int,
    conflict_start: str,
) -> pd.DataFrame:
    """
    Compute pre/post features for one city/orbit/time window combination.

    Mirrors compute_features_for_window() in extract_features_local.py exactly.
    Only difference: uses city-specific conflict_start instead of GAZA_WAR_START.
    """
    df = df.copy()
    df["s1_date"] = pd.to_datetime(df["system:time_start"], unit="ms").dt.date.astype(str)
    df["date_first_severe"] = df["date_first_severe"].astype(str)

    # Label assignment — Dietrich et al. eq. 1
    end_post = post_period[1]
    label = 0 if end_post <= conflict_start else 1

    # For label=1: only keep points where damage confirmed before end_post
    if label == 1:
        df = df[df["date_first_severe"] <= end_post].copy()

    if len(df) == 0:
        return pd.DataFrame()

    prefix_pre = f"pre_{EXTRACT_WINDOW}"
    prefix_post = f"post_{EXTRACT_WINDOW}"

    # Filter to pre and post date ranges
    pre_df = df[(df["s1_date"] >= pre_period[0]) & (df["s1_date"] <= pre_period[1])]
    post_df = df[(df["s1_date"] >= post_period[0]) & (df["s1_date"] <= post_period[1])]

    # Point metadata
    meta = df.groupby("unosat_id").first()[["damage", "aoi", "date_first_severe", "site_id"]].reset_index()
    meta = meta.rename(columns={"date_first_severe": "date"})

    results = meta.copy()
    results["label"] = label
    results["orbit"] = orbit
    results["start_pre"] = pre_period[0]
    results["end_pre"] = pre_period[1]
    results["start_post"] = post_period[0]
    results["end_post"] = post_period[1]

    # Compute 7 statistics for each band and period
    for band in S1_BANDS:
        for period_df, prefix in [(pre_df, prefix_pre), (post_df, prefix_post)]:
            if len(period_df) > 0:
                period_df = period_df.copy()
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


def extract_features_city(city_id: str, cfg: dict) -> pd.DataFrame:
    """
    Extract features for all orbits and time windows for one city.

    Mirrors extract_features_local() in extract_features_local.py.
    """
    all_features = []
    pre_period = cfg["pre_period"]
    post_periods = cfg["post_periods"]
    conflict_start = cfg["conflict_start"]

    print(f"\n  Pre-period:     {pre_period[0]} → {pre_period[1]}")
    print(f"  Post-windows:   {len(post_periods)} total")
    print(f"  Conflict start: {conflict_start}")

    label_0 = [(s, e) for s, e in post_periods if e <= conflict_start]
    label_1 = [(s, e) for s, e in post_periods if e > conflict_start]
    print(f"  label=0 windows: {len(label_0)}")
    print(f"  label=1 windows: {len(label_1)}")

    for orbit in cfg["orbits"]:
        fp = TRANSFER_CACHE_DIR / f"{city_id}_orbit{orbit}.parquet"
        assert fp.exists(), f"Cache file not found: {fp}"

        print(f"\n  Loading {city_id}_orbit{orbit}...")
        df = pd.read_parquet(fp)
        print(f"  {len(df):,} rows in cache")

        for post_period in tqdm(post_periods, desc=f"    {city_id}_orbit{orbit}"):
            features = compute_features_for_window(df, pre_period, post_period, orbit, conflict_start)
            if len(features) > 0:
                all_features.append(features)

    if not all_features:
        print(f"  WARNING: No features extracted for {city_id}!")
        return pd.DataFrame()

    result = pd.concat(all_features, ignore_index=True)
    print(f"\n  {city_id}: {len(result):,} rows, {len(result.columns)} columns")

    # Label distribution
    label_counts = result["label"].value_counts().sort_index()
    print(f"  Label distribution: {label_counts.to_dict()}")

    return result


def extract_all_transfer_features() -> None:
    """Extract features for all three transfer cities."""
    TRANSFER_FEATURES_DIR.mkdir(parents=True, exist_ok=True)

    for city_id, cfg in TRANSFER_CITIES.items():
        print(f"\n{'='*60}")
        print(f"{city_id} — {cfg['city_name']} ({cfg['country']})")
        print(f"{'='*60}")

        fp_out = TRANSFER_FEATURES_DIR / f"{city_id}_features.parquet"
        if fp_out.exists():
            print(f"  Already exists — skipping: {fp_out.name}")
            continue

        features = extract_features_city(city_id, cfg)

        if len(features) == 0:
            print(f"  WARNING: No features for {city_id} — skipping save")
            continue

        features.to_parquet(fp_out)
        print(f"  Saved -> {fp_out}")

    # Final summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    for city_id in TRANSFER_CITIES:
        fp = TRANSFER_FEATURES_DIR / f"{city_id}_features.parquet"
        if fp.exists():
            df = pd.read_parquet(fp)
            label_counts = df["label"].value_counts().sort_index().to_dict()
            print(f"  {city_id}: {len(df):,} rows — labels: {label_counts}")
        else:
            print(f"  {city_id}: MISSING")


if __name__ == "__main__":
    extract_all_transfer_features()
