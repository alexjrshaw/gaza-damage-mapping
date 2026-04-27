"""
Feature importance analysis for Gaza damage mapping.

Extracts and saves feature importances from the trained Random Forest,
following Dietrich et al. (2025) methodology.

Produces:
    data/runs/{run_name}/feature_importance.csv
        - feature: feature name
        - importance: mean decrease in impurity (MDI)
        - rank: rank by importance (1 = most important)
        - band: VV or VH
        - period: pre or post
        - reducer: mean, stdDev, median, min, max, skew, kurtosis

Usage:
    python3 src/classification/feature_importance.py
"""

import pickle
from pathlib import Path

import pandas as pd
from omegaconf import OmegaConf

from src.constants import DATA_PATH, PRE_PERIOD
from src.classification.utils import get_features_names

RUN_NAME = "rf_s1_2months_50trees_1x1_all7reducers_baseline"
RUNS_DIR = DATA_PATH / "runs"


def get_feature_importance(run_name: str = RUN_NAME) -> pd.DataFrame:
    fp_model = RUNS_DIR / run_name / "model.pkl"
    assert fp_model.exists(), f"Model not found: {fp_model}"

    with open(fp_model, "rb") as f:
        clf = pickle.load(f)

    cfg = OmegaConf.create(dict(
        data=dict(
            s1=dict(subset_bands=None),
            s2=None,
            extract_winds="1x1",
            time_periods=dict(pre=PRE_PERIOD, post="2months"),
        ),
        reducer_names=["mean", "stdDev", "median", "min", "max", "skew", "kurtosis"],
    ))
    feature_cols = get_features_names(cfg)

    df = pd.DataFrame({
        "feature": feature_cols,
        "importance": clf.feature_importances_,
    })

    df["band"]    = df["feature"].apply(lambda x: x.split("_")[0])
    df["period"]  = df["feature"].apply(lambda x: x.split("_")[1])
    df["window"]  = df["feature"].apply(lambda x: x.split("_")[2])
    df["reducer"] = df["feature"].apply(lambda x: x.split("_")[3])

    df = df.sort_values("importance", ascending=False).reset_index(drop=True)
    df["rank"] = df.index + 1
    df["importance"] = df["importance"].round(6)

    return df[["rank", "feature", "importance", "band", "period", "window", "reducer"]]


def print_summary(df: pd.DataFrame) -> None:
    print("\n" + "=" * 65)
    print("FEATURE IMPORTANCE — TOP 10")
    print("=" * 65)
    print(f"{'Rank':<6} {'Feature':<30} {'Importance':>12}")
    print("-" * 65)
    for _, row in df.head(10).iterrows():
        print(f"{int(row['rank']):<6} {row['feature']:<30} {row['importance']:>12.4f}")

    print("\n" + "=" * 65)
    print("FEATURE IMPORTANCE — BOTTOM 5")
    print("=" * 65)
    print(f"{'Rank':<6} {'Feature':<30} {'Importance':>12}")
    print("-" * 65)
    for _, row in df.tail(5).iterrows():
        print(f"{int(row['rank']):<6} {row['feature']:<30} {row['importance']:>12.4f}")

    print("\n" + "=" * 40)
    print("IMPORTANCE BY PERIOD")
    print("=" * 40)
    by_period = df.groupby("period")["importance"].sum().sort_values(ascending=False)
    for period, imp in by_period.items():
        print(f"  {period:<10} {imp:.4f} ({imp*100:.1f}%)")

    print("\n" + "=" * 40)
    print("IMPORTANCE BY BAND")
    print("=" * 40)
    by_band = df.groupby("band")["importance"].sum().sort_values(ascending=False)
    for band, imp in by_band.items():
        print(f"  {band:<10} {imp:.4f} ({imp*100:.1f}%)")

    print("\n" + "=" * 40)
    print("IMPORTANCE BY REDUCER")
    print("=" * 40)
    by_reducer = df.groupby("reducer")["importance"].sum().sort_values(ascending=False)
    for reducer, imp in by_reducer.items():
        print(f"  {reducer:<12} {imp:.4f} ({imp*100:.1f}%)")


if __name__ == "__main__":
    df = get_feature_importance()
    print_summary(df)
    fp_out = RUNS_DIR / RUN_NAME / "feature_importance.csv"
    df.to_csv(fp_out, index=False)
    print(f"\nSaved to {fp_out}")
