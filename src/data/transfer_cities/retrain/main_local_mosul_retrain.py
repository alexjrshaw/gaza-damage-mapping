"""
Local training pipeline for Mosul, as a retraining comparison against
the Gaza-trained zero-shot transfer result.

Mirrors src/classification/main_local.py's training step exactly, with
two adaptations, both noted at point of use below:
    1. Data source: data/transfer_cities/features_ready/MOS_features.parquet
       instead of Gaza's data/features_ready/s1_*.parquet.
    2. Train/test split: spatial east/west split on longitude (Tigris River,
       lon = 43.1262), instead of Gaza's AOI-based split. Chosen to approximate
       Gaza's own train/test point ratio (46.4% / 53.6%): the west-bank/train
       split yields 46.0% / 54.0%, the closest match found by sweeping
       candidate longitudes against the actual point distribution.

Train (west bank, lon < 43.1262): historic Old City, west-bank neighbourhoods.
Test  (east bank, lon >= 43.1262): east-bank neighbourhoods.

This script only trains and saves the model (model.pkl), matching the
output location and format expected by Gaza's local_pixel_inference.py.
Pixel-level evaluation is intentionally NOT done here: doing so on the
tabular point-level features would not be comparable to the zero-shot
Mosul result, which is pixel-level (3x3 max window, sampled from
classified rasters). Instead:

    [this script] (train, save model.pkl)
        -> mosul_retrain_pixel_inference.py (classify Mosul's existing
           feature rasters with this model)
        -> evaluate_pixel_transfer.py (unmodified; the same script used
           for the zero-shot Mosul result, run on east-bank test points
           only - see usage note in that script's call)

Usage:
    python3 src/data/transfer_cities/retrain/main_local_mosul_retrain.py
"""

import geopandas as gpd
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

LON_SPLIT = 43.1262  # approximate Tigris River line

FEATURES_FP = "data/transfer_cities/features_ready/MOS_features.parquet"
LABELS_FP = "test_sites/processed/mos/unosat_labels.geojson"
MODEL_OUT_FP = "data/transfer_cities/runs/MOS_retrained/model.pkl"

REDUCER_NAMES = ["mean", "stdDev", "median", "min", "max", "skew", "kurtosis"]
S1_BANDS = ["VV", "VH"]


def get_feature_cols() -> list[str]:
    """Same 28-feature set and order as Gaza: 7 stats x 2 bands x (pre + post)."""
    cols = []
    for band in S1_BANDS:
        for period in ["pre_1x1", "post_1x1"]:
            for stat in REDUCER_NAMES:
                cols.append(f"{band}_{period}_{stat}")
    return cols


def load_and_split_mosul() -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load Mosul features, join geometry, apply east/west spatial split.

    Adaptation 2 (see module docstring): spatial split on longitude,
    not Gaza's AOI-based split, since Mosul has no AOI subdivisions.

    Returns (df_train, df_test). df_test is retained here only to report
    its size; pixel-level test-set geometry filtering happens later, in
    evaluate_pixel_transfer.py, on the east-bank UNOSAT points directly.
    """
    print("Loading Mosul features...")
    df = pd.read_parquet(FEATURES_FP)
    print(f"  {len(df):,} rows, {df['unosat_id'].nunique():,} unique points")

    print("Loading UNOSAT labels for geometry...")
    labels = gpd.read_file(LABELS_FP)
    lon_lookup = labels.set_index("unosat_id").geometry.x.to_dict()

    df["lon"] = df["unosat_id"].map(lon_lookup)
    assert df["lon"].notna().all(), "Some unosat_id values could not be matched to geometry"

    is_train = df["lon"] < LON_SPLIT
    df_train = df[is_train].copy()
    df_test = df[~is_train].copy()

    n_train_pts = df_train["unosat_id"].nunique()
    n_test_pts = df_test["unosat_id"].nunique()
    n_total_pts = n_train_pts + n_test_pts
    print(f"\nSpatial split at lon = {LON_SPLIT} (approx. Tigris River):")
    print(f"  Train (west bank): {n_train_pts:,} points ({n_train_pts / n_total_pts * 100:.1f}%)")
    print(f"  Test  (east bank): {n_test_pts:,} points ({n_test_pts / n_total_pts * 100:.1f}%)")
    print("  (Gaza reference ratio: 46.4% / 53.6%)")

    return df_train, df_test


def train_classifier(df_train: pd.DataFrame, feature_cols: list[str]) -> RandomForestClassifier:
    """
    Train RF with identical hyperparameters to Gaza/Dietrich et al.

    Matches src/classification/models_local.py::classifier_factory_local
    exactly: n_estimators=50, min_samples_leaf=3, max_leaf_nodes=10000,
    class_weight=None. max_features is left at sklearn's default ('sqrt'),
    which is what Gaza's model also uses implicitly (it is never set in
    classifier_factory_local).
    """
    df_train = df_train.dropna(subset=feature_cols)
    X_train = df_train[feature_cols].values
    y_train = df_train["label"].values

    print(f"\nTraining set: {len(df_train):,} rows")
    print(f"Label distribution: {pd.Series(y_train).value_counts().sort_index().to_dict()}")

    clf = RandomForestClassifier(
        n_estimators=50,
        min_samples_leaf=3,
        max_leaf_nodes=10_000,
        class_weight=None,
        n_jobs=-1,
        random_state=0,
        verbose=1,
    )
    print("Fitting Random Forest...")
    clf.fit(X_train, y_train)
    print("Training complete.")
    return clf


def main():
    df_train, df_test = load_and_split_mosul()
    feature_cols = get_feature_cols()

    clf = train_classifier(df_train, feature_cols)

    import pickle
    from pathlib import Path

    out_fp = Path(MODEL_OUT_FP)
    out_fp.parent.mkdir(exist_ok=True, parents=True)
    with open(out_fp, "wb") as f:
        pickle.dump(clf, f)
    print(f"\nModel saved to {out_fp}")
    print(
        "\nNext steps:\n"
        "  1. python3 src/data/transfer_cities/retrain/mosul_retrain_pixel_inference.py\n"
        "  2. python3 src/data/transfer_cities/pixel_inference/evaluate_pixel_transfer.py "
        "--city MOS_RETRAINED_EAST_ONLY"
    )


if __name__ == "__main__":
    main()
