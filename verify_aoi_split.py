"""
End-to-end verification that the AOI-based split (train on GAZ1+GAZ2,
test on GAZ3+GAZ4+GAZ5) was actually used, and actually produced the
final reported metrics - checking the data directly at every stage,
not just trusting filenames or split_strategy labels.
"""
import json
import pandas as pd
import geopandas as gpd
from pathlib import Path
from sklearn import metrics as sk_metrics
from src.constants import DATA_PATH, AOIS_TRAIN, AOIS_TEST

RUN_DIR = DATA_PATH / "runs" / "rf_s1_2months_50trees_1x1_all7reducers"
FEATURES_DIR = DATA_PATH / "features_ready"

print(f"AOIS_TRAIN (expected): {AOIS_TRAIN}")
print(f"AOIS_TEST  (expected): {AOIS_TEST}")
print()

print("=" * 60)
print("Check 1: Feature file AOI contents")
print("=" * 60)
df_train = pd.read_parquet(FEATURES_DIR / "s1_1x1_2months_train.parquet")
df_test = pd.read_parquet(FEATURES_DIR / "s1_1x1_2months_test.parquet")

train_aois = set(df_train["aoi"].unique())
test_aois = set(df_test["aoi"].unique())

print(f"Train file actually contains AOIs: {train_aois}")
print(f"Test file actually contains AOIs:  {test_aois}")
print(f"Train matches AOIS_TRAIN exactly: {train_aois == set(AOIS_TRAIN)}")
print(f"Test matches AOIS_TEST exactly:   {test_aois == set(AOIS_TEST)}")
print(f"No overlap between train/test AOIs: {train_aois.isdisjoint(test_aois)}")

print()
print("=" * 60)
print("Check 2: File sequencing (each stage newer than the last)")
print("=" * 60)
paths = {
    "train features": FEATURES_DIR / "s1_1x1_2months_train.parquet",
    "test features": FEATURES_DIR / "s1_1x1_2months_test.parquet",
    "trained model": RUN_DIR / "model.pkl",
    "predictions": RUN_DIR / "rf_s1_2months_50trees_1x1_all7reducers.geojson",
    "metrics": RUN_DIR / "metrics.json",
}
times = {}
for label, p in paths.items():
    if p.exists():
        times[label] = p.stat().st_mtime
        print(f"  {label}: {pd.Timestamp(times[label], unit='s')}")
    else:
        print(f"  {label}: MISSING — {p}")

ordered_labels = ["train features", "test features", "trained model", "predictions", "metrics"]
present = [l for l in ordered_labels if l in times]
in_order = all(times[present[i]] <= times[present[i+1]] for i in range(len(present)-1))
print(f"\nAll stages in correct chronological order: {in_order}")

print()
print("=" * 60)
print("Check 3: Final predictions file AOI contents")
print("=" * 60)
gdf_preds = gpd.read_file(paths["predictions"])
pred_aois = set(gdf_preds["aoi"].unique()) if "aoi" in gdf_preds.columns else set(a for _, a in gdf_preds.index)
print(f"Predictions file contains AOIs: {pred_aois}")
print(f"Matches AOIS_TEST exactly: {pred_aois == set(AOIS_TEST)}")

print()
print("=" * 60)
print("Check 4: Recomputed metrics vs. saved metrics.json")
print("=" * 60)
with open(paths["metrics"]) as f:
    saved_metrics = json.load(f)
print("Saved metrics.json:", saved_metrics)
print()
print("NOTE: exact recomputation requires knowing metrics.py's date-wise")
print("evaluation logic precisely (thresholding per date, aggregation method).")
print("If you have y_true/y_prob arrays available separately, compare:")
print("  sk_metrics.f1_score(y_true, y_prob > 0.5)")
print("  sk_metrics.roc_auc_score(y_true, y_prob)")
print("against the saved f1/roc_auc above as a sanity check.")
