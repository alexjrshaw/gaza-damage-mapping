"""
Test mtry (max_features) values from 16 to 25 for OOB error.
Extends the ablation study to confirm the optimal mtry value.

Usage:
    python3 alex/tmp/test_mtry.py
"""

import json
from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from omegaconf import OmegaConf

from src.constants import DATA_PATH, PRE_PERIOD
from src.classification.dataset_local import get_dataset_ready_local
from src.classification.utils import get_features_names

ABLATION_DIR = DATA_PATH / "ablation_runs"
OUT_FP = ABLATION_DIR / "mtry_extended_results.json"

# Load known results from previous run
known_results = {
    1: 0.2789, 2: 0.2699, 3: 0.2671, 4: 0.2661,
    5: 0.2654, 6: 0.2655, 7: 0.2652, 8: 0.2652,
    9: 0.2652, 10: 0.2652, 11: 0.2650, 12: 0.2651,
    13: 0.2652, 14: 0.2650, 15: 0.2650, 28: 0.2666,
}

print("Loading train features...")
df_train = get_dataset_ready_local(
    sat="s1", split="train", post_dates="2months", extract_wind="1x1"
)

cfg = OmegaConf.create(dict(
    data=dict(
        s1=dict(subset_bands=None), s2=None, extract_winds="1x1",
        time_periods=dict(pre=PRE_PERIOD, post="2months"),
    ),
    reducer_names=["mean", "stdDev", "median", "min", "max", "skew", "kurtosis"],
))
feature_cols = get_features_names(cfg)
df_train = df_train.dropna(subset=feature_cols)
X = df_train[feature_cols].values
y = df_train["label"].values
print(f"Training data: {X.shape[0]:,} rows, {X.shape[1]} features")

# Test mtry 16-25
new_results = {}
for mtry in range(16, 26):
    print(f"Testing mtry={mtry}...")
    clf = RandomForestClassifier(
        n_estimators=50,
        min_samples_leaf=3,
        max_leaf_nodes=10000,
        max_features=mtry,
        oob_score=True,
        n_jobs=-1,
        random_state=0,
    )
    clf.fit(X, y)
    oob_error = 1 - clf.oob_score_
    new_results[mtry] = oob_error
    print(f"  mtry={mtry}: OOB error={oob_error:.4f}")

# Combine all results
all_results = {**known_results, **new_results}
all_results = dict(sorted(all_results.items()))

# Print full table
print("\n=== Full mtry results ===")
print(f"{'mtry':>6} {'OOB error':>12} {'note':>20}")
print("-" * 42)
min_error = min(all_results.values())
for mtry, error in all_results.items():
    note = "<-- minimum" if error == min_error else ""
    print(f"{mtry:>6} {error:>12.4f} {note:>20}")

# Find optimal
optimal_mtry = min(all_results, key=all_results.get)
print(f"\nOptimal mtry: {optimal_mtry} (OOB error={all_results[optimal_mtry]:.4f})")
print(f"Default sqrt(28)=5: OOB error={all_results[5]:.4f}")
print(f"Difference: {all_results[5] - all_results[optimal_mtry]:.4f}")

# Save
ABLATION_DIR.mkdir(exist_ok=True, parents=True)
with open(OUT_FP, "w") as f:
    json.dump({str(k): v for k, v in all_results.items()}, f, indent=2)
print(f"\nSaved to {OUT_FP}")
