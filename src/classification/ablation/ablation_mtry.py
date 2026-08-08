"""
Test mtry (max_features) values 1-28 for OOB error, on the full training set.
Resumable: loads any values already saved from an interrupted run and only
computes what's missing. Otherwise identical to the original fresh sweep.

Usage:
    screen -S mtry_sweep
    python3 src/classification/ablation/ablation_mtry.py
    # Ctrl+A D to detach
"""
import json
from pathlib import Path

from omegaconf import OmegaConf
from sklearn.ensemble import RandomForestClassifier
from tqdm.auto import tqdm

# Paths
from src.constants import DATA_PATH, PRE_PERIOD
from src.classification.dataset_local import get_dataset_ready_local
from src.classification.utils import get_features_names

ABLATION_DIR = DATA_PATH / "ablation_runs"
OUT_FP = ABLATION_DIR / "mtry_full_sweep_results.json"

# Load any progress already saved from this same run, if it was interrupted
all_results = {}
if OUT_FP.exists():
    with open(OUT_FP) as f:
        all_results = {int(k): v for k, v in json.load(f).items()}
    print(f"Resuming: {len(all_results)} values already saved: {sorted(all_results.keys())}")

missing = [m for m in range(1, 29) if m not in all_results]
if not missing:
    print("All 28 values already present - nothing to do.")
else:
    print(f"Still need: {missing}")

    print("Loading train features...")
    df_train = get_dataset_ready_local(
        sat="s1", split="train", post_dates="2months", extract_wind="1x1"
    )
    cfg = OmegaConf.create(
        dict(
            data=dict(
                s1=dict(subset_bands=None),
                extract_winds="1x1",
                time_periods=dict(pre=PRE_PERIOD, post="2months"),
            ),
            reducer_names=["mean", "stdDev", "median", "min", "max", "skew", "kurtosis"],
        )
    )
    feature_cols = get_features_names(cfg)
    df_train = df_train.dropna(subset=feature_cols)
    X = df_train[feature_cols].values
    y = df_train["label"].values
    print(f"Training data: {X.shape[0]:,} rows, {X.shape[1]} features")
    assert X.shape[1] == 28, f"Expected 28 features, found {X.shape[1]} - check reducer_names/config"

    for mtry in tqdm(missing, desc="mtry sweep (resuming)"):
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
        all_results[mtry] = oob_error
        print(f"  mtry={mtry}: OOB error={oob_error:.4f}")

        # Save incrementally after every value, in case the run is interrupted again
        with open(OUT_FP, "w") as f:
            json.dump({str(k): v for k, v in sorted(all_results.items())}, f, indent=2)

assert list(sorted(all_results.keys())) == list(range(1, 29)), "Sweep did not complete 1-28"

# Print full table
print(f"{'mtry':>6} {'OOB error':>12} {'note':>20}")
print("-" * 42)
min_error = min(all_results.values())
for mtry, error in sorted(all_results.items()):
    note = "<-- minimum" if error == min_error else ""
    print(f"{mtry:>6} {error:>12.4f} {note:>20}")

optimal_mtry = min(all_results, key=all_results.get)
print(f"\nOptimal mtry: {optimal_mtry} (OOB error={all_results[optimal_mtry]:.4f})")
print(f"Default sqrt(28)=5: OOB error={all_results[5]:.4f}")
print(f"Difference: {all_results[5] - all_results[optimal_mtry]:.4f}")
print(f"\nSaved to {OUT_FP}")
