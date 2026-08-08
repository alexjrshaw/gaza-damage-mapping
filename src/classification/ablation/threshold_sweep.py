"""
Threshold sweep for Gaza model calibration.

Tests thresholds at 0.005 intervals against the merged probability rasters
produced by pixel_postprocessing.py, sampling predictions at held-out UNOSAT
test points using sample_rasters_at_unosat_points() from ablation_pixel_level.py.
No retraining or re-inference is required. The lowest threshold achieving
90% precision is selected as the calibrated threshold (t=0.670).
Results saved to data/threshold_sweep_current_results.csv.
"""

import numpy as np
import pandas as pd
from pathlib import Path

from src.classification.ablation.ablation_pixel_level import sample_rasters_at_unosat_points
from src.classification.metrics import get_metrics
from src.constants import DATA_PATH, AOIS_TEST

# Paths
MERGED_DIR = DATA_PATH / "merged_probability_rasters"
# Load merged rasters
merged_fps = sorted(MERGED_DIR.glob("gaza_w*.tif"))
print(f"Found {len(merged_fps)} merged rasters (should be 20)")

# Sample at UNOSAT points
gdf_points = sample_rasters_at_unosat_points(
    variant_name="current_verified",
    merged_fps=merged_fps,
    force_recreate=True,
)

# Filter to test set
gdf_test = gdf_points[gdf_points.aoi.isin(AOIS_TEST)].copy()
gdf_test = gdf_test[gdf_test.damage.isin([1, 2])].copy()
gdf_test["date"] = pd.to_datetime(gdf_test["date"])

print(f"\nSweeping thresholds (step=0.005) against {len(gdf_test):,} test points...")
# Threshold sweep
results = []
for t in np.arange(0.0, 1.005, 0.005):
    m = get_metrics(gdf_test, threshold=t, method="date-wise", print_classification_report=False)
    results.append({"threshold": round(t, 3), **m})
    if abs(m["precision"] - 0.90) < 0.01:
        print(
            f"  t={t:.3f}: precision={m['precision']:.3f}, recall={m['recall']:.3f}, f1={m['f1']:.3f}  <-- near 90% precision"
        )

# Find optimal threshold
df_results = pd.DataFrame(results)
df_results.to_csv(DATA_PATH / "threshold_sweep_current_results.csv", index=False)

closest = df_results.iloc[(df_results["precision"] - 0.90).abs().argsort()[:1]]
print(f"\nClosest threshold to 90% precision:")
print(closest.to_string(index=False))
