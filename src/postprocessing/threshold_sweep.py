"""
Fine-grained threshold sweep against the CURRENT, freshly-regenerated
merged probability rasters (from tonight's pixel_postprocessing.py run).
Reuses sample_rasters_at_unosat_points() from ablation_pixel_level.py
directly, avoiding any retraining or re-inference.
"""

import numpy as np
import pandas as pd
from pathlib import Path

from src.classification.ablation_pixel_level import sample_rasters_at_unosat_points
from src.classification.metrics import get_metrics
from src.constants import DATA_PATH, AOIS_TEST

MERGED_DIR = DATA_PATH / "merged_probability_rasters"
merged_fps = sorted(MERGED_DIR.glob("gaza_w*.tif"))
print(f"Found {len(merged_fps)} merged rasters (should be 20)")

gdf_points = sample_rasters_at_unosat_points(
    variant_name="current_verified",
    merged_fps=merged_fps,
    force_recreate=True,
)

gdf_test = gdf_points[gdf_points.aoi.isin(AOIS_TEST)].copy()
gdf_test = gdf_test[gdf_test.damage.isin([1, 2])].copy()
gdf_test["date"] = pd.to_datetime(gdf_test["date"])

print(f"\nSweeping thresholds (step=0.005) against {len(gdf_test):,} test points...")
results = []
for t in np.arange(0.0, 1.005, 0.005):
    m = get_metrics(gdf_test, threshold=t, method="date-wise", print_classification_report=False)
    results.append({"threshold": round(t, 3), **m})
    if abs(m["precision"] - 0.90) < 0.01:
        print(
            f"  t={t:.3f}: precision={m['precision']:.3f}, recall={m['recall']:.3f}, f1={m['f1']:.3f}  <-- near 90% precision"
        )

df_results = pd.DataFrame(results)
df_results.to_csv(DATA_PATH / "threshold_sweep_current_results.csv", index=False)

closest = df_results.iloc[(df_results["precision"] - 0.90).abs().argsort()[:1]]
print(f"\nClosest threshold to 90% precision:")
print(closest.to_string(index=False))
