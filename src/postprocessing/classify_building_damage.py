"""
Apply Equation 3 from Dietrich et al. (2025) to produce final binary
building damage classifications.

Equation 3:
    ŷ_j = 1  if max(T_{post}) >= t AND max(T_{pre}) < t
    ŷ_j = 0  otherwise

Where:
    - T_{post} = post-war windows (>= 2023-10-07)
    - T_{pre}  = pre-war windows  (<  2023-10-07)
    - t        = threshold (0.5 × 255 = 127 on 0-255 scale)

This mirrors metrics.py's date-wise evaluation logic exactly,
applied to the pixel-level buildings_preds.parquet output.

Output:
    data/pixel_postprocessing/buildings_damage.parquet
        - binary 'damaged' column (0 or 1)
        - max_post: max prediction across post-war windows
        - max_pre:  max prediction across pre-war windows
        - all building metadata

Usage:
    python3 src/postprocessing/apply_equation3.py
"""

import pandas as pd
from src.constants import DATA_PATH, GAZA_WAR_START

THRESHOLD = int(0.670 * 255)  # = nearly matches Dietrich et al. (0.655)
INPUT_FP  = DATA_PATH / "pixel_postprocessing/buildings_preds.parquet"
OUTPUT_FP = DATA_PATH / "pixel_postprocessing/buildings_damage.parquet"


def apply_equation3(threshold: int = THRESHOLD) -> pd.DataFrame:
    """
    Apply Dietrich et al. Equation 3 to produce binary building damage.

    Args:
        threshold: Damage threshold on 0-255 scale. Default 127 (= 0.5).

    Returns:
        DataFrame with binary damage classification per building.
    """
    print(f"Loading {INPUT_FP}...")
    df = pd.read_parquet(INPUT_FP)

    # Identify date columns
    date_cols = [c for c in df.columns if isinstance(c, str) and len(c) == 10 and c[4] == "-"]
    pre_cols  = [c for c in date_cols if c <  GAZA_WAR_START]
    post_cols = [c for c in date_cols if c >= GAZA_WAR_START]

    print(f"Pre-war windows ({len(pre_cols)}):  {pre_cols[0]} to {pre_cols[-1]}")
    print(f"Post-war windows ({len(post_cols)}): {post_cols[0]} to {post_cols[-1]}")
    print(f"Threshold: {threshold} (= {threshold/255:.2f} probability)")

    # Equation 3
    max_post = df[post_cols].max(axis=1)
    max_pre  = df[pre_cols].max(axis=1)
    damaged  = ((max_post >= threshold) & (max_pre < threshold)).astype(int)

    # Build output
    meta_cols = ["area_m2", "lon", "lat", "adm2_name", "adm2_id"]
    result = df[meta_cols].copy()
    result["max_pre"]  = max_pre
    result["max_post"] = max_post
    result["damaged"]  = damaged

    # Summary
    print(f"\nTotal buildings: {len(result):,}")
    print(f"Damaged (Eq. 3): {damaged.sum():,} ({damaged.mean()*100:.1f}%)")
    print(f"Undamaged:       {(1-damaged).sum():,} ({(1-damaged).mean()*100:.1f}%)")

    print("\n=== Per-governorate damage summary ===")
    summary = result.groupby("adm2_name").agg(
        n_buildings=("damaged", "count"),
        n_damaged=("damaged", "sum"),
        pct_damaged=("damaged", lambda x: x.mean() * 100),
        mean_max_post=("max_post", "mean"),
    ).round(2)
    print(summary.to_string())

    # Save
    result.to_parquet(OUTPUT_FP)
    print(f"\nSaved to {OUTPUT_FP}")

    return result


if __name__ == "__main__":
    apply_equation3()
