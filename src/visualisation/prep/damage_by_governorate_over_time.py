"""
Per-governorate cumulative damage over time, for grounding the Results
"spatial evolution of damage" paragraph in real, verified numbers rather
than unverified narrative claims -- mirrors the kind of governorate-by-
governorate, date-by-date figures Scher and Van Den Hoek (2025b) report
for Gaza (e.g. "by the end of November 2023, over half of North Gaza...").

CORRECTED VERSION: the original version of this script only checked
whether a building's post-war prediction crossed the threshold in some
window, without also checking the pre-war exclusion that Dietrich et
al.'s Equation 3 requires (max_pre < threshold). This meant a building
whose PRE-war windows also happened to cross the threshold (a likely
false positive, not genuine war damage) was still counted as "damaged",
inflating every governorate's final-window total above the validated
Table 3 figures (e.g. Rafah: 91.4% here vs the correct 82.6% in Table 3).
This version applies the full Equation 3 condition -- a building is only
ever eligible to be classified as damaged if its pre-war maximum stays
below the threshold -- so the final window's totals now agree exactly
with Table 3.

Usage:
    python3 src/visualisation/prep/damage_by_governorate_over_time.py
"""

import pandas as pd

from src.constants import DATA_PATH

PREDS_FP = DATA_PATH / "pixel_postprocessing/buildings_preds.parquet"
OUT_FP = DATA_PATH / "ablation_runs/figures/damage_by_governorate_over_time.csv"

THRESHOLD = int(0.670 * 255)

PRE_WINDOW_COLS = [
    "2021-10-07",
    "2022-10-07",
    "2022-12-07",
    "2023-02-07",
    "2023-04-07",
    "2023-06-07",
    "2023-08-07",
]
POST_WINDOW_COLS = [
    "2023-10-07",
    "2023-12-07",
    "2024-02-07",
    "2024-04-07",
    "2024-06-07",
    "2024-08-07",
    "2024-10-07",
    "2024-12-07",
    "2025-02-07",
    "2025-04-07",
    "2025-06-07",
    "2025-08-07",
    "2025-10-07",
]
# End dates, matching Results Table 4 exactly
WINDOW_END_DATES = [
    "2023-12-06",
    "2024-02-06",
    "2024-04-06",
    "2024-06-06",
    "2024-08-06",
    "2024-10-06",
    "2024-12-06",
    "2025-02-06",
    "2025-04-06",
    "2025-06-06",
    "2025-08-06",
    "2025-10-06",
    "2025-12-06",
]


def first_damaged_window_index(
    row: pd.Series, pre_cols: list[str], post_cols: list[str]
) -> int | None:
    """
    First post-war window index at which this building is classified as
    damaged, applying the full Equation 3 condition: eligible only if the
    pre-war maximum stays below threshold, then the first post-war window
    crossing the threshold.
    """
    max_pre = row[pre_cols].max()
    if max_pre >= THRESHOLD:
        return None  # excluded by Equation 3's pre-war condition -- never "damaged"
    for i, col in enumerate(post_cols):
        if row[col] >= THRESHOLD:
            return i
    return None


def main():
    print("Loading building predictions...")
    df = pd.read_parquet(PREDS_FP)
    print(f"  {len(df):,} buildings")
    print(f"  Governorates: {sorted(df['adm2_name'].unique())}")

    print(
        "Finding first-detected window index per building (Equation 3, pre-war exclusion applied)..."
    )
    df["first_idx"] = df.apply(
        lambda row: first_damaged_window_index(row, PRE_WINDOW_COLS, POST_WINDOW_COLS), axis=1
    )

    n_damaged_final = df["first_idx"].notna().sum()
    print(
        f"  Total damaged (final window): {n_damaged_final:,} ({n_damaged_final/len(df)*100:.1f}%)"
    )
    print("  (Should match Table 1's total: 151,368 / 220,820 = 68.5%)")

    governorates = sorted(df["adm2_name"].unique())
    results = []

    for gov in governorates:
        gov_df = df[df["adm2_name"] == gov]
        n_total = len(gov_df)
        for i, end_date in enumerate(WINDOW_END_DATES):
            n_damaged = (gov_df["first_idx"] <= i).sum()
            pct = n_damaged / n_total * 100
            results.append(
                {
                    "governorate": gov,
                    "window_end_date": end_date,
                    "n_total": n_total,
                    "n_damaged_cumulative": int(n_damaged),
                    "pct_damaged_cumulative": round(pct, 1),
                }
            )

    results_df = pd.DataFrame(results)

    pivot = results_df.pivot(
        index="governorate", columns="window_end_date", values="pct_damaged_cumulative"
    )
    pivot = pivot[WINDOW_END_DATES]  # keep chronological column order
    print(pivot.to_string())

    print(
        "Table 3 (validated): North Gaza 81.2, Rafah 81.9, Gaza 71.3, Khan Younis 71.3, Deir al-Balah 36.6"
    )
    print(f"This script (final window, {WINDOW_END_DATES[-1]}):")
    print(pivot[WINDOW_END_DATES[-1]].to_string())

    OUT_FP.parent.mkdir(exist_ok=True, parents=True)
    results_df.to_csv(OUT_FP, index=False)
    print(f"\nSaved full data to {OUT_FP}")


if __name__ == "__main__":
    main()
