"""
Summarise all local classification runs.

Reads metrics.json and cfg.yaml from each run directory and produces
a single summary CSV and printed comparison table.

Mirrors the summary table produced by classification.ipynb in Dietrich et al. (2025),
adapted for the local sklearn pipeline.

Usage:
    python3 src/classification/summarise_runs.py
"""

import json
from pathlib import Path

import pandas as pd
from omegaconf import OmegaConf

from src.constants import DATA_PATH

RUNS_DIR = DATA_PATH / "runs"
OUT_FP = RUNS_DIR / "runs_summary.csv"

# Ukraine baseline from Dietrich et al. Table 1
UKRAINE_BASELINE = {
    "run_name": "Ukraine (Dietrich et al. 2025)",
    "f1": 0.749,
    "precision": 0.671,
    "recall": 0.846,
    "roc_auc": 0.813,
    "accuracy": None,
    "model": "random_forest",
    "numberOfTrees": 50,
    "split_strategy": "aoi",
    "class_weight": None,
    "run_suffix": None,
}


def load_run(run_dir: Path) -> dict | None:
    """
    Load metrics and config from a run directory.

    Args:
        run_dir: Path to the run directory.

    Returns:
        dict with run metadata and metrics, or None if incomplete.
    """
    fp_metrics = run_dir / "metrics.json"
    fp_cfg = run_dir / "cfg.yaml"

    if not fp_metrics.exists():
        return None

    with open(fp_metrics) as f:
        metrics = json.load(f)

    record = {"run_name": run_dir.name}
    record.update(metrics)

    # Load config if available
    if fp_cfg.exists():
        cfg = OmegaConf.load(fp_cfg)
        record["model"] = cfg.get("model_name", None)
        record["numberOfTrees"] = cfg.model_kwargs.get("numberOfTrees", None)
        record["split_strategy"] = cfg.data.get("split_strategy", "aoi")
        record["class_weight"] = cfg.model_kwargs.get("class_weight", None)
        record["run_suffix"] = cfg.get("run_suffix", None)
    else:
        record["model"] = None
        record["numberOfTrees"] = None
        record["split_strategy"] = None
        record["class_weight"] = None
        record["run_suffix"] = None

    return record


def summarise_runs() -> pd.DataFrame:
    """
    Load all runs and produce summary DataFrame.

    Returns:
        DataFrame with one row per run, sorted by F1 descending.
    """
    records = [UKRAINE_BASELINE]

    for run_dir in sorted(RUNS_DIR.iterdir()):
        if not run_dir.is_dir():
            continue
        record = load_run(run_dir)
        if record is not None:
            records.append(record)

    df = pd.DataFrame(records)

    # Sort by F1 descending, Ukraine first
    ukraine_row = df[df["run_name"].str.startswith("Ukraine")]
    gaza_rows = df[~df["run_name"].str.startswith("Ukraine")].sort_values("f1", ascending=False)
    df = pd.concat([ukraine_row, gaza_rows], ignore_index=True)

    return df


def print_comparison_table(df: pd.DataFrame) -> None:
    """Print formatted comparison table."""
    print("\n" + "=" * 80)
    print("CLASSIFICATION RESULTS — ALL RUNS")
    print("=" * 80)
    print(f"{'Run':<55} {'F1':>6} {'Prec':>6} {'Rec':>6} {'AUC':>6}")
    print("-" * 80)

    for _, row in df.iterrows():
        name = row["run_name"]
        if len(name) > 54:
            name = name[:51] + "..."
        f1   = f"{row['f1']:.3f}" if pd.notna(row.get("f1")) else "—"
        prec = f"{row['precision']:.3f}" if pd.notna(row.get("precision")) else "—"
        rec  = f"{row['recall']:.3f}" if pd.notna(row.get("recall")) else "—"
        auc  = f"{row['roc_auc']:.3f}" if pd.notna(row.get("roc_auc")) else "—"
        print(f"{name:<55} {f1:>6} {prec:>6} {rec:>6} {auc:>6}")

    print("=" * 80)


if __name__ == "__main__":
    df = summarise_runs()
    print_comparison_table(df)

    df.to_csv(OUT_FP, index=False)
    print(f"\nSummary saved to {OUT_FP}")
