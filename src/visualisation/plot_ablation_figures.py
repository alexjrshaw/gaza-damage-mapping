"""
Plot ablation study figures for Gaza damage mapping.
Reads only pixel-level results from data/ablation_runs/pixel_level/results.json,
produced by src/classification/ablation_pixel_level.py.
Replaces the hybrid point-level/pixel-level version in src/classification/.
Figures saved to data/ablation_runs/figures/
Usage:
    python3 src/visualisation/plot_ablation_figures.py
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from src.constants import DATA_PATH

FIGURES_DIR = DATA_PATH / "ablation_runs" / "figures"
PIXEL_JSON = DATA_PATH / "ablation_runs" / "pixel_level" / "results.json"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

DPI = 200
BLUE = "#377eb8"
ORANGE = "#ff7f00"

# Baseline: VV+VH, all 7 reducers, n=50 trees, t=0.670
BASELINE_F1 = 0.879
BASELINE_AUC = 0.889


def load_results() -> dict:
    with open(PIXEL_JSON) as f:
        return json.load(f)


# ── Figure 1: OOB error vs n_trees ────────────────────────────────────────────
def plot_oob_ntrees(data: dict):
    oob = data["oob_n_trees"]
    n_trees = oob["n_trees"]
    oob_error = oob["oob_error"]

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(n_trees, oob_error, color=BLUE, marker="o", markersize=4)
    ax.axvline(50, color="grey", linestyle="--", linewidth=0.8, label="n=50 (chosen)")
    ax.set_xlabel("Number of trees")
    ax.set_ylabel("OOB error")
    ax.set_title("OOB error vs number of trees")
    ax.legend()
    fig.tight_layout()
    fp = FIGURES_DIR / "oob_vs_ntrees.png"
    fig.savefig(fp, dpi=DPI)
    plt.close(fig)
    print(f"Saved: {fp}")


# ── Figure 2: OOB error vs mtry ───────────────────────────────────────────────
def plot_oob_mtry(data: dict):
    oob = data["oob_mtry"]
    mtry = oob["mtry"]
    oob_error = oob["oob_error"]

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(mtry, oob_error, color=ORANGE, marker="o", markersize=4)
    ax.set_xlabel("mtry (features considered per split)")
    ax.set_ylabel("OOB error")
    ax.set_title("OOB error vs mtry")
    fig.tight_layout()
    fp = FIGURES_DIR / "oob_vs_mtry.png"
    fig.savefig(fp, dpi=DPI)
    plt.close(fig)
    print(f"Saved: {fp}")


# ── Figure 3: Band ablation ────────────────────────────────────────────────────
def plot_band_ablation(data: dict):
    bands = {
        "VV only": data["ablation_bands_VV"],
        "VH only": data["ablation_bands_VH"],
        "VV + VH\n(chosen)": {"f1": BASELINE_F1, "auc": BASELINE_AUC},
    }
    labels = list(bands.keys())
    f1_vals = [bands[k]["f1"] for k in labels]
    auc_vals = [bands[k]["auc"] for k in labels]
    x = np.arange(len(labels))
    w = 0.35

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(x - w / 2, f1_vals, w, color=BLUE, label="F1", edgecolor="white")
    ax.bar(x + w / 2, auc_vals, w, color=ORANGE, label="Balanced accuracy", edgecolor="white")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Score (t=0.670)")
    ax.set_title("Band ablation (pixel-level)")
    ax.set_ylim(0.7, 1.0)
    ax.legend()
    fig.tight_layout()
    fp = FIGURES_DIR / "band_ablation.png"
    fig.savefig(fp, dpi=DPI)
    plt.close(fig)
    print(f"Saved: {fp}")


# ── Figure 4: Feature subset ablation ─────────────────────────────────────────
def plot_reducer_ablation(data: dict):
    configs = {
        "Mean + SD": data["ablation_reducers_mean_std"],
        "Mean + SD\n+ median": data["ablation_reducers_mean_std_median"],
        "5 statistics\n(no skew/kurt)": data["ablation_reducers_no_skew_kurt"],
        "All 7\n(chosen)": {"f1": BASELINE_F1, "auc": BASELINE_AUC},
    }
    labels = list(configs.keys())
    f1_vals = [configs[k]["f1"] for k in labels]
    auc_vals = [configs[k]["auc"] for k in labels]
    x = np.arange(len(labels))
    w = 0.35

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(x - w / 2, f1_vals, w, color=BLUE, label="F1", edgecolor="white")
    ax.bar(x + w / 2, auc_vals, w, color=ORANGE, label="Balanced accuracy", edgecolor="white")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("Score (t=0.670)")
    ax.set_title("Feature subset ablation (pixel-level)")
    ax.set_ylim(0.7, 1.0)
    ax.legend()
    fig.tight_layout()
    fp = FIGURES_DIR / "reducer_ablation.png"
    fig.savefig(fp, dpi=DPI)
    plt.close(fig)
    print(f"Saved: {fp}")


# ── Figure 5: F1 vs n_trees ───────────────────────────────────────────────────
def plot_f1_ntrees(data: dict):
    tree_keys = sorted(
        [k for k in data.keys() if k.startswith("ablation_ntrees_")],
        key=lambda k: int(k.split("_")[-1]),
    )
    n_trees = [int(k.split("_")[-1]) for k in tree_keys]
    f1_vals = [data[k]["f1"] for k in tree_keys]
    auc_vals = [data[k]["auc"] for k in tree_keys]

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(n_trees, f1_vals, color=BLUE, marker="o", markersize=4, label="F1")
    ax.plot(n_trees, auc_vals, color=ORANGE, marker="s", markersize=4, label="Balanced accuracy")
    ax.axvline(50, color="grey", linestyle="--", linewidth=0.8, label="n=50 (chosen)")
    ax.axhline(BASELINE_F1, color=BLUE, linestyle=":", linewidth=0.8)
    ax.axhline(BASELINE_AUC, color=ORANGE, linestyle=":", linewidth=0.8)
    ax.set_xlabel("Number of trees")
    ax.set_ylabel("Score (t=0.670)")
    ax.set_title("Pixel-level performance vs number of trees")
    ax.legend()
    fig.tight_layout()
    fp = FIGURES_DIR / "f1_vs_ntrees.png"
    fig.savefig(fp, dpi=DPI)
    plt.close(fig)
    print(f"Saved: {fp}")


# ── Figure 6: Summary bar chart ────────────────────────────────────────────────
def plot_summary(data: dict):
    """Single bar chart of F1 across all ablation settings at t=0.670."""
    labels, f1_vals, colours = [], [], []

    # Bands
    for k, label in [
        ("ablation_bands_VV", "VV only"),
        ("ablation_bands_VH", "VH only"),
    ]:
        labels.append(label)
        f1_vals.append(data[k]["f1"])
        colours.append(BLUE)

    # n_trees
    tree_keys = sorted(
        [k for k in data.keys() if k.startswith("ablation_ntrees_")],
        key=lambda k: int(k.split("_")[-1]),
    )
    for k in tree_keys:
        n = int(k.split("_")[-1])
        labels.append(f"n={n}")
        f1_vals.append(data[k]["f1"])
        colours.append(ORANGE)

    # Feature subsets
    for k, label in [
        ("ablation_reducers_mean_std", "Mean+SD"),
        ("ablation_reducers_mean_std_median", "Mean+SD+med"),
        ("ablation_reducers_no_skew_kurt", "5 stats"),
    ]:
        labels.append(label)
        f1_vals.append(data[k]["f1"])
        colours.append("#4daf4a")

    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.bar(x, f1_vals, color=colours, edgecolor="white")
    ax.axhline(BASELINE_F1, color="black", linestyle="--", linewidth=0.8,
               label=f"Baseline F1 (t=0.670): {BASELINE_F1:.3f}")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("F1-score (t=0.670)")
    ax.set_title("Ablation study summary (pixel-level, t=0.670)")
    ax.set_ylim(0.7, 1.0)
    ax.legend()
    fig.tight_layout()
    fp = FIGURES_DIR / "ablation_summary.png"
    fig.savefig(fp, dpi=DPI)
    plt.close(fig)
    print(f"Saved: {fp}")


# ── Main ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    data = load_results()
    plot_oob_ntrees(data)
    plot_oob_mtry(data)
    plot_band_ablation(data)
    plot_reducer_ablation(data)
    plot_f1_ntrees(data)
    plot_summary(data)
    print("\nAll ablation figures saved to", FIGURES_DIR)
