"""
Plot ablation study figures for Gaza damage mapping.

Primary data: data/ablation_runs/ablation_results.json (point-level,
mirrors Dietrich et al. 2025 Supplementary Note 6 methodology exactly).

Pixel-level overlay: data/ablation_runs/pixel_level/results.json
(where available — produced by ablation_pixel_level.py).

Figures saved to data/ablation_runs/figures/

Usage:
    python3 plot_ablation_figures.py
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

# ── Paths ──────────────────────────────────────────────────────────────────────
ABLATION_DIR = Path("data/ablation_runs")
FIGURES_DIR = ABLATION_DIR / "figures"
POINT_JSON = ABLATION_DIR / "ablation_results.json"
PIXEL_JSON = ABLATION_DIR / "pixel_level" / "results.json"
MTRY_EXTENDED_JSON = ABLATION_DIR / "mtry_extended_results.json"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# Baselines from evaluation.ipynb pixel-level run at t=0.655 (matches Dietrich et al.)
PIXEL_BASELINE_F1 = 0.884  # pixel-level t=0.655
PIXEL_BASELINE_AUC = 0.893  # pixel-level t=0.655

# Point-level baseline (VV+VH, all 7 reducers, n=50 trees) at t=0.655
POINT_BASELINE_F1 = 0.665  # from ablation_results.json bands["VV+VH (baseline)"]["t0.655"]["f1"]

# ── Style ──────────────────────────────────────────────────────────────────────
BLUE = "steelblue"
ORANGE = "darkorange"
RED = "crimson"
GREY = "dimgrey"
DPI = 150

plt.rcParams.update(
    {
        "font.size": 11,
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "legend.fontsize": 10,
        "figure.dpi": DPI,
    }
)

# ── Load results ───────────────────────────────────────────────────────────────


def load_point_results() -> dict:
    with open(POINT_JSON) as f:
        return json.load(f)


def load_pixel_results() -> dict:
    if PIXEL_JSON.exists():
        with open(PIXEL_JSON) as f:
            return json.load(f)
    return {}


def load_mtry_extended() -> dict:
    with open(MTRY_EXTENDED_JSON) as f:
        return json.load(f)


# ── Figure 1: OOB error vs n_trees ────────────────────────────────────────────


def plot_oob_n_trees(rp: dict, rx: dict) -> None:
    pt = rp["oob_n_trees"]
    fig, ax = plt.subplots(figsize=(8, 5))

    ax.plot(
        pt["tree_counts"],
        pt["oob_errors"],
        color=BLUE,
        linewidth=2,
        marker="o",
        markersize=4,
        label="Gaza (point-level, this study)",
    )

    # Pixel-level OOB if available
    if "oob_n_trees" in rx:
        px = rx["oob_n_trees"]
        ax.plot(
            px["n_trees"],
            px["oob_error"],
            color=ORANGE,
            linewidth=2,
            marker="s",
            markersize=4,
            linestyle="--",
            label="Gaza (pixel-level, this study)",
        )

    optimal_n = pt["tree_counts"][int(np.argmin(pt["oob_errors"]))]
    ax.axvline(50, color=RED, linestyle="--", linewidth=1, label="Dietrich et al. (50 trees)")
    ax.axvline(optimal_n, color="green", linestyle=":", linewidth=1.5, label=f"Optimal ({optimal_n} trees)")

    ax.set_xlabel("Number of trees")
    ax.set_ylabel("OOB error rate")
    ax.set_title("OOB error vs number of trees (Gaza)")
    ax.legend()
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.3f"))
    fig.tight_layout()
    fp = FIGURES_DIR / "oob_vs_n_trees_gaza.png"
    fig.savefig(fp, dpi=DPI)
    plt.close()
    print(f"Saved: {fp}")


# ── Figure 2: OOB error vs mtry ───────────────────────────────────────────────


def plot_oob_mtry(rp: dict, rx: dict, mtry_ext: dict) -> None:
    pt = rp["oob_mtry"]
    fig, ax = plt.subplots(figsize=(8, 5))

    # Extended point-level sweep (dense, 1-28)
    ext_x = [int(k) for k in mtry_ext.keys()]
    ext_y = list(mtry_ext.values())
    ax.plot(
        ext_x, ext_y, color=GREY, linewidth=1.5, linestyle="--", label="Gaza point-level (extended sweep)", alpha=0.7
    )

    # Main point-level sweep
    ax.plot(
        pt["mtry_values"], pt["oob_errors"], color=BLUE, linewidth=2, marker="o", markersize=5, label="Gaza point-level"
    )

    # Pixel-level OOB if available
    if "oob_mtry" in rx:
        px = rx["oob_mtry"]
        ax.plot(
            px["mtry"],
            px["oob_error"],
            color=ORANGE,
            linewidth=2,
            marker="s",
            markersize=5,
            linestyle="--",
            label="Gaza pixel-level",
        )

    sqrt_p = int(np.sqrt(28))
    ax.axvline(sqrt_p, color=RED, linestyle="--", linewidth=1, label=f"sqrt(p)={sqrt_p} (sklearn default)")

    ax.set_xlabel("mtry (max_features)")
    ax.set_ylabel("OOB error rate")
    ax.set_title("OOB error vs mtry parameter (Gaza)")
    ax.legend()
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.3f"))
    fig.tight_layout()
    fp = FIGURES_DIR / "oob_vs_mtry_gaza.png"
    fig.savefig(fp, dpi=DPI)
    plt.close()
    print(f"Saved: {fp}")


# ── Figure 3: Band ablation ────────────────────────────────────────────────────


def plot_band_ablation(rp: dict, rx: dict) -> None:
    labels = ["VV only", "VH only", "VV+VH\n(baseline)"]
    pt_keys = ["VV only", "VH only", "VV+VH (baseline)"]
    px_keys = ["ablation_bands_VV", "ablation_bands_VH", None]

    pt_f1 = [rp["bands"][k]["t0.655"]["f1"] for k in pt_keys]
    pt_auc = [rp["bands"][k]["t0.655"]["roc_auc"] for k in pt_keys]

    px_f1 = [rx.get(k, {}).get("f1", np.nan) if k else PIXEL_BASELINE_F1 for k in px_keys]
    px_auc = [rx.get(k, {}).get("auc", np.nan) if k else PIXEL_BASELINE_AUC for k in px_keys]

    x = np.arange(len(labels))
    w = 0.35
    fig, ax = plt.subplots(figsize=(8, 5))

    bars1 = ax.bar(x - w / 2, pt_f1, w, color=BLUE, label="Point-level (t=0.655)", edgecolor="white")
    bars2 = ax.bar(x + w / 2, px_f1, w, color=ORANGE, label="Pixel-level (t=0.655)", edgecolor="white")

    pass  # no annotations

    ax.axhline(POINT_BASELINE_F1, color=BLUE, linestyle=":", linewidth=1, alpha=0.6)
    ax.axhline(PIXEL_BASELINE_F1, color=ORANGE, linestyle=":", linewidth=1, alpha=0.6)

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("F1-score (t=0.655)")
    ax.set_title("Band ablation — input polarisation (Gaza)")
    ax.set_ylim(0, 1.05)
    ax.legend()
    fig.tight_layout()
    fp = FIGURES_DIR / "band_ablation_gaza.png"
    fig.savefig(fp, dpi=DPI)
    plt.close()
    print(f"Saved: {fp}")


# ── Figure 4: Feature subset ablation ─────────────────────────────────────────


def plot_feature_ablation(rp: dict, rx: dict) -> None:
    pt_configs = [
        ("mean+std", "mean\n+std"),
        ("+median", "+median"),
        ("+min/max", "+min/max"),
        ("+skew", "+skew"),
        ("all 7 (baseline)", "all 7\n(baseline)"),
    ]
    px_configs = [
        "ablation_reducers_mean_std",
        "ablation_reducers_mean_std_median",
        "ablation_reducers_no_skew_kurt",
        None,  # +skew not run separately
        None,  # baseline
    ]

    labels = [c[1] for c in pt_configs]
    pt_f1 = [rp["features"][c[0]]["t0.655"]["f1"] for c in pt_configs]
    px_f1 = []
    for i, key in enumerate(px_configs):
        if key is None and pt_configs[i][0] == "all 7 (baseline)":
            px_f1.append(PIXEL_BASELINE_F1)
        elif key is None:
            px_f1.append(np.nan)
        else:
            px_f1.append(rx.get(key, {}).get("f1", np.nan))

    x = np.arange(len(labels))
    w = 0.35
    fig, ax = plt.subplots(figsize=(9, 5))

    ax.bar(x - w / 2, pt_f1, w, color=BLUE, label="Point-level (t=0.655)", edgecolor="white")
    ax.bar(x + w / 2, px_f1, w, color=ORANGE, label="Pixel-level (t=0.655)", edgecolor="white")

    ax.axhline(POINT_BASELINE_F1, color=BLUE, linestyle=":", linewidth=1, alpha=0.6)
    ax.axhline(PIXEL_BASELINE_F1, color=ORANGE, linestyle=":", linewidth=1, alpha=0.6)

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("F1-score (t=0.655)")
    ax.set_title("Feature subset ablation — reducers (Gaza)")
    ax.set_ylim(0, 1.05)
    ax.legend()
    fig.tight_layout()
    fp = FIGURES_DIR / "feature_ablation_gaza.png"
    fig.savefig(fp, dpi=DPI)
    plt.close()
    print(f"Saved: {fp}")


# ── Figure 5: F1 vs n_trees ───────────────────────────────────────────────────


def plot_f1_n_trees(rp: dict, rx: dict) -> None:
    pt_trees = sorted(int(k) for k in rp["n_trees"].keys())
    pt_f1 = [rp["n_trees"][str(n)]["t0.655"]["f1"] for n in pt_trees]

    # Pixel-level variants
    px_map = {
        10: "ablation_ntrees_10",
        25: "ablation_ntrees_25",
        50: None,
        100: "ablation_ntrees_100",
        200: "ablation_ntrees_200",
        300: "ablation_ntrees_300",
    }
    px_trees = sorted(px_map.keys())
    px_f1 = []
    for n in px_trees:
        key = px_map[n]
        if key is None:
            px_f1.append(PIXEL_BASELINE_F1)
        else:
            px_f1.append(rx.get(key, {}).get("f1", np.nan))

    fig, ax = plt.subplots(figsize=(9, 5))

    ax.plot(pt_trees, pt_f1, color=BLUE, linewidth=2, marker="o", markersize=6, label="Point-level (t=0.655)")

    # Only plot pixel-level line where we have data
    px_available = [(n, f) for n, f in zip(px_trees, px_f1) if not np.isnan(f)]
    if px_available:
        px_x, px_y = zip(*px_available)
        ax.plot(
            px_x,
            px_y,
            color=ORANGE,
            linewidth=2,
            marker="s",
            markersize=6,
            linestyle="--",
            label="Pixel-level (t=0.655)",
        )

    ax.axvline(50, color=RED, linestyle="--", linewidth=1, label="Dietrich et al. (50 trees)")
    ax.axhline(POINT_BASELINE_F1, color=BLUE, linestyle=":", linewidth=1, alpha=0.5)
    ax.axhline(PIXEL_BASELINE_F1, color=ORANGE, linestyle=":", linewidth=1, alpha=0.5)

    ax.set_xlabel("Number of trees")
    ax.set_ylabel("F1-score (t=0.655)")
    ax.set_title("F1-score vs number of trees (Gaza)")
    ax.set_ylim(0, 1.05)
    ax.legend()
    fig.tight_layout()
    fp = FIGURES_DIR / "f1_vs_n_trees_gaza.png"
    fig.savefig(fp, dpi=DPI)
    plt.close()
    print(f"Saved: {fp}")


# ── Dietrich et al. Fig S4 equivalent — summary bar chart ─────────────────────


def plot_ablation_summary(rp: dict) -> None:
    """
    Single bar chart of F1 across all ablation settings at t=0.655.
    Mirrors Dietrich et al. Fig S4 exactly.
    """
    labels, f1_vals, colors = [], [], []

    # Bands
    for k, lbl in [("VV only", "VV only"), ("VH only", "VH only"), ("VV+VH (baseline)", "VV+VH\n(baseline)")]:
        labels.append(lbl)
        f1_vals.append(rp["bands"][k]["t0.655"]["f1"])
        colors.append(ORANGE if "baseline" in k else BLUE)

    # n_trees
    for n in [10, 25, 50, 75, 100]:
        labels.append(f"{n} trees")
        f1_vals.append(rp["n_trees"][str(n)]["t0.655"]["f1"])
        colors.append(ORANGE if n == 50 else BLUE)

    # Features
    for k, lbl in [
        ("mean+std", "mean+std"),
        ("+median", "+median"),
        ("+min/max", "+min/max"),
        ("+skew", "+skew"),
        ("all 7 (baseline)", "all 7\n(baseline)"),
    ]:
        labels.append(lbl)
        f1_vals.append(rp["features"][k]["t0.655"]["f1"])
        colors.append(ORANGE if "baseline" in k else BLUE)

    # Also collect t=0.5 values
    f1_05_vals = []
    for k, lbl in [("VV only", "VV only"), ("VH only", "VH only"), ("VV+VH (baseline)", "VV+VH\n(baseline)")]:
        f1_05_vals.append(rp["bands"][k]["t0.5"]["f1"])
    for n in [10, 25, 50, 75, 100]:
        f1_05_vals.append(rp["n_trees"][str(n)]["t0.5"]["f1"])
    for k, lbl in [
        ("mean+std", "mean+std"),
        ("+median", "+median"),
        ("+min/max", "+min/max"),
        ("+skew", "+skew"),
        ("all 7 (baseline)", "all 7\n(baseline)"),
    ]:
        f1_05_vals.append(rp["features"][k]["t0.5"]["f1"])

    # Extraction window
    if "extraction_window" in rp:
        for k, lbl in [("1x1 (baseline)", "1x1\n(baseline)"), ("3x3", "3x3"), ("1x1+3x3", "1x1+3x3")]:
            labels.append(lbl)
            f1_vals.append(rp["extraction_window"][k]["t0.655"]["f1"])
            f1_05_vals.append(rp["extraction_window"][k]["t0.5"]["f1"])
            colors.append(ORANGE if "baseline" in k else BLUE)

    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.bar(x - 0.2, f1_05_vals, 0.4, label="t=0.5", color="steelblue")
    ax.bar(x + 0.2, f1_vals, 0.4, label="t=0.655", color="orange")
    baseline_f1_05 = rp["bands"]["VV+VH (baseline)"]["t0.5"]["f1"]
    ax.axhline(baseline_f1_05, color=RED, linestyle="--", linewidth=1, label=f"Baseline F1={baseline_f1_05:.3f}")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("F1-score (t=0.655)")
    ax.set_title("Ablation study results (Gaza, point-level — mirrors Dietrich et al. Fig S4)")
    ax.set_ylim(0, 1.0)
    ax.legend()
    fig.tight_layout()
    fp = FIGURES_DIR / "ablation_summary_gaza.png"
    fig.savefig(fp, dpi=DPI)
    plt.close()
    print(f"Saved: {fp}")


# ── Main ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Loading results...")
    rp = load_point_results()
    rx = load_pixel_results()
    mtry_ext = load_mtry_extended()

    px_done = [k for k in rx if k.startswith("ablation_")]
    print(f"Point-level data: complete")
    print(f"Pixel-level variants completed: {len(px_done)} — {px_done}")

    print("\nGenerating figures...")
    plot_oob_n_trees(rp, rx)
    plot_oob_mtry(rp, rx, mtry_ext)
    plot_band_ablation(rp, rx)
    plot_feature_ablation(rp, rx)
    plot_f1_n_trees(rp, rx)
    plot_ablation_summary(rp)

    print(f"\nAll figures saved to {FIGURES_DIR}")
