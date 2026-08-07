"""
Plot OOB error vs number of trees (full training set).
Saves to data/ablation_runs/figures/oob_vs_ntrees_full.png

Usage:
    cd /scratch/s1214882/gaza-damage-mapping
    source alex/bin/activate
    python3 src/visualisation/plot/plot_oob_ntrees.py
"""
import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

# Load data
FP = Path("data/ablation_runs/oob_ntrees_full_results.json")
OUT = Path("data/ablation_runs/figures/oob_vs_ntrees_full.png")

with open(FP) as f:
    raw = json.load(f)

n_trees_vals = raw["n_trees"]
oob_vals     = raw["oob_error"]

# Style
plt.rcParams.update({
    "font.family": "Times New Roman",
    "font.size": 12,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 0.8,
    "xtick.major.size": 4,
    "ytick.major.size": 4,
})

fig, ax = plt.subplots(figsize=(8.5, 5.2))
fig.patch.set_facecolor("white")
ax.set_facecolor("white")

# Horizontal grid lines
ax.yaxis.grid(True, color="#cccccc", linewidth=0.6, zorder=0)
ax.set_axisbelow(True)

# Main line
ax.plot(n_trees_vals, oob_vals, color="#1d4ed8", linewidth=2.2,
        zorder=3, solid_capstyle="round")

# Data points
ax.scatter(n_trees_vals, oob_vals, color="#1d4ed8", s=40, zorder=4, alpha=0.85)

# Highlight baseline (n=50)
baseline_oob = oob_vals[n_trees_vals.index(50)]
ax.scatter([50], [baseline_oob], color="#dc2626", s=60, zorder=5)

# Vertical dotted line at n=50
ax.axvline(50, color="#dc2626", linewidth=1.4, linestyle=(0, (4, 3)), zorder=2)

# Label to the right of dotted line
ax.text(52, 0.318, "Baseline\n50 trees",
        ha="left", va="top", fontsize=14, color="#dc2626", fontweight="bold",
        transform=ax.transData, clip_on=True)

# Axis limits and ticks
ax.set_xlim(0, 320)
ax.set_ylim(0.255, 0.320)
ax.set_xticks(n_trees_vals)
ax.set_xticklabels(n_trees_vals, fontsize=12)
ax.set_yticks(np.arange(0.26, 0.314, 0.005))
ax.set_yticklabels([f"{v:.3f}" for v in np.arange(0.26, 0.314, 0.005)], fontsize=12)

# Axis titles
ax.set_xlabel("Number of trees", fontsize=14, labelpad=8, color="black")
ax.set_ylabel("OOB error", fontsize=14, labelpad=8, color="black")

# Chart title and subtitle
fig.text(0.06, 0.97,
         "Adding trees beyond 50 brings diminishing returns",
         fontsize=22, fontweight="bold", color="black", va="top", ha="left")
fig.text(0.06, 0.90,
         "Out-of-bag (OOB) error by number of trees in the Random Forest",
         fontsize=16, color="black", va="top", ha="left")

# Layout and save
plt.tight_layout(rect=[0, 0.02, 1, 0.86])
OUT.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT, dpi=200, bbox_inches="tight", facecolor="white")
plt.close()
print(f"Saved to {OUT}")