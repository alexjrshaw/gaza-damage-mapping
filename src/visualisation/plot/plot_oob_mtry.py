"""
Plot OOB error vs mtry (full training set).
Saves to data/ablation_runs/figures/oob_vs_mtry_full.png

Usage:
    cd /scratch/s1214882/gaza-damage-mapping
    source alex/bin/activate
    python3 src/visualisation/plot/plot_oob_mtry.py
"""
import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

# Load data
FP = Path("data/ablation_runs/mtry_full_sweep_results.json")
OUT = Path("data/ablation_runs/figures/oob_vs_mtry_full.png")

with open(FP) as f:
    raw = json.load(f)

mtry_vals = sorted(int(k) for k in raw.keys())
oob_vals  = [raw[str(m)] for m in mtry_vals]

# Global style
plt.rcParams.update({
    "font.family": "Times New Roman",
    "font.size": 14,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 0.8,
    "xtick.major.size": 4,
    "ytick.major.size": 4,
})

fig, ax = plt.subplots(figsize=(8.5, 5.2))
fig.patch.set_facecolor("white")
ax.set_facecolor("white")

# Stable region shading (mtry 5–23)
ax.axvspan(5, 23, color="#3b82f6", alpha=0.10, zorder=0)

# Horizontal grid lines
ax.yaxis.grid(True, color="#cccccc", linewidth=0.6, zorder=0)
ax.set_axisbelow(True)

# Main line
ax.plot(mtry_vals, oob_vals, color="#1d4ed8", linewidth=2.2,
        zorder=3, solid_capstyle="round")

# Data points
ax.scatter(mtry_vals, oob_vals, color="#1d4ed8", s=30, zorder=4, alpha=0.85)

# Highlighted points
ax.scatter([5],  [raw["5"]],  color="#dc2626", s=60, zorder=5)
ax.scatter([17], [raw["17"]], color="#4b5563", s=60, zorder=5)

# Vertical dotted reference lines
ax.axvline(5,  color="#dc2626", linewidth=1.4, linestyle=(0, (4, 3)), zorder=2)
ax.axvline(17, color="#4b5563", linewidth=1.4, linestyle=(0, (4, 3)), zorder=2)

# Labels to the right of each dotted line
ax.text(5.3,  0.280, "Default\nmtry: 5",       ha="left", va="bottom",
        fontsize=14, color="#dc2626", fontweight="bold")
ax.text(17.3, 0.280, "Lowest error\nmtry: 17", ha="left", va="bottom",
        fontsize=14, color="#4b5563", fontweight="bold")

# Stable region annotation
ax.text(11, 0.274, "OOB error is stable\nat mtry 5–23",
        fontsize=14, color="#1d4ed8", fontweight="bold",
        ha="center", va="center", zorder=5,
        bbox=dict(facecolor="white", alpha=0.7, edgecolor="none", pad=2))

# Axis limits and ticks
ax.set_xlim(0, 29)
ax.set_ylim(0.264, 0.282)
ax.set_xticks(range(0, 29, 2))
ax.set_yticks(np.arange(0.264, 0.281, 0.002))
ax.set_yticklabels([f"{v:.3f}" for v in np.arange(0.264, 0.281, 0.002)], fontsize=12)
ax.set_xticklabels(range(0, 29, 2), fontsize=12)

# Axis titles in black
ax.set_xlabel("mtry (features per split)", fontsize=14, labelpad=8, color="black")
ax.set_ylabel("OOB error", fontsize=14, labelpad=8, color="black")

# Chart title and subtitle in black
fig.text(0.06, 0.97,
         "The model needs few features per split to perform well",
         fontsize=22, fontweight="bold", color="black", va="top", ha="left")
fig.text(0.06, 0.90,
         "Out-of-bag (OOB) error by number of features considered at each decision tree split (mtry)",
         fontsize=15, color="black", va="top", ha="left")

# Layout and save
plt.tight_layout(rect=[0, 0.02, 1, 0.86])
OUT.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT, dpi=200, bbox_inches="tight", facecolor="white")
plt.close()
print(f"Saved to {OUT}")