"""
Plot balanced accuracy at t=0.67 for all ablation variants.
Saves to data/ablation_runs/figures/ablation_summary_balanced_acc.png

Usage:
    cd /scratch/s1214882/gaza-damage-mapping
    source alex/bin/activate
    python3 src/visualisation/plot_ablation_summary.py
"""
import json
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# Data
fp = Path("data/ablation_runs/pixel_level/results.json")
with open(fp) as f:
    data = json.load(f)

variants = [
    ("baseline",                         "Baseline",             "baseline"),
    ("ablation_ntrees_10",               "10 trees",             "trees"),
    ("ablation_ntrees_25",               "25 trees",             "trees"),
    ("ablation_ntrees_75",               "75 trees",             "trees"),
    ("ablation_ntrees_100",              "100 trees",            "trees"),
    ("ablation_ntrees_200",              "200 trees",            "trees"),
    ("ablation_ntrees_300",              "300 trees",            "trees"),
    ("ablation_bands_VV",                "VV only",              "bands"),
    ("ablation_bands_VH",                "VH only",              "bands"),
    ("ablation_reducers_mean_std",       "Mean + std only",      "stats"),
    ("ablation_reducers_mean_std_median","Mean + std + median",  "stats"),
    ("ablation_reducers_no_skew_kurt",   "No skew/kurt",         "stats"),
]

baseline_val = 88.9

group_colours = {
    "baseline": "#dc2626",
    "trees":    "#1d4ed8",
    "bands":    "#16a34a",
    "stats":    "#9333ea",
}

labels, values, colours = [], [], []
for key, label, group in variants:
    labels.append(label)
    values.append(baseline_val if key == "baseline" else round(data[key]["auc"] * 100, 1))
    colours.append(group_colours[group])

# Style
plt.rcParams.update({
    "font.family": "Times New Roman",
    "font.size": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.spines.left": False,
    "axes.linewidth": 0.8,
    "xtick.major.size": 0,
    "ytick.major.size": 3,
})

# Figure and axis (single plot, white background)
fig, ax = plt.subplots(figsize=(9, 6.0))
fig.patch.set_facecolor("white")
ax.set_facecolor("white")

# Horizontal legend
legend_items = [
    mpatches.Patch(color=group_colours["baseline"], label="Baseline"),
    mpatches.Patch(color=group_colours["trees"],    label="Number of trees"),
    mpatches.Patch(color=group_colours["bands"],    label="Polarisation"),
    mpatches.Patch(color=group_colours["stats"],    label="SAR statistics"),
]
# Horizontal legend above chart
legend_items = [
    mpatches.Patch(color=group_colours["baseline"], label="Baseline"),
    mpatches.Patch(color=group_colours["trees"],    label="Number of trees"),
    mpatches.Patch(color=group_colours["bands"],    label="Polarisation"),
    mpatches.Patch(color=group_colours["stats"],    label="SAR statistics"),
]
ax.legend(handles=legend_items, loc="upper center",
          bbox_to_anchor=(0.5, 1.08), ncol=4,
          fontsize=12, frameon=False,
          handlelength=1.2, handletextpad=0.5, columnspacing=1.5)

# Grid
ax.xaxis.grid(True, color="#cccccc", linewidth=0.6, zorder=0)
ax.set_axisbelow(True)

# Bars
y_pos = np.arange(len(labels))
bars = ax.barh(y_pos, values, color=colours, alpha=0.85,
               height=0.6, zorder=3)

# Baseline reference line (drawn on top of value labels)
ax.axvline(baseline_val, color="#dc2626", linewidth=1.2,
           linestyle=(0, (4, 3)), zorder=5)

# Value labels
for bar, val, (key, label, group) in zip(bars, values, variants):
    colour = "#dc2626" if key == "baseline" else "#333" # Baseline value in red
    ax.text(val + 0.35, bar.get_y() + bar.get_height() / 2,
            f"{val:.1f}%", va="center", ha="left", fontsize=14, fontweight="bold",
            color=colour, bbox=dict(facecolor="white", edgecolor="none", pad=1.0))

# Y axis
ax.set_yticks(y_pos)
ax.set_yticklabels(labels, fontsize=14)
ax.invert_yaxis()

# X axis
ax.set_xlim(79, 91)
ax.set_xlabel("Balanced accuracy", fontsize=14, labelpad=8, color="black")
ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{int(x)}%"))
ax.xaxis.set_tick_params(labelsize=12)

# Group separators
for s in [0.5, 6.5, 8.5]:
    ax.axhline(s, color="#cccccc", linewidth=0.8, zorder=2)

# Title and subtitle
fig.text(0.06, 0.99,
         "Polarisation matters. Other components, less so",
         fontsize=22, fontweight="bold", color="black", va="top", ha="left")
fig.text(0.06, 0.92,
         "Balanced accuracy for each ablation variant",
         fontsize=18, color="black", va="top", ha="left")

# Save
plt.tight_layout(rect=[0, 0.02, 1, 0.88])
OUT = Path("data/ablation_runs/figures/ablation_summary_balanced_acc.png")
OUT.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT, dpi=200, bbox_inches="tight", facecolor="white")
plt.close()
print(f"Saved to {OUT}")