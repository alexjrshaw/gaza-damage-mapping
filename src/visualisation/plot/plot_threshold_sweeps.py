"""
Plot threshold sweep results for Gaza and Mosul (retrained).
Saves to data/ablation_runs/figures/threshold_sweep_gaza.png
             data/ablation_runs/figures/threshold_sweep_mosul.png

Usage:
    cd /scratch/s1214882/gaza-damage-mapping
    source alex/bin/activate
    python3 src/visualisation/plot/plot_threshold_sweeps.py
"""
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from pathlib import Path
import numpy as np

# Data
gz_fp  = Path("data/threshold_sweep_current_results.csv")
mos_fp = Path("data/transfer_cities/runs/mosul_threshold_sweep_results.csv")

gz  = pd.read_csv(gz_fp)
mos = pd.read_csv(mos_fp)

# Rename roc_auc to balanced_accuracy for Gaza (established convention)
gz  = gz.rename(columns={"roc_auc": "balanced_accuracy"})

# Metrics to plot
metrics = {
    "balanced_accuracy":  {"label": "Balanced accuracy",   "color": "#000000", "ls": "-"},
    "f1":                 {"label": "F1",                  "color": "#8B4513", "ls": "--"},
    "precision":          {"label": "Precision",           "color": "#E69F00", "ls": "-."},
    "recall":             {"label": "Recall",              "color": "#CC79A7", "ls": (0,(5,1))},
    "accuracy":           {"label": "Overall accuracy",    "color": "#56B4E9", "ls": (0,(3,1,1,1))},
}

# Style
plt.rcParams.update({
    "font.family": "Times New Roman",
    "font.size": 12,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 0.8,
    "xtick.major.size": 3,
    "ytick.major.size": 3,
})

def make_threshold_chart(df, title, subtitle, vlines, out_fp):
    """
    df:      dataframe with threshold and metric columns
    title:   chart title
    subtitle: chart subtitle
    vlines:  list of (x, label, color, y_pos) tuples for vertical reference lines
    out_fp:  output path
    """
    fig, ax = plt.subplots(figsize=(9, 5.5))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    # Horizontal legend above chart
    legend_items = [
        mlines.Line2D([], [], color=m["color"], linestyle=m["ls"],
                      linewidth=1.8, label=m["label"])
        for m in metrics.values()
    ]
    ax.legend(handles=legend_items, loc="upper center",
              bbox_to_anchor=(0.5, 1.15), ncol=5,
              fontsize=12, frameon=False,
              handlelength=2.0, handletextpad=0.5, columnspacing=1.5)

    # Grid
    ax.yaxis.grid(True, color="#cccccc", linewidth=0.6, zorder=0)
    ax.set_axisbelow(True)

    # Plot metric lines
    for col, m in metrics.items():
        if col in df.columns:
            ax.plot(df["threshold"], df[col] * 100,
                    color=m["color"], linestyle=m["ls"],
                    linewidth=1.8, zorder=3)

    # Vertical reference lines
    for x, label, color, y_pos in vlines:
        ax.axvline(x, color=color, linewidth=1.3,
                   linestyle=(0, (4, 3)), zorder=4)
        is_left = "Mosul optimal" in label
        ax.text(x - 0.01 if is_left else x + 0.01,
                y_pos, label,
                fontsize=14, color=color, fontweight="bold",
                va="top",
                ha="right" if is_left else "left")

    # Axes
    ax.set_xlim(0, 0.95)
    ax.set_ylim(0, 101)
    ax.set_xticks(np.arange(0, 1.0, 0.1))
    ax.set_xticklabels([f"{v:.1f}" for v in np.arange(0, 1.0, 0.1)], fontsize=12)
    ax.set_yticks(range(0, 101, 20))
    ax.set_yticklabels([f"{v}%" for v in range(0, 101, 20)], fontsize=12)
    ax.set_xlabel("Threshold", fontsize=14, labelpad=8, color="black")
    ax.set_ylabel("Performance", fontsize=14, labelpad=8, color="black")

    # Title and subtitle
    fig.text(0.06, 0.99, title,
             fontsize=22, fontweight="bold", color="black", va="top", ha="left")
    fig.text(0.06, 0.93, subtitle,
             fontsize=16, color="black", va="top", ha="left")

    # Save
    plt.tight_layout(rect=[0, 0.02, 1, 0.88])
    out_fp.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_fp, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"Saved to {out_fp}")


# Gaza chart
make_threshold_chart(
    df=gz,
    title="A higher bar means fewer false alarms",
    subtitle="Performance by damage classification threshold, Gaza model",
    vlines=[
        (0.5,  "Default:\n0.5",         "#6b7280", 15),
        (0.67, "Calibrated:\n0.67",     "#dc2626", 15),
    ],
    out_fp=Path("data/ablation_runs/figures/threshold_sweep_gaza.png"),
)

# Mosul chart
make_threshold_chart(
    df=mos,
    title="Gaza's threshold is too cautious for Mosul",
    subtitle="Performance by damage classification threshold, Mosul retrained model",
    vlines=[
        (0.44, "Mosul optimal:\n0.44",   "#0072B2", 15),
        (0.5,  "Default:\n0.5",          "#6b7280", 53),
        (0.67, "Gaza calibrated:\n0.67", "#dc2626", 37),
    ],
    out_fp=Path("data/ablation_runs/figures/threshold_sweep_mosul.png"),
)