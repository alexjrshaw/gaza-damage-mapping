"""
Plot the pixel-level threshold sweep: precision, recall, and F1 vs
threshold, with the chosen optimal threshold (0.670) marked.
"""

import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

DATA_PATH = Path("/scratch/s1214882/gaza-damage-mapping/data")
df = pd.read_csv(DATA_PATH / "threshold_sweep_current_results.csv")

fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(df["threshold"], df["precision"], label="Precision", color="#1f77b4", linewidth=2)
ax.plot(df["threshold"], df["recall"], label="Recall", color="#ff7f0e", linewidth=2)
ax.plot(df["threshold"], df["f1"], label="F1", color="#2ca02c", linewidth=2)
ax.plot(
    df["threshold"], df["roc_auc"], label="'AUC' (balanced accuracy)", color="#9467bd", linewidth=2
)
ax.plot(df["threshold"], df["accuracy"], label="Accuracy", color="#8c564b", linewidth=2)

ax.axvline(0.670, color="grey", linestyle="--", linewidth=1, label="Chosen threshold (t=0.670)")
ax.axvline(0.500, color="firebrick", linestyle="--", linewidth=1, label="Default (t=0.5)")
ax.axhline(0.90, color="lightgrey", linestyle=":", linewidth=1, label="90% precision target")

ax.set_xlabel("Threshold")
ax.set_ylabel("Score")
ax.set_title("Gaza pixel-level threshold sweep")
ax.legend(loc="lower left", fontsize=9)
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.grid(alpha=0.3)

fig.tight_layout()
fig.savefig(DATA_PATH.parent / "threshold_sweep_plot.png", dpi=150)
print(f"Saved to {DATA_PATH.parent / 'threshold_sweep_plot.png'}")
