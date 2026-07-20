"""Plot OOB error vs n_trees and vs mtry, both already computed."""

import json
import matplotlib.pyplot as plt
from pathlib import Path

DATA_PATH = Path("/scratch/s1214882/gaza-damage-mapping/data")
with open(DATA_PATH / "ablation_runs/pixel_level/results.json") as f:
    results = json.load(f)

fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

axes[0].plot(
    results["oob_n_trees"]["n_trees"],
    results["oob_n_trees"]["oob_error"],
    marker="o",
    color="#1f77b4",
)
axes[0].axvline(50, color="grey", linestyle="--", linewidth=1, label="Chosen (50 trees)")
axes[0].set_xlabel("Number of trees")
axes[0].set_ylabel("OOB error")
axes[0].set_title("OOB error vs number of trees")
axes[0].legend()
axes[0].grid(alpha=0.3)

axes[1].plot(
    results["oob_mtry"]["mtry"], results["oob_mtry"]["oob_error"], marker="o", color="#ff7f0e"
)
axes[1].axvline(5, color="grey", linestyle="--", linewidth=1, label="sqrt(28) default")
axes[1].set_xlabel("mtry (features per split)")
axes[1].set_ylabel("OOB error")
axes[1].set_title("OOB error vs mtry")
axes[1].legend()
axes[1].grid(alpha=0.3)

fig.tight_layout()
fig.savefig(DATA_PATH.parent / "oob_plots.png", dpi=150)
print(f"Saved to {DATA_PATH.parent / 'oob_plots.png'}")
