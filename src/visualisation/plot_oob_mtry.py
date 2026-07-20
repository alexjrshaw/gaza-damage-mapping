import json
import matplotlib.pyplot as plt
from src.constants import DATA_PATH
import numpy as np

FIGURES_DIR = DATA_PATH / "ablation_runs/figures"

with open(DATA_PATH / "ablation_runs/mtry_extended_results.json") as f:
    results = {int(k): v for k, v in json.load(f).items()}

mtry_values = sorted(results.keys())
oob_errors = [results[m] for m in mtry_values]

optimal_mtry = min(results, key=results.get)
sqrt_p = int(np.sqrt(28))

fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(mtry_values, oob_errors, color="blue", linewidth=2, marker="o", markersize=4)
ax.axvline(
    sqrt_p,
    color="red",
    linestyle="--",
    linewidth=1,
    label=f"sqrt(p)={sqrt_p} (sklearn default, OOB={results[sqrt_p]:.4f})",
)
ax.axvline(
    optimal_mtry,
    color="green",
    linestyle=":",
    linewidth=1.5,
    label=f"Optimal mtry={optimal_mtry} (OOB={results[optimal_mtry]:.4f})",
)
ax.set_xlabel("mtry (max_features)")
ax.set_ylabel("OOB error rate")
ax.set_title("OOB error vs mtry parameter (Gaza)")
ax.legend()
fig.tight_layout()
fp = FIGURES_DIR / "oob_vs_mtry.png"
fig.savefig(fp, dpi=150)
plt.close()
print(f"Saved: {fp}")
