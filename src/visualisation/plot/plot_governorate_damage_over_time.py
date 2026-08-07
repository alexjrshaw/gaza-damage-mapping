"""
Plot share of buildings damaged by governorate over time.
Saves to data/ablation_runs/figures/governorate_damage_lines.png

Usage:
    cd /scratch/s1214882/gaza-damage-mapping
    source alex/bin/activate
    python3 src/visualisation/plot/plot_governorate_damage_over_time.py
"""
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
from pathlib import Path
import datetime

# Data
fp = Path("data/outputs/temporal_damage_datawrapper.csv")
df = pd.read_csv(fp)
df["Date"] = pd.to_datetime(df["Date"], format="%Y-%m-%d")
df = df.sort_values("Date")
df = df.rename(columns={"Khan Younis": "Khan Yunis"})

# Colourblind-safe palette + distinct line styles
# Avoids red/green confusion + distinct dashes for greyscale printing
gov_styles = {
    "North Gaza":    {"color": "#000000", "ls": "-",        "lw": 1.8},
    "Gaza City":     {"color": "#0072B2", "ls": "--",       "lw": 1.8},
    "Khan Yunis":    {"color": "#E69F00", "ls": "-.",       "lw": 1.8},
    "Rafah":         {"color": "#CC79A7", "ls": (0,(5,1)),  "lw": 1.8},
    "Deir al-Balah": {"color": "#56B4E9", "ls": (0,(3,1,1,1)), "lw": 1.8},
    "Gaza Strip":    {"color": "#dc2626", "ls": (0,(8,2)),  "lw": 2.5},
}

govs = ["North Gaza", "Gaza City", "Khan Yunis", "Rafah", "Deir al-Balah", "Gaza Strip"]

# Style
plt.rcParams.update({
    "font.family": "Times New Roman",
    "font.size": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 0.8,
    "xtick.major.size": 3,
    "ytick.major.size": 3,
})

#  Figure (single axes, legend positioned above chart)
fig, ax = plt.subplots(figsize=(11, 6.5))
fig.patch.set_facecolor("white")
ax.set_facecolor("white")

# Horizontal legend above chart
legend_items = [
    mlines.Line2D([], [], color=gov_styles[g]["color"],
                  linestyle=gov_styles[g]["ls"],
                  linewidth=1.8, label=g)
    for g in govs
]
ax.legend(handles=legend_items, loc="upper center",
          bbox_to_anchor=(0.5, 1.08), ncol=6,
          fontsize=14, frameon=False,
          handlelength=2.0, handletextpad=0.5, columnspacing=1.2)

# Ceasefire shading
ceasefire_colour = "#3b82f6"
ceasefire_alpha  = 0.10

# Temporary ceasefire: 19 Jan to 18 Mar 2025
ax.axvspan(datetime.date(2025, 1, 19), datetime.date(2025, 3, 18),
           color=ceasefire_colour, alpha=ceasefire_alpha, zorder=0)

# Ongoing ceasefire: 10 Oct 2025 to end of data
ax.axvspan(datetime.date(2025, 10, 10), datetime.date(2025, 12, 31),
           color=ceasefire_colour, alpha=ceasefire_alpha, zorder=0)

# Ceasefire labels (placed below the lines to avoid overlap)
ax.text(datetime.date(2025, 1, 21), 3,
        "Ceasefire\n(temporary)",
        fontsize=14, fontweight="bold", color="#1d4ed8", va="bottom", ha="left", zorder=5)
ax.text(datetime.date(2025, 10, 12), 3,
        "Ceasefire\n(ongoing)",
        fontsize=14, fontweight="bold", color="#1d4ed8", va="bottom", ha="left", zorder=5)

# Vertical event lines
ax.axvline(datetime.date(2023, 12, 1), color="#6b7280",
           linewidth=1.2, linestyle=(0, (4, 3)), zorder=2)
ax.text(datetime.date(2023, 12, 4), 55,
        "Battle for\nKhan Yunis\nbegins",
        fontsize=14, fontweight="bold", color="#6b7280", va="top", ha="left")

ax.axvline(datetime.date(2024, 5, 6), color="#6b7280",
           linewidth=1.2, linestyle=(0, (4, 3)), zorder=2)
ax.text(datetime.date(2024, 5, 9), 55,
        "Rafah\nground\ninvasion",
        fontsize=14, fontweight="bold", color="#6b7280", va="top", ha="left")

# Grid
ax.yaxis.grid(True, color="#cccccc", linewidth=0.6, zorder=0)
ax.set_axisbelow(True)

# Plot lines
for gov in govs:
    s = gov_styles[gov]
    ax.plot(df["Date"], df[gov],
            color=s["color"], linestyle=s["ls"], linewidth=s["lw"],
            marker="o", markersize=3, zorder=3, label=gov)

# End-of-line labels — staggered to avoid overlap
last_date = df["Date"].iloc[-1]
# Sort by final value to assign staggered y offsets
final_vals = {gov: df[gov].iloc[-1] for gov in govs}
sorted_govs = sorted(final_vals, key=final_vals.get)

# Assign staggered positions with minimum spacing of 3pp
min_gap = 4.0
positions = {}
prev = -999
for gov in sorted_govs:
    pos = max(final_vals[gov], prev + min_gap)
    positions[gov] = pos
    prev = pos

for gov in govs:
    ax.text(last_date + pd.Timedelta(days=10),
            positions[gov],
            f"{final_vals[gov]:.1f}%",
            fontsize=14, color=gov_styles[gov]["color"],
            fontweight="bold", va="center", ha="left")

# Axes
ax.set_xlim(df["Date"].min() - pd.Timedelta(days=10),
            last_date + pd.Timedelta(days=75))
ax.set_ylim(0, 90)
ax.set_ylabel("Buildings damaged", fontsize=14, labelpad=8, color="black")
ax.yaxis.set_tick_params(labelsize=12)
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{int(x)}%"))
ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
plt.setp(ax.get_xticklabels(), rotation=30, ha="right", fontsize=12)

# Title and subtitle
fig.text(0.06, 0.99,
         "Rafah: from last refuge to most destroyed",
         fontsize=26, fontweight="bold", color="black", va="top", ha="left")
fig.text(0.06, 0.93,
         "Share of buildings damaged in each governorate, October 2023 to December 2025",
         fontsize=18, color="black", va="top", ha="left")
plt.tight_layout(rect=[0, 0.02, 1, 0.88])

# Save
OUT = Path("data/ablation_runs/figures/governorate_damage_lines.png")
OUT.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT, dpi=200, bbox_inches="tight", facecolor="white")
plt.close()
print(f"Saved to {OUT}")