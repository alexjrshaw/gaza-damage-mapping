"""
Dissertation figures for Gaza damage mapping.

Figures:
    1. OSM building footprints map with governorate counts
    2. Bivariate choropleth - predicted vs observed damage % per grid cell
    3. First damage detection map (buildings coloured by month)
    4. Cumulative % buildings damaged over time with military events
    5. Ceasefire damage rate table
    6. CDF of days between initial and secondary damage detection
    7. UNOSAT map series at each assessment date

All figures use threshold t=0.655 (90% precision target).

Usage:
    python3 plot_dissertation_figures.py
"""

from pathlib import Path
import warnings
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import matplotlib.ticker as mticker
from matplotlib.colors import LinearSegmentedColormap
from shapely.wkb import loads as wkb_loads
from shapely.geometry import box
from shapely.affinity import rotate as shapely_rotate

from src.constants import DATA_PATH, GAZA_WAR_START

# -- Paths -------------
BUILDINGS_FP   = DATA_PATH / "pixel_postprocessing/buildings_preds.parquet"
OVERTURE_FP    = DATA_PATH / "overture_buildings/gaza_buildings.parquet"
UNOSAT_FP      = DATA_PATH / "unosat_labels.geojson"
AOIS_FP        = DATA_PATH / "unosat_aois.geojson"
FIGURES_DIR    = DATA_PATH / "ablation_runs/figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# -- Constants -----------
THRESHOLD     = 0.655
THRESHOLD_RAW = THRESHOLD * 255
WAR_START     = GAZA_WAR_START  # "2023-10-07"
DPI           = 150

# Gaza Strip is rotated ~10� from vertical - rotate map to align N-S
ROTATION_ANGLE = 10.0  # degrees clockwise

GOV_ORDER  = ["North Gaza", "Gaza", "Deir Al-Balah", "Khan Yunis", "Rafah"]
GOV_COLORS = {
    "North Gaza":    "#e41a1c",
    "Gaza":          "#377eb8",
    "Deir Al-Balah": "#4daf4a",
    "Khan Yunis":    "#984ea3",
    "Rafah":         "#ff7f00",
}
GOV_LINE_STYLES = {
    "North Gaza":    (0, (1, 1)),
    "Gaza":          (0, (3, 1)),
    "Deir Al-Balah": (0, (5, 1)),
    "Khan Yunis":    (0, (5, 2, 1, 2)),
    "Rafah":         (0, (7, 2)),
}

# Military events for Figure 4
INVASIONS = [
    ("2023-10-27", "Ground\ninvasions"),
    ("2023-12-01", "Khan Younis"),
    ("2023-12-26", "Deir al-Balah\nencircled"),
    ("2024-05-06", "Rafah\ninvasion"),
]
CEASEFIRE = ("2023-11-24", "2023-11-30")

plt.rcParams.update({
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "legend.fontsize": 9,
    "font.family": "serif",
})


# -- Helpers ----------
def load_buildings_with_geom() -> gpd.GeoDataFrame:
    df = pd.read_parquet(OVERTURE_FP)
    df["geometry"] = df["geometry_wkb"].apply(lambda x: wkb_loads(bytes(x)))
    return gpd.GeoDataFrame(df, geometry="geometry", crs="EPSG:4326")


def load_predictions() -> pd.DataFrame:
    return pd.read_parquet(BUILDINGS_FP)


def load_aois() -> gpd.GeoDataFrame:
    gdf = gpd.read_file(AOIS_FP)
    # Normalise governorate name column
    if "governorate" in gdf.columns:
        gdf = gdf.rename(columns={"governorate": "adm2_name"})
    return gdf


def get_post_war_cols(df: pd.DataFrame) -> list:
    return sorted([c for c in df.columns if len(c) == 10 and c >= WAR_START])


def rotate_gdf(gdf: gpd.GeoDataFrame, angle: float = ROTATION_ANGLE) -> gpd.GeoDataFrame:
    """Rotate GeoDataFrame around its centroid for map display."""
    cx = gdf.total_bounds[[0, 2]].mean()
    cy = gdf.total_bounds[[1, 3]].mean()
    gdf = gdf.copy()
    gdf["geometry"] = gdf.geometry.apply(
        lambda g: shapely_rotate(g, -angle, origin=(cx, cy), use_radians=False)
    )
    return gdf


# -- Figure 1: Building footprints map ------

def plot_building_footprints() -> None:
    """
    OSM building footprints with governorate boundaries and building counts.
    Mirrors Scher & Van Den Hoek Fig 8.
    """
    print("  Loading buildings...")
    gdf_b = load_buildings_with_geom()
    gdf_a = load_aois()

    # Count buildings per governorate
    counts = gdf_b.groupby("adm2_name").size().to_dict()

    # Rotate for display
    gdf_b_r = rotate_gdf(gdf_b)
    gdf_a_r = rotate_gdf(gdf_a)

    fig, axes = plt.subplots(1, 2, figsize=(12, 14),
                             gridspec_kw={"width_ratios": [1, 0.55]})

    # Left: Gaza City detail inset (zoomed)
    ax_detail = axes[0]
    gaza_bounds = gdf_a_r[gdf_a_r["adm2_name"] == "Gaza"].total_bounds
    cx = (gaza_bounds[0] + gaza_bounds[2]) / 2
    cy = (gaza_bounds[1] + gaza_bounds[3]) / 2
    half = 0.025
    detail_mask = (
        (gdf_b_r.geometry.centroid.x > cx - half) &
        (gdf_b_r.geometry.centroid.x < cx + half) &
        (gdf_b_r.geometry.centroid.y > cy - half) &
        (gdf_b_r.geometry.centroid.y < cy + half)
    )
    gdf_b_r[detail_mask].plot(ax=ax_detail, color="black", linewidth=0.1,
                               edgecolor="black", alpha=0.8)
    gdf_a_r.boundary.plot(ax=ax_detail, color="black", linewidth=0.8)
    ax_detail.set_xlim(cx - half, cx + half)
    ax_detail.set_ylim(cy - half, cy + half)
    ax_detail.set_title("Gaza City Detail", fontsize=11)
    ax_detail.set_axis_off()

    # Right: Full Gaza Strip
    ax_full = axes[1]
    gdf_b_r.plot(ax=ax_full, color="black", linewidth=0.05,
                  edgecolor="black", alpha=0.6, markersize=0.1)
    gdf_a_r.boundary.plot(ax=ax_full, color="black", linewidth=1.0)

    # Add building count labels per governorate
    for _, row in gdf_a_r.iterrows():
        gov = row["adm2_name"]
        cnt = counts.get(gov, 0)
        cx_g = row.geometry.centroid.x + 0.05
        cy_g = row.geometry.centroid.y
        ax_full.annotate(f"{gov}\n{cnt:,} Buildings",
                         xy=(cx_g, cy_g), fontsize=8,
                         ha="left", va="center")

    ax_full.set_axis_off()
    ax_full.set_title("Gaza Strip", fontsize=11)

    # Scale bar
    total_height = gdf_a_r.total_bounds[3] - gdf_a_r.total_bounds[1]
    ax_full.plot([gdf_a_r.total_bounds[0], gdf_a_r.total_bounds[0] + 0.036],
                 [gdf_a_r.total_bounds[1] - 0.01] * 2,
                 color="black", linewidth=2)
    ax_full.text(gdf_a_r.total_bounds[0] + 0.018,
                 gdf_a_r.total_bounds[1] - 0.02,
                 "2 km", ha="center", fontsize=8)

    fig.suptitle("OSM building footprints - Gaza Strip", fontsize=12)
    fig.tight_layout()
    fp = FIGURES_DIR / "building_footprints_map.png"
    fig.savefig(fp, dpi=DPI, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {fp}")


# -- Figure 2: Bivariate choropleth ---------

def plot_bivariate_choropleth() -> None:
    """
    Bivariate choropleth of predicted vs observed damage % per grid cell.
    Follows Ballinger (2025) Fig 8 approach.
    Red = overprediction, blue = underprediction, dark = both high.
    """
    print("  Computing grid cells...")
    preds = load_predictions()
    post_cols = get_post_war_cols(preds)
    gdf_a = load_aois()

    # Load UNOSAT - all damage categories combined
    unosat = gpd.read_file(UNOSAT_FP)
    unosat = unosat[unosat["damage"].isin([1, 2, 3, 4])].copy()

    # Create regular grid over Gaza (500m � 500m H 0.0045�)
    bounds = gdf_a.total_bounds
    cell = 0.005
    xs = np.arange(bounds[0], bounds[2], cell)
    ys = np.arange(bounds[1], bounds[3], cell)

    rows = []
    for x in xs:
        for y in ys:
            cell_box = box(x, y, x + cell, y + cell)
            # Predicted: buildings in cell with max post-war pred >= threshold
            mask_p = (
                (preds["lon"] >= x) & (preds["lon"] < x + cell) &
                (preds["lat"] >= y) & (preds["lat"] < y + cell)
            )
            sub_p = preds[mask_p]
            if len(sub_p) < 3:
                continue
            pred_pct = (sub_p[post_cols].max(axis=1) >= THRESHOLD_RAW).mean() * 100

            # Observed: UNOSAT points in cell
            mask_u = (
                (unosat.geometry.x >= x) & (unosat.geometry.x < x + cell) &
                (unosat.geometry.y >= y) & (unosat.geometry.y < y + cell)
            )
            sub_u = unosat[mask_u]
            if len(sub_u) == 0:
                obs_pct = 0.0
            else:
                obs_pct = min(len(sub_u) / max(len(sub_p), 1) * 100, 100)

            rows.append({
                "geometry": cell_box,
                "pred_pct": pred_pct,
                "obs_pct": obs_pct,
            })

    grid = gpd.GeoDataFrame(rows, crs="EPSG:4326")
    grid = gpd.clip(grid, gdf_a.unary_union)
    grid_r = rotate_gdf(grid)
    aois_r = rotate_gdf(gdf_a)

    # Bivariate colour: percentile bins
    n_bins = 4
    pred_bins = np.percentile(grid["pred_pct"], np.linspace(0, 100, n_bins + 1))
    obs_bins  = np.percentile(grid["obs_pct"],  np.linspace(0, 100, n_bins + 1))
    grid["pred_bin"] = np.digitize(grid["pred_pct"], pred_bins[1:-1])
    grid["obs_bin"]  = np.digitize(grid["obs_pct"],  obs_bins[1:-1])
    grid_r["pred_bin"] = grid["pred_bin"].values
    grid_r["obs_bin"]  = grid["obs_bin"].values

    # Colour matrix: rows=pred (0=low, 3=high), cols=obs (0=low, 3=high)
    # White=both low, dark=both high, red=high pred/low obs, blue=low pred/high obs
    def bivariate_color(pred_bin, obs_bin):
        r = 0.2 + 0.8 * (pred_bin / (n_bins - 1))
        b = 0.2 + 0.8 * (obs_bin  / (n_bins - 1))
        g = 0.05
        brightness = 1 - (pred_bin + obs_bin) / (2 * (n_bins - 1)) * 0.7
        if pred_bin > obs_bin:      # overprediction - red
            return (brightness * r, brightness * g, brightness * g)
        elif obs_bin > pred_bin:    # underprediction - blue
            return (brightness * g, brightness * g, brightness * b)
        else:                       # agreement - dark purple
            return (brightness * r * 0.6, brightness * g, brightness * b * 0.6)

    grid_r["color"] = [
        bivariate_color(r["pred_bin"], r["obs_bin"])
        for _, r in grid_r.iterrows()
    ]

    fig, ax = plt.subplots(1, 1, figsize=(6, 12))
    for _, row in grid_r.iterrows():
        gpd.GeoDataFrame([row], crs="EPSG:4326").plot(
            ax=ax, color=[row["color"]], edgecolor="none"
        )
    aois_r.boundary.plot(ax=ax, color="black", linewidth=0.8)
    ax.set_axis_off()
    ax.set_title("Predicted vs observed damage\n(bivariate choropleth)", fontsize=11)

    # Legend
    legend_ax = fig.add_axes([0.05, 0.05, 0.18, 0.18])
    legend_data = np.zeros((n_bins, n_bins, 3))
    for i in range(n_bins):
        for j in range(n_bins):
            legend_data[i, j] = bivariate_color(i, j)
    legend_ax.imshow(legend_data, origin="lower", aspect="auto")
    legend_ax.set_xlabel("Observed\nDamage %", fontsize=7)
    legend_ax.set_ylabel("Predicted\nDamage %", fontsize=7)
    legend_ax.set_xticks([0, n_bins - 1])
    legend_ax.set_xticklabels(["0", "100"], fontsize=6)
    legend_ax.set_yticks([0, n_bins - 1])
    legend_ax.set_yticklabels(["0", "100"], fontsize=6)

    fig.tight_layout()
    fp = FIGURES_DIR / "bivariate_choropleth.png"
    fig.savefig(fp, dpi=DPI, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {fp}")


# -- Figure 3: First detection map ------------

def plot_first_detection_map() -> None:
    """
    Buildings coloured by month of first damage detection.
    Mirrors Scher & Van Den Hoek Fig 3.
    """
    print("  Loading predictions for first detection map...")
    preds = load_predictions()
    post_cols = get_post_war_cols(preds)

    # Find first window where prediction >= threshold
    preds["first_detected"] = None
    for col in post_cols:
        mask = (preds["first_detected"].isna()) & (preds[col] >= THRESHOLD_RAW)
        preds.loc[mask, "first_detected"] = col

    # Load building geometries
    gdf_b = load_buildings_with_geom()[["building_id", "geometry", "adm2_name"]]
    gdf_b = gdf_b.merge(
        preds[["first_detected"]].reset_index(),
        left_on="building_id", right_on="building_id", how="left"
    ) if "building_id" in preds.index.names else gdf_b.assign(
        first_detected=preds["first_detected"].values
    )
    gdf_a = load_aois()

    # Colour by detection month using viridis_r (Oct 2023=dark, latest=yellow)
    unique_dates = sorted([d for d in preds["first_detected"].unique() if d is not None])
    cmap = plt.cm.get_cmap("viridis_r", len(unique_dates))
    date_to_idx = {d: i for i, d in enumerate(unique_dates)}

    gdf_b_r = rotate_gdf(gdf_b)
    gdf_a_r = rotate_gdf(gdf_a)

    fig, ax = plt.subplots(figsize=(6, 12))

    # Undamaged
    undamaged = gdf_b_r[gdf_b_r["first_detected"].isna()]
    undamaged.plot(ax=ax, color="whitesmoke", edgecolor="lightgrey",
                   linewidth=0.1, alpha=0.5)

    # Damaged - coloured by first detection date
    for date in unique_dates:
        sub = gdf_b_r[gdf_b_r["first_detected"] == date]
        if len(sub) == 0:
            continue
        sub.plot(ax=ax, color=cmap(date_to_idx[date]),
                 edgecolor="none", alpha=0.9)

    # Governorate boundaries
    gdf_a_r.boundary.plot(ax=ax, color="black", linewidth=1.0)

    # North arrow
    ax.annotate("N", xy=(0.05, 0.97), xycoords="axes fraction",
                fontsize=14, ha="center", va="top",
                arrowprops=dict(arrowstyle="-|>", color="black"),
                xytext=(0.05, 0.94))

    # Colourbar
    sm = plt.cm.ScalarMappable(
        cmap="viridis_r",
        norm=mcolors.BoundaryNorm(range(len(unique_dates) + 1), len(unique_dates))
    )
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, orientation="vertical",
                        fraction=0.03, pad=0.01,
                        ticks=np.arange(0.5, len(unique_dates)))
    cbar.set_ticklabels(
        [pd.to_datetime(d).strftime("%b\n%Y") for d in unique_dates],
        fontsize=7
    )
    cbar.ax.invert_yaxis()

    ax.set_title("Aggregate damage detected - Gaza\n(buildings coloured by month of first detection)",
                 fontsize=10)
    ax.set_axis_off()

    # Scale bar
    tb = gdf_a_r.total_bounds
    ax.plot([tb[0] + 0.005, tb[0] + 0.005 + 0.018], [tb[1] + 0.005] * 2,
            color="black", linewidth=2)
    ax.text(tb[0] + 0.005 + 0.009, tb[1] + 0.008, "2 km",
            ha="center", fontsize=8)

    fig.tight_layout()
    fp = FIGURES_DIR / "first_detection_map.png"
    fig.savefig(fp, dpi=DPI, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {fp}")


# -- Figure 4: Cumulative damage over time ----------

def plot_cumulative_damage() -> None:
    """
    % buildings damaged over time by governorate + Gaza Strip.
    With military event markers. Mirrors Scher & Van Den Hoek Fig 4.
    """
    preds = load_predictions()
    post_cols = get_post_war_cols(preds)
    dates = pd.to_datetime(post_cols)

    fig, ax = plt.subplots(figsize=(12, 6))

    # Ceasefire shading
    ax.axvspan(pd.to_datetime(CEASEFIRE[0]), pd.to_datetime(CEASEFIRE[1]),
               color="grey", alpha=0.25, label="Ceasefire (24-30 Nov 2023)")

    # Invasion vertical lines
    for date_str, label in INVASIONS:
        ax.axvline(pd.to_datetime(date_str), color="red",
                   linestyle="--", linewidth=1.2, alpha=0.8)
        ax.text(pd.to_datetime(date_str), 72, label,
                rotation=90, va="top", ha="right", fontsize=7, color="red")

    # Per-governorate lines
    for gov in GOV_ORDER:
        sub = preds[preds["adm2_name"] == gov]
        if len(sub) == 0:
            continue
        n = len(sub)
        pct = []
        for col in post_cols:
            cols_to_date = [c for c in post_cols if c <= col]
            pct.append((sub[cols_to_date].max(axis=1) >= THRESHOLD_RAW).sum() / n * 100)
        ax.plot(dates, pct,
                color=GOV_COLORS[gov],
                linestyle=GOV_LINE_STYLES[gov],
                linewidth=1.8, label=gov)

    # Gaza Strip total
    n_total = len(preds)
    gaza_pct = []
    for col in post_cols:
        cols_to_date = [c for c in post_cols if c <= col]
        gaza_pct.append((preds[cols_to_date].max(axis=1) >= THRESHOLD_RAW).sum() / n_total * 100)
    ax.plot(dates, gaza_pct, color="black", linewidth=2.5,
            linestyle="-", label="Gaza Strip")

    ax.set_xlabel("Date")
    ax.set_ylabel("Buildings damaged (%)")
    ax.set_title(f"Percentage of buildings damaged over time (t={THRESHOLD})")
    ax.set_ylim(0, 80)
    ax.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter("%Y-%m"))
    ax.legend(loc="upper left", frameon=True)
    fig.autofmt_xdate()
    fig.tight_layout()
    fp = FIGURES_DIR / "cumulative_damage_over_time.png"
    fig.savefig(fp, dpi=DPI)
    plt.close()
    print(f"  Saved: {fp}")

    return preds, post_cols, gaza_pct


# -- Figure 5: Ceasefire table -----------------

def compute_ceasefire_table(preds: pd.DataFrame, post_cols: list) -> None:
    """
    Table: % decrease in rate of new damage during ceasefire vs first 6 weeks.
    """
    # First 6 weeks of war: windows up to ~2023-11-17 (~42 days after Oct 7)
    # Our windows: 2023-10-07, 2023-12-07 - ceasefire falls within the gap
    # Rate = new buildings damaged per day in each period

    # Pre-ceasefire: 2023-10-07 to 2023-11-24 (~48 days)
    pre_cf_col = "2023-10-07"
    # Ceasefire: 2023-11-24 to 2023-11-30 (6 days)
    # Post-ceasefire proxy: 2023-12-07 window

    rows = []
    for gov in GOV_ORDER + ["Gaza Strip"]:
        if gov == "Gaza Strip":
            sub = preds
        else:
            sub = preds[preds["adm2_name"] == gov]
        n = len(sub)

        # New damage in first window (proxy for pre-ceasefire rate)
        new_pre = (sub[pre_cf_col] >= THRESHOLD_RAW).sum()
        days_pre = 48  # Oct 7 to Nov 24
        rate_pre = new_pre / days_pre / n * 100

        # New damage in Dec window not in Oct window (proxy for post-ceasefire)
        if "2023-12-07" in preds.columns:
            new_post = (
                (sub["2023-12-07"] >= THRESHOLD_RAW) &
                (sub[pre_cf_col] < THRESHOLD_RAW)
            ).sum()
            days_post = 61 - 6  # Dec window minus ceasefire days
            rate_post = new_post / days_post / n * 100
            change = (rate_post - rate_pre) / rate_pre * 100 if rate_pre > 0 else float("nan")
        else:
            rate_post = float("nan")
            change = float("nan")

        rows.append({
            "Governorate": gov,
            "Pre-ceasefire rate (%/day)": f"{rate_pre:.4f}",
            "Post-ceasefire rate (%/day)": f"{rate_post:.4f}",
            "Change (%)": f"{change:+.1f}%" if not np.isnan(change) else "N/A",
        })

    df_table = pd.DataFrame(rows)
    print("\n" + "="*70)
    print("TABLE 5: Ceasefire damage rate comparison")
    print("="*70)
    print(df_table.to_string(index=False))

    fp = FIGURES_DIR / "ceasefire_table.csv"
    df_table.to_csv(fp, index=False)
    print(f"\n  Saved: {fp}")


# -- Figure 6: CDF of confirmation lag ---------------

def plot_confirmation_lag_cdf() -> None:
    """
    CDF of days between initial and secondary damage detection.
    Mirrors Scher & Van Den Hoek Fig 6.
    """
    preds = load_predictions()
    post_cols = get_post_war_cols(preds)
    dates = pd.to_datetime(post_cols)

    # For each building first detected as damaged, find first subsequent
    # window where it is also >= threshold (secondary confirmation)
    lags = []
    for i, col in enumerate(post_cols[:-1]):
        # Buildings first detected in this window
        prev_cols = post_cols[:i]
        if prev_cols:
            first_here = (
                (preds[col] >= THRESHOLD_RAW) &
                (preds[prev_cols].max(axis=1) < THRESHOLD_RAW)
            )
        else:
            first_here = preds[col] >= THRESHOLD_RAW

        first_date = dates[i]
        sub = preds[first_here]
        if len(sub) == 0:
            continue

        # Find first subsequent window with confirmation
        for j, next_col in enumerate(post_cols[i + 1:], start=i + 1):
            confirmed = sub[next_col] >= THRESHOLD_RAW
            lag_days = (dates[j] - first_date).days
            n_confirmed = confirmed.sum()
            lags.extend([lag_days] * int(n_confirmed))
            sub = sub[~confirmed]  # remove confirmed buildings
            if len(sub) == 0:
                break

    if not lags:
        print("  No confirmation lags found - skipping CDF")
        return

    lags = np.array(sorted(lags))
    cdf = np.arange(1, len(lags) + 1) / len(lags) * 100

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(lags, cdf, color="black", linewidth=2)

    # Mark key percentiles
    milestones = [7, 14, 31, 176]
    for days in milestones:
        idx = np.searchsorted(lags, days)
        if idx < len(cdf):
            pct = cdf[idx]
            ax.axvline(days, color="red", linestyle="--", linewidth=1, alpha=0.7)
            ax.axhline(pct,  color="red", linestyle="--", linewidth=1, alpha=0.7)
            ax.text(days + 3, pct - 3, f"{pct:.1f}%", color="red", fontsize=8)

    ax.set_xlabel("Days to confirm initial damage detection")
    ax.set_ylabel("Cumulative probability (%)")
    ax.set_title("CDF - days between initial and secondary damage detection")
    ax.set_xlim(0, max(lags) + 10)
    ax.set_ylim(0, 105)
    fig.tight_layout()
    fp = FIGURES_DIR / "confirmation_lag_cdf.png"
    fig.savefig(fp, dpi=DPI)
    plt.close()
    print(f"  Saved: {fp}")
    print(f"  Total confirmations: {len(lags):,}")
    print(f"  Lag at  7 days: {cdf[np.searchsorted(lags,  7)]:.1f}%")
    print(f"  Lag at 31 days: {cdf[np.searchsorted(lags, 31)]:.1f}%")
    print(f"  Lag at 99%: {lags[np.searchsorted(cdf, 99.0)]:.0f} days")


# -- Figure 7: UNOSAT map series ------------------------

def plot_unosat_map_series() -> None:
    """
    UNOSAT data at each assessment date aggregated to pixel grid.
    New damage = orange, old damage = blue. Mirrors Scher & Van Den Hoek Fig 9.
    """
    print("  Loading UNOSAT...")
    unosat = gpd.read_file(UNOSAT_FP)
    unosat = unosat[unosat["damage"].isin([1, 2, 3, 4])].copy()
    unosat["date"] = pd.to_datetime(unosat["date"])

    gdf_a = load_aois()
    aois_r = rotate_gdf(gdf_a)

    # Get UNOSAT assessment dates within our study period
    war_start_dt = pd.to_datetime(WAR_START)
    assessment_dates = sorted(unosat[unosat["date"] >= war_start_dt]["date"].unique())
    # Limit to dates up to end of our study period
    assessment_dates = [d for d in assessment_dates if d <= pd.to_datetime("2025-10-07")]

    # Grid resolution: 0.005� H 500m
    cell = 0.005
    bounds = gdf_a.total_bounds
    xs = np.arange(bounds[0], bounds[2], cell)
    ys = np.arange(bounds[1], bounds[3], cell)

    # Pre-compute grid cell assignments for all UNOSAT points
    unosat["x_bin"] = np.digitize(unosat.geometry.x, xs) - 1
    unosat["y_bin"] = np.digitize(unosat.geometry.y, ys) - 1
    unosat["cell_id"] = unosat["x_bin"].astype(str) + "_" + unosat["y_bin"].astype(str)

    n_dates = len(assessment_dates)
    ncols = 3
    nrows = int(np.ceil(n_dates / ncols))

    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 3.5, nrows * 7))
    axes = axes.flatten()

    seen_cells = set()

    for i, date in enumerate(assessment_dates):
        ax = axes[i]

        # Points up to this date
        current = unosat[unosat["date"] == date]
        current_cells = set(current["cell_id"].unique())
        new_cells = current_cells - seen_cells
        old_cells = current_cells & seen_cells
        seen_cells.update(current_cells)

        # Build grid GeoDataFrame
        def cells_to_gdf(cell_ids):
            rows = []
            for cid in cell_ids:
                xi, yi = map(int, cid.split("_"))
                if 0 <= xi < len(xs) and 0 <= yi < len(ys):
                    rows.append({"geometry": box(xs[xi], ys[yi],
                                                 xs[xi] + cell, ys[yi] + cell)})
            if rows:
                return rotate_gdf(gpd.GeoDataFrame(rows, crs="EPSG:4326"))
            return None

        new_gdf = cells_to_gdf(new_cells)
        old_gdf = cells_to_gdf(old_cells)

        aois_r.boundary.plot(ax=ax, color="black", linewidth=0.5)
        if old_gdf is not None:
            old_gdf.plot(ax=ax, color="#4393c3", alpha=0.8, edgecolor="none")
        if new_gdf is not None:
            new_gdf.plot(ax=ax, color="#f4a40a", alpha=0.9, edgecolor="none")

        ax.set_title(date.strftime("%Y-%m-%d"), fontsize=9)
        ax.set_axis_off()

    # Hide unused axes
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    # Legend
    new_patch = mpatches.Patch(color="#f4a40a", label="New Damage")
    old_patch = mpatches.Patch(color="#4393c3", label="Old Damage")
    fig.legend(handles=[new_patch, old_patch], loc="lower right",
               fontsize=10, frameon=True)

    fig.suptitle("UNOSAT damage assessments - Gaza (aggregated to 500m grid)",
                 fontsize=12)
    fig.tight_layout()
    fp = FIGURES_DIR / "unosat_map_series.png"
    fig.savefig(fp, dpi=DPI, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {fp}")


# -- Main ----------------

if __name__ == "__main__":
    print("Figure 1: Building footprints map...")
    plot_building_footprints()

    print("\nFigure 2: Bivariate choropleth...")
    plot_bivariate_choropleth()

    print("\nFigure 3: First detection map...")
    plot_first_detection_map()

    print("\nFigure 4: Cumulative damage over time...")
    preds, post_cols, gaza_pct = plot_cumulative_damage()

    print("\nFigure 5: Ceasefire table...")
    compute_ceasefire_table(preds, post_cols)

    print("\nFigure 6: Confirmation lag CDF...")
    plot_confirmation_lag_cdf()

    print("\nFigure 7: UNOSAT map series...")
    plot_unosat_map_series()

    print(f"\nAll figures saved to {FIGURES_DIR}")
