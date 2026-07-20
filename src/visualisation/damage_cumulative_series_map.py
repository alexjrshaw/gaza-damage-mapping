"""
Gaza cumulative damage map series: 2x7 version.
Reverts to scatter centroids (footprints are 0.24px at this scale,
physically invisible).
"""

import geopandas as gpd
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "Arial"
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from shapely.affinity import rotate as shapely_rotate
from shapely.wkb import loads as wkb_loads

from src.constants import DATA_PATH

PREDS_FP = DATA_PATH / "pixel_postprocessing/buildings_preds.parquet"
ADMIN2_FP = DATA_PATH / "raw/pse_admin2.geojson"
OUT_FP = DATA_PATH / "ablation_runs/figures/damage_cumulative_series.png"

THRESHOLD = int(0.670 * 255)  # Gaza-calibrated threshold: int(0.670 * 255) = 170
UTM_CRS = "EPSG:32636"
MARKER_SIZE = 0.1

GOVERNORATE_ALIASES = {
    "khan yunis": "khan younis",
    "khan younis": "khan younis",
    "deir al-balah": "deir al-balah",
    "deir al balah": "deir al-balah",
    "north gaza": "north gaza",
    "gaza": "gaza",
    "rafah": "rafah",
}


def normalize_governorate_name(name):
    return GOVERNORATE_ALIASES.get(name.lower().strip(), name.lower().strip())


PRE_WINDOW_COLS = [
    "2021-10-07",
    "2022-10-07",
    "2022-12-07",
    "2023-02-07",
    "2023-04-07",
    "2023-06-07",
    "2023-08-07",
]
POST_WINDOW_COLS = [
    "2023-10-07",
    "2023-12-07",
    "2024-02-07",
    "2024-04-07",
    "2024-06-07",
    "2024-08-07",
    "2024-10-07",
    "2024-12-07",
    "2025-02-07",
    "2025-04-07",
    "2025-06-07",
    "2025-08-07",
    "2025-10-07",
]
PANEL_LABELS = [
    "Dec '23",
    "Feb '24",
    "Apr '24",
    "Jun '24",
    "Aug '24",
    "Oct '24",
    "Dec '24",
    "Feb '25",
    "Apr '25",
    "Jun '25",
    "Aug '25",
    "Oct '25",
    "Dec '25",
]

COLOR_OLD = "#377eb8"
COLOR_NEW = "#ff7f00"


def first_damaged_window_index(row, pre_cols, post_cols):
    if row[pre_cols].max() >= THRESHOLD:
        return None
    for i, col in enumerate(post_cols):
        if row[col] >= THRESHOLD:
            return i
    return None


def compute_rotation_angle(centroids):
    centred = centroids - centroids.mean(axis=0)
    cov = np.cov(centred.T)
    eigvals, eigvecs = np.linalg.eigh(cov)
    principal = eigvecs[:, np.argmax(eigvals)]
    return 90 - np.degrees(np.arctan2(principal[1], principal[0]))


def rotate_points(coords, angle_deg, origin):
    theta = np.radians(angle_deg)
    cos_t, sin_t = np.cos(theta), np.sin(theta)
    cx, cy = origin
    dx, dy = coords[:, 0] - cx, coords[:, 1] - cy
    return np.column_stack(
        [
            dx * cos_t - dy * sin_t + cx,
            dx * sin_t + dy * cos_t + cy,
        ]
    )


def boundary_xy_parts(geom):
    if geom.geom_type == "Polygon":
        yield geom.exterior.xy
    elif geom.geom_type == "MultiPolygon":
        for part in geom.geoms:
            yield part.exterior.xy


def main():
    print("Loading building predictions...")
    df = pd.read_parquet(PREDS_FP)
    print(f"  {len(df):,} buildings")

    print("Finding first-detected window per building (full Equation 3)...")
    df["first_idx"] = df.apply(
        lambda row: first_damaged_window_index(row, PRE_WINDOW_COLS, POST_WINDOW_COLS),
        axis=1,
    )
    n_damaged = df["first_idx"].notna().sum()
    print(f"  {n_damaged:,} damaged ({n_damaged/len(df)*100:.1f}%)")

    print("Building GeoDataFrame, reprojecting to UTM...")
    df["geometry"] = df["geometry_wkb"].apply(lambda x: wkb_loads(bytes(x)))
    gdf = gpd.GeoDataFrame(df, geometry="geometry", crs="EPSG:4326").to_crs(UTM_CRS)

    print("Loading and dissolving Gaza boundary...")
    admin2_raw = gpd.read_file(ADMIN2_FP).to_crs(UTM_CRS)
    name_col = [c for c in admin2_raw.columns if "name" in c.lower() and "ar" not in c.lower()][0]
    gov_names = set(gdf["adm2_name"].apply(normalize_governorate_name))
    admin2 = admin2_raw[admin2_raw[name_col].apply(normalize_governorate_name).isin(gov_names)]
    gaza_boundary = admin2.dissolve().geometry.iloc[0]
    print(f"  {len(admin2)} governorates matched")

    print("Computing north-up rotation...")
    centroids = np.array([(g.centroid.x, g.centroid.y) for g in gdf.geometry])
    rotation_deg = compute_rotation_angle(centroids)
    center = (centroids[:, 0].mean(), centroids[:, 1].mean())
    rot_cents = rotate_points(centroids, rotation_deg, center)

    if np.corrcoef(centroids[:, 1], rot_cents[:, 1])[0, 1] < 0:
        rotation_deg += 180
        rot_cents = rotate_points(centroids, rotation_deg, center)
    print(f"  Final rotation: {rotation_deg:.1f}°")

    gaza_boundary_rot = shapely_rotate(gaza_boundary, rotation_deg, origin=center)
    boundary_parts = list(boundary_xy_parts(gaza_boundary_rot))
    x_rot, y_rot = rot_cents[:, 0], rot_cents[:, 1]
    first_idx_arr = df["first_idx"].values

    print("Plotting 4×4 series (A4 portrait)...")
    fig, axes = plt.subplots(
        2,
        7,
        figsize=(8.27, 11.69),
        gridspec_kw={"wspace": 0.02, "hspace": 0.00},
    )
    axes_flat = axes.flat

    for i, label in enumerate(PANEL_LABELS):
        ax = axes_flat[i]

        has_old = (first_idx_arr < i) & ~pd.isna(first_idx_arr)
        has_new = (first_idx_arr == i) & ~pd.isna(first_idx_arr)

        ax.scatter(x_rot[has_old], y_rot[has_old], s=MARKER_SIZE, color=COLOR_OLD, linewidths=0)
        ax.scatter(x_rot[has_new], y_rot[has_new], s=MARKER_SIZE, color=COLOR_NEW, linewidths=0)

        for part_x, part_y in boundary_parts:
            ax.plot(part_x, part_y, color="black", linewidth=0.4)

        ax.set_title(label, fontsize=9, pad=2)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_aspect("equal")
        for spine in ax.spines.values():
            spine.set_visible(False)

    legend_ax = axes_flat[13]
    legend_ax.axis("off")
    legend_ax.legend(
        handles=[
            Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                markerfacecolor=COLOR_OLD,
                markersize=10,
                label="Old damage",
            ),
            Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                markerfacecolor=COLOR_NEW,
                markersize=10,
                label="New damage",
            ),
        ],
        loc="center",
        fontsize=10,
        frameon=False,
    )

    fig.suptitle("Gaza: cumulative building damage by assessment window", fontsize=13, y=0.99)

    OUT_FP.parent.mkdir(exist_ok=True, parents=True)
    fig.savefig(OUT_FP, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved to {OUT_FP}")


if __name__ == "__main__":
    main()
