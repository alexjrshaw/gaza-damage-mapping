#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# In[ ]:


from collections import defaultdict
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
import xarray as xr

from src.constants import DATA_PATH, AOIS_TRAIN, AOIS_TEST
from src.data.unosat import load_unosat_labels, load_unosat_geo
from src.data.utils import read_fp_within_geo, get_all_aois
from src.utils.time import timeit


# In[ ]:


run_name = "pixel_postprocessing"
MERGED_RASTERS_DIR = DATA_PATH / "merged_probability_rasters"


# # Evaluate predictions vs UNOSAT labels
#
# Gaza adaptation of Dietrich et al. evaluation.ipynb.
# Uses pixel-level probability rasters from local_pixel_inference.py.

# ## Helper: find post dates
# Adapted from postprocessing/utils.py find_post_dates() to use Gaza window naming convention.

# In[ ]:


import re


def find_post_dates_gaza(merged_rasters_dir=MERGED_RASTERS_DIR):
    """
    Find post dates from merged probability raster filenames.
    Gaza adaptation of find_post_dates() — uses gaza_w{n}_{start}_{end}.tif naming.
    Returns list of (start, end) tuples sorted by date.
    """
    post_dates = []
    date_pattern = re.compile(r"(\d{4}-\d{2}-\d{2})_(\d{4}-\d{2}-\d{2})")
    for fp in sorted(merged_rasters_dir.glob("gaza_*.tif")):
        match = date_pattern.search(fp.stem)
        if match:
            post_dates.append((match.group(1), match.group(2)))
    return sorted(post_dates)


post_dates = find_post_dates_gaza()
print(f"Found {len(post_dates)} post dates:")
for d in post_dates:
    print(f"  {d[0]} to {d[1]}")


def extract_raster_value(point, raster):
    value = raster.sel(x=point.x, y=point.y, method="nearest").item()
    return value


def get_preds_geo(geo, run_name):

    post_dates = find_post_dates_gaza()
    post_dates_ = [p[0] for p in post_dates]  # keep only first date for reference

    # Read and stack preds for each date
    fp_preds = [
        DATA_PATH / run_name / f'ukraine_{"_".join(post_date)}.tif' for post_date in post_dates
    ]
    dates = xr.Variable("date", pd.to_datetime(post_dates_))
    preds = xr.concat([read_fp_within_geo(fp, geo) for fp in fp_preds], dim=dates).squeeze()
    return preds


# ## Assign predictions to each UNOSAT label
# Mirrors combine_all_unosat_points_with_preds() exactly. Extracts raster value at each UNOSAT point location for each time window.

# In[ ]:


def get_preds_geo_gaza(geo, merged_rasters_dir=MERGED_RASTERS_DIR):
    """
    Load and stack probability rasters for a given geometry.
    Gaza adaptation of get_preds_geo() — uses gaza_w{n}_{start}_{end}.tif naming.
    Mirrors Dietrich et al. exactly.
    """
    post_dates = find_post_dates_gaza(merged_rasters_dir)
    post_dates_ = [p[0] for p in post_dates]

    fp_preds = [
        merged_rasters_dir / f"gaza_w{str(i+1).zfill(2)}_{start}_{end}.tif"
        for i, (start, end) in enumerate(post_dates)
    ]
    dates = xr.Variable("date", pd.to_datetime(post_dates_))
    preds = xr.concat([read_fp_within_geo(fp, geo) for fp in fp_preds], dim=dates).squeeze()
    return preds


def extract_raster_value(point, raster):
    """Extract raster value at point location. Unchanged from Dietrich et al."""
    return raster.sel(x=point.x, y=point.y, method="nearest").item()


def extract_raster_value_with_window(point, raster, window=1, agg="mean"):
    """Extract raster value with optional spatial window. Unchanged from Dietrich et al."""
    if window == 1:
        return extract_raster_value(point, raster)
    else:
        half_window = window // 2
        x_idx = np.argmin(np.abs(raster.x.values - point.x))
        y_idx = np.argmin(np.abs(raster.y.values - point.y))
        raster_wind = raster.isel(
            x=slice(x_idx - half_window, x_idx + half_window + 1),
            y=slice(y_idx - half_window, y_idx + half_window + 1),
        )
        if agg == "mean":
            return raster_wind.mean().item()
        elif agg == "max":
            return raster_wind.max().item()
        else:
            raise ValueError(f"agg={agg} not supported")


def extract_raster_value_with_window_or_nan(point, raster, window, agg):
    """Unchanged from Dietrich et al."""
    try:
        return extract_raster_value_with_window(point, raster, window, agg)
    except:
        return None


@timeit
def combine_all_unosat_points_with_preds_gaza(window=1, agg="mean"):
    """
    Assign pixel predictions to each UNOSAT point for all AOIs.
    Gaza adaptation of combine_all_unosat_points_with_preds() — mirrors Dietrich et al. exactly.
    """
    folder = DATA_PATH / run_name / "aoi_preds"
    folder.mkdir(exist_ok=True, parents=True)

    if window == 1:
        fp = folder / "unosat_points_with_preds.geojson"
    else:
        fp = folder / f"unosat_points_with_preds_window_{window}_{agg}.geojson"

    if fp.exists():
        print(f"File {fp} already exists.")
        return

    post_dates = find_post_dates_gaza()
    gdf_labels_ = None

    for aoi in tqdm(get_all_aois()):
        geo = load_unosat_geo(aoi)
        preds = get_preds_geo_gaza(geo)

        gdf_labels = load_unosat_labels(aoi, labels_to_keep=None, combine_epoch="first_severe")[
            ["geometry", "date", "aoi", "damage"]
        ]
        for post_date in post_dates:
            date = post_date[0]
            gdf_labels[f"pred_{date}"] = gdf_labels.geometry.apply(
                lambda x: extract_raster_value_with_window_or_nan(
                    x, preds.sel(date=date), window, agg
                )
            )
        gdf_labels_ = (
            pd.concat([gdf_labels_, gdf_labels]) if gdf_labels_ is not None else gdf_labels
        )

    gdf_labels_.fillna(0, inplace=True)
    gdf_labels_.to_file(fp, driver="GeoJSON")
    print(f"Saved {len(gdf_labels_):,} points to {fp}")


def load_unosat_points_with_preds_gaza(window=1, agg=None):
    """
    Load UNOSAT points with predictions. Gaza adaptation of load_unosat_points_with_preds().
    Mirrors Dietrich et al. exactly.
    """
    folder = DATA_PATH / run_name / "aoi_preds"
    if window == 1:
        fp = folder / "unosat_points_with_preds.geojson"
    else:
        fp = folder / f"unosat_points_with_preds_window_{window}_{agg}.geojson"
    if not fp.exists():
        combine_all_unosat_points_with_preds_gaza(window, agg)
    return gpd.read_file(fp)


# In[ ]:


# Gaza adaptation: window=3, agg="max" — unchanged from Dietrich et al.
_ = load_unosat_points_with_preds_gaza(window=3, agg="max")


# ## Sweep over different thresholds
# Key Gaza adaptation: only_2022_for_pos=False since Gaza war started Oct 2023, not 2022.

# In[ ]:


from src.classification.metrics import get_metrics

THRESHOLDS = np.arange(0.1, 0.95, 0.005)  # unchanged from Dietrich et al.


def unosat_vs_preds_comparison_gaza(
    window=3,
    agg="max",
    labels_to_keep=[1, 2],
    only_2022_for_pos=False,  # Gaza adaptation: False (war started Oct 2023, not 2022)
    metrics="all",
):
    """
    Compare UNOSAT labels vs pixel predictions across thresholds.
    Gaza adaptation of unosat_vs_preds_comparaison() — mirrors Dietrich et al. exactly
    except only_2022_for_pos=False.
    """
    gdf_points = load_unosat_points_with_preds_gaza(window=window, agg=agg)
    gdf_train = gdf_points[gdf_points.aoi.isin(AOIS_TRAIN)]
    gdf_test = gdf_points[gdf_points.aoi.isin(AOIS_TEST)]

    if labels_to_keep is not None:
        gdf_train = gdf_train[gdf_train.damage.isin(labels_to_keep)]
        gdf_test = gdf_test[gdf_test.damage.isin(labels_to_keep)]

    # Convert date column to datetime — required by get_metrics()
    gdf_train["date"] = pd.to_datetime(gdf_train["date"])
    gdf_test["date"] = pd.to_datetime(gdf_test["date"])

    d_metrics_list = defaultdict(list)
    for t in THRESHOLDS:
        d_metrics = get_metrics(
            gdf_test,
            threshold=t,
            method="date-wise",
            print_classification_report=False,
            only_2022_for_pos=only_2022_for_pos,
            pos_year="2023",
            return_preds=False,
        )
        if metrics != "all":
            d_metrics = {k: v for k, v in d_metrics.items() if k in metrics}
        for k, v in d_metrics.items():
            d_metrics_list[k].append(v)

    return d_metrics_list


d_metrics_list = unosat_vs_preds_comparison_gaza(
    window=3, agg="max", labels_to_keep=[1, 2], only_2022_for_pos=False
)


# ## Best threshold
# Unchanged from Dietrich et al. — finds threshold closest to 90% precision.

# In[ ]:


def find_best_threshold(d_metrics_list, metric="precision", target=0.9):
    """Unchanged from Dietrich et al."""
    diff = np.array(d_metrics_list[metric]) - target
    idx_min = np.abs(diff).argmin()
    best_threshold = THRESHOLDS[idx_min] if diff[idx_min] > 0 else THRESHOLDS[idx_min + 1]
    print(f"Best threshold for {metric}@{target} = {best_threshold:.3f}")
    return best_threshold


best_threshold = find_best_threshold(d_metrics_list, metric="precision", target=0.9)
print(f"Dietrich et al. threshold (Ukraine): 0.655")
print(f"Gaza optimal threshold:              {best_threshold:.3f}")


# ## Plot
# Unchanged from Dietrich et al. — mirrors Fig. S1 for Gaza.

# In[ ]:


def plot_metrics_vs_thresholds(
    d_metrics_list,
    vlines=None,
    label_with_max=False,
    ax=None,
    style="seaborn-v0_8",
    save_fp=None,
):
    """Unchanged from Dietrich et al."""
    with plt.style.context(style):
        if ax is None:
            fig, ax = plt.subplots(figsize=(12, 6))

        for label, scores in d_metrics_list.items():
            if label_with_max:
                label += f"(max@{THRESHOLDS[np.argmax(scores)]:.2f}={np.max(scores):.2f})"
            ax.plot(THRESHOLDS, scores, label=label, linewidth=2.5)

        if vlines is not None:
            for vline in vlines:
                ax.axvline(vline, color="black", linestyle="--", alpha=0.5)

        ax.set_xlabel("Threshold", fontsize=18)
        ax.set_ylabel("Metrics", fontsize=18)
        ax.tick_params(axis="both", labelsize=16)
        ax.set_xlim([0.1, 0.9])
        ax.set_ylim([0, 1.001])
        ax.legend(loc="lower left", frameon=True, fontsize=16)

        if save_fp is not None:
            plt.savefig(save_fp, dpi=300, bbox_inches="tight")
        plt.show()


plot_metrics_vs_thresholds(
    d_metrics_list,
    vlines=[0.5, best_threshold, 0.670],
    label_with_max=True,
    save_fp=DATA_PATH / "ablation_runs/figures/evaluation_threshold_sweep.png",
)
