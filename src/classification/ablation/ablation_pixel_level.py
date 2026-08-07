"""
Pixel-level ablation study for Gaza damage mapping.

Runs all ablation variants using the pixel-level evaluation method from
evaluation.ipynb (Dietrich et al. 2025 methodology):
    1. Train RF variant
    2. Run full pixel inference - probability rasters
    3. Merge tiles - Gaza-wide GeoTIFFs
    4. Sample rasters at UNOSAT test points (3x3 window, max agg)
    5. Compute metrics at t=0.670 (90% precision target, re-verified 5 July)

OOB plots (n_trees, mtry) use sklearn oob_score_ - no inference needed.

Results saved incrementally to data/ablation_runs/pixel_level/results.json.
Fully resumable: skips any variant whose results are already in the JSON.

Usage:
    screen -S ablation
    python3 src/ablation_pixel_level.py 2>&1 | tee logs/ablation_pixel_level.log
    # Ctrl+A D to detach
"""

import json
import pickle
import warnings

import geopandas as gpd
import numpy as np
import pandas as pd
import sklearn.metrics as sk_metrics
import xarray as xr
from omegaconf import OmegaConf
from sklearn.ensemble import RandomForestClassifier
from tqdm.auto import tqdm

from src.classification.dataset_local import get_dataset_ready_local
from src.classification.metrics import get_metrics
from src.classification.utils import get_features_names
from src.constants import AOIS_TEST, DATA_PATH, PRE_PERIOD
from src.data.unosat import load_unosat_labels
from src.data.utils import read_fp_within_geo

# Paths

ABLATION_DIR = DATA_PATH / "ablation_runs" / "pixel_level"
PROB_RASTERS_BASE = DATA_PATH / "probability_rasters_ablation"
MERGED_RASTERS_BASE = DATA_PATH / "merged_probability_rasters_ablation"
FEATURE_RASTERS_DIR = DATA_PATH / "feature_rasters"
RESULTS_JSON = ABLATION_DIR / "results.json"

ABLATION_DIR.mkdir(parents=True, exist_ok=True)

# Orbits / feature config

ORBITS = [87, 94, 160]
ALL_REDUCERS = ["mean", "stdDev", "median", "min", "max", "skew", "kurtosis"]
THRESHOLD_TARGET = 0.670
WINDOW_AGG = "max"
SPATIAL_WINDOW = 3

# Results I/O


def load_results() -> dict:
    if RESULTS_JSON.exists():
        with open(RESULTS_JSON) as f:
            return json.load(f)
    return {}


def save_results(results: dict) -> None:
    with open(RESULTS_JSON, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved {RESULTS_JSON}")


# Training


def train_variant(
    variant_name: str,
    n_trees: int = 50,
    max_features: str | int = "sqrt",
    subset_bands: list | None = None,
    reducer_names: list | None = None,
    class_weight=None,
    force_recreate: bool = False,
) -> tuple:
    """
    Train RF variant and return (clf, feature_cols, cfg).
    Saves model to data/runs/{variant_name}/model.pkl.
    """
    run_dir = DATA_PATH / "runs" / variant_name
    fp_model = run_dir / "model.pkl"
    run_dir.mkdir(parents=True, exist_ok=True)

    if reducer_names is None:
        reducer_names = ALL_REDUCERS

    cfg = OmegaConf.create(
        dict(
            aggregation_method="mean",
            model_name="random_forest",
            model_kwargs=dict(numberOfTrees=n_trees, minLeafPopulation=3, maxNodes=1e4),
            data=dict(
                s1=dict(subset_bands=subset_bands),
                s2=None,
                aois_test=list(AOIS_TEST),
                damages_to_keep=[1, 2],
                extract_winds="1x1",
                time_periods=dict(pre=PRE_PERIOD, post="2months"),
                split_strategy="aoi",
            ),
            reducer_names=reducer_names,
            seed=0,
            local_folder=DATA_PATH / "runs",
            train_on_all_data=False,
        )
    )

    feature_cols = get_features_names(cfg)

    if fp_model.exists() and not force_recreate:
        print(f"  Loading existing model: {variant_name}")
        with open(fp_model, "rb") as f:
            clf = pickle.load(f)
        return clf, feature_cols, cfg

    print(f"  Training: {variant_name} (n_trees={n_trees}, features={len(feature_cols)})")
    df_train = get_dataset_ready_local(
        sat="s1",
        split="train",
        post_dates="2months",
        extract_wind="1x1",
        split_strategy="aoi",
    )

    # Filter to subset_bands features if needed
    if subset_bands:
        feature_cols = [c for c in feature_cols if any(b in c for b in subset_bands)]

    X = df_train[feature_cols].values
    y = df_train["label"].values

    clf = RandomForestClassifier(
        n_estimators=n_trees,
        max_features=max_features,
        min_samples_leaf=3,
        oob_score=True,
        n_jobs=4,
        random_state=0,
        class_weight=class_weight,
    )
    clf.fit(X, y)
    print(f"  OOB score: {clf.oob_score_:.4f}")

    with open(fp_model, "wb") as f:
        pickle.dump(clf, f)
    print(f"  Model saved � {fp_model}")

    return clf, feature_cols, cfg


# Inference


def classify_tile_variant(data, band_names, clf, feature_cols):
    """Classify one tile with variant feature set."""
    n_bands, H, W = data.shape
    band_index = {name: i for i, name in enumerate(band_names)}
    # Only use bands present in both tile and feature_cols
    available = [c for c in feature_cols if c in band_index]
    if not available:
        return np.full((H, W), np.nan, dtype=np.float32)
    order = [band_index[c] for c in available]
    data_sub = data[order]
    X = data_sub.reshape(len(order), -1).T
    valid = ~np.any(np.isnan(X), axis=1)
    prob_flat = np.full(H * W, np.nan, dtype=np.float32)
    if valid.any():
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            prob_flat[valid] = clf.predict_proba(X[valid])[:, 1]
    return prob_flat.reshape(H, W)


def run_inference_variant(
    variant_name: str,
    clf,
    feature_cols: list,
    force_recreate: bool = False,
) -> None:
    """Run pixel inference for all windows using variant model."""
    import rasterio

    prob_dir = PROB_RASTERS_BASE / variant_name
    windows = sorted(d.name for d in FEATURE_RASTERS_DIR.iterdir() if d.is_dir())
    print(f"  Inference: {len(windows)} windows {prob_dir}")

    for window_str in tqdm(windows, desc=f"Inference {variant_name}"):
        out_dir = prob_dir / window_str
        out_dir.mkdir(parents=True, exist_ok=True)

        tile_ids = set()
        for orbit in ORBITS:
            od = FEATURE_RASTERS_DIR / window_str / f"orbit{orbit}"
            if od.exists():
                tile_ids.update(fp.stem.replace("qk_", "") for fp in od.glob("qk_*.tif"))

        for qk_id in sorted(tile_ids):
            fp_out = out_dir / f"qk_{qk_id}.tif"
            if fp_out.exists() and not force_recreate:
                continue

            orbit_probs, ref_profile = [], None
            for orbit in ORBITS:
                fp = FEATURE_RASTERS_DIR / window_str / f"orbit{orbit}" / f"qk_{qk_id}.tif"
                if not fp.exists():
                    continue
                with rasterio.open(fp) as src:
                    data = src.read().astype(np.float32)
                    band_names = list(src.descriptions)
                    profile = src.profile.copy()
                if ref_profile is None:
                    ref_profile = profile
                prob = classify_tile_variant(data, band_names, clf, feature_cols)
                orbit_probs.append(prob)

            if not orbit_probs or ref_profile is None:
                continue

            stack = np.stack(orbit_probs, axis=0)
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore")
                prob_agg = np.nanmean(stack, axis=0)

            prob_uint8 = np.where(np.isnan(prob_agg), 0, prob_agg * 255).astype(np.uint8)
            out_profile = ref_profile.copy()
            out_profile.update(dtype=rasterio.uint8, count=1, nodata=0, compress="lzw")
            with rasterio.open(fp_out, "w", **out_profile) as dst:
                dst.write(prob_uint8[np.newaxis])


# Merge tiles


def merge_tiles_variant(variant_name: str, force_recreate: bool = False) -> list:
    """Merge quadkey tiles - Gaza-wide GeoTIFFs for this variant."""
    from osgeo import gdal

    prob_dir = PROB_RASTERS_BASE / variant_name
    merged_dir = MERGED_RASTERS_BASE / variant_name
    merged_dir.mkdir(parents=True, exist_ok=True)

    merged_fps = []
    windows = sorted(d.name for d in prob_dir.iterdir() if d.is_dir())
    for window_str in windows:
        fp_out = merged_dir / f"gaza_{window_str}.tif"
        if fp_out.exists() and not force_recreate:
            merged_fps.append(fp_out)
            continue
        tifs = sorted(str(fp) for fp in (prob_dir / window_str).glob("qk_*.tif"))
        if not tifs:
            continue
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            gdal.Warp(str(fp_out), tifs, format="GTiff")
        merged_fps.append(fp_out)

    print(f"  Merged {len(merged_fps)} windows {merged_dir}")
    return merged_fps


# Pixel sampling


def extract_with_window(point, raster, window=3, agg="max"):
    """Sample raster at point with spatial window. Unchanged from evaluation.ipynb."""
    if window == 1:
        return raster.sel(x=point.x, y=point.y, method="nearest").item()
    half = window // 2
    xi = int(np.argmin(np.abs(raster.x.values - point.x)))
    yi = int(np.argmin(np.abs(raster.y.values - point.y)))
    patch = raster.isel(
        x=slice(xi - half, xi + half + 1),
        y=slice(yi - half, yi + half + 1),
    )
    return patch.max().item() if agg == "max" else patch.mean().item()


def sample_rasters_at_unosat_points(
    variant_name: str,
    merged_fps: list,
    force_recreate: bool = False,
) -> gpd.GeoDataFrame:
    """
    Sample merged probability rasters at UNOSAT test point locations.
    Mirrors combine_all_unosat_points_with_preds_gaza() exactly.
    """
    aoi_preds_dir = ABLATION_DIR / variant_name / "aoi_preds"
    aoi_preds_dir.mkdir(parents=True, exist_ok=True)
    fp_out = (
        aoi_preds_dir / f"unosat_points_with_preds_window_{SPATIAL_WINDOW}_{WINDOW_AGG}.geojson"
    )

    if fp_out.exists() and not force_recreate:
        print(f"  Loading existing sampled points: {variant_name}")
        return gpd.read_file(fp_out)

    print(f"  Sampling rasters at UNOSAT points: {variant_name}")

    # Get post dates from merged filenames
    post_dates = []
    for fp in sorted(merged_fps):
        parts = fp.stem.replace("gaza_", "").split("_")
        post_dates.append((parts[1], parts[2]))  # (start, end)

    # Load all UNOSAT labels
    gdf_labels = load_unosat_labels(
        combine_epoch="first_severe",
        labels_to_keep=None,
    )[["geometry", "date", "aoi", "damage"]].copy()
    gdf_labels["date"] = pd.to_datetime(gdf_labels["date"])

    # Stack rasters per AOI and sample
    all_aois = sorted(gdf_labels["aoi"].unique())
    gdf_out = None

    for aoi in tqdm(all_aois, desc="  Sampling AOIs"):
        gdf_aoi = gdf_labels[gdf_labels["aoi"] == aoi].copy()
        from shapely.geometry import box

        bounds = box(*gdf_aoi.total_bounds)

        dates_var = xr.Variable("date", pd.to_datetime([p[0] for p in post_dates]))
        preds = xr.concat(
            [read_fp_within_geo(fp, bounds) for fp in sorted(merged_fps)],
            dim=dates_var,
        ).squeeze()

        for start, _ in post_dates:
            date = start
            gdf_aoi[f"pred_{date}"] = gdf_aoi.geometry.apply(
                lambda pt: _safe_extract(pt, preds.sel(date=date))
            )

        gdf_out = pd.concat([gdf_out, gdf_aoi]) if gdf_out is not None else gdf_aoi

    gdf_out.fillna(0, inplace=True)
    gdf_out.to_file(fp_out, driver="GeoJSON")
    print(f"  Saved {len(gdf_out):,} points {fp_out}")
    return gpd.read_file(fp_out)


def _safe_extract(pt, raster):
    try:
        return extract_with_window(pt, raster, SPATIAL_WINDOW, WINDOW_AGG)
    except Exception:
        return None


# Evaluation


def evaluate_variant(gdf_points: gpd.GeoDataFrame, threshold: float = THRESHOLD_TARGET) -> dict:
    """
    Compute F1, precision, recall, AUC at given threshold.
    Uses get_metrics() from metrics.py.
    """
    gdf_test = gdf_points[gdf_points.aoi.isin(AOIS_TEST)].copy()
    gdf_test = gdf_test[gdf_test.damage.isin([1, 2])].copy()
    gdf_test["date"] = pd.to_datetime(gdf_test["date"])

    m, y_preds, y_trues = get_metrics(
        gdf_test,
        threshold=threshold,
        method="date-wise",
        print_classification_report=False,
        only_2022_for_pos=False,
        pos_year="2023",
        return_preds=True,
    )
    auc = sk_metrics.roc_auc_score(y_trues, y_preds)
    rep = sk_metrics.classification_report(
        y_trues,
        y_preds,
        labels=[0, 1],
        target_names=["Undamaged", "Damaged"],
        output_dict=True,
        zero_division=0,
    )
    return {
        "f1": rep["Damaged"]["f1-score"],
        "precision": rep["Damaged"]["precision"],
        "recall": rep["Damaged"]["recall"],
        "accuracy": rep["accuracy"],
        "auc": auc,
        "threshold": threshold,
    }


# Full variant pipeline


def run_variant_full(
    variant_name: str,
    n_trees: int = 50,
    max_features="sqrt",
    subset_bands=None,
    reducer_names=None,
    results: dict = None,
    force_recreate: bool = False,
) -> dict:
    """Train infer merge sample evaluate one variant."""
    if results and variant_name in results:
        print(f"  Skipping {variant_name} (already in results)")
        return results[variant_name]

    print(f"\n{'='*60}")
    print(f"VARIANT: {variant_name}")
    print(f"{'='*60}")

    import sys

    clf, feature_cols, cfg = train_variant(
        variant_name,
        n_trees,
        max_features,
        subset_bands,
        reducer_names,
        force_recreate=force_recreate,
    )
    print(f"  [checkpoint] train done")
    sys.stdout.flush()
    run_inference_variant(variant_name, clf, feature_cols, force_recreate)
    print(f"  [checkpoint] inference done")
    sys.stdout.flush()
    del clf
    merged_fps = merge_tiles_variant(variant_name, force_recreate)
    print(f"  [checkpoint] merge done, {len(merged_fps)} windows")
    sys.stdout.flush()
    gdf_points = sample_rasters_at_unosat_points(variant_name, merged_fps, force_recreate)
    print(f"  [checkpoint] sampling done, {len(gdf_points)} points")
    sys.stdout.flush()
    metrics = evaluate_variant(gdf_points)
    print(f"  [checkpoint] evaluation done")
    sys.stdout.flush()

    print(
        f"  F1={metrics['f1']:.3f}  P={metrics['precision']:.3f}  "
        f"R={metrics['recall']:.3f}  AUC={metrics['auc']:.3f}"
    )
    return metrics


# OOB study (training-time only, no inference)


def run_oob_study(results: dict) -> dict:
    """OOB error vs n_trees and vs max_features. No inference needed."""
    if "oob_n_trees" in results:
        print("OOB studies already complete - skipping")
        return results

    print("\n" + "=" * 60)
    print("OOB STUDY")
    print("=" * 60)

    df_train = get_dataset_ready_local(
        sat="s1",
        split="train",
        post_dates="2months",
        extract_wind="1x1",
        split_strategy="aoi",
    )
    cfg_base = OmegaConf.create(
        dict(
            data=dict(
                s1=dict(subset_bands=None),
                s2=None,
                time_periods=dict(pre=PRE_PERIOD, post="2months"),
                extract_winds="1x1",
            ),
            reducer_names=ALL_REDUCERS,
        )
    )
    feature_cols = get_features_names(cfg_base)
    df_train = df_train.dropna(subset=feature_cols)
    df_train = df_train.sample(n=min(200_000, len(df_train)), random_state=0)
    print(f"  Subsampled to {len(df_train):,} rows for OOB study")
    X = df_train[feature_cols].values
    y = df_train["label"].values

    # OOB vs n_trees
    if "oob_n_trees" not in results:
        print("OOB vs n_trees...")
        n_trees_vals = [10, 25, 50, 75, 100, 200, 300]
        oob_scores = []
        for n in tqdm(n_trees_vals, desc="n_trees"):
            clf = RandomForestClassifier(
                n_estimators=n,
                max_features="sqrt",
                min_samples_leaf=3,
                oob_score=True,
                n_jobs=4,
                random_state=0,
            )
            clf.fit(X, y)
            oob_scores.append(1 - clf.oob_score_)  # OOB error
            print(f"  n_trees={n}: OOB error={1-clf.oob_score_:.4f}")
        results["oob_n_trees"] = {"n_trees": n_trees_vals, "oob_error": oob_scores}

    return results


# Main

if __name__ == "__main__":
    results = load_results()

    # 1. OOB studies (fast - training only)
    results = run_oob_study(results)
    save_results(results)

    # 2. F1 vs n_trees (pixel-level inference per variant, ~1hr each)
    n_trees_variants = [10, 25, 75, 100, 200, 300]  # 50 is baseline, skip if already run
    for n in n_trees_variants:
        vname = f"ablation_ntrees_{n}"
        m = run_variant_full(vname, n_trees=n, results=results)
        results[vname] = m
        save_results(results)
        import gc

        gc.collect()

    # 3. Band ablation (VV only, VH only - VV+VH is baseline)
    band_variants = {
        "ablation_bands_VV": ["VV"],
        "ablation_bands_VH": ["VH"],
    }
    for vname, bands in band_variants.items():
        m = run_variant_full(vname, subset_bands=bands, results=results)
        results[vname] = m
        save_results(results)
        import gc

        gc.collect()

    # 4. Feature subset ablation
    reducer_variants = {
        "ablation_reducers_mean_std": ["mean", "stdDev"],
        "ablation_reducers_mean_std_median": ["mean", "stdDev", "median"],
        "ablation_reducers_no_skew_kurt": ["mean", "stdDev", "median", "min", "max"],
    }
    for vname, reducers in reducer_variants.items():
        m = run_variant_full(vname, reducer_names=reducers, results=results)
        results[vname] = m
        save_results(results)
        import gc

        gc.collect()

    print("\n" + "=" * 60)
    print("ALL ABLATION VARIANTS COMPLETE")
    print("=" * 60)
    print(json.dumps(results, indent=2))
