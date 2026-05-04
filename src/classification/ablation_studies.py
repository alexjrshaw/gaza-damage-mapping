"""
Ablation studies for Gaza damage mapping pipeline.

Replicates and extends Dietrich et al. (2025) Supplementary Notes 2 and 6:

    Supp. Note 2: Threshold sweep — precision/recall/F1 vs threshold
    Supp. Note 6: Ablation studies:
        - Number of trees (with OOB error curve)
        - mtry / max_features (with OOB error curve) [extends Dietrich et al.]
        - Input bands: VV only, VH only, VV+VH
        - Feature subsets: mean+std, +median, +min/max, all 7
        - Extraction window: 1x1 vs 3x3

All results saved to data/ablation_runs/ablation_results.json
Plots saved to data/ablation_runs/figures/

Usage:
    python3 src/classification/ablation_studies.py
"""

import json
import pickle
import warnings
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from omegaconf import OmegaConf
from sklearn.ensemble import RandomForestClassifier

from src.classification.dataset_local import get_dataset_ready_local
from src.classification.main_local import _format_predictions, full_pipeline_local
from src.classification.metrics import get_metrics
from src.classification.utils import get_features_names
from src.constants import DATA_PATH, PRE_PERIOD, AOIS_TEST

RUNS_DIR    = DATA_PATH / "runs"
ABLATION_DIR = DATA_PATH / "ablation_runs"
FIGURES_DIR  = ABLATION_DIR / "figures"
ABLATION_DIR.mkdir(exist_ok=True, parents=True)
FIGURES_DIR.mkdir(exist_ok=True, parents=True)

BASELINE_RUN = "rf_s1_2months_50trees_1x1_all7reducers_baseline"
ALL_REDUCERS = ["mean", "stdDev", "median", "min", "max", "skew", "kurtosis"]


# ==================== HELPERS ====================

def load_baseline_gdf() -> gpd.GeoDataFrame:
    """Load baseline predictions GeoDataFrame."""
    fp = RUNS_DIR / BASELINE_RUN / f"{BASELINE_RUN}.geojson"
    return gpd.read_file(fp).set_index(["unosat_id", "aoi"])


def make_cfg(
    reducer_names=None,
    n_trees=50,
    extract_winds="1x1",
    subset_bands=None,
    class_weight=None,
):
    """Create config dict for a given ablation setting."""
    if reducer_names is None:
        reducer_names = ALL_REDUCERS
    return OmegaConf.create(dict(
        aggregation_method="mean",
        model_name="random_forest",
        model_kwargs=dict(
            numberOfTrees=n_trees,
            minLeafPopulation=3,
            maxNodes=1e4,
            class_weight=class_weight,
        ),
        data=dict(
            s1=dict(subset_bands=subset_bands),
            s2=None,
            aois_test=AOIS_TEST,
            damages_to_keep=[1, 2],
            extract_winds=extract_winds,
            time_periods=dict(pre=PRE_PERIOD, post="2months"),
            split_strategy="aoi",
        ),
        reducer_names=reducer_names,
        seed=0,
        local_folder=RUNS_DIR,
        train_on_all_data=False,
    ))


def load_features(split_strategy="aoi", extract_winds="1x1"):
    """Load train and test feature DataFrames."""
    suffix = "" if split_strategy == "aoi" else f"_{split_strategy}"
    train = get_dataset_ready_local(
        sat="s1", split="train", post_dates="2months",
        extract_wind=extract_winds, split_strategy=split_strategy,
    )
    test = get_dataset_ready_local(
        sat="s1", split="test", post_dates="2months",
        extract_wind=extract_winds, split_strategy=split_strategy,
    )
    return train, test


def train_and_evaluate(cfg, df_train, df_test) -> dict:
    """Train RF with given config and return metrics at t=0.5 and t=0.655."""
    feature_cols = get_features_names(cfg)

    df_train = df_train.dropna(subset=feature_cols)
    df_test  = df_test.dropna(subset=feature_cols)

    clf = RandomForestClassifier(
        n_estimators=int(cfg.model_kwargs.numberOfTrees),
        min_samples_leaf=int(cfg.model_kwargs.minLeafPopulation),
        max_leaf_nodes=int(cfg.model_kwargs.maxNodes),
        n_jobs=-1,
        random_state=cfg.seed,
    )
    clf.fit(df_train[feature_cols].values, df_train["label"].values)

    y_prob = clf.predict_proba(df_test[feature_cols].values)[:, 1]
    df_test = df_test.copy()
    df_test["prob"] = y_prob
    gdf = _format_predictions(df_test, cfg)

    m05  = get_metrics(gdf, threshold=0.5,   method="date-wise", print_classification_report=False)
    m655 = get_metrics(gdf, threshold=0.655, method="date-wise", print_classification_report=False)

    return {"t0.5": m05, "t0.655": m655}


# ==================== STUDY 1: THRESHOLD SWEEP ====================

def threshold_sweep(gdf: gpd.GeoDataFrame) -> dict:
    """
    Replicate Dietrich et al. Supplementary Note 2 Fig. S1.
    Sweep thresholds from 0.1 to 0.95 and record F1, precision, recall, AUC.
    """
    print("\n=== Study 1: Threshold sweep ===")
    thresholds = np.arange(0.1, 0.96, 0.025)
    results = []

    for t in thresholds:
        m = get_metrics(gdf, threshold=float(t), method="date-wise",
                       print_classification_report=False)
        results.append({"threshold": float(t), **m})
        print(f"  t={t:.3f}: F1={m['f1']:.3f}, Prec={m['precision']:.3f}, "
              f"Rec={m['recall']:.3f}, AUC={m['roc_auc']:.3f}")

    # Find threshold closest to 90% precision
    df = pd.DataFrame(results)
    above_90 = df[df["precision"] >= 0.90]
    if len(above_90) > 0:
        optimal = above_90.loc[above_90["f1"].idxmax()]
        print(f"\n  Optimal threshold (precision>=90%): {optimal['threshold']:.3f} "
              f"(F1={optimal['f1']:.3f}, Prec={optimal['precision']:.3f}, "
              f"Rec={optimal['recall']:.3f})")

    # Plot — mirrors Fig. S1
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(df["threshold"], df["f1"],        label="F1-score",  color="blue")
    ax.plot(df["threshold"], df["precision"],  label="precision", color="green")
    ax.plot(df["threshold"], df["recall"],     label="recall",    color="red")
    ax.plot(df["threshold"], df["roc_auc"],    label="roc_auc",   color="purple")
    ax.plot(df["threshold"], df["accuracy"],   label="accuracy",  color="orange")
    ax.axvline(0.5,   color="grey", linestyle="--", linewidth=1)
    ax.axvline(0.655, color="grey", linestyle="--", linewidth=1)
    ax.set_xlabel("Threshold")
    ax.set_ylabel("Metrics")
    ax.set_title("Performance of the model for different thresholds (Gaza)")
    ax.legend()
    ax.set_xlim(0.1, 0.95)
    ax.set_ylim(0, 1.0)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "threshold_sweep.png", dpi=150)
    plt.close()
    print(f"  Saved: threshold_sweep.png")

    return results


# ==================== STUDY 2: OOB ERROR VS N_TREES ====================

def oob_vs_n_trees(df_train: pd.DataFrame, feature_cols: list) -> dict:
    """
    OOB error vs number of trees — justifies choice of 50 trees.
    Uses warm_start to add trees incrementally.
    Extends Dietrich et al. who tested 10/25/50/75/100 trees.
    """
    print("\n=== Study 2: OOB error vs n_trees ===")

    X = df_train[feature_cols].dropna().values
    # Re-filter labels to match dropped rows
    y = df_train.dropna(subset=feature_cols)["label"].values

    tree_counts = list(range(1, 151, 5))  # 1 to 150 in steps of 5
    oob_errors = []

    clf = RandomForestClassifier(
        n_estimators=1,
        min_samples_leaf=3,
        max_leaf_nodes=10000,
        oob_score=True,
        warm_start=True,
        n_jobs=-1,
        random_state=0,
    )

    for n in tree_counts:
        clf.set_params(n_estimators=n)
        clf.fit(X, y)
        oob_errors.append(1 - clf.oob_score_)
        print(f"  n_trees={n:3d}: OOB error={oob_errors[-1]:.4f}")

    # Detect optimal n_trees - first point where marginal improvement < epsilon
    epsilon = 0.0005
    optimal_n = tree_counts[0]
    for i in range(1, len(oob_errors)):
        improvement = oob_errors[i-1] - oob_errors[i]
        if improvement < epsilon:
            optimal_n = tree_counts[i-1]
            break
    else:
        optimal_n = tree_counts[-1]

    print(f"\n  Optimal n_trees (elbow): {optimal_n}")
    print(f"  OOB error at optimal: {oob_errors[tree_counts.index(optimal_n)]:.4f}")
    print(f"  OOB error at 50 trees: {oob_errors[tree_counts.index(50)]:.4f}")

    # Plot
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(tree_counts, oob_errors, color="blue", linewidth=2)
    ax.axvline(50, color="red", linestyle="--", linewidth=1, label="Dietrich et al. (50 trees)")
    ax.axvline(optimal_n, color="green", linestyle=":", linewidth=1.5,
           label=f"Auto-detected optimum ({optimal_n} trees)")
    ax.set_xlabel("Number of trees")
    ax.set_ylabel("OOB error rate")
    ax.set_title("OOB error vs number of trees (Gaza)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "oob_vs_n_trees.png", dpi=150)
    plt.close()
    print(f"  Saved: oob_vs_n_trees.png")

    return {"tree_counts": tree_counts, "oob_errors": oob_errors, "optimal_n": optimal_n}


# ==================== STUDY 3: OOB ERROR VS MTRY ====================

def oob_vs_mtry(df_train: pd.DataFrame, feature_cols: list) -> dict:
    """
    OOB error vs max_features (mtry) — extends Dietrich et al.
    Tests sqrt(p), log2(p), and fractions of total features.
    """
    print("\n=== Study 3: OOB error vs mtry (max_features) ===")

    X = df_train[feature_cols].dropna().values
    y = df_train.dropna(subset=feature_cols)["label"].values
    n_features = X.shape[1]

    mtry_values = sorted(set([
        1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 14,
        int(np.sqrt(n_features)),   # default sqrt(p) = 5
        int(np.log2(n_features)),   # log2(p) = 4
        n_features // 3,
        n_features // 2,
        n_features,
    ]))

    oob_errors = []
    for mtry in mtry_values:
        clf = RandomForestClassifier(
            n_estimators=50,
            min_samples_leaf=3,
            max_leaf_nodes=10000,
            max_features=mtry,
            oob_score=True,
            n_jobs=-1,
            random_state=0,
        )
        clf.fit(X, y)
        oob_errors.append(1 - clf.oob_score_)
        print(f"  mtry={mtry:3d}: OOB error={oob_errors[-1]:.4f}")

    sqrt_p = int(np.sqrt(n_features))
    # Plot
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(mtry_values, oob_errors, color="blue", linewidth=2, marker="o")
    ax.axvline(sqrt_p, color="red", linestyle="--", linewidth=1,
               label=f"sqrt(p)={sqrt_p} (sklearn default)")
    ax.set_xlabel("mtry (max_features)")
    ax.set_ylabel("OOB error rate")
    ax.set_title("OOB error vs mtry parameter (Gaza)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "oob_vs_mtry.png", dpi=150)
    plt.close()
    print(f"  Saved: oob_vs_mtry.png")

    return {"mtry_values": mtry_values, "oob_errors": oob_errors}


# ==================== STUDY 4: INPUT BANDS ====================

def ablation_bands(df_train, df_test) -> dict:
    """VV only vs VH only vs VV+VH — mirrors Dietrich et al. Supp. Note 6."""
    print("\n=== Study 4: Input bands ===")
    results = {}

    for bands, label in [
        (["VV"], "VV only"),
        (["VH"], "VH only"),
        (None,   "VV+VH (baseline)"),
    ]:
        cfg = make_cfg(subset_bands=bands)
        m = train_and_evaluate(cfg, df_train, df_test)
        results[label] = m
        print(f"  {label}: F1@0.5={m['t0.5']['f1']:.3f}, "
              f"F1@0.655={m['t0.655']['f1']:.3f}")

    return results


# ==================== STUDY 5: FEATURE SUBSETS ====================

def ablation_features(df_train, df_test) -> dict:
    """
    Feature subsets — mirrors Dietrich et al. Supp. Note 6.
    Tests progressive addition of reducers.
    """
    print("\n=== Study 5: Feature subsets ===")
    results = {}

    configs = [
        (["mean", "stdDev"],                                        "mean+std"),
        (["mean", "stdDev", "median"],                              "+median"),
        (["mean", "stdDev", "median", "min", "max"],                "+min/max"),
        (["mean", "stdDev", "median", "min", "max", "skew"],        "+skew"),
        (["mean", "stdDev", "median", "min", "max", "skew", "kurtosis"], "all 7 (baseline)"),
    ]

    for reducers, label in configs:
        cfg = make_cfg(reducer_names=reducers)
        m = train_and_evaluate(cfg, df_train, df_test)
        results[label] = m
        print(f"  {label}: F1@0.5={m['t0.5']['f1']:.3f}, "
              f"F1@0.655={m['t0.655']['f1']:.3f}")

    return results


# ==================== STUDY 6: NUMBER OF TREES (F1) ====================

def ablation_n_trees(df_train, df_test) -> dict:
    """
    F1 vs number of trees — directly mirrors Dietrich et al. Supp. Note 6.
    Tests 10, 25, 50, 75, 100 trees.
    """
    print("\n=== Study 6: Number of trees (F1) ===")
    results = {}

    feature_cols = get_features_names(make_cfg())

    for n_trees in [10, 25, 50, 75, 100]:
        cfg = make_cfg(n_trees=n_trees)
        m = train_and_evaluate(cfg, df_train, df_test)
        results[str(n_trees)] = m
        print(f"  n_trees={n_trees}: F1@0.5={m['t0.5']['f1']:.3f}, "
              f"F1@0.655={m['t0.655']['f1']:.3f}")

    return results


# ==================== PLOT ABLATION SUMMARY (Fig. S4 equivalent) ====================

def plot_ablation_summary(results: dict) -> None:
    """
    Bar chart of F1 scores across ablation settings — mirrors Fig. S4.
    """
    labels, f1_05, f1_655 = [], [], []

    for study, study_results in results.items():
        if study in ("threshold_sweep", "oob_n_trees", "oob_mtry"):
            continue
        for setting, metrics in study_results.items():
            labels.append(f"{setting}")
            f1_05.append(metrics["t0.5"]["f1"])
            f1_655.append(metrics["t0.655"]["f1"])

    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.bar(x - 0.2, f1_05,  0.4, label="t=0.5",   color="steelblue")
    ax.bar(x + 0.2, f1_655, 0.4, label="t=0.655",  color="orange")

    # Baseline line
    baseline_f1 = results["bands"]["VV+VH (baseline)"]["t0.5"]["f1"]
    ax.axhline(baseline_f1, color="red", linestyle="--", linewidth=1,
               label=f"Baseline F1={baseline_f1:.3f}")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=9)
    ax.set_ylabel("F1-score")
    ax.set_title("Ablation study results (Gaza) — mirrors Dietrich et al. Fig. S4")
    ax.legend()
    ax.set_ylim(0, 1.0)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "ablation_summary.png", dpi=150)
    plt.close()
    print(f"\nSaved: ablation_summary.png")

# ==================== STUDY 7: PIXEL-LEVEL THRESHOLD (Dietrich et al. method) ====================

def pixel_level_threshold_sweep() -> dict:
    """
    Find optimal threshold using pixel-level raster predictions vs UNOSAT labels.
    Mirrors Dietrich et al. evaluation.ipynb exactly.
    Uses window=3 (3x3 pixel spatial window), agg="max".
    """
    print("\n=== Study 7: Pixel-level threshold sweep (Dietrich et al. method) ===")

    from collections import defaultdict
    import geopandas as gpd
    from src.constants import AOIS_TEST

    # Load pre-computed UNOSAT points with pixel predictions
    fp = DATA_PATH / "pixel_postprocessing/aoi_preds/unosat_points_with_preds_window_3_max.geojson"
    if not fp.exists():
        print(f"  File not found: {fp}")
        print("  Run evaluation.ipynb first to generate UNOSAT points with pixel predictions.")
        return {}

    gdf_points = gpd.read_file(fp)
    gdf_points["date"] = pd.to_datetime(gdf_points["date"])
    gdf_test = gdf_points[gdf_points["aoi"].isin(AOIS_TEST)]
    gdf_test = gdf_test[gdf_test["damage"].isin([1, 2])]

    print(f"  Test points: {len(gdf_test):,}")

    thresholds = np.arange(0.1, 0.95, 0.005)
    d_metrics_list = defaultdict(list)

    for t in thresholds:
        m = get_metrics(
            gdf_test,
            threshold=t,
            method="date-wise",
            print_classification_report=False,
            only_2022_for_pos=False,
            pos_year="2023",
            return_preds=False,
        )
        for k, v in m.items():
            d_metrics_list[k].append(v)

    # Find optimal threshold at precision=0.9
    diff = np.array(d_metrics_list["precision"]) - 0.9
    idx_min = np.abs(diff).argmin()
    optimal_t = thresholds[idx_min] if diff[idx_min] > 0 else thresholds[min(idx_min + 1, len(thresholds)-1)]
    print(f"\n  Optimal threshold (pixel-level, precision>=90%): {optimal_t:.3f}")
    print(f"  F1={d_metrics_list['f1'][idx_min]:.3f}, "
          f"Precision={d_metrics_list['precision'][idx_min]:.3f}, "
          f"Recall={d_metrics_list['recall'][idx_min]:.3f}")
    print(f"  Dietrich et al. (Ukraine): 0.655")
    print(f"  Point-level optimal (Gaza): 0.600")
    print(f"  Pixel-level optimal (Gaza): {optimal_t:.3f}")

    # Plot — mirrors evaluation.ipynb plot
    fig, ax = plt.subplots(figsize=(12, 6))
    colors = {"f1": "blue", "precision": "green", "recall": "red",
              "roc_auc": "purple", "accuracy": "orange"}
    for metric, scores in d_metrics_list.items():
        ax.plot(thresholds, scores, label=metric, linewidth=2.5,
                color=colors.get(metric, "grey"))
    ax.axvline(0.5,       color="black", linestyle="--", alpha=0.5)
    ax.axvline(0.655,     color="grey",  linestyle=":",  alpha=0.7, label="Ukraine t=0.655")
    ax.axvline(optimal_t, color="red",   linestyle="--", alpha=0.7,
               label=f"Gaza pixel optimal t={optimal_t:.3f}")
    ax.set_xlabel("Threshold", fontsize=14)
    ax.set_ylabel("Metrics", fontsize=14)
    ax.set_title("Pixel-level threshold sweep — Gaza vs UNOSAT labels")
    ax.set_xlim([0.1, 0.9])
    ax.set_ylim([0, 1.001])
    ax.legend(loc="lower left", fontsize=12)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "pixel_threshold_sweep.png", dpi=150)
    plt.close()
    print(f"  Saved: pixel_threshold_sweep.png")

    return {"thresholds": list(thresholds), "metrics": dict(d_metrics_list),
            "optimal_threshold": float(optimal_t)}

# ==================== MAIN ====================

if __name__ == "__main__":
    print("Loading data...")
    df_train, df_test = load_features()
    gdf_baseline = load_baseline_gdf()
    feature_cols = get_features_names(make_cfg())

    all_results = {}

    # Study 1: Threshold sweep
    all_results["threshold_sweep"] = threshold_sweep(gdf_baseline)

    # Study 2: OOB vs n_trees
    all_results["oob_n_trees"] = oob_vs_n_trees(df_train, feature_cols)

    # Study 3: OOB vs mtry
    all_results["oob_mtry"] = oob_vs_mtry(df_train, feature_cols)

    # Study 4: Input bands
    all_results["bands"] = ablation_bands(df_train, df_test)

    # Study 5: Feature subsets
    all_results["features"] = ablation_features(df_train, df_test)

    # Study 6: Number of trees (F1)
    all_results["n_trees"] = ablation_n_trees(df_train, df_test)

    # Plot summary
    plot_ablation_summary(all_results)

    # Study 7: Pixel-level threshold sweep
    all_results["pixel_threshold"] = pixel_level_threshold_sweep()

    # Save all results
    fp_out = ABLATION_DIR / "ablation_results.json"
    with open(fp_out, "w") as f:
        json.dump(all_results, f, indent=2, default=float)
    print(f"\nAll results saved to {fp_out}")
    print(f"Figures saved to {FIGURES_DIR}")
