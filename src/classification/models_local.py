"""
Local equivalent of models.py.

Replaces GEE SMILE Random Forest with scikit-learn Random Forest.
Mirrors the classifier_factory() interface and hyperparameters from
Dietrich et al. (2025).

Gaza adaptation: sklearn instead of GEE SMILE RF.
"""

import pickle
from pathlib import Path

import pandas as pd
from sklearn.ensemble import RandomForestClassifier


def classifier_factory_local(
    model_name: str = "random_forest",
    seed: int = 0,
    verbose: int = 1,
    **kwargs,
) -> RandomForestClassifier:
    """
    Create a local sklearn classifier.

    Local equivalent of classifier_factory() in models.py.
    Mirrors the same hyperparameter names as GEE SMILE RF.

    Args:
        model_name (str): Currently only 'random_forest' supported.
        seed (int): Random seed. Defaults to 0.
        verbose (int): Verbosity. Defaults to 1.
        **kwargs: Hyperparameters matching GEE SMILE RF names:
            numberOfTrees, minLeafPopulation, maxNodes.

    Returns:
        RandomForestClassifier: Untrained sklearn classifier.
    """
    if model_name == "random_forest":
        clf = RandomForestClassifier(
            n_estimators=kwargs.get("numberOfTrees", 50),
            min_samples_leaf=kwargs.get("minLeafPopulation", 3),
            max_leaf_nodes=int(kwargs.get("maxNodes", 10000)),
            class_weight=kwargs.get("class_weight", None), # Line added after first Gaza results
            n_jobs=-1,
            random_state=seed,
            verbose=verbose,
        )
    else:
        raise NotImplementedError(f"Model {model_name} not implemented locally.")

    if verbose:
        print(f"Classifier {model_name} created.")
    return clf


def save_classifier_local(clf: RandomForestClassifier, fp: Path) -> None:
    """
    Save classifier to disk as pickle.

    Local equivalent of export_classifier() in models.py.

    Args:
        clf: Trained sklearn classifier.
        fp (Path): Output file path.
    """
    fp.parent.mkdir(exist_ok=True, parents=True)
    with open(fp, "wb") as f:
        pickle.dump(clf, f)
    print(f"Classifier saved to {fp}")


def load_classifier_local(fp: Path) -> RandomForestClassifier:
    """
    Load classifier from disk.

    Local equivalent of load_classifier() in models.py.

    Args:
        fp (Path): Path to pickle file.

    Returns:
        RandomForestClassifier: Loaded classifier.
    """
    assert fp.exists(), f"Classifier not found: {fp}"
    with open(fp, "rb") as f:
        clf = pickle.load(f)
    print(f"Classifier loaded from {fp}")
    return clf
