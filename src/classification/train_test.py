import time

from omegaconf import OmegaConf
from sklearn.ensemble import RandomForestClassifier

from src.classification.dataset_local import get_dataset_ready_local
from src.classification.utils import get_features_names
from src.constants import PRE_PERIOD

cfg = OmegaConf.create(
    dict(
        data=dict(
            s1=dict(subset_bands=None), s2=None, time_periods=dict(pre=PRE_PERIOD, post="2months"), extract_winds="1x1"
        ),
        reducer_names=["mean", "stdDev", "median", "min", "max", "skew", "kurtosis"],
    )
)
feature_cols = get_features_names(cfg)
print("Loading data...", flush=True)
df = get_dataset_ready_local(sat="s1", split="train", post_dates="2months", extract_wind="1x1", split_strategy="aoi")
df = df.dropna(subset=feature_cols)
print(f"Rows: {len(df):,}", flush=True)
X = df[feature_cols].values
y = df["label"].values
for n_trees in [200, 300]:
    print(f"Training {n_trees} trees...", flush=True)
    t0 = time.time()
    clf = RandomForestClassifier(
        n_estimators=n_trees, max_features="sqrt", min_samples_leaf=3, oob_score=True, n_jobs=4, random_state=0
    )
    clf.fit(X, y)
    print(f"n_trees={n_trees}: Done in {time.time()-t0:.1f}s, OOB={clf.oob_score_:.4f}", flush=True)
