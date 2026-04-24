"""
Local equivalent of dataset.py.
Replaces GEE asset loading with local parquet files on Forth.
Mirrors the interface of get_dataset_ready() exactly.
Gaza adaptation: features stored locally instead of GEE assets.
"""
import pandas as pd
from pathlib import Path
from src.constants import DATA_PATH

FEATURES_DIR = DATA_PATH / "features_ready"
EXTRACT_WIND = "1x1"


def get_dataset_ready_local(
    sat: str = "s1",
    split: str = "train",
    post_dates: str = "2months",
    extract_wind: str = "1x1",
    split_strategy: str = "aoi",
) -> pd.DataFrame:
    """
    Load feature DataFrame from local parquet.

    Local equivalent of get_dataset_ready() in dataset.py.
    Mirrors the same interface but reads from local parquet
    instead of GEE FeatureCollection assets.

    Args:
        sat (str): Satellite to use. Currently only 's1' supported.
        split (str): 'train' or 'test'.
        post_dates (str): Time period label. Currently only '2months' supported.
        extract_wind (str): Extraction window. Currently only '1x1' supported.
        split_strategy (str): 'aoi' (default), 'random_all', or 'random_per_aoi'.

    Returns:
        pd.DataFrame: Feature DataFrame with label column.
    """
    assert sat == "s1", "Only s1 supported for Gaza local pipeline."
    assert post_dates == "2months", "Only 2months supported for Gaza local pipeline."
    assert split_strategy in ("aoi", "random_all", "random_per_aoi"), \
        f"Unknown split_strategy: {split_strategy}"

    suffix = "" if split_strategy == "aoi" else f"_{split_strategy}"
    fp = FEATURES_DIR / f"{sat}_{extract_wind}_{post_dates}_{split}{suffix}.parquet"
    
    assert fp.exists(), (
        f"Features not found: {fp}. "
        f"Run src/data/sentinel1/extract_features_local.py first."
    )
    df = pd.read_parquet(fp)
    print(f"  Loaded {split} set ({split_strategy}): {len(df):,} rows")
    return df
