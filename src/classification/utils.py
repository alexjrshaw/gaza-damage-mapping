from omegaconf import DictConfig
from src.constants import S1_BANDS


def get_run_name(cfg: DictConfig) -> str:
    """
    Generate a run name based on the configuration.
    Ex: rf_s1_only_VV_2months_50trees_1x1_seed0_all7reducers
    Args:
        cfg (DictConfig): The configuration.
    Returns:
        str: The run name.
    """
    clf_name = cfg.model_name
    clf_name = "".join([n[0] for n in clf_name.split("_")]) if "_" in clf_name else clf_name
    name = f"{clf_name}_"
    if cfg.data.s1 is not None:
        name += "s1_"
        if cfg.data.s1.subset_bands:
            name += "only_" + "_".join(cfg.data.s1.subset_bands) + "_"
    name += cfg.data.time_periods["post"]
    if "numberOfTrees" in cfg.model_kwargs:
        name += f"_{cfg.model_kwargs['numberOfTrees']}trees"
    if isinstance(cfg.data.extract_winds, str):
        wind = cfg.data.extract_winds
    else:
        wind = "_".join(cfg.data.extract_winds)
    if wind != "3x3" and cfg.data.s1 is not None:
        name += f"_{wind}"
    if cfg.seed != 0:
        name += f"_seed{cfg.seed}"
    if cfg.reducer_names != ["mean", "stdDev", "median", "min", "max", "skew", "kurtosis"]:
        name += f"_reducer{'_'.join(cfg.reducer_names)}"
    else:
        name += "_all7reducers"
    if cfg.train_on_all_data:
        name += "_all_data"
    if cfg.get("run_suffix", None):
        name += f"_{cfg.run_suffix}"
    return name


def get_features_names(cfg: DictConfig) -> list:
    """Get feature names for Classifier from the config. Only S1 is used in this study."""
    if cfg.data.s1 is not None:
        bands = cfg.data.s1.subset_bands if cfg.data.s1.subset_bands else S1_BANDS
    else:
        bands = []
    assert len(bands) > 0, "No bands selected"
    names = cfg.data.time_periods.keys()
    winds = cfg.data.extract_winds
    winds = [winds] if isinstance(winds, str) else winds
    reducers = cfg.reducer_names
    return [f"{b}_{n}_{w}_{r}" for b in bands for n in names for w in winds for r in reducers]


def get_sat_from_cfg(cfg: DictConfig) -> str:
    """Get the satellite name from the config. Only S1 is used in this study."""
    return "s1" if cfg.data.s1 is not None else ""
