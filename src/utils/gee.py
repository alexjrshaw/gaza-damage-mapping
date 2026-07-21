import ee

from src.classification.reducers import get_reducers


def init_gee(project: str = "gaza-damage-mapping") -> None:
    """
    Initialize GEE. Works also when working through ssh

    Args:
        project (str, optional): Name of the project. Defaults to "gaza-damage-mapping".
    """
    ee.Initialize(project=project)


def fc_to_list(fc: ee.FeatureCollection) -> ee.List:
    """Transforms a feature collection to a list."""
    return fc.toList(fc.size())


def fill_nan_with_mean(col: ee.FeatureCollection) -> ee.FeatureCollection:
    """Fill NaN values with the mean of the column."""
    col_mean = col.reduce(ee.Reducer.mean())

    def _fill_nan_with_mean(img):
        mask = img.mask().Not()
        filled_img = img.unmask().add(col_mean.multiply(mask))
        filled_img = filled_img.copyProperties(img, img.propertyNames())
        return filled_img

    return col.map(_fill_nan_with_mean)


# Asset management
def asset_exists(asset_id: str) -> bool:
    """Check if an asset exists."""
    try:
        ee.data.getAsset(asset_id)
        return True
    except ee.ee_exception.EEException:
        return False


def delete_asset(asset_id: str) -> bool:
    """Delete an asset."""
    try:
        ee.data.deleteAsset(asset_id)
        print(f"{asset_id} deleted")
        return True
    except ee.ee_exception.EEException:
        return False


def rename_asset(original_path: str, new_path: str) -> None:
    """Rename an asset."""
    try:
        ee.data.renameAsset(original_path, new_path)
        print(f"Asset renamed from {original_path} to {new_path}")
    except Exception as e:
        print(f"Error renaming asset: {e}")


def create_folder(folder_path: str, verbose: int = 1) -> None:
    """Create a folder in GEE."""
    try:
        ee.data.createAsset({"type": "FOLDER"}, folder_path)
        if verbose:
            print(f"Folder created at {folder_path}")
    except Exception as e:
        if verbose:
            print(f"Error creating folder: {e}")


def list_assets(folder_path: str, print_list: bool = False) -> list[str]:
    """List all assets in a folder."""
    try:
        asset_list = [a["id"] for a in ee.data.getList({"id": folder_path})]
        if print_list:
            print(f"Assets in {folder_path}: {asset_list}")
        return asset_list
    except Exception as e:
        print(f"Error listing assets: {e}")


def create_folders_recursively(full_path: str, last_one_is_asset: bool = False):
    """Create folders recursively."""

    if last_one_is_asset:
        # ignore asset_id
        full_path = "/".join(full_path.split("/")[:-1])

    folders_to_create = []
    current_path = full_path

    # Traverse up until we find an existing folder
    while not asset_exists(current_path):
        folders_to_create.append(current_path)
        current_path = "/".join(current_path.split("/")[:-1])
        # Stop if there's no more parent (root reached)
        assert current_path != "projects", "Problem, we should never reach the root !"

    if not folders_to_create:
        return

    # Create the folders from top to bottom
    for folder in reversed(folders_to_create):
        create_folder(folder)


# Functions moved from src/inference/dense_inference.py (now deleted). See git history for original.


def col_to_features(
    col: ee.ImageCollection,
    reducer_names: list[str],
    time_periods: dict[str, tuple[str, str]],
    extract_window: str,
) -> ee.Image:
    """
    Convert an ImageCollection to a single ee.Image where each band is a feature.

    Args:
        col (ee.ImageCollection): The collection to convert.
        reducer_names (list[str]): The reducer names.
        time_periods (dict[str, tuple[str, str]]): The time periods. ({pre: (start, end), post: (start, end)})
        extract_window (str): The window to extract, eg 1x1.

    Returns:
        ee.Image: The image with all features.
    """
    s1_features = None

    reducer_names = list(reducer_names)  # GEE does not like ListConfig
    reducer = get_reducers(reducer_names)
    original_col_names = [f"{b}_{r}" for b in ["VV", "VH"] for r in reducer_names]

    if int(extract_window[0]) > 1:
        # convolve (similar to looking at a larger window) with radius (eg 15m for 3x3 window)
        col = convolve_collection(col, 10 * int(extract_window[0]) // 2, "square", "meters")

    # Extract features for each time period
    for name_period, (start, end) in time_periods.items():
        s1_dates = col.filterDate(start, end)
        prefix = f"{name_period}_{extract_window}"

        # Reduce to features, and rename the bands
        _s1_features = s1_dates.reduce(reducer)
        _s1_features = _s1_features.select(
            original_col_names, get_new_names(original_col_names, prefix)
        )
        s1_features = _s1_features if s1_features is None else s1_features.addBands(_s1_features)

    return s1_features


def find_orbits(
    s1: ee.FeatureCollection,
    time_periods: dict[str, tuple[str, str]],
    min_number: int = 5,
) -> ee.List:
    """Find all orbits that appear at least min_number in each time period."""
    list_orbits = []
    for _, (start, end) in time_periods.items():
        s1_ = s1.filterDate(start, end)
        orbits_counts = s1_.aggregate_histogram("relativeOrbitNumber_start")
        # At least 5 images per orbit (two months of data)
        orbits_counts = orbits_counts.map(
            lambda k, v: ee.Algorithms.If(ee.Number(v).gte(min_number), k, None)
        )
        orbits_inference = orbits_counts.keys().map(
            lambda k: ee.Number.parse(k)
        )  # cast keys back to number
        list_orbits.append(orbits_inference)
    return list_orbits[0].filter(ee.Filter.inList("item", list_orbits[1]))


def convolve_collection(
    img_col: ee.ImageCollection,
    radius: int,
    kernel_type: str = "square",
    units: str = "meters",
) -> ee.ImageCollection:
    """Convolve each image in the collection with a focal mean of radius `radius`"""

    def _convolve_mean(img):
        return img.focalMean(radius, kernel_type, units=units).set(
            "system:time_start", img.get("system:time_start")
        )

    return img_col.map(_convolve_mean)


def get_new_names(bands: list[str], prefix: str) -> list[str]:
    """Add the prefix (pre or post and window) between the band and the reducer name."""
    new_bands = []
    for b in bands:
        b_, r = b.split("_")
        new_bands.append(f"{b_}_{prefix}_{r}")
    return new_bands
