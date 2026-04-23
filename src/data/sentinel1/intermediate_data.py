import ee

from src.constants import ASSETS_PATH
from src.data.sentinel1.collection import get_s1_collection
from src.data.unosat import load_unosat_geo_gee, load_unosat_labels_gee
from src.utils.gee import asset_exists, create_folders_recursively, fill_nan_with_mean, init_gee

init_gee()

# Pre-war baseline start through end of conflict period studied
START_DATE = "2021-10-07" # One-year pre-conflict baseline
END_DATE = "2025-12-07"  # One day after last post window end


def create_fc_aoi_orbit(
    aoi: str,
    orbit: str,
    scale: int = 10,
    export: bool = True,
    ) -> ee.FeatureCollection:
    """
    Creates a feature collection with all Sentinel-1 band values
    for each date and each UNOSAT point (one row per point per image).

    Uses ee.Image.reduceRegions() instead of nested map for performance.
    This reduces GEE operations from ~3 million to ~91 for Gaza-scale data.

    Args:
        aoi (str): The area of interest (e.g. 'GAZ1').
        orbit (str): The Sentinel-1 relative orbit number.
        scale (int): Pixel scale in metres. Defaults to 10.
        export (bool): Whether to export to GEE asset. Defaults to True.

    Returns:
        ee.FeatureCollection: One feature per (point, image) combination.
    """
    extract = f"{scale // 10}x{scale // 10}"
    folder = ASSETS_PATH + f"intermediate_features/ts_s1_{extract}"
    asset_id = folder + f"/{aoi}_orbit{orbit}"
    if asset_exists(asset_id):
        print(f"Asset {asset_id} already exists.")
        return
    create_folders_recursively(folder)

    # Load UNOSAT labels (classes 1+2 only)
    labels = load_unosat_labels_gee(aoi, False)
    geo = load_unosat_geo_gee(aoi)

    # Load S1 collection for this AOI and orbit
    s1 = get_s1_collection(geo, START_DATE, END_DATE).filterMetadata(
        "relativeOrbitNumber_start", "equals", orbit
    )
    s1 = fill_nan_with_mean(s1)

    def extract_image(img):
        """Extract VV and VH at all points for one image using reduceRegions."""
        return img.select(["VV", "VH"]).reduceRegions(
            collection=labels,
            reducer=ee.Reducer.mean(),
            scale=scale,
        ).map(lambda f: f.set("system:time_start", img.get("system:time_start")))

    # Apply to all images and flatten
    fc_extracted = s1.map(extract_image).flatten()

    if export:
        ee.batch.Export.table.toAsset(
            collection=fc_extracted,
            description=f"{aoi}_orbit{orbit}_{scale}m",
            assetId=asset_id,
        ).start()
        print(f"Exporting {aoi}_orbit{orbit}_{scale}m")

    return fc_extracted


if __name__ == "__main__":
    from src.data.utils import aoi_orbit_iterator

    scale = 10
    for aoi, orbit in aoi_orbit_iterator():
        create_fc_aoi_orbit(aoi, orbit, scale=scale, export=True)
