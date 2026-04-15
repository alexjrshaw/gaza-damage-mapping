import io
import zipfile

import geopandas as gpd
import osmnx as ox
import requests
from pyproj import Transformer
from shapely import Geometry
from shapely.geometry import MultiPolygon, Polygon
from shapely.ops import transform

from src.constants import DATA_PATH


def load_country_boundaries(country: str) -> tuple[Polygon, MultiPolygon]:
    """
    Load shapefile with country boundaries.

    If file does not exist, download it from OSM.

    Args:
        country (str): Name of the country (eg Palestine, Iraq, ...)

    Returns:
        Tuple[Polygon, MultiPolygon]: The boundaries
    """

    folder = DATA_PATH / "countries"
    folder.mkdir(exist_ok=True)

    fp = folder / f"{country}.shp"

    if not fp.exists():
        print(f"The file with {country} boundaries does not exist. Downloading it now...")
        gdf = ox.geocode_to_gdf(country)
        gdf[["geometry"]].to_file(fp)
        print("Done")

    return gpd.read_file(fp).iloc[0].geometry


def download_gaza_admin_boundaries() -> None:
    """
    Download Gaza/Palestine admin boundaries from OCHA HDX if not already present.

    Source: OCHA COD-AB Palestine admin boundaries (GeoJSON)
    URL: https://data.humdata.org/dataset/cod-ab-pse
    """
    folder = DATA_PATH / "raw"
    folder.mkdir(exist_ok=True)

    # Check if already downloaded
    if (folder / "pse_admin2.geojson").exists():
        return

    print("Downloading Palestine admin boundaries from OCHA HDX...")
    url = (
        "https://data.humdata.org/dataset/cod-ab-pse/resource/"
        "ca372385-4c79-4378-abf1-cb506fb98023/download/"
        "pse_admin_boundaries.geojson.zip"
    )
    r = requests.get(url)
    z = zipfile.ZipFile(io.BytesIO(r.content))
    z.extractall(folder)
    print("Done")


def load_gaza_admin_polygons(adm_level: int = 2) -> gpd.GeoDataFrame:
    """
    Load Gaza Strip admin polygons at the specified level.

    Auto-downloads from OCHA HDX if not already present.

    Args:
        adm_level (int): Admin level. 1=Gaza Strip boundary, 2=Governorates.
            Defaults to 2.

    Returns:
        gpd.GeoDataFrame: The admin polygons with admin_id index.
    """
    assert adm_level in [1, 2], "Only admin levels 1 and 2 are available for Gaza"

    # Auto-download if not present
    download_gaza_admin_boundaries()

    fp = DATA_PATH / f"raw/pse_admin{adm_level}.geojson"
    assert fp.exists(), f"Admin boundaries not found at {fp}"

    gdf = gpd.read_file(fp)

    # Filter to Gaza Strip only
    gdf = gdf[gdf["adm1_name"] == "Gaza Strip"].copy()

    if adm_level == 2:
        # Normalise spelling to match UNOSAT data
        gdf["adm2_name"] = gdf["adm2_name"].replace("Khan Younis", "Khan Yunis")

    gdf.index.name = "admin_id"
    gdf.reset_index(inplace=True)
    gdf["admin_id"] = gdf["admin_id"].apply(lambda x: f"{adm_level}_{x}")

    return gdf


def load_gaza_strip_boundary() -> Polygon:
    """
    Load the official Gaza Strip boundary from OCHA admin1 file.

    Auto-downloads from OCHA HDX if not already present.

    Returns:
        Polygon: The Gaza Strip boundary.
    """
    gdf = load_gaza_admin_polygons(adm_level=1)
    return gdf.iloc[0].geometry


def reproject_geo(geo: Geometry, current_crs: str, target_crs: str) -> Geometry:
    """Reprojects a Shapely geometry from the current CRS to a new CRS."""
    transformer = Transformer.from_crs(current_crs, target_crs, always_xy=True)
    return transform(transformer.transform, geo)


def get_best_utm_crs_from_gdf(gdf: gpd.GeoDataFrame) -> str:
    """Get the best UTM CRS for the given GeoDataFrame."""
    mean_lon = gdf.geometry.unary_union.centroid.x
    mean_lat = gdf.geometry.unary_union.centroid.y
    return get_best_utm_crs_from_lon_lat(mean_lon, mean_lat)


def get_best_utm_crs_from_lon_lat(lon: float, lat: float) -> str:
    """Get the best UTM CRS for the given lon and lat."""
    utm_zone = int(((lon + 180) / 6) % 60) + 1
    utm_crs = f"EPSG:326{utm_zone}" if lat > 0 else f"EPSG:327{utm_zone}"
    return utm_crs
