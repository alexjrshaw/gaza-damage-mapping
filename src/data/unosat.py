import ee
import geopandas as gpd
import pandas as pd
from pathlib import Path
from shapely.geometry import Polygon

from src.constants import DATA_PATH, OLD_ASSETS_PATH, ASSETS_PATH

# ==================== LOCAL DATA ====================

def load_unosat_labels(
    aoi: str | list[str] | None = None,
    labels_to_keep: list[int] = [1, 2],
    combine_epoch: bool = "last",
) -> gpd.GeoDataFrame:
    """
    Load UNOSAT labels processed.

    Args:
        aoi (str | list[str] | None): Which AOIs to keep. Default to None (all)
        labels_to_keep (list[int]): Which labels to keep. Default to [1,2] (destroyed, major damage)
        combine_epoch (bool): For points that have multiple observations, we keep only one label.
            Either the 'last' one or the 'min' one (eg the strongest label). Default to 'last'

    Returns:
        gpd.GeoDataFrame: The GeoDataFrame with all UNOSAT labels
    """
    labels_fp = DATA_PATH / "unosat_labels.geojson"
    assert labels_fp.exists(), "The GeoDataFrame has not been created yet."

    gdf = gpd.read_file(labels_fp).set_index("unosat_id")

    if combine_epoch is not None:
        if combine_epoch == "last":
            # Only keep most recent epoch for each point
            gdf = gdf.loc[gdf.groupby(gdf.geometry.to_wkt())["ep"].idxmax()]
        elif combine_epoch == "min":
            # Only keep strongest label for each point
            gdf = gdf.loc[gdf.groupby(gdf.geometry.to_wkt())["damage"].idxmin()]
        else:
            raise ValueError("combine_epoch must be 'last' or 'min'")

    if labels_to_keep is not None:
        # Only keep some labels
        gdf = gdf[gdf.damage.isin(labels_to_keep)]

    if aoi is not None:
        # Only keep some AOIs
        aoi = [aoi] if isinstance(aoi, str) else aoi
        gdf = gdf[gdf.aoi.isin(aoi)]

    return gdf


def load_unosat_aois() -> gpd.GeoDataFrame:
    """
    Load GeoDataFrame with all AOIs from UNOSAT.

    Returns:
        gpd.GeoDataFrame: The GeoDataFrame with column 'aoi' and 'geometry'
    """
    aoi_fp = DATA_PATH / "unosat_aois.geojson"
    assert aoi_fp.exists(), "The GeoDataFrame has not been created yet."
    return gpd.read_file(aoi_fp)


def load_unosat_geo(aoi: str) -> Polygon:
    """
    Get the geometry of the given AOI.

    Args:
        aoi (str): The area of interest.

    Returns:
        Polygon: The geometry of the AOI
    """
    aois = load_unosat_aois().set_index("aoi")
    geo = aois.loc[aoi].geometry
    return geo


# ==================== GEE DATA ====================

def load_unosat_labels_gee(aoi: str, all_labels: bool = False) -> ee.FeatureCollection:
    """
    Get the UNOSAT labels for the given AOI.

    Args:
        aoi (str): The area of interest.
        all_labels (bool, optional): If True, return all labels. Otherwise only 1 and 2.
            Defaults to False.

    Returns:
        ee.FeatureCollection: The UNOSAT labels.
    """
    if all_labels:
        return ee.FeatureCollection(ASSETS_PATH + f"UNOSAT_labels/{aoi}_full")
    else:
        return ee.FeatureCollection(ASSETS_PATH + f"UNOSAT_labels/{aoi}")


def load_unosat_geo_gee(aoi: str) -> ee.FeatureCollection:
    """
    Get the AOI geometry in GEE.

    Args:
        aoi (str): The area of interest.

    Returns:
        ee.FeatureCollection: The AOI geometry.
    """
    return ee.FeatureCollection(ASSETS_PATH + f"AOIs/{aoi}").geometry()


# ==================== GAZA PREPROCESSING ====================

# Raw GDB downloaded from UNOSAT
GDB_PATH = DATA_PATH / "raw/UNOSAT_GazaStrip_CDA_11October2025.gdb"
LAYER_NAME = "Damage_Sites_GazaStrip_20251011"
N_EPOCHS = 14

# Map governorates to AOI IDs
GOVERNORATE_TO_AOI = {
    "North Gaza":    "GAZ1",
    "Gaza":          "GAZ2",
    "Deir Al-Balah": "GAZ3",
    "Khan Yunis":    "GAZ4",
    "Rafah":         "GAZ5",
}


def preprocess_gaza_unosat(
    gdb_path: Path = GDB_PATH,
    labels_to_keep: list[int] = [1, 2],
) -> None:
    """
    Convert raw UNOSAT Gaza GDB into the two GeoJSON files the pipeline expects:
      - data/unosat_labels.geojson  (one row per point per epoch)
      - data/unosat_aois.geojson    (one convex-hull polygon per governorate)

    The GDB uses wide format (one row per site, up to 14 epoch columns).
    This function converts it to long format (one row per site per epoch),
    which matches the schema used by load_unosat_labels() above.

    Damage classes (from Main_Damage_Site_Class field):
        1 = Destroyed (123,464 structures as of Oct 2025)
        2 = Severely Damaged (17,116)
        3 = Moderately Damaged (33,857)
        4 = Possibly Damaged (21,669)
        11 = Possible Damage From Adjacent Impact/Debris (2,167) - excluded
        6 = No Visible Damage (35) - excluded
    Pipeline uses labels_to_keep=[1, 2] by default, matching Ukraine methodology
    combine_epoch='last' in load_unosat_labels() selects epoch 14 as definitive label

    Args:
        gdb_path (Path): Path to the raw .gdb folder.
        labels_to_keep (list[int]): Damage classes to include. Defaults to [1, 2].
    """

    from shapely.ops import transform

    print(f"Loading {LAYER_NAME} from {gdb_path} ...")
    gdf_raw = gpd.read_file(gdb_path, layer=LAYER_NAME)
    # Drop Z dimension (compatible with older geopandas)
    gdf_raw.geometry = gdf_raw.geometry.apply(
        lambda geom: transform(lambda x, y, *z: (x, y), geom)
    )
    gdf_raw = gdf_raw.set_crs("EPSG:4326", allow_override=True)
    print(f"  Loaded {len(gdf_raw):,} points across {gdf_raw['Governorate'].nunique()} governorates")

    # --- Convert wide format to long format ---
    print("Converting wide format to long format ...")
    records = []
    for _, row in gdf_raw.iterrows():
        aoi = GOVERNORATE_TO_AOI.get(row["Governorate"])
        if aoi is None:
            continue

        # Collect all epochs into a consistent list of (date, damage_class, epoch_num)
        epochs = [(row.get("SensorDate"), row.get("Main_Damage_Site_Class"), 1)]
        for i in range(2, N_EPOCHS + 1):
            epochs.append((
                row.get(f"SensorDate_{i}"),
                row.get(f"Main_Damage_Site_Class_{i}"),
                i,
            ))

        for sensor_date, damage_class, ep_num in epochs:
            if pd.isna(damage_class):
                continue
            records.append({
                "unosat_id": f"{row['SiteID']}_{ep_num}",
                "site_id":   row["SiteID"],
                "aoi":       aoi,
                "damage":    int(damage_class),
                "ep":        ep_num,
                "date":      pd.to_datetime(sensor_date).date() if pd.notna(sensor_date) else None,
                "geometry":  row["geometry"],
            })

    gdf_long = gpd.GeoDataFrame(records, geometry="geometry", crs="EPSG:4326")
    print(f"  Long format: {len(gdf_long):,} rows across all damage classes")

    # --- Save labels (all damage classes, filtering happens in load_unosat_labels()) ---
    out_labels = DATA_PATH / "unosat_labels.geojson"
    gdf_long.set_index("unosat_id").to_file(out_labels, driver="GeoJSON")
    print(f"Saved {out_labels} ({len(gdf_long):,} rows)")

    # --- Create one AOI polygon per governorate (convex hull of all its points) ---
    print("Creating AOI polygons ...")
    aoi_records = []
    for gov, aoi_id in GOVERNORATE_TO_AOI.items():
        pts = gdf_raw[gdf_raw["Governorate"] == gov]
        if len(pts) == 0:
            print(f"  Warning: no points for {gov}")
            continue
        aoi_records.append({
            "aoi":        aoi_id,
            "governorate": gov,
            "geometry":   pts.geometry.unary_union.convex_hull,
        })
    gdf_aois = gpd.GeoDataFrame(aoi_records, geometry="geometry", crs="EPSG:4326")
    out_aois = DATA_PATH / "unosat_aois.geojson"
    gdf_aois.to_file(out_aois, driver="GeoJSON")
    print(f"Saved {out_aois} ({len(gdf_aois)} AOIs)")

    # --- Summary ---
    print("\n── Label counts by AOI and damage class ──")
    print(gdf_long.groupby(["aoi", "damage"]).size().to_string())


if __name__ == "__main__":
    preprocess_gaza_unosat()