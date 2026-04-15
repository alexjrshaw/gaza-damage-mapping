import geopandas as gpd
import pandas as pd
import ee

from src.constants import DATA_PATH, AOIS


def get_valid_orbits(aoi: str) -> list[int]:
    """
    Get the valid orbits for a given AOI.

    Args:
        aoi (str): The AOI name

    Returns:
        List[int]: The list of valid orbits
    """
    df_orbits = load_df_orbits()
    return df_orbits.loc[aoi, "valid_orbits"]


def load_df_orbits() -> pd.DataFrame:
    """
    Load the DataFrame with the valid orbits for each AOI.

    If the file does not exist, it is created by querying GEE.

    Returns:
        pd.DataFrame: The DataFrame
    """
    fp = DATA_PATH / "s1_aoi_orbits.csv"

    if not fp.exists():
        print("Orbits file not found. Querying GEE to find valid orbits...")
        create_orbits_file(fp)

    df_orbits = pd.read_csv(fp)
    df_orbits.valid_orbits = df_orbits.valid_orbits.apply(
        lambda x: [int(i) for i in x.split(",")]
    )
    df_orbits.set_index("aoi", inplace=True)
    return df_orbits


def create_orbits_file(fp) -> None:
    """
    Query GEE to find valid Sentinel-1 orbits for each Gaza AOI.

    A valid orbit is one that has at least 10 images over the AOI
    during the pre-war baseline period.

    Args:
        fp: Path to save the CSV file
    """
    from src.constants import PRE_PERIOD
    from src.data.unosat import load_unosat_geo_gee
    from src.utils.gee import init_gee
    init_gee(project="gaza-damage-mapping")

    records = []
    for aoi in AOIS:
        print(f"  Finding orbits for {aoi}...")
        geo = load_unosat_geo_gee(aoi)

        # Query S1 collection over AOI during pre-war period
        s1 = (
            ee.ImageCollection("COPERNICUS/S1_GRD")
            .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VV"))
            .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VH"))
            .filter(ee.Filter.eq("instrumentMode", "IW"))
            .filterBounds(geo)
            .filterDate(PRE_PERIOD[0], PRE_PERIOD[1])
        )

        # Count images per orbit
        orbit_counts = s1.aggregate_histogram("relativeOrbitNumber_start").getInfo()
        print(f"    All orbits: {orbit_counts}")

        # Keep orbits with at least 10 images (roughly one per month)
        valid_orbits = [
            int(float(orbit)) for orbit, count in orbit_counts.items()
            if int(float(count)) >= 10
        ]
        valid_orbits.sort()
        print(f"    Valid orbits (>=10 images): {valid_orbits}")

        records.append({
            "aoi": aoi,
            "valid_orbits": ",".join(str(o) for o in valid_orbits),
        })

    df = pd.DataFrame(records)
    df.to_csv(fp, index=False)
    print(f"Saved orbits file to {fp}")


if __name__ == "__main__":
    from src.utils.gee import init_gee
    init_gee(project="gaza-damage-mapping")
    fp = DATA_PATH / "s1_aoi_orbits.csv"
    if fp.exists():
        fp.unlink()  # delete so we force recreation
    load_df_orbits()