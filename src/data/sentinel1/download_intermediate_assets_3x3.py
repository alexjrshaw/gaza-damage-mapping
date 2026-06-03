"""Download 3x3 GEE intermediate assets to local parquet cache."""
import ee
import pandas as pd
from src.data.sentinel1.download_intermediate_assets import download_intermediate_asset
from src.constants import DATA_PATH, AOIS
from src.utils.gee import init_gee
init_gee()

CACHE_DIR_3x3 = DATA_PATH / "intermediate_features_cache_3x3"
ORBITS = [87, 94, 160]

def download_3x3(aoi: str, orbit: int, force: bool = False):
    import time
    from src.utils.gdrive import drive_to_local
    from src.constants import ASSETS_PATH
    CACHE_DIR_3x3.mkdir(exist_ok=True, parents=True)
    fp = CACHE_DIR_3x3 / f"{aoi}_orbit{orbit}.parquet"
    if fp.exists() and not force:
        print(f"  {aoi}_orbit{orbit}: already cached ✓")
        return fp
    print(f"  {aoi}_orbit{orbit}: exporting to Drive...")
    asset_id = ASSETS_PATH + f"intermediate_features/ts_s1_3x3/{aoi}_orbit{orbit}"
    fc = ee.FeatureCollection(asset_id)
    description = f"{aoi}_orbit{orbit}_features_3x3"
    drive_folder = "gaza_intermediate_features_3x3"
    task = ee.batch.Export.table.toDrive(
        collection=fc,
        description=description,
        folder=drive_folder,
        fileFormat="CSV",
    )
    task.start()
    print(f"  {aoi}_orbit{orbit}: waiting for export...")
    while True:
        status = task.status()
        state = status["state"]
        if state == "COMPLETED":
            break
        elif state in ["FAILED", "CANCELLED"]:
            raise RuntimeError(f"Export failed: {status}")
        import time
        time.sleep(30)
    print(f"  {aoi}_orbit{orbit}: downloading from Drive...")
    tmp_dir = CACHE_DIR_3x3 / "tmp"
    tmp_dir.mkdir(exist_ok=True)
    drive_to_local(drive_folder, tmp_dir, delete_in_drive=True, verbose=0)
    csv_fp = tmp_dir / f"{description}.csv"
    df = pd.read_csv(csv_fp)
    df.to_parquet(fp)
    print(f"  {aoi}_orbit{orbit}: {len(df):,} rows saved ✓")
    return fp

if __name__ == "__main__":
    print("Downloading 3x3 intermediate assets...")
    for aoi in AOIS:
        print(f"\n{aoi}:")
        for orbit in ORBITS:
            download_3x3(aoi, orbit)
    print("\nAll 3x3 assets downloaded.")
