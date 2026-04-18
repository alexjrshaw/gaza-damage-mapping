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
            Either the 'last' one, the 'min' one (eg the strongest label), or 'first_severe'
            (earliest epoch where damage is class 1 or 2 — Gaza adaptation). Default to 'last'

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
        elif combine_epoch == "first_severe":
            # Keep earliest epoch where damage is class 1 or 2.
            # This is tunosat in Dietrich et al. (2025) eq. 1 —
            # the date severe damage was first confirmed by UNOSAT.
            # Gaza-specific adaptation: Ukraine had one epoch per point
            # so combine_epoch='last' naturally gave the detection date.
            # Gaza has 14 epochs so we must explicitly find first severe.
            severe = gdf[gdf["damage"].isin([1, 2])]
            gdf = severe.loc[
                severe.groupby(severe.geometry.to_wkt())["ep"].idxmin()
            ]
        else:
            raise ValueError("combine_epoch must be 'last', 'min' or 'first_severe'")

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
    gdb_path: Path = GDB_PATH,) -> None:
    """
    Convert raw UNOSAT Gaza GDB into unosat_labels.geojson and unosat_aois.geojson.

    Follows Dietrich et al. (2025) methodology as closely as possible.
    The underlying file stores ALL damage classes and ALL epochs (long format),
    consistent with the Ukraine pipeline where load_unosat_labels() filters
    to classes [1,2] at load time rather than preprocessing time.

    Gaza-specific adaptation: because the same points are assessed across
    14 epochs (unlike Ukraine where each point had one assessment), we
    pre-compute four summary date fields per point to support correct
    label assignment per Dietrich et al. equation 1:

        y = 0 if end_post <= conflict_start
        y = 1 if end_post > tunosat  (= date_first_severe here)
        y = -1 otherwise (discard)

    Summary fields added to every row:
        date_first           - first date any damage was recorded (any class)
        date_first_severe    - first date class 1 or 2 was recorded
                               THIS IS tunosat in Dietrich et al. eq. 1
        date_first_destroyed - first date class 1 was recorded (None if never)
        damage_max           - most severe class ever recorded (lowest number)

    Args:
        gdb_path (Path): Path to the raw .gdb folder.
    """

    print(f"Loading {LAYER_NAME} from {gdb_path} ...")
    gdf_raw = gpd.read_file(gdb_path, layer=LAYER_NAME)
    from shapely.ops import transform
    gdf_raw.geometry = gdf_raw.geometry.apply(lambda geom: transform(lambda x, y, *args: (x, y), geom))
    gdf_raw = gdf_raw.to_crs("EPSG:4326")
    print(f"  Loaded {len(gdf_raw):,} points across "
          f"{gdf_raw['Governorate'].nunique()} governorates")

    # --- Convert wide format to long format ---
    # Ukraine data was already long format (one row per point).
    # Gaza data is wide format (one row per point, 14 epoch columns).
    # We convert to long format to match the Ukraine schema exactly.
    print("Converting wide format to long format ...")
    records = []
    for _, row in gdf_raw.iterrows():
        aoi = GOVERNORATE_TO_AOI.get(row["Governorate"])
        if aoi is None:
            continue

        # Collect all epochs into a list of (date, damage_class, epoch_num)
        epochs = [(row.get("SensorDate"), row.get("Main_Damage_Site_Class"), 1)]
        for i in range(2, N_EPOCHS + 1):
            epochs.append((
                row.get(f"SensorDate_{i}"),
                row.get(f"Main_Damage_Site_Class_{i}"),
                i,
            ))

        # Build per-epoch records — skip epochs with no assessment
        point_records = []
        for sensor_date, damage_class, ep_num in epochs:
            if pd.isna(damage_class):
                continue
            point_records.append({
                "unosat_id":  f"{row['SiteID']}_{ep_num}",
                "site_id":    row["SiteID"],
                "aoi":        aoi,
                "damage":     int(damage_class),
                "ep":         ep_num,
                "date":       str(pd.to_datetime(sensor_date).date())
                              if pd.notna(sensor_date) else None,
                "geometry":   row["geometry"],
            })

        if not point_records:
            continue

        # --- Compute summary fields across all epochs for this point ---
        # These follow the spirit of Dietrich et al. eq. 1 where tunosat
        # is the date the post-event image was acquired confirming damage.
        dates = [pd.to_datetime(r["date"]) for r in point_records
                 if r["date"] is not None]
        severe = [r for r in point_records if r["damage"] in [1, 2]]
        destroyed = [r for r in point_records if r["damage"] == 1]
        damage_classes = [r["damage"] for r in point_records]

        date_first = str(min(dates).date()) if dates else None

        date_first_severe = str(min(
            pd.to_datetime(r["date"]) for r in severe
            if r["date"] is not None
        ).date()) if severe else None

        date_first_destroyed = str(min(
            pd.to_datetime(r["date"]) for r in destroyed
            if r["date"] is not None
        ).date()) if destroyed else None

        damage_max = min(damage_classes)

        # Add summary fields to every row for this point
        for r in point_records:
            r["date_first"] = date_first
            r["date_first_severe"] = date_first_severe
            r["date_first_destroyed"] = date_first_destroyed
            r["damage_max"] = damage_max

        records.extend(point_records)

    gdf_long = gpd.GeoDataFrame(records, geometry="geometry", crs="EPSG:4326")
    print(f"  Long format: {len(gdf_long):,} rows across all damage classes")

    # --- Save labels ---
    # Store ALL damage classes — filtering to [1,2] happens at load time
    # in load_unosat_labels(), exactly as in the Ukraine pipeline.
    out_labels = DATA_PATH / "unosat_labels.geojson"
    gdf_long.set_index("unosat_id").to_file(out_labels, driver="GeoJSON")
    print(f"Saved {out_labels} ({len(gdf_long):,} rows)")

    # --- Create AOI polygons from OCHA admin boundaries ---
    # Use official boundaries rather than convex hulls of damage points
    print("Creating AOI polygons from OCHA admin boundaries ...")
    from src.utils.geo import load_gaza_admin_polygons
    adm2 = load_gaza_admin_polygons(adm_level=2)
    aoi_records = []
    for _, row in adm2.iterrows():
        gov = row["adm2_name"]
        aoi_id = GOVERNORATE_TO_AOI.get(gov)
        if aoi_id is None:
            continue
        aoi_records.append({
            "aoi":         aoi_id,
            "governorate": gov,
            "geometry":    row["geometry"],
        })
    gdf_aois = gpd.GeoDataFrame(aoi_records, geometry="geometry", crs="EPSG:4326")
    out_aois = DATA_PATH / "unosat_aois.geojson"
    gdf_aois.to_file(out_aois, driver="GeoJSON")
    print(f"Saved {out_aois} ({len(gdf_aois)} AOIs)")

    # --- Summary ---
    print("\n── Label counts by AOI and damage class ──")
    severe_only = gdf_long[gdf_long["damage"].isin([1, 2])]
    first_severe = severe_only.loc[
        severe_only.groupby("site_id")["ep"].idxmin()
    ]
    print(first_severe.groupby(["aoi", "damage"]).size().to_string())
    print(f"\nTotal unique severely damaged points: {len(first_severe):,}")
    print("\nDistribution of date_first_severe:")
    print(
        first_severe["date_first_severe"]
        .value_counts()
        .sort_index()
        .to_string()
    )

def export_gaza_unosat_per_aoi() -> None:
    """
    Export preprocessed UNOSAT Gaza labels as separate GeoJSON files per AOI,
    ready for upload to GEE via the earthengine command line tool.

    Output files are saved to data/gee_upload/
    """
    import json

    print("Exporting per-AOI GeoJSON files for GEE upload...")
    labels_fp = DATA_PATH / "unosat_labels.geojson"
    aois_fp   = DATA_PATH / "unosat_aois.geojson"
    assert labels_fp.exists(), "Run preprocess_gaza_unosat() first"
    assert aois_fp.exists(),   "Run preprocess_gaza_unosat() first"

    gdf_labels = gpd.read_file(labels_fp)
    gdf_aois   = gpd.read_file(aois_fp)

    out_dir = DATA_PATH / "gee_upload"
    out_dir.mkdir(exist_ok=True, parents=True)

    for aoi in GOVERNORATE_TO_AOI.values():
        print(f"\nProcessing {aoi} ...")

        # Labels (use first epoch where damage is class 1 or 2)
        pts = gdf_labels[
            (gdf_labels["aoi"] == aoi) &
            (gdf_labels["damage"].isin([1, 2]))
        ].copy()
        pts = pts.loc[pts.groupby(pts.geometry.to_wkt())["ep"].idxmin()]
        fp = out_dir / f"UNOSAT_labels_{aoi}.geojson"
        pts.to_file(fp, driver="GeoJSON")
        print(f"  Saved {len(pts)} points to {fp}")

        # Use first epoch per point (for full labels)
        pts_full = gdf_labels[gdf_labels["aoi"] == aoi].copy()
        pts_full = pts_full.loc[pts_full.groupby(pts_full.geometry.to_wkt())["ep"].idxmin()]
        fp_full = out_dir / f"UNOSAT_labels_{aoi}_full.geojson"
        pts_full.to_file(fp_full, driver="GeoJSON")
        print(f"  Saved {len(pts_full)} points (all classes) to {fp_full}")

        # AOI boundary
        aoi_row = gdf_aois[gdf_aois["aoi"] == aoi].copy()
        fp_aoi = out_dir / f"AOI_{aoi}.geojson"
        aoi_row.to_file(fp_aoi, driver="GeoJSON")
        print(f"  Saved AOI boundary to {fp_aoi}")

    print(f"\nAll files saved to {out_dir}")
    print("\nTo upload to GEE, run the following commands:")
    print("source alex/bin/activate")
    for aoi in GOVERNORATE_TO_AOI.values():
        asset_path = ASSETS_PATH + f"UNOSAT_labels/{aoi}"
        fp = out_dir / f"UNOSAT_labels_{aoi}.geojson"
        print(f"earthengine upload table --asset_id={asset_path} {fp}")

        asset_path_full = ASSETS_PATH + f"UNOSAT_labels/{aoi}_full"
        fp_full = out_dir / f"UNOSAT_labels_{aoi}_full.geojson"
        print(f"earthengine upload table --asset_id={asset_path_full} {fp_full}")

        asset_path_aoi = ASSETS_PATH + f"AOIs/{aoi}"
        fp_aoi = out_dir / f"AOI_{aoi}.geojson"
        print(f"earthengine upload table --asset_id={asset_path_aoi} {fp_aoi}")

def upload_gaza_unosat_to_gee() -> None:
    """
    Upload preprocessed UNOSAT Gaza labels and AOI boundaries to GEE assets.
    Files under 10MB upload directly; larger files are uploaded in chunks and merged.
    """
    import time
    import geemap
    from src.utils.gee import asset_exists, create_folder, init_gee
    init_gee(project="gaza-damage-mapping")

    labels_fp = DATA_PATH / "unosat_labels.geojson"
    aois_fp   = DATA_PATH / "unosat_aois.geojson"
    assert labels_fp.exists(), "Run preprocess_gaza_unosat() first"
    assert aois_fp.exists(),   "Run preprocess_gaza_unosat() first"

    gdf_labels = gpd.read_file(labels_fp)
    gdf_aois   = gpd.read_file(aois_fp)

    # Ensure folders exist
    for folder in ["UNOSAT_labels", "AOIs"]:
        folder_path = ASSETS_PATH + folder
        if not asset_exists(folder_path):
            create_folder(folder_path)

    def upload_direct(gdf, asset_id, description):
        """Upload a small GeoDataFrame directly to GEE."""
        if asset_exists(asset_id):
            print(f"  {asset_id.split('/')[-1]} already exists, skipping")
            return
        fc = geemap.geopandas_to_ee(gdf)
        task = ee.batch.Export.table.toAsset(
            collection=fc,
            description=description,
            assetId=asset_id,
        )
        task.start()
        print(f"  Uploading {len(gdf)} features → {asset_id.split('/')[-1]}")
        while not asset_exists(asset_id):
            time.sleep(5)
        print(f"  ✓ Done")

    def upload_chunked(gdf, asset_id, description, chunk_size=5000):
        """Upload a large GeoDataFrame in chunks, then merge into one asset."""
        if asset_exists(asset_id):
            print(f"  {asset_id.split('/')[-1]} already exists, skipping")
            return

        n = len(gdf)
        chunks = [gdf.iloc[i:i+chunk_size] for i in range(0, n, chunk_size)]
        print(f"  Uploading {n} features in {len(chunks)} chunks...")

        chunk_ids = []
        for i, chunk in enumerate(chunks):
            chunk_id = asset_id + f"_tmp_chunk{i}"
            if not asset_exists(chunk_id):
                fc = geemap.geopandas_to_ee(chunk.copy())
                task = ee.batch.Export.table.toAsset(
                    collection=fc,
                    description=f"{description}_chunk{i}",
                    assetId=chunk_id,
                )
                task.start()
            chunk_ids.append(chunk_id)

        # Wait for all chunks
        print(f"  Waiting for chunks to complete...")
        for chunk_id in chunk_ids:
            while not asset_exists(chunk_id):
                time.sleep(5)
        print(f"  All chunks uploaded, merging...")

        # Merge and export as single asset
        merged = ee.FeatureCollection(
            [ee.FeatureCollection(cid) for cid in chunk_ids]
        ).flatten()
        merge_task = ee.batch.Export.table.toAsset(
            collection=merged,
            description=description,
            assetId=asset_id,
        )
        merge_task.start()
        while not asset_exists(asset_id):
            time.sleep(5)

        # Clean up temporary chunks
        for chunk_id in chunk_ids:
            ee.data.deleteAsset(chunk_id)
        print(f"  ✓ Done ({n} features merged)")

    # Feature count threshold above which we use chunked upload
    CHUNK_THRESHOLD = 30000

    for aoi in GOVERNORATE_TO_AOI.values():
        print(f"\nProcessing {aoi}...")

        # AOI boundary (always tiny — direct upload)
        aoi_row = gdf_aois[gdf_aois["aoi"] == aoi].copy()
        upload_direct(aoi_row, ASSETS_PATH + f"AOIs/{aoi}", f"AOI_{aoi}")

        # Labels (classes 1+2, latest epoch per point)
        pts = gdf_labels[
            (gdf_labels["aoi"] == aoi) &
            (gdf_labels["damage"].isin([1, 2]))
        ].copy()
        pts = pts.loc[pts.groupby(pts.geometry.to_wkt())["ep"].idxmax()]
        asset_id = ASSETS_PATH + f"UNOSAT_labels/{aoi}"
        if len(pts) > CHUNK_THRESHOLD:
            upload_chunked(pts, asset_id, f"UNOSAT_labels_{aoi}")
        else:
            upload_direct(pts, asset_id, f"UNOSAT_labels_{aoi}")

        # Labels full (all classes, latest epoch per point) - skipped for now, not needed for core pipeline
        #pts_full = gdf_labels[gdf_labels["aoi"] == aoi].copy()
        #pts_full = pts_full.loc[pts_full.groupby(pts_full.geometry.to_wkt())["ep"].idxmax()]
        #asset_id_full = ASSETS_PATH + f"UNOSAT_labels/{aoi}_full"

        # upload when needed by changing skip_full=False
        skip_full = True
        if not skip_full:
            pts_full = gdf_labels[gdf_labels["aoi"] == aoi].copy()
            pts_full = pts_full.loc[pts_full.groupby(pts_full.geometry.to_wkt())["ep"].idxmax()]
            asset_id_full = ASSETS_PATH + f"UNOSAT_labels/{aoi}_full"
            if len(pts_full) > CHUNK_THRESHOLD:
                upload_chunked(pts_full, asset_id_full, f"UNOSAT_labels_{aoi}_full")
            else:
                upload_direct(pts_full, asset_id_full, f"UNOSAT_labels_{aoi}_full")

    print("\nAll uploads complete.")

if __name__ == "__main__":
    preprocess_gaza_unosat()
    
    response = input("\nExport GeoJSON files for GEE upload? (y/n): ")
    if response.lower() == "y":
        export_gaza_unosat_per_aoi()
        
        response2 = input("\nUpload to GEE? This will take a long time. (y/n): ")
        if response2.lower() == "y":
            upload_gaza_unosat_to_gee()
