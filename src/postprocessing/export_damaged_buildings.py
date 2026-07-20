import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.wkb import loads as wkb_loads
from pathlib import Path

DATA_PATH      = Path("/scratch/s1214882/gaza-damage-mapping/data")
OUTPUT_DIR     = DATA_PATH / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

THRESHOLD      = int(0.670 * 255)
GAZA_WAR_START = "2023-10-07"

print("Loading data...")
df_preds  = pd.read_parquet(DATA_PATH / "pixel_postprocessing/buildings_preds.parquet")
df_damage = pd.read_parquet(DATA_PATH / "pixel_postprocessing/buildings_damage.parquet")

date_cols = sorted([c for c in df_preds.columns
                    if isinstance(c, str) and len(c) == 10 and c[4] == "-"])
post_cols = [c for c in date_cols if c >= GAZA_WAR_START]

# Filter to damaged buildings and RESET INDEX — this is the critical fix
damaged_idx = df_damage[df_damage["damaged"] == 1].index
df_sub = df_preds.loc[damaged_idx].copy().reset_index(drop=True)
print(f"Damaged buildings: {len(df_sub):,}")

# First post-conflict window above threshold
post_arr       = df_sub[post_cols].values
above          = post_arr >= THRESHOLD
has_any        = above.any(axis=1)
first_col      = np.argmax(above, axis=1)
post_arr_lbls  = np.array(post_cols)
first_end_date = np.where(has_any, post_arr_lbls[first_col], "unknown")
window_num     = np.where(has_any, first_col + 1, -1).astype(int)
window_labels  = {col: f"T{i+7:02d} ({col})" for i, col in enumerate(post_cols)}
window_label   = np.array([window_labels.get(w, "unknown") for w in first_end_date])

# Decode WKB geometry — index now matches (both 0-based)
print("Decoding WKB geometry...")
geom_col    = "geometry_wkb" if "geometry_wkb" in df_sub.columns else "geometry"
geom_series = gpd.GeoSeries(
    df_sub[geom_col].apply(lambda x: wkb_loads(bytes(x))),
    crs="EPSG:4326"
)
print(f"  Sample geometry type: {geom_series.iloc[0].geom_type}")
print(f"  All valid: {geom_series.is_valid.all()}")
print(f"  Any None: {geom_series.isna().any()}")

# Build GeoDataFrame — index alignment now guaranteed
gdf = gpd.GeoDataFrame({
    "fid":              range(1, len(df_sub) + 1),
    "area_m2":          df_sub["area_m2"].values,
    "governorate":      df_sub["adm2_name"].values,
    "first_window_end": first_end_date,
    "window_num":       window_num,
    "window_label":     window_label,
}, geometry=geom_series, crs="EPSG:4326")

assert len(gdf) == 151368, f"Expected 151,368, got {len(gdf):,}"
assert not gdf.geometry.isna().any(), "Null geometries detected"
print(f"\n✓ {len(gdf):,} valid polygon features")

print("\nBuildings by first-detected window:")
summary = gdf.groupby(["window_num", "window_label"]).size().rename("n_buildings")
print(summary.to_string())

out_path = OUTPUT_DIR / "damaged_buildings_polygons_t067.gpkg"
print(f"\nExporting to {out_path}...")
gdf.to_file(out_path, driver="GPKG", layer="damaged_buildings")
print(f"Done. File size: {out_path.stat().st_size / 1e6:.1f} MB")