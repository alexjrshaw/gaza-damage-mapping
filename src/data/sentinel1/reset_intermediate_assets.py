import ee
from src.utils.gee import asset_exists, delete_asset, init_gee
from src.data.sentinel1.intermediate_data import create_fc_aoi_orbit
from src.data.utils import aoi_orbit_iterator
from src.constants import AOIS

init_gee(project="gaza-damage-mapping")

BASE = "projects/gaza-damage-mapping/assets/gaza-mapping-tool/intermediate_features/ts_s1_1x1"

# --- Step 1: Delete existing assets ---
print("Deleting existing intermediate assets...")
for aoi, orbit in aoi_orbit_iterator():
    asset_id = f"{BASE}/{aoi}_orbit{orbit}"
    if asset_exists(asset_id):
        delete_asset(asset_id)
    else:
        print(f"  {aoi}_orbit{orbit} — not found, skipping")

# --- Step 2: Resubmit orbit tasks ---
print("\nSubmitting new orbit tasks...")
for aoi, orbit in aoi_orbit_iterator():
    create_fc_aoi_orbit(aoi, orbit, scale=10, export=True)
    print(f"  Submitted {aoi}_orbit{orbit}")

print("\nAll tasks submitted. Monitor progress in GEE task manager.")
