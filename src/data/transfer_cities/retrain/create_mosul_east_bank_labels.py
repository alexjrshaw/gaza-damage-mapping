"""
Create an east-bank-only UNOSAT labels file for Mosul, for fair
evaluation of the retrained model.

Adaptation needed because evaluate_pixel_transfer.py (unmodified)
evaluates against whichever UNOSAT labels file is given to it. For the
zero-shot comparison, that is the full Mosul UNOSAT file (the whole city
is "unseen" by the Gaza-trained model). For the retrained comparison,
evaluating on the full file would be unfair: the retrained model has
already seen the west-bank points during training, so including them
in evaluation would inflate its apparent performance relative to the
zero-shot result, which is tested on points it never saw.

This script filters Mosul's UNOSAT labels to east-bank points only
(lon >= 43.1262, the same split boundary used in main_local_mosul_retrain.py)
and saves them as a new labels file, referenced by a new
"MOS_RETRAINED_EAST_ONLY" entry in constants_transfer.py (added manually;
see note printed at the end of this script).

Usage:
# Apply spatial split
    python3 src/data/transfer_cities/retrain/create_mosul_east_bank_labels.py
"""

import geopandas as gpd

# Spatial split configuration
LON_SPLIT = 43.1262  # must match main_local_mosul_retrain.py exactly

INPUT_FP = "test_sites/processed/mos/unosat_labels.geojson"
OUTPUT_FP = "test_sites/processed/mos/unosat_labels_east_bank_only.geojson"


def main():
    print(f"Loading {INPUT_FP}...")
    gdf = gpd.read_file(INPUT_FP)
    print(f"  {len(gdf):,} total points")

    east_bank = gdf[gdf.geometry.x >= LON_SPLIT].copy()
    print(f"  {len(east_bank):,} east-bank points (lon >= {LON_SPLIT}) retained")
    print(f"  {len(gdf) - len(east_bank):,} west-bank points excluded (used for training)")

    east_bank.to_file(OUTPUT_FP, driver="GeoJSON")
    print(f"\nSaved to {OUTPUT_FP}")
    print(
        "\nManual step required: add an entry to "
        "src/data/transfer_cities/constants_transfer.py, e.g.:\n\n"
        "  TRANSFER_CITIES['MOS_RETRAINED_EAST_ONLY'] = {\n"
        "      **TRANSFER_CITIES['MOS'],\n"
        "      'unosat_labels': "
        f'"{OUTPUT_FP}",\n'
        "  }\n\n"
        "Then run:\n"
        "  python3 src/data/transfer_cities/pixel_inference/evaluate_pixel_transfer.py "
        "--city MOS_RETRAINED_EAST_ONLY\n\n"
        "Note: evaluate_pixel_transfer.py also needs TRANSFER_PROB_BASE / city_id to "
        "resolve to the retrained probability rasters. Since city_id here is "
        "'MOS_RETRAINED_EAST_ONLY' but the retrained rasters were saved "
        "under 'MOS_retrained' (see mosul_retrain_pixel_inference.py), either rename "
        "that output folder to match, or symlink it:\n"
        "  ln -s data/transfer_cities/probability_rasters/MOS_retrained "
        "data/transfer_cities/probability_rasters/MOS_RETRAINED_EAST_ONLY"
    )


# Entry point
if __name__ == "__main__":
    main()
