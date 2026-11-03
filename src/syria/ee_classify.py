#!/usr/bin/env python3
"""
Classify Sentinel-1 windows INSIDE Earth Engine with the converted Gaza forest, and test
that this reproduces the local pipeline on Aleppo.

This is Dietrich et al.'s architecture (features + classification in Earth Engine, only a
1-band probability tile leaves) running YOUR Gaza model: the 50 trees uploaded by
forest_to_gee.py as the table asset projects/gaza-damage-mapping/assets/gaza_rf_trees.

For one tile and window it follows exactly what the local pipeline does:
    per orbit: 28 features (same col_to_features, same S1 filter, same pre/post windows)
               -> forest probability
    then the MEAN over orbits (masked pixels are skipped, like np.nanmean)
    then x255 and cast to Uint8.

Commands (run from the repository root on Forth):

  python3 src/syria/ee_classify.py export  --city ALP --windows w02 w11 w14
        Starts Earth Engine export tasks (2 per tile and window: a Float32 probability
        and the Uint8 tile) into Drive folder syria_parity/<city>/<window>.

  python3 src/syria/ee_classify.py eecu
        Lists those tasks with state, run time and EECU-seconds (the compute cost).

  python3 src/syria/ee_classify.py compare --city ALP --windows w02 w11 w14
        Downloads the finished tiles (and deletes them from Drive) and compares them with
        data/transfer_cities/probability_rasters/<city>/<window>/qk_<id>.tif, the output of
        the local pipeline. Prints how many pixels agree and how the disagreements look.
"""
import argparse
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

FOREST_ASSET = "projects/gaza-damage-mapping/assets/gaza_rf_trees"
N_TREES = 50
REDUCER_NAMES = ["mean", "stdDev", "median", "min", "max", "skew", "kurtosis"]
EXTRACT_WINDOW = "1x1"
QUADKEY_ZOOM = 12
SCALE = 10
THRESHOLD_U8 = 170  # int(0.670 * 255), the damage threshold used by classify_building_damage.py
DRIVE_BASE = "syria_parity"
TASK_PREFIX = "parity_"
DEFAULT_QK = "122112100312"  # central Aleppo (old city), one of the 11 ALP tiles
DEFAULT_WINDOWS = ["w02", "w11", "w14"]  # baseline, early conflict, assessment date


# --------------------------------------------------------------------------- windows
def window_table(city_id):
    """[(window_str, (start, end))] exactly as the local export/inference scripts name them."""
    from src.data.transfer_cities.constants_transfer import TRANSFER_CITIES

    cfg = TRANSFER_CITIES[city_id]
    periods = [cfg["pre_period"]] + list(cfg["post_periods"])
    return [(f"w{i + 1:02d}_{p[0]}_{p[1]}", p) for i, p in enumerate(periods)]


def pick_windows(city_id, prefixes):
    table = window_table(city_id)
    chosen = [(w, p) for w, p in table if any(w.startswith(x) for x in prefixes)]
    if not chosen:
        sys.exit(f"No window of {city_id} starts with {prefixes}. Windows are: {[w for w, _ in table]}")
    return chosen


# --------------------------------------------------------------------------- Earth Engine
def window_probability(ee, classifier, s1, orbits, pre, post):
    """Mean over orbits of the forest probability, as an ee.Image with one band 'p' (0..1)."""
    from src.utils.gee import col_to_features

    per_orbit = []
    for orbit in orbits:
        s1_orbit = s1.filter(ee.Filter.eq("relativeOrbitNumber_start", orbit))
        feats = col_to_features(s1_orbit, REDUCER_NAMES, dict(pre=pre, post=post), EXTRACT_WINDOW)
        per_orbit.append(feats.classify(classifier).rename("p"))
    # ImageCollection.mean() ignores masked pixels, like np.nanmean over orbits.
    return ee.ImageCollection.fromImages(per_orbit).mean().rename("p")


def start_export(ee, image, description, drive_folder, prefix, region):
    task = ee.batch.Export.image.toDrive(
        image=image,
        description=description[:100],
        folder=drive_folder,
        fileNamePrefix=prefix,
        region=region,
        scale=SCALE,
        maxPixels=1e13,
        fileFormat="GeoTIFF",
    )
    task.start()
    return task


def cmd_export(args):
    import ee

    from src.data.sentinel1.collection import get_s1_collection
    from src.data.transfer_cities.constants_transfer import TRANSFER_CITIES, TRANSFER_GEE_FOLDER
    from src.syria.forest_to_gee import classifier_from_table
    from src.utils.gdrive import create_drive_folder
    from src.utils.gee import init_gee

    init_gee()
    city = args.city.upper()
    cfg = TRANSFER_CITIES[city]
    windows = pick_windows(city, args.windows)

    grids = ee.FeatureCollection(TRANSFER_GEE_FOLDER + f"quadkeys/{city}_zoom{QUADKEY_ZOOM}")
    known = set(grids.aggregate_array("qk").getInfo())
    for qk in args.qk:
        if qk not in known:
            sys.exit(f"Tile {qk} is not in the {city} grid. Tiles: {sorted(known)}")

    classifier = classifier_from_table(ee.FeatureCollection(FOREST_ASSET), args.n_trees)
    # Same collection construction as export_feature_rasters_transfer.py (bounds = whole AOI grid).
    s1 = get_s1_collection(grids.geometry(), cfg["gee_start"], cfg["gee_end"])

    n = 0
    for window_str, post in windows:
        drive_folder = f"{DRIVE_BASE}/{city}/{window_str}"
        for folder in (f"{DRIVE_BASE}/{city}", drive_folder):
            try:
                create_drive_folder(folder)
            except Exception:
                pass
        prob = window_probability(ee, classifier, s1, cfg["orbits"], cfg["pre_period"], post)
        as_float = prob.toFloat()
        as_u8 = prob.multiply(255).toUint8()  # Dietrich et al.: .multiply(255).toUint8()
        for qk in args.qk:
            region = grids.filter(ee.Filter.eq("qk", qk)).geometry()
            for kind, image in (("float", as_float), ("u8", as_u8)):
                desc = f"{TASK_PREFIX}{city}_{window_str[:3]}_{qk}_{kind}"
                task = start_export(ee, image, desc, drive_folder, f"qk_{qk}_{kind}", region)
                n += 1
                print(f"  started {desc}  ->  Drive: {drive_folder}/qk_{qk}_{kind}.tif")
    print(f"\n{n} tasks started. Watch https://code.earthengine.google.com/tasks ; when they have finished run:")
    print(f"  python3 src/syria/ee_classify.py eecu\n  python3 src/syria/ee_classify.py compare --city {city}"
          f" --windows {' '.join(a for a in args.windows)}" + "".join(f" --qk {q}" for q in args.qk))


def cmd_eecu(args):
    import ee

    from src.utils.gee import init_gee

    init_gee()
    try:
        ops = ee.data.listOperations()
    except Exception as err:  # older/newer client without this call
        sys.exit(f"Could not list operations ({err!r}). Read the numbers from the Tasks tab instead: "
                 "click a finished task and copy 'batchEecuUsageSeconds' and the run time.")
    rows = []
    for op in ops:
        meta = op.get("metadata", {}) or {}
        desc = meta.get("description", "")
        if not desc.startswith(args.prefix):
            continue
        rows.append((desc, meta.get("state", "?"), meta.get("startTime"), meta.get("endTime"),
                     meta.get("batchEecuUsageSeconds")))
    if not rows:
        sys.exit(f"No tasks whose description starts with '{args.prefix}' were returned "
                 f"(operations seen: {len(ops)}).")
    from datetime import datetime

    def secs(a, b):
        try:
            fmt = lambda s: datetime.fromisoformat(s.replace("Z", "+00:00"))
            return (fmt(b) - fmt(a)).total_seconds()
        except Exception:
            return None

    print(f"{'task':<46}{'state':<12}{'run s':>8}{'EECU-s':>10}")
    total = 0.0
    for desc, state, a, b, eecu in sorted(rows):
        run = secs(a, b) if a and b else None
        total += float(eecu or 0)
        print(f"{desc:<46}{state:<12}{(f'{run:.0f}' if run is not None else '-'):>8}"
              f"{(f'{float(eecu):.0f}' if eecu is not None else '-'):>10}")
    print(f"\nTotal EECU-seconds: {total:.0f}  (= {total / 3600:.3f} EECU-hours; Community tier gives 150 per month)")


# --------------------------------------------------------------------------- comparison
def compare_arrays(local_u8, ee_u8, ee_float):
    """Numbers describing how well the Earth Engine tile reproduces the local tile."""
    out = {"shape_local": local_u8.shape, "shape_ee": ee_u8.shape}
    if local_u8.shape != ee_u8.shape:
        return out
    lv, ev = local_u8 > 0, ee_u8 > 0
    both = lv & ev
    out.update(n_pixels=local_u8.size, local_valid=int(lv.sum()), ee_valid=int(ev.sum()),
               both_valid=int(both.sum()), only_local=int((lv & ~ev).sum()), only_ee=int((ev & ~lv).sum()))
    if not both.any():
        return out
    d = ee_u8[both].astype(int) - local_u8[both].astype(int)
    ad = np.abs(d)
    out.update(
        exact=float((d == 0).mean()), within1=float((ad <= 1).mean()), over1=int((ad > 1).sum()),
        over5=int((ad > 5).sum()), max_abs=int(ad.max()), mean_diff=float(d.mean()),
    )
    x = ee_float[both].astype(float) * 255.0
    out.update(
        local_matches_trunc=float((np.floor(x) == local_u8[both]).mean()),
        local_matches_round=float((np.round(x) == local_u8[both]).mean()),
        ee_u8_is_trunc=float((np.floor(x) == ee_u8[both]).mean()),
        ee_u8_is_round=float((np.round(x) == ee_u8[both]).mean()),
    )
    la, ea = local_u8[both] >= THRESHOLD_U8, ee_u8[both] >= THRESHOLD_U8
    out.update(flagged_local=int(la.sum()), flagged_ee=int(ea.sum()), flag_disagree=int((la != ea).sum()))
    return out


def verdict(s):
    if "both_valid" not in s or s["both_valid"] == 0:
        return "cannot judge: no pixels valid in both"
    mask_ok = (s["only_local"] + s["only_ee"]) <= 0.005 * max(s["local_valid"], 1)
    value_ok = s["within1"] >= 0.999
    flag_ok = s["flag_disagree"] <= 0.001 * s["both_valid"]
    if mask_ok and value_ok and flag_ok:
        return "PASS (masks agree, >=99.9% of pixels within 1 grey level, <=0.1% threshold flips)"
    why = [n for n, ok in (("valid-pixel masks differ", mask_ok), ("values differ by >1 grey level", value_ok),
                           ("threshold flags differ", flag_ok)) if not ok]
    return "CHECK: " + "; ".join(why)


def print_stats(label, s):
    print(f"\n=== {label}")
    if s["shape_local"] != s["shape_ee"]:
        print(f"  Tiles have different shapes: local {s['shape_local']} vs Earth Engine {s['shape_ee']}. "
              "The pixel grids differ; paste this back.")
        return
    pct = lambda a, b: f"{100 * a / max(b, 1):.2f}%"
    print(f"  Valid pixels (>0): local {s['local_valid']:,}, Earth Engine {s['ee_valid']:,}, both {s['both_valid']:,}; "
          f"only local {s['only_local']:,} ({pct(s['only_local'], s['local_valid'])}), only Earth Engine {s['only_ee']:,}")
    if "exact" not in s:
        return
    print(f"  Uint8 difference (EE - local) on shared pixels: identical {100 * s['exact']:.2f}%, within +-1 "
          f"{100 * s['within1']:.3f}%, >1 grey level {s['over1']:,}, >5 {s['over5']:,}, max {s['max_abs']}, "
          f"mean {s['mean_diff']:+.3f}")
    print(f"  Float check: local tile = floor(255*p_EE) for {100 * s['local_matches_trunc']:.2f}% of pixels "
          f"(round: {100 * s['local_matches_round']:.2f}%)")
    print(f"               EE Uint8 tile = floor(255*p) for {100 * s['ee_u8_is_trunc']:.2f}% "
          f"(round: {100 * s['ee_u8_is_round']:.2f}%)  <- how Earth Engine casts to Uint8")
    print(f"  Pixels at or above the damage threshold ({THRESHOLD_U8}): local {s['flagged_local']:,}, "
          f"Earth Engine {s['flagged_ee']:,}, disagree {s['flag_disagree']:,}")
    print(f"  Verdict: {verdict(s)}")


def read_band(fp):
    import rasterio

    with rasterio.open(fp) as src:
        return src.read(1), src.transform, src.crs


def download_folder(drive_folder, local_dir, delete):
    from src.utils.gdrive import drive, get_folder_id

    folder_id = get_folder_id(drive_folder)
    files = drive.ListFile({"q": f"'{folder_id}' in parents and trashed=false"}).GetList()
    tifs = [(f["title"], f["id"]) for f in files if f["title"].endswith(".tif")]
    local_dir.mkdir(parents=True, exist_ok=True)
    for name, file_id in tifs:
        f = drive.CreateFile({"id": file_id})
        f.GetContentFile(str(local_dir / name))
        if delete:
            f.Delete()
    return [n for n, _ in tifs]


def cmd_compare(args):
    from src.constants import DATA_PATH

    city = args.city.upper()
    for window_str, _ in pick_windows(city, args.windows):
        local_dir = DATA_PATH / "syria" / "parity" / city / window_str
        if not args.no_download:
            try:
                got = download_folder(f"{DRIVE_BASE}/{city}/{window_str}", local_dir, delete=not args.keep)
                print(f"{window_str}: downloaded {len(got)} files")
            except Exception as err:
                print(f"{window_str}: nothing downloaded yet ({err}). Are the Earth Engine tasks finished?")
        for qk in args.qk:
            fp_f, fp_u = local_dir / f"qk_{qk}_float.tif", local_dir / f"qk_{qk}_u8.tif"
            fp_l = DATA_PATH / "transfer_cities" / "probability_rasters" / city / window_str / f"qk_{qk}.tif"
            missing = [str(p) for p in (fp_f, fp_u, fp_l) if not p.exists()]
            if missing:
                print(f"\n=== {window_str} / {qk}: missing {missing}")
                continue
            (l_arr, l_tf, l_crs), (u_arr, u_tf, u_crs), (f_arr, _, _) = read_band(fp_l), read_band(fp_u), read_band(fp_f)
            stats = compare_arrays(l_arr, u_arr, f_arr)
            print_stats(f"{window_str} / tile {qk}", stats)
            if stats["shape_local"] == stats["shape_ee"] and not (np.allclose(tuple(l_tf)[:6], tuple(u_tf)[:6], rtol=0, atol=1e-9) and l_crs == u_crs):
                print(f"  NOTE: same shape but the georeferencing differs.\n    local: {l_tf} {l_crs}\n    EE:    {u_tf} {u_crs}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("export", "compare"):
        sp = sub.add_parser(name)
        sp.add_argument("--city", default="ALP")
        sp.add_argument("--windows", nargs="+", default=DEFAULT_WINDOWS, help="window prefixes, e.g. w02 w14")
        sp.add_argument("--qk", nargs="+", default=[DEFAULT_QK], help="z12 quadkey ids")
        if name == "export":
            sp.add_argument("--n-trees", type=int, default=N_TREES)
        else:
            sp.add_argument("--no-download", action="store_true", help="use files already downloaded")
            sp.add_argument("--keep", action="store_true", help="keep the files on Drive after downloading")
    sp = sub.add_parser("eecu")
    sp.add_argument("--prefix", default=TASK_PREFIX, help="task description prefix to list")
    args = ap.parse_args(argv)
    {"export": cmd_export, "compare": cmd_compare, "eecu": cmd_eecu}[args.cmd](args)


if __name__ == "__main__":
    main()
