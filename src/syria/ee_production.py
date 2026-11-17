#!/usr/bin/env python3
"""
Syria-wide classification in Earth Engine: one task per z13 tile, all assessment windows in
one task (one Uint8 band per window), restricted to z14 tiles with enough buildings.

It runs the Gaza forest unchanged (same code path as the Aleppo parity test), with
  * baseline : A = 2014-10-06 .. 2015-10-05 (default, Aleppo's) or B = 2015-10-21 .. 2016-10-20 (Raqqa's)
  * windows  : 2-month windows 7 Aug .. 6 Oct of the years in --years, OR the window list of a
               transfer city (--city ALP / RAQ: the same windows as the local transfer workflow)
  * region   : optional --boundary FILE [--where FIELD=VALUE] (an administrative layer) or --bbox W S E N keeps only that area (a region trial)
  * orbits   : per z13 tile and window, from orbit_survey.py --zoom 13:
               valid = >= 10 baseline images covering the tile; used in a window if it
               has >= 2 images there. A window with no usable orbit is left as nodata (0).
  * region   : the bounding box of the tile's z14 children that have >= --min-buildings
               buildings (a rectangle; clipping scattered tiles saved little in the pilot).

Commands (run from the repository root on Forth):

  python3 src/syria/ee_production.py plan [--years 2016 2017 ... | --city ALP] [--baseline A|B] [--boundary FILE --where FIELD=VALUE | --bbox W S E N]
        No Earth Engine. Builds the task list (data/syria/ee_run/plan.csv) and prints the number
        of tasks and z14 tiles, tiles skipped, and the expected cost in EECU-hours.

  python3 src/syria/ee_production.py submit --limit 20
        Starts up to 20 tasks (a first small batch) and records them in data/syria/ee_run/ledger.csv.

  python3 src/syria/ee_production.py status
        Updates the ledger from Earth Engine and prints counts and EECU-hours used.

  python3 src/syria/ee_production.py watch --max-active 300
        Keeps about 300 tasks active, resubmitting as they finish, until the plan is done or the
        EECU-hour limit (--max-eecu-hours, default 800) is reached. Safe to stop with Ctrl-C and
        start again: the ledger remembers every task.

Every command takes --tag NAME, which keeps a separate run apart: files in data/syria/ee_run_NAME/,
Drive folder syria_run_NAME, task names syria_NAME_<tile>. Without --tag the main run is used.

Output: Drive folder syria_run/, one GeoTIFF qk_<z13 quadkey>.tif per task with one band per window (w2016, ... or w01, w02, ...)
(Uint8 probability x 255, 0 = no data), same encoding as the local rasters.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

BASELINES = {"A": ("2014-10-06", "2015-10-05"), "B": ("2015-10-21", "2016-10-20")}
DEFAULT_YEARS = [2016, 2017, 2018, 2020, 2023, 2025]
MIN_BASELINE_IMAGES = 10
MIN_WINDOW_IMAGES = 2
TASK_ZOOM = 13
TASK_PREFIX = "syria_"
DRIVE_FOLDER = "syria_run"
OUT_DIR = Path("data/syria/ee_run")
# Cost model from the Aleppo pilots (EECU-seconds per z14 tile): first window ~59, each further
# window ~10.5 (low) to ~21 (high). Per orbit-count differences are not modelled.
COST_FIRST, COST_EXTRA_LOW, COST_EXTRA_HIGH = 59.0, 10.5, 21.0
ACTIVE = ("PENDING", "RUNNING", "READY", "CANCEL_REQUESTED")
DONE = ("SUCCEEDED",)
MAX_ATTEMPTS = 3


def configure(tag=""):
    """Point the module at a tagged run (separate plan, ledger, Drive folder and task names)."""
    global OUT_DIR, TASK_PREFIX, DRIVE_FOLDER
    tag = (tag or "").strip().lower()
    OUT_DIR = Path("data/syria/ee_run" + (f"_{tag}" if tag else ""))
    TASK_PREFIX = "syria_" + (f"{tag}_" if tag else "")
    DRIVE_FOLDER = "syria_run" + (f"_{tag}" if tag else "")


def window_for_year(year):
    return f"w{year}", (f"{year}-08-07", f"{year}-10-06")


def year_windows(years):
    """Windows as dicts: name (band name), period (name in the orbit survey), start, end."""
    out = []
    for y in years:
        name, (a, b) = window_for_year(y)
        out.append(dict(name=name, period=f"win{y}", start=a, end=b))
    return out


def city_windows(city):
    """The window list of a transfer city exactly as the local workflow has it, and its baseline key."""
    from src.data.transfer_cities.constants_transfer import TRANSFER_CITIES

    cfg = TRANSFER_CITIES[city.upper()]
    pre = tuple(cfg["pre_period"])
    key = next((k for k, v in BASELINES.items() if tuple(v) == pre), None)
    if key is None:
        sys.exit(f"{city}: its pre-period {pre} is neither baseline A nor B.")
    periods = [tuple(p) for p in cfg["post_periods"]]
    if not periods or periods[0] != pre:  # the constants already start with the pre-period as window 1 (T0); do not add it twice
        periods = [pre] + periods
    wins = [dict(name=f"w{i + 1:02d}", period=f"win{i + 1:02d}", start=a, end=b) for i, (a, b) in enumerate(periods)]
    return wins, key, str(cfg.get("conflict_start", ""))


def save_windows(meta):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "windows.json").write_text(json.dumps(meta, indent=2))


def load_windows():
    f = OUT_DIR / "windows.json"
    if f.exists():
        return json.loads(f.read_text())
    plan_f = OUT_DIR / "plan.csv"
    if plan_f.exists():  # a plan made before windows.json existed: year windows, baseline A
        cols = pd.read_csv(plan_f, nrows=0).columns
        years = sorted(int(c.split("_w")[1]) for c in cols if c.startswith("orbits_w") and c.split("_w")[1].isdigit())
        if years:
            meta = dict(baseline=list(BASELINES["A"]), baseline_key="A", windows=year_windows(years), city="", conflict_start="", bbox=None)
            save_windows(meta)
            return meta
    sys.exit(f"{f} not found. Run:  python3 src/syria/ee_production.py plan")


def quadkey_bounds(qk):
    z, x, y = len(qk), 0, 0
    for i, ch in enumerate(qk):
        bit = z - 1 - i
        d = int(ch)
        x |= (d & 1) << bit
        y |= ((d >> 1) & 1) << bit
    import math

    n = 2**z
    lat = lambda yy: math.degrees(math.atan(math.sinh(math.pi * (1 - 2 * yy / n))))
    return x / n * 360 - 180, lat(y + 1), (x + 1) / n * 360 - 180, lat(y)


# ----------------------------------------------------------------------------- region selection
def load_region(path, where=None):
    """
    One shapely geometry (EPSG:4326) from a boundary file (GeoJSON, GeoPackage, shapefile).
    where: optional list of "FIELD=VALUE" filters (value compared without case or spaces, any of several
    comma-separated values allowed), e.g. ["admin1Name_en=Aleppo"] picks the Aleppo governorate out of an ADM1 layer.
    """
    import geopandas as gpd
    import numpy as np
    import shapely

    gdf = gpd.read_file(path)
    for cond in where or []:
        field, _, value = cond.partition("=")
        field = field.strip()
        if field not in gdf.columns:
            sys.exit(f"{path} has no field '{field}'. Fields: {', '.join(c for c in gdf.columns if c != 'geometry')}")
        wanted = {v.strip().lower() for v in value.split(",")}
        hit = gdf[field].astype(str).str.strip().str.lower().isin(wanted)
        if not hit.any():
            vals = sorted(gdf[field].dropna().astype(str).unique())
            sys.exit(f"No feature with {field} = {value}. Values in that field: {', '.join(vals[:40])}{' ...' if len(vals) > 40 else ''}")
        gdf = gdf[hit]
    gdf = gdf[gdf.geometry.notna()]
    if gdf.empty:
        sys.exit(f"No geometry left in {path} after the filters.")
    if gdf.crs is not None and gdf.crs.to_epsg() != 4326:
        gdf = gdf.to_crs(4326)
    return shapely.union_all(shapely.make_valid(np.asarray(gdf.geometry.values)))


def tiles_in_region(quadkeys, bbox=None, region=None):
    """Boolean list over z14 quadkeys: the tile's centre is inside the box AND the tile touches the region polygon."""
    import numpy as np
    import shapely

    b = np.array([quadkey_bounds(q) for q in quadkeys], dtype=float).reshape(-1, 4)
    keep = np.ones(len(b), dtype=bool)
    if bbox is not None:
        w, s_, e, n = bbox
        keep &= (w <= (b[:, 0] + b[:, 2]) / 2) & ((b[:, 0] + b[:, 2]) / 2 <= e) & (s_ <= (b[:, 1] + b[:, 3]) / 2) & ((b[:, 1] + b[:, 3]) / 2 <= n)
    if region is not None and keep.any():
        boxes = shapely.box(b[:, 0], b[:, 1], b[:, 2], b[:, 3])
        shapely.prepare(region)
        keep &= shapely.intersects(region, boxes)
    return keep.tolist()


# ----------------------------------------------------------------------------- the plan
def build_plan(tiles, survey, windows, baseline_key, min_buildings, bbox=None, region=None):
    """
    tiles    : DataFrame with quadkey (z14) and n_buildings
    survey   : images_per_tile_orbit.csv (tile, orbit, n_full, period) at z13
    windows  : list of dicts with name and period (see year_windows / city_windows)
    bbox     : optional (west, south, east, north); only z14 tiles whose centre lies inside are kept
    region   : optional shapely geometry (an administrative boundary); only z14 tiles touching it are kept
    Returns (plan DataFrame, skipped DataFrame). plan has one row per task.
    """
    t = tiles[tiles.n_buildings >= min_buildings].copy()
    if bbox is not None or region is not None:
        t = t[tiles_in_region(t.quadkey.tolist(), bbox, region)]
    t["z13"] = t.quadkey.str[:TASK_ZOOM]
    survey = survey.copy()
    survey["tile"] = survey.tile.astype(str)
    base = survey[(survey.period == f"base{baseline_key}") & (survey.n_full >= MIN_BASELINE_IMAGES)]
    valid = base.groupby("tile").orbit.apply(lambda s: sorted(set(map(int, s)))).to_dict()
    win = survey[survey.period.isin([w["period"] for w in windows]) & (survey.n_full >= MIN_WINDOW_IMAGES)]
    per = {}
    for r in win.itertuples():
        per.setdefault((r.tile, r.period), set()).add(int(r.orbit))

    rows, skipped = [], []
    for z13, g in t.groupby("z13"):
        orbits = valid.get(z13, [])
        if not orbits:
            skipped.append((z13, len(g), "no valid baseline orbit"))
            continue
        bounds = [quadkey_bounds(q) for q in g.quadkey]
        row = {
            "z13": z13,
            "n_z14": len(g),
            "n_buildings": int(g.n_buildings.sum()),
            "west": min(b[0] for b in bounds), "south": min(b[1] for b in bounds),
            "east": max(b[2] for b in bounds), "north": max(b[3] for b in bounds),
            "baseline_orbits": ",".join(map(str, orbits)),
        }
        for w in windows:
            ok = sorted(per.get((z13, w["period"]), set()) & set(orbits))
            row[f"orbits_{w['name']}"] = ",".join(map(str, ok))
        rows.append(row)
    plan = pd.DataFrame(rows)
    return plan, pd.DataFrame(skipped, columns=["z13", "n_z14", "reason"])


def cost_range(n_z14, n_windows):
    low = n_z14 * (COST_FIRST + (n_windows - 1) * COST_EXTRA_LOW) / 3600
    high = n_z14 * (COST_FIRST + (n_windows - 1) * COST_EXTRA_HIGH) / 3600
    return low, high


def check_survey_periods(survey_path, key, windows, by_city):
    """
    The survey names its periods (win01 ...) and records their dates in periods.json next to the csv.
    A city's window list can change between versions, so the dates must agree with the plan's windows,
    or orbits would be taken from the wrong period.
    """
    f = Path(survey_path).parent / "periods.json"
    if not f.exists():
        if by_city:
            sys.exit(f"{survey_path} has no periods.json next to it, so I cannot check that its windows match this plan. "
                     f"Re-run orbit_survey.py with the same --city (a current copy of the script writes the file).")
        return
    rec = json.loads(f.read_text())
    wrong = [(n, rec.get(n), list(d)) for n, d in [(f"base{key}", BASELINES[key])] + [(w["period"], (w["start"], w["end"])) for w in windows]
             if n in rec and list(rec[n]) != list(d)]
    if wrong:
        sys.exit("The survey and the plan disagree on these periods (name, survey dates, plan dates): " + "; ".join(map(str, wrong))
                 + ". Re-run orbit_survey.py with the same --city.")


def cmd_plan(args):
    tiles = pd.read_csv(args.tiles_csv, dtype={"quadkey": str})
    survey = pd.read_csv(args.survey, dtype={"tile": str})
    if survey.tile.str.len().iloc[0] != TASK_ZOOM:
        sys.exit(f"{args.survey} is not a z{TASK_ZOOM} survey. Run: python3 src/syria/orbit_survey.py --zoom {TASK_ZOOM} ...")
    conflict_start = ""
    if args.city:
        windows, key, conflict_start = city_windows(args.city)
        if args.baseline and args.baseline.upper() != key:
            alt = args.baseline.upper()
            first = min(w["start"] for w in windows)
            if first <= BASELINES[alt][1]:
                sys.exit(f"{args.city} uses baseline {key}. Baseline {alt} could only replace it if every window started after {BASELINES[alt][1]}, "
                         f"but {args.city}'s windows start {first}. Drop --baseline or use --baseline {key}.")
            print(f"Note: using baseline {alt} {BASELINES[alt]} instead of {args.city}'s own baseline {key}.")
            key = alt
    else:
        windows, key = year_windows(args.years or DEFAULT_YEARS), (args.baseline or "A").upper()
    check_survey_periods(args.survey, key, windows, bool(args.city))
    have = set(survey.period.unique())
    missing = [p for p in [f"base{key}"] + [w["period"] for w in windows] if p not in have]
    if missing:
        sys.exit(f"The survey {args.survey} has no data for {missing}. Re-run orbit_survey.py with the same "
                 f"{'--city ' + args.city if args.city else '--windows'} and --zoom {TASK_ZOOM}.")
    bbox = tuple(args.bbox) if args.bbox else None
    boundary, where = getattr(args, "boundary", None), getattr(args, "where", None)
    region = load_region(boundary, where) if boundary else None
    plan, skipped = build_plan(tiles, survey, windows, key, args.min_buildings, bbox, region)
    if plan.empty:
        sys.exit("No tasks: no z14 tile with enough buildings (and a valid orbit) in that area.")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    plan.to_csv(OUT_DIR / "plan.csv", index=False)
    skipped.to_csv(OUT_DIR / "skipped.csv", index=False)
    save_windows(dict(baseline=list(BASELINES[key]), baseline_key=key, windows=windows, city=(args.city or "").upper(),
                      conflict_start=conflict_start, bbox=list(bbox) if bbox else None,
                      boundary=(str(boundary) if boundary else None), where=list(where or [])))
    n_z14 = int(plan.n_z14.sum())
    lo, hi = cost_range(n_z14, len(windows))
    print(f"Tasks (z13 tiles): {len(plan):,}   z14 tiles inside them: {n_z14:,}   windows: {len(windows)}   baseline {key} {BASELINES[key]}")
    if bbox:
        print(f"Region (west, south, east, north): {bbox}")
    if region is not None:
        print(f"Boundary: {boundary}{'  where ' + ', '.join(where) if where else ''}")
    print(f"Skipped (no valid baseline orbit): {len(skipped):,} z13 tiles, {int(skipped.n_z14.sum()) if len(skipped) else 0:,} z14 tiles")
    print(f"Expected cost: {lo:,.0f} to {hi:,.0f} EECU-hours  (Contributor quota: 1,000 per month)")
    for c in [c for c in plan.columns if c.startswith("orbits_")]:
        print(f"  {c}: tasks with no usable orbit in this window: {(plan[c] == '').sum():,}")
    print(f"\nWrote {OUT_DIR / 'plan.csv'}, skipped.csv and windows.json")
    return 0


# ----------------------------------------------------------------------------- the ledger
def load_ledger():
    f = OUT_DIR / "ledger.csv"
    if f.exists():
        return pd.read_csv(f, dtype={"z13": str, "state": str, "submitted": str}).set_index("z13")
    return pd.DataFrame(columns=["z13", "state", "eecu_s", "attempts", "submitted"]).astype({"z13": str}).set_index("z13")


def save_ledger(ledger):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    ledger.reset_index().to_csv(OUT_DIR / "ledger.csv", index=False)


def refresh_ledger(ledger, ops):
    """Update states and EECU from a list of Earth Engine operations (dicts with metadata)."""
    latest = {}
    for op in ops:
        meta = op.get("metadata", {}) or {}
        desc = meta.get("description", "")
        if not desc.startswith(TASK_PREFIX):
            continue
        z13 = desc[len(TASK_PREFIX):]
        if z13 not in ledger.index:
            continue
        stamp = meta.get("createTime") or meta.get("startTime") or ""
        if z13 not in latest or stamp >= latest[z13][0]:
            latest[z13] = (stamp, meta.get("state", "?"), meta.get("batchEecuUsageSeconds"))
    for z13, (_, state, eecu) in latest.items():
        ledger.loc[z13, "state"] = state
        if eecu is not None:
            ledger.loc[z13, "eecu_s"] = float(eecu)
    return ledger


def to_submit(plan, ledger, n_slots):
    """z13 ids to start now: never submitted, or FAILED/CANCELLED with attempts left."""
    out = []
    for z13 in plan.z13:
        if len(out) >= n_slots:
            break
        if z13 not in ledger.index:
            out.append(z13)
        else:
            st, att = ledger.loc[z13, "state"], int(ledger.loc[z13, "attempts"])
            if st in ("FAILED", "CANCELLED") and att < MAX_ATTEMPTS:
                out.append(z13)
    return out


def summary(plan, ledger):
    counts = ledger.state.value_counts().to_dict() if len(ledger) else {}
    used = float(ledger.eecu_s.fillna(0).sum()) / 3600 if len(ledger) else 0.0
    left = len(plan) - sum(counts.get(s, 0) for s in DONE)
    return counts, used, left


# ----------------------------------------------------------------------------- Earth Engine
def make_image(ee, classifier, row, meta):
    from src.data.sentinel1.collection import get_s1_collection
    from src.syria.ee_classify import window_probability

    baseline = tuple(meta["baseline"])
    region = ee.Geometry.Rectangle([row.west, row.south, row.east, row.north], proj="EPSG:4326", geodesic=False)
    last = max([baseline[1]] + [w["end"] for w in meta["windows"]])
    s1 = get_s1_collection(region, baseline[0], last)
    bands = []
    for w in meta["windows"]:
        orbits = [int(o) for o in str(getattr(row, f"orbits_{w['name']}")).split(",") if o != ""]
        if orbits:
            prob = window_probability(ee, classifier, s1, orbits, baseline, (w["start"], w["end"]))
            bands.append(prob.multiply(255).toUint8().rename(w["name"]))
        else:
            bands.append(ee.Image.constant(0).toUint8().rename(w["name"]))
    return ee.Image.cat(bands), region


def start_task(ee, classifier, row, meta):
    from src.syria.ee_classify import start_export

    image, region = make_image(ee, classifier, row, meta)
    start_export(ee, image, f"{TASK_PREFIX}{row.z13}", DRIVE_FOLDER, f"qk_{row.z13}", region)


def z13_with_labels(names, plan_z13):
    """z13 tiles of the plan that hold UNOSAT-labelled points of the named cities (ALP, RAQ)."""
    from src.syria import footprint_tiles as ft

    ids = set()
    for name in names:
        path = ft.labels_dir() / name.lower() / "unosat_labels.geojson"
        if not path.exists():
            sys.exit(f"{path} not found.")
        lon, lat = ft.read_points(path, "damage", ["1", "2"])
        x, y = ft.lonlat_to_tile(lon, lat, TASK_ZOOM)
        ids |= set(map(str, ft.quadkeys(x, y, TASK_ZOOM)))
    return sorted(ids & set(plan_z13))


def read_plan():
    f = OUT_DIR / "plan.csv"
    if not f.exists():
        sys.exit(f"{f} not found. Run:  python3 src/syria/ee_production.py plan")
    plan = pd.read_csv(f, dtype=str).fillna("")
    for c in ("n_z14", "n_buildings"):
        plan[c] = plan[c].astype(int)
    for c in ("west", "south", "east", "north"):
        plan[c] = plan[c].astype(float)
    return plan


def ee_ops(ee):
    try:
        return ee.data.listOperations()
    except Exception as err:
        sys.exit(f"Could not list Earth Engine operations ({err!r}). Nothing was submitted.")


def do_submit(args, ee, classifier, plan, ledger, ops=None):
    ops = ee_ops(ee) if ops is None else ops
    ledger = refresh_ledger(ledger, ops)
    only = getattr(args, "only", None)
    if only is not None:
        plan = plan[plan.z13.isin(only)].reset_index(drop=True)
    if getattr(args, "shuffle", False):
        plan = plan.sample(frac=1, random_state=0).reset_index(drop=True)  # same order every run
    active = int(ledger.state.isin(ACTIVE).sum()) if len(ledger) else 0
    counts, used, left = summary(plan, ledger)
    budget_left = args.max_eecu_hours - used
    slots = min(args.max_active - active, args.limit if args.limit else 10**9)
    meta = load_windows()
    pending_cost = active * (plan.n_z14.mean() * (COST_FIRST + (len(meta['windows']) - 1) * COST_EXTRA_HIGH)) / 3600
    if budget_left - pending_cost <= 0:
        print(f"EECU limit reached: used {used:,.1f} h (+ about {pending_cost:,.1f} h still running) of {args.max_eecu_hours:,.0f} h.")
        return 0, ledger
    todo = to_submit(plan, ledger, max(slots, 0))
    started = 0
    for z13 in todo:
        row = next(plan[plan.z13 == z13].itertuples())
        try:
            start_task(ee, classifier, row, meta)
        except Exception as err:
            print(f"  could not start {z13}: {err!r}")
            continue
        att = int(ledger.loc[z13, "attempts"]) + 1 if z13 in ledger.index else 1
        ledger.loc[z13, ["state", "attempts", "submitted"]] = ["PENDING", att, time.strftime("%Y-%m-%d %H:%M:%S")]
        started += 1
        if started % 25 == 0:
            save_ledger(ledger)
    save_ledger(ledger)
    return started, ledger


def cmd_submit(args):
    import ee

    from src.syria.forest_to_gee import classifier_from_table
    from src.syria.ee_classify import FOREST_ASSET
    from src.utils.gdrive import create_drive_folder
    from src.utils.gee import init_gee

    init_gee()
    plan, ledger = read_plan(), load_ledger()
    if getattr(args, "labels", None):
        args.only = z13_with_labels(args.labels, plan.z13)
        print(f"{len(args.only)} tasks hold UNOSAT-labelled points of {', '.join(args.labels)}.")
    try:
        create_drive_folder(DRIVE_FOLDER)
    except Exception:
        pass
    classifier = classifier_from_table(ee.FeatureCollection(FOREST_ASSET), args.n_trees)
    started, ledger = do_submit(args, ee, classifier, plan, ledger)
    counts, used, left = summary(plan, ledger)
    print(f"Started {started} task(s). States: {counts}.  EECU used so far: {used:,.2f} h.  Not yet succeeded: {left:,}")
    return 0


def cmd_status(args):
    import ee

    from src.utils.gee import init_gee

    init_gee()
    plan, ledger = read_plan(), load_ledger()
    ledger = refresh_ledger(ledger, ee_ops(ee))
    save_ledger(ledger)
    counts, used, left = summary(plan, ledger)
    print(f"Plan: {len(plan):,} tasks.  Ledger states: {counts}")
    done = ledger[ledger.state.isin(DONE)].dropna(subset=["eecu_s"])
    if len(done):
        tiles_done = int(plan.set_index("z13").loc[done.index, "n_z14"].sum())
        per_tile = done.eecu_s.sum() / tiles_done
        print(f"Finished {len(done):,} tasks ({tiles_done:,} z14 tiles): {done.eecu_s.sum() / 3600:,.2f} EECU-h, {per_tile:,.0f} EECU-s per z14 tile")
        print(f"Projected for the whole plan ({int(plan.n_z14.sum()):,} z14 tiles): {per_tile * plan.n_z14.sum() / 3600:,.0f} EECU-h"
              f"  (plan estimate {cost_range(int(plan.n_z14.sum()), len(load_windows()['windows']))[0]:,.0f}-{cost_range(int(plan.n_z14.sum()), len(load_windows()['windows']))[1]:,.0f};"
              f" limit {args.max_eecu_hours:,.0f} per month)")
    print(f"EECU-hours used: {used:,.2f}.  Not yet succeeded: {left:,}")
    return 0


def cmd_watch(args):
    import ee

    from src.syria.forest_to_gee import classifier_from_table
    from src.syria.ee_classify import FOREST_ASSET
    from src.utils.gdrive import create_drive_folder
    from src.utils.gee import init_gee

    init_gee()
    plan, ledger = read_plan(), load_ledger()
    if getattr(args, "labels", None):
        args.only = z13_with_labels(args.labels, plan.z13)
        print(f"{len(args.only)} tasks hold UNOSAT-labelled points of {', '.join(args.labels)}.")
    try:
        create_drive_folder(DRIVE_FOLDER)
    except Exception:
        pass
    classifier = classifier_from_table(ee.FeatureCollection(FOREST_ASSET), args.n_trees)
    try:
        while True:
            started, ledger = do_submit(args, ee, classifier, plan, ledger)
            sel = plan if getattr(args, "only", None) is None else plan[plan.z13.isin(args.only)]
            counts, used, left = summary(sel, ledger)
            print(f"{time.strftime('%H:%M:%S')}  started {started}  states {counts}  used {used:,.1f} h  left {left:,}", flush=True)
            if left == 0:
                print("All tasks have succeeded.")
                return 0
            retry_left = len(to_submit(plan, ledger, 1))
            active = int(ledger.state.isin(ACTIVE).sum())
            if active == 0 and not retry_left and started == 0:
                print("Nothing is running and nothing can be resubmitted (EECU limit or three failed attempts).")
                return 1
            time.sleep(args.interval)
    except KeyboardInterrupt:
        save_ledger(ledger)
        print("\nStopped. The ledger is saved; run watch again to carry on.")
        return 1


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.strip().split("\n\n")[0])
    sub = p.add_subparsers(dest="cmd", required=True)

    def common(sp):
        sp.add_argument("--max-eecu-hours", type=float, default=800)
        sp.add_argument("--n-trees", type=int, default=50)
        sp.add_argument("--tag", default="", help="name of a separate run (default: the main run)")

    sp = sub.add_parser("plan")
    sp.add_argument("--tiles-csv", default="data/syria/footprints/tiles_z14.csv")
    sp.add_argument("--survey", default="data/syria/orbit_survey_z13/images_per_tile_orbit.csv")
    sp.add_argument("--min-buildings", type=int, default=10)
    sp.add_argument("--years", nargs="+", type=int, help=f"window years (default {DEFAULT_YEARS})")
    sp.add_argument("--city", help="use this transfer city's windows and baseline (ALP or RAQ) instead of --years")
    sp.add_argument("--baseline", choices=["A", "B", "a", "b"], help="A = Oct 2014-Oct 2015 (default), B = Oct 2015-Oct 2016")
    sp.add_argument("--bbox", nargs=4, type=float, metavar=("W", "S", "E", "N"), help="keep only tasks inside this box")
    sp.add_argument("--boundary", help="keep only z14 tiles touching this boundary file (an administrative layer)")
    sp.add_argument("--where", nargs="+", metavar="FIELD=VALUE", help="pick features of the boundary file, e.g. admin1Name_en=Aleppo")
    sp.add_argument("--tag", default="", help="name of a separate run (default: the main run)")
    sp = sub.add_parser("submit"); common(sp)
    sp.add_argument("--limit", type=int, default=0, help="start at most this many tasks")
    sp.add_argument("--max-active", type=int, default=300)
    sp.add_argument("--shuffle", action="store_true", help="pick tasks in a fixed random order (a spread-out test batch)")
    sp.add_argument("--labels", nargs="+", metavar="CITY", help="only tasks that hold UNOSAT-labelled points of these cities, e.g. ALP RAQ")
    sp = sub.add_parser("status"); common(sp)
    sp = sub.add_parser("watch"); common(sp)
    sp.add_argument("--max-active", type=int, default=300)
    sp.add_argument("--limit", type=int, default=0)
    sp.add_argument("--shuffle", action="store_true")
    sp.add_argument("--labels", nargs="+", metavar="CITY")
    sp.add_argument("--interval", type=int, default=300, help="seconds between checks")
    args = p.parse_args(argv)
    configure(getattr(args, "tag", ""))
    return {"plan": cmd_plan, "submit": cmd_submit, "status": cmd_status, "watch": cmd_watch}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
