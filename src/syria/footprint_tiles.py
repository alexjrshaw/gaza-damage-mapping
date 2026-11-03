"""
Tile list for the Syria run: the zoom-14 tiles that hold buildings.

Classifying every pixel of Syria would spend most of the compute on desert, so the run is
restricted to tiles that contain buildings. Overture Maps building footprints (a merge of
OpenStreetMap, Microsoft and Google footprints) decide which tiles hold buildings. Tiles
that the footprints miss are added where GHSL (2015 epoch) or ESA WorldCover show built-up
land, and where a gazetteer of populated places (for example OCHA's) lists a settlement. Only
the footprint boxes are used, so the download reads a few columns instead of the
full geometries (hundreds of MB instead of tens of GB).

Usage, from the repository root:
    python -m src.syria.footprint_tiles selftest    Offline checks of the tile maths and file handling
    python -m src.syria.footprint_tiles download    Overture building boxes for Syria -> local Parquet (login node)
    python -m src.syria.footprint_tiles builtup     GHSL and WorldCover built-up area per footprint-free tile (Earth Engine)
    python -m src.syria.footprint_tiles tiles       Final tile list: CSV, summary and optional GeoJSON
    python -m src.syria.footprint_tiles coverage    Share of UNOSAT-damaged buildings (Aleppo, Raqqa) inside the list

Order: download, builtup (optional), tiles, coverage. `builtup`, `tiles` and `coverage` need a
Syria boundary polygon (any GeoJSON, GeoPackage or shapefile, for example ADM0 from geoBoundaries
or OCHA COD-AB). What the polygon contains decides whether the Israeli-held Golan Heights are included.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator

import numpy as np
import pandas as pd

DEFAULT_ZOOM = 14
DEFAULT_RELEASE = "2026-10-21.0"
S3_BUCKET = "overturemaps-us-west-2"
S3_REGION = "us-west-2"
STAC_URL = "https://stac.overturemaps.org/catalog.json"
SYRIA_BBOX = (35.5, 32.2, 42.5, 37.4)  # west, south, east, north; generous box round the country
BBOX_FIELDS = ("xmin", "ymin", "xmax", "ymax")

DEFAULT_PROJECT = "gaza-damage-mapping"
EE_HIGH_VOLUME_URL = "https://earthengine-highvolume.googleapis.com"
GHSL_ASSET = "JRC/GHSL/P2023A/GHS_BUILT_S"  # Built-up surface, m2 per 100 m cell, one image per epoch
WORLDCOVER_ASSET = "ESA/WorldCover/v200"  # 10 m land cover for 2021, band "Map"
WORLDCOVER_BUILT = 50

MAX_LAT = 85.0511287798066
TOUCH_EPS = 1e-9  # Degrees; a box that only touches a tile edge still counts in that tile (inclusive, so nothing is lost)
EARTH_RADIUS_KM = 6371.0088

BUILDINGS_FILE = "overture_buildings_bbox.parquet"
BUILTUP_FILE = "builtup_candidates_z{zoom}.csv"
BUILTUP_HEADER = "quadkey,x,y,ghsl_m2,worldcover_m2"  # An earlier version wrote wc_m2, which was 20-25% too high
TILES_FILE = "tiles_z{zoom}.csv"

# Places used by `builtup --check`: name, longitude, latitude, and whether the tile should be mostly built-up
# (the three city centres) or almost empty (the desert).
CHECK_POINTS = [
    ("Aleppo centre", 37.16, 36.20, True),
    ("Raqqa centre", 39.01, 35.95, True),
    ("Damascus centre", 36.29, 33.51, True),
    ("Open desert east of Palmyra", 39.60, 34.20, False),
]


# Paths


def default_dir() -> Path:
    try:
        from src.constants import DATA_PATH
    except Exception:
        return Path("data") / "syria" / "footprints"
    return Path(DATA_PATH) / "syria" / "footprints"


def labels_dir() -> Path:
    try:
        from src.constants import PROJECT_PATH
    except Exception:
        return Path("test_sites") / "processed"
    return Path(PROJECT_PATH) / "test_sites" / "processed"


# Tile maths (Web Mercator slippy-map tiles, the same scheme as Bing quadkeys)


def lonlat_to_tile(lon, lat, zoom: int) -> tuple[np.ndarray, np.ndarray]:
    """Tile column x and row y (row 0 is the northernmost) containing each point."""
    n = 1 << zoom
    lon = np.asarray(lon, dtype=np.float64)
    lat = np.clip(np.asarray(lat, dtype=np.float64), -MAX_LAT, MAX_LAT)
    x = np.floor((lon + 180.0) / 360.0 * n)
    y = np.floor((1.0 - np.arcsinh(np.tan(np.radians(lat))) / np.pi) / 2.0 * n)
    return np.clip(x, 0, n - 1).astype(np.int64), np.clip(y, 0, n - 1).astype(np.int64)


def tile_bounds(x, y, zoom: int):
    """West, south, east, north in degrees for each tile."""
    n = float(1 << zoom)
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    west = x / n * 360.0 - 180.0
    east = (x + 1.0) / n * 360.0 - 180.0
    north = np.degrees(np.arctan(np.sinh(np.pi * (1.0 - 2.0 * y / n))))
    south = np.degrees(np.arctan(np.sinh(np.pi * (1.0 - 2.0 * (y + 1.0) / n))))
    return west, south, east, north


def tile_area_km2(x, y, zoom: int) -> np.ndarray:
    """Tile area on a sphere (within about 0.3% of the ellipsoidal area)."""
    west, south, east, north = tile_bounds(x, y, zoom)
    return EARTH_RADIUS_KM**2 * np.radians(east - west) * (np.sin(np.radians(north)) - np.sin(np.radians(south)))


def quadkeys(x, y, zoom: int) -> np.ndarray:
    """Bing quadkey string for each tile."""
    x = np.asarray(x, dtype=np.int64)
    y = np.asarray(y, dtype=np.int64)
    shifts = np.arange(zoom - 1, -1, -1, dtype=np.int64)
    digits = ((x[:, None] >> shifts) & 1) + 2 * ((y[:, None] >> shifts) & 1)
    chars = np.ascontiguousarray((digits + ord("0")).astype(np.uint8))
    return chars.view(f"S{zoom}").ravel().astype(str)


def quadkey_to_tile(quadkey: str) -> tuple[int, int, int]:
    x = y = 0
    for ch in quadkey:
        x = (x << 1) | (int(ch) & 1)
        y = (y << 1) | (int(ch) >> 1)
    return x, y, len(quadkey)


# Boundary and tile grid


def load_boundary(path: str | Path):
    """All polygons in the file merged into one shapely geometry (EPSG:4326)."""
    import geopandas as gpd
    import shapely

    gdf = gpd.read_file(path)
    gdf = gdf[gdf.geometry.notna()]
    if gdf.empty:
        raise SystemExit(f"No geometries in {path}")
    if gdf.crs is not None and gdf.crs.to_epsg() != 4326:
        gdf = gdf.to_crs(4326)
    geoms = shapely.make_valid(np.asarray(gdf.geometry.values))
    return shapely.union_all(geoms)


def area_km2(geom) -> float:
    import shapely
    from pyproj import Transformer

    transformer = Transformer.from_crs(4326, 6933, always_xy=True)  # Equal-area projection

    def project(coords):
        x, y = transformer.transform(coords[:, 0], coords[:, 1])
        return np.column_stack([x, y])

    return shapely.transform(geom, project).area / 1e6


class Grid:
    """Every tile in the boundary's bounding box, with a mask of those touching the boundary."""

    def __init__(self, zoom: int, x0: int, y0: int, inside: np.ndarray):
        self.zoom = zoom
        self.x0 = x0
        self.y0 = y0
        self.inside = inside

    @property
    def shape(self) -> tuple[int, int]:
        return self.inside.shape

    def xy(self, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        rows, cols = np.nonzero(mask)
        return cols + self.x0, rows + self.y0


def make_grid(boundary, zoom: int) -> Grid:
    import shapely

    west, south, east, north = boundary.bounds
    x0, y_south = lonlat_to_tile(west, south, zoom)
    x1, y_north = lonlat_to_tile(east, north, zoom)
    xs = np.arange(int(x0), int(x1) + 1)
    ys = np.arange(int(y_north), int(y_south) + 1)
    gx, gy = np.meshgrid(xs, ys)
    w, s, e, n = tile_bounds(gx.ravel(), gy.ravel(), zoom)
    boxes = shapely.box(w, s, e, n)
    shapely.prepare(boundary)
    inside = shapely.intersects(boundary, boxes).reshape(gx.shape)
    return Grid(zoom, int(xs[0]), int(ys[0]), inside)


def dilate(mask: np.ndarray, rings: int) -> np.ndarray:
    """Grow a boolean tile mask by `rings` tiles in all eight directions."""
    out = mask.copy()
    for _ in range(rings):
        p = np.pad(out, 1)
        out = (
            p[:-2, :-2] | p[:-2, 1:-1] | p[:-2, 2:]
            | p[1:-1, :-2] | p[1:-1, 1:-1] | p[1:-1, 2:]
            | p[2:, :-2] | p[2:, 1:-1] | p[2:, 2:]
        )
    return out


# Reading building boxes


def detect_mode(names: list[str]) -> tuple[str, list[str]]:
    """How the box is stored: four flat columns, an Overture-style bbox struct, or only a WKB geometry."""
    if all(f in names for f in BBOX_FIELDS):
        return "flat", list(BBOX_FIELDS)
    if "bbox" in names:
        return "struct", ["bbox"]
    if "geometry" in names:
        return "wkb", ["geometry"]
    raise SystemExit(f"Found neither bbox nor geometry columns. Columns: {names}")


def _numpy(column) -> np.ndarray:
    if isinstance(column, np.ndarray):
        return column.astype(np.float64)
    return np.asarray(column.to_numpy(zero_copy_only=False), dtype=np.float64)


def bounds_from_batch(batch, mode: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """xmin, ymin, xmax, ymax as float64 arrays; rows without a finite box are dropped."""
    if mode == "flat":
        cols = [batch.column(f) for f in BBOX_FIELDS]
    elif mode == "struct":
        struct = batch.column("bbox")
        by_name = {field.name: arr for field, arr in zip(struct.type, struct.flatten())}
        cols = [by_name[f] for f in BBOX_FIELDS]
    else:
        import shapely

        geoms = shapely.from_wkb(batch.column("geometry").to_numpy(zero_copy_only=False))
        b = shapely.bounds(geoms)
        cols = [b[:, 0], b[:, 1], b[:, 2], b[:, 3]]
    arrays = [_numpy(c) for c in cols]
    ok = np.isfinite(arrays[0]) & np.isfinite(arrays[1]) & np.isfinite(arrays[2]) & np.isfinite(arrays[3])
    return tuple(a[ok] for a in arrays)


def iter_bounds(path: str | Path, batch_size: int = 500_000, window: tuple[float, float, float, float] | None = None) -> Iterator[tuple]:
    """Building boxes from a Parquet file or folder, batch by batch, optionally limited to a west, south, east, north window."""
    import pyarrow.dataset as ds

    dataset = ds.dataset(str(path), format="parquet")
    mode, cols = detect_mode(dataset.schema.names)
    for batch in dataset.to_batches(columns=cols, batch_size=batch_size):
        xmin, ymin, xmax, ymax = bounds_from_batch(batch, mode)
        if window is not None:
            w, s, e, n = window
            keep = (xmax >= w) & (xmin <= e) & (ymax >= s) & (ymin <= n)
            xmin, ymin, xmax, ymax = xmin[keep], ymin[keep], xmax[keep], ymax[keep]
        yield xmin, ymin, xmax, ymax


def tile_counts(batches, zoom: int, max_span: int = 100):
    """Buildings per tile; a building counts in every tile its box overlaps or touches.

    Returns tile x, tile y, count arrays and a stats dict. Boxes spanning more than `max_span`
    tiles (or with reversed corners) are skipped and counted under `invalid`.
    """
    n = 1 << zoom
    keys, counts = [], []
    stats = {"rows": 0, "multi_tile": 0, "invalid": 0}
    for xmin, ymin, xmax, ymax in batches:
        stats["rows"] += len(xmin)
        x0, y_south = lonlat_to_tile(xmin - TOUCH_EPS, ymin - TOUCH_EPS, zoom)
        x1, y_north = lonlat_to_tile(xmax + TOUCH_EPS, ymax + TOUCH_EPS, zoom)
        span = np.where((x1 >= x0) & (y_south >= y_north), (x1 - x0 + 1) * (y_south - y_north + 1), 0)
        stats["invalid"] += int(((span < 1) | (span > max_span)).sum())
        single = span == 1
        parts = [x0[single] * n + y_north[single]]
        multi = np.flatnonzero((span > 1) & (span <= max_span))
        stats["multi_tile"] += len(multi)
        if len(multi):
            extra = [
                xx * n + yy
                for i in multi
                for xx in range(int(x0[i]), int(x1[i]) + 1)
                for yy in range(int(y_north[i]), int(y_south[i]) + 1)
            ]
            parts.append(np.asarray(extra, dtype=np.int64))
        k = np.concatenate(parts)
        if len(k):
            u, c = np.unique(k, return_counts=True)
            keys.append(u)
            counts.append(c)
    if not keys:
        empty = np.zeros(0, dtype=np.int64)
        return empty, empty, empty, stats
    all_keys = np.concatenate(keys)
    all_counts = np.concatenate(counts)
    unique, inverse = np.unique(all_keys, return_inverse=True)
    total = np.bincount(inverse, weights=all_counts).astype(np.int64)
    return unique // n, unique % n, total, stats


def counts_on_grid(grid: Grid, x, y, count) -> np.ndarray:
    rows = np.asarray(y) - grid.y0
    cols = np.asarray(x) - grid.x0
    ny, nx = grid.shape
    ok = (rows >= 0) & (rows < ny) & (cols >= 0) & (cols < nx)
    out = np.zeros(grid.shape, dtype=np.int64)
    np.add.at(out, (rows[ok], cols[ok]), np.asarray(count)[ok])
    return out


def footprint_tiles(boundary, buildings: Path, zoom: int, min_buildings: int):
    """Grid, buildings per tile, mask of footprint tiles inside the boundary, and read statistics."""
    grid = make_grid(boundary, zoom)
    x, y, count, stats = tile_counts(iter_bounds(buildings), zoom)
    counts = counts_on_grid(grid, x, y, count)
    fp = grid.inside & (counts >= min_buildings)
    return grid, counts, fp, stats


# Command: download


def resolve_release(release: str) -> str:
    if release != "latest":
        return release
    import urllib.request

    with urllib.request.urlopen(STAC_URL, timeout=30) as response:
        return json.load(response)["latest"]


def parse_bbox(text: str) -> tuple[float, float, float, float]:
    try:
        w, s, e, n = (float(v) for v in text.split(","))
    except ValueError:
        raise SystemExit(f"--bbox needs west,south,east,north; got {text!r}")
    return w, s, e, n


def open_overture(args, release: str):
    import pyarrow.dataset as ds
    import pyarrow.fs as pafs

    fmt = ds.ParquetFileFormat(default_fragment_scan_options=ds.ParquetFragmentScanOptions(pre_buffer=True))
    if args.source_path:
        return ds.dataset(args.source_path, format=fmt)
    options = dict(anonymous=True, region=S3_REGION, connect_timeout=30, request_timeout=120)
    proxy = args.proxy or os.environ.get("HTTPS_PROXY") or os.environ.get("https_proxy")
    if proxy and proxy.lower() == "none":
        proxy = None
    if proxy:
        options["proxy_options"] = proxy
        print(f"Using proxy {proxy}")
    fs = pafs.S3FileSystem(**options)
    root = f"{S3_BUCKET}/release/{release}/theme=buildings/type=building/"
    try:
        return ds.dataset(root, filesystem=fs, format=fmt)
    except Exception as exc:
        print(f"Could not open s3://{root}\n  {exc}")
        for prefix in (f"{S3_BUCKET}/release/", f"{S3_BUCKET}/release/{release}/", f"{S3_BUCKET}/release/{release}/theme=buildings/"):
            try:
                names = sorted(Path(i.path).name for i in fs.get_file_info(pafs.FileSelector(prefix, recursive=False)))
                print(f"  Contents of {prefix}: {', '.join(names[-12:])}")
            except Exception as inner:
                print(f"  Could not list {prefix}: {inner}")
        raise SystemExit("Check the release name (see https://docs.overturemaps.org/getting-data) and your internet access.")


def flatten_batch(batch, extras: list[str]):
    import pyarrow as pa

    struct = batch.column("bbox")
    by_name = {field.name: arr for field, arr in zip(struct.type, struct.flatten())}
    arrays = [by_name[f] for f in BBOX_FIELDS] + [batch.column(name) for name in extras]
    return pa.RecordBatch.from_arrays(arrays, names=list(BBOX_FIELDS) + extras)


def cmd_download(args) -> int:
    import pyarrow as pa
    import pyarrow.compute as pc
    import pyarrow.parquet as pq

    release = resolve_release(args.release)
    if args.bbox:
        bbox = parse_bbox(args.bbox)
    elif args.boundary:
        w, s, e, n = load_boundary(args.boundary).bounds
        bbox = (w - 0.05, s - 0.05, e + 0.05, n + 0.05)
    else:
        bbox = SYRIA_BBOX
    extras = [c for c in (args.extra_columns or "").split(",") if c]
    out = Path(args.out) if args.out else default_dir() / BUILDINGS_FILE
    out.parent.mkdir(parents=True, exist_ok=True)
    pa.set_io_thread_count(args.io_threads)

    source = args.source_path or f"s3://{S3_BUCKET}/release/{release}/theme=buildings/type=building/"
    print(f"Source: {source}\nBox: west {bbox[0]}, south {bbox[1]}, east {bbox[2]}, north {bbox[3]}")
    if int(pa.__version__.split(".")[0]) < 14:
        print(f"WARNING: pyarrow {pa.__version__} is old; it may fail to skip row groups outside the box and read far too much. Upgrade to 14 or later.")
    dataset = open_overture(args, release)
    schema = dataset.schema
    files = getattr(dataset, "files", None)
    if files:
        print(f"{len(files)} Parquet files found")
    print("Columns: " + ", ".join(f"{f.name}: {f.type}" for f in schema))
    missing = [c for c in ["bbox"] + extras if c not in schema.names]
    if missing:
        raise SystemExit(f"Column(s) not in the dataset: {missing}. Paste the column list above back to me if the schema has changed.")
    bbox_type = schema.field("bbox").type
    members = [bbox_type.field(i).name for i in range(bbox_type.num_fields)] if pa.types.is_struct(bbox_type) else []
    if not all(f in members for f in BBOX_FIELDS):
        raise SystemExit(f"bbox is not a struct with {BBOX_FIELDS} (it is {bbox_type}). Paste the column list above back to me.")

    west, south, east, north = bbox
    window = (
        (pc.field("bbox", "xmin") <= east)
        & (pc.field("bbox", "xmax") >= west)
        & (pc.field("bbox", "ymin") <= north)
        & (pc.field("bbox", "ymax") >= south)
    )
    scanner = dataset.scanner(columns=["bbox"] + extras, filter=window, batch_size=args.batch_size, use_threads=True)
    print("Reading file footers and skipping row groups outside the box; the first rows can take a few minutes to appear.")

    tmp = out.with_name(out.name + ".part")
    writer = None
    rows = 0
    start = last_print = time.time()
    try:
        for batch in scanner.to_batches():
            if batch.num_rows == 0:
                continue
            flat = flatten_batch(batch, extras)
            if writer is None:
                writer = pq.ParquetWriter(tmp, flat.schema, compression="zstd")
            writer.write_batch(flat)
            rows += flat.num_rows
            if time.time() - last_print > 10:
                print(f"  {rows:,} buildings after {time.time() - start:.0f} s")
                last_print = time.time()
    except BaseException:
        if writer is not None:
            writer.close()
        tmp.unlink(missing_ok=True)
        raise
    if writer is None:
        raise SystemExit("No buildings fell inside the box. Check --bbox and --release.")
    writer.close()
    os.replace(tmp, out)

    meta = {
        "release": release,
        "source": source,
        "bbox_west_south_east_north": list(bbox),
        "buildings": rows,
        "columns": list(pq.read_schema(out).names),
        "file_mb": round(out.stat().st_size / 1e6, 1),
        "downloaded_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "seconds": round(time.time() - start),
        "pyarrow": pa.__version__,
    }
    out.with_suffix(".json").write_text(json.dumps(meta, indent=2))
    print(f"\n{rows:,} buildings -> {out} ({meta['file_mb']} MB, {meta['seconds']} s)\nRelease {release}. Details in {out.with_suffix('.json').name}.")
    return 0


# Command: builtup (Earth Engine)


def start_ee(project: str):
    import ee

    try:
        ee.Initialize(project=project, opt_url=EE_HIGH_VOLUME_URL)
    except Exception:
        try:
            ee.Initialize(project=project)
        except Exception as exc:
            raise SystemExit(f"Could not start Earth Engine ({exc}).\nRun `earthengine authenticate` on the login node, then try again.")
    return ee


def builtup_images(ee, epoch: int):
    ghsl = ee.Image(f"{GHSL_ASSET}/{epoch}").select("built_surface")
    # Ground area, not pixel counts: a 20 m pixel on WorldCover's lat/lon grid is only about 16 m wide at Syria's
    # latitudes, so counting pixels and multiplying by 400 m2 would be 20-25% too high.
    worldcover = ee.ImageCollection(WORLDCOVER_ASSET).first().select("Map").eq(WORLDCOVER_BUILT).multiply(ee.Image.pixelArea())
    return ghsl, worldcover


def builtup_chunk(ee, images, xs, ys, zoom: int, wc_scale: int) -> list[tuple]:
    """GHSL built-up area (m2) and WorldCover built-up area (m2) for one batch of tiles."""
    west, south, east, north = tile_bounds(xs, ys, zoom)
    features = [
        ee.Feature(
            ee.Geometry.Rectangle(coords=[float(w), float(s), float(e), float(n)], proj="EPSG:4326", geodesic=False),
            {"x": int(x), "y": int(y)},
        )
        for x, y, w, s, e, n in zip(xs, ys, west, south, east, north)
    ]
    collection = ee.FeatureCollection(features)

    def total(image, scale):
        reduced = image.reduceRegions(collection=collection, reducer=ee.Reducer.sum(), scale=scale)
        info = reduced.select(["x", "y", "sum"], retainGeometry=False).getInfo()
        return {(int(f["properties"]["x"]), int(f["properties"]["y"])): float(f["properties"].get("sum") or 0.0) for f in info["features"]}

    ghsl, worldcover = images
    g = total(ghsl, 100)
    w = total(worldcover, wc_scale)
    return [(int(x), int(y), g.get((int(x), int(y)), 0.0), w.get((int(x), int(y)), 0.0)) for x, y in zip(xs, ys)]


def validate_images(ee, images) -> None:
    """Fail at once, with the asset name, if a dataset or band is not what the code expects."""
    for label, image in zip((f"{GHSL_ASSET} band built_surface", f"{WORLDCOVER_ASSET} band Map"), images):
        try:
            image.bandNames().getInfo()
        except Exception as exc:
            raise SystemExit(f"Could not read {label} in Earth Engine: {str(exc)[:300]}")


def with_retries(fn, attempts: int = 5, base: float = 3.0):
    for k in range(attempts):
        try:
            return fn()
        except Exception as exc:
            if k == attempts - 1:
                raise
            wait = base * 2**k + random.random()
            print(f"  retry {k + 1}/{attempts - 1} in {wait:.0f} s ({str(exc)[:120]})")
            time.sleep(wait)


def cmd_builtup(args) -> int:
    ee = start_ee(args.project)
    images = builtup_images(ee, args.epoch)
    validate_images(ee, images)

    if args.check:
        print(f"Smoke test: built-up area (m2) in the zoom-{args.zoom} tile under each point")
        lon = np.array([p[1] for p in CHECK_POINTS])
        lat = np.array([p[2] for p in CHECK_POINTS])
        xs, ys = lonlat_to_tile(lon, lat, args.zoom)
        tile_m2 = tile_area_km2(xs, ys, args.zoom) * 1e6
        rows = with_retries(lambda: builtup_chunk(ee, images, xs, ys, args.zoom, args.wc_scale))
        print(f"  {'place':30s}{'tile':>12s}{'GHSL ' + str(args.epoch):>14s}{'WorldCover':>14s}")
        problems = []
        for (name, _, _, is_city), (_, _, g, w), area in zip(CHECK_POINTS, rows, tile_m2):
            print(f"  {name:30s}{area:12,.0f}{g:14,.0f}{w:14,.0f}   ({100 * g / area:.0f}% and {100 * w / area:.0f}% of the tile)")
            if max(g, w) > 1.02 * area:
                problems.append(f"{name}: more built-up area than the tile has, so the units are wrong")
            elif is_city and (g < 0.10 * area or w < 0.30 * area):
                problems.append(f"{name}: a city centre should be mostly built-up")
            elif not is_city and max(g, w) > 0.20 * area:
                problems.append(f"{name}: the desert tile should be almost empty")
        if problems:
            print("WARNING: " + "; ".join(problems) + ". Do not run the full pass yet; send this output back.")
            return 1
        print("Looks right: the city centres are mostly built-up and the desert tile is almost empty.")
        return 0

    if not args.boundary:
        raise SystemExit("builtup needs --boundary (a Syria polygon file)")
    buildings = Path(args.buildings) if args.buildings else default_dir() / BUILDINGS_FILE
    if not buildings.exists():
        raise SystemExit(f"{buildings} not found. Run `download` first.")
    boundary = load_boundary(args.boundary)
    grid, counts, fp, stats = footprint_tiles(boundary, buildings, args.zoom, args.min_buildings)
    candidates = grid.inside & ~fp
    cx, cy = grid.xy(candidates)
    print(f"{int(grid.inside.sum()):,} tiles touch Syria; {int(fp.sum()):,} have footprints; {len(cx):,} have none and will be checked")

    out = Path(args.out) if args.out else default_dir() / BUILTUP_FILE.format(zoom=args.zoom)
    out.parent.mkdir(parents=True, exist_ok=True)
    done: set[tuple[int, int]] = set()
    header = out.read_text().splitlines()[:1] if out.exists() else []
    if header and header[0] != BUILTUP_HEADER:
        raise SystemExit(
            f"{out.name} was made by an older version of this script, whose WorldCover areas were 20-25% too high "
            f"(its columns: {header[0]}). Delete it and run `builtup` again."
        )
    if header:
        previous = pd.read_csv(out)
        done = set(zip(previous["x"].astype(int), previous["y"].astype(int)))
        print(f"Resuming: {len(done):,} tiles already in {out.name}")
    else:
        out.write_text(BUILTUP_HEADER + "\n")
    todo = [(int(x), int(y)) for x, y in zip(cx, cy) if (int(x), int(y)) not in done]
    chunks = [todo[i : i + args.chunk] for i in range(0, len(todo), args.chunk)]
    print(f"{len(todo):,} tiles in {len(chunks)} requests, {args.workers} at a time")

    def work(chunk):
        xs = np.array([c[0] for c in chunk])
        ys = np.array([c[1] for c in chunk])
        return with_retries(lambda: builtup_chunk(ee, images, xs, ys, args.zoom, args.wc_scale))

    failed = 0
    finished = 0
    start = time.time()
    pool = ThreadPoolExecutor(max_workers=args.workers)
    try:
        with open(out, "a") as handle:
            futures = [pool.submit(work, chunk) for chunk in chunks]
            for future in as_completed(futures):
                try:
                    rows = future.result()
                except Exception as exc:
                    failed += 1
                    print(f"  request failed for good: {str(exc)[:200]}")
                    if failed >= 3 and finished == 0:
                        print("Three requests in a row failed and none succeeded, so something is wrong; stopping.")
                        break
                    continue
                keys = quadkeys([r[0] for r in rows], [r[1] for r in rows], args.zoom)
                for key, (x, y, g, w) in zip(keys, rows):
                    handle.write(f"{key},{x},{y},{g:.1f},{w:.1f}\n")
                handle.flush()
                finished += len(rows)
                print(f"  {finished:,}/{len(todo):,} tiles after {time.time() - start:.0f} s")
    finally:
        pool.shutdown(wait=False, cancel_futures=True)
    if failed:
        print(f"\n{failed} request(s) failed. Run the same command again to fetch the missing tiles.")
        return 1
    print(f"\nDone -> {out}")
    return 0


# Command: tiles


def count_bins(counts: np.ndarray) -> list[tuple[str, int, int]]:
    edges = [(1, 1, "1 building"), (2, 4, "2-4"), (5, 19, "5-19"), (20, 99, "20-99"), (100, None, "100 or more")]
    rows = []
    for low, high, label in edges:
        sel = counts >= low if high is None else (counts >= low) & (counts <= high)
        rows.append((label, int(sel.sum()), int(counts[sel].sum())))
    return rows


def cmd_tiles(args) -> int:
    zoom = args.zoom
    buildings = Path(args.buildings) if args.buildings else default_dir() / BUILDINGS_FILE
    if not buildings.exists():
        raise SystemExit(f"{buildings} not found. Run `download` first, or pass --buildings.")
    if not args.boundary:
        raise SystemExit("tiles needs --boundary, a polygon file for Syria (GeoJSON, GeoPackage or shapefile)")
    boundary = load_boundary(args.boundary)
    country_km2 = area_km2(boundary)
    print(f"Boundary area {country_km2:,.0f} km2 (the whole country is about 185,000 km2)")
    if not 150_000 < country_km2 < 200_000:
        print("  WARNING: that is not the size of Syria. Check that the file holds the whole country in one or more polygons.")

    grid, counts, fp, stats = footprint_tiles(boundary, buildings, zoom, args.min_buildings)
    ny, nx = grid.shape
    print(
        f"Buildings read: {stats['rows']:,} (touching more than one tile: {stats['multi_tile']:,}; "
        f"skipped as invalid or huge: {stats['invalid']:,})"
    )
    print(f"Tiles touching the boundary: {int(grid.inside.sum()):,} (zoom {zoom}, grid {nx} x {ny})")
    print(f"Tiles with at least {args.min_buildings} footprint(s): {int(fp.sum()):,}")
    for label, n_tiles, n_buildings in count_bins(counts[grid.inside & (counts > 0)]):
        print(f"    {label:12s}{n_tiles:8,} tiles{n_buildings:12,} buildings")

    out_dir = Path(args.out_dir) if args.out_dir else default_dir()
    out_dir.mkdir(parents=True, exist_ok=True)

    place_counts = np.zeros(grid.shape, dtype=np.int64)
    place_extra = np.zeros_like(fp)
    places_summary = None
    if args.places:
        places = read_point_frame(Path(args.places))
        px, py = lonlat_to_tile(places["lon"].to_numpy(), places["lat"].to_numpy(), zoom)
        prow, pcol = py - grid.y0, px - grid.x0
        in_box = (prow >= 0) & (prow < ny) & (pcol >= 0) & (pcol < nx)
        usable = np.zeros(len(places), dtype=bool)
        usable[in_box] = grid.inside[prow[in_box], pcol[in_box]]
        covered = np.zeros(len(places), dtype=bool)
        covered[usable] = fp[prow[usable], pcol[usable]]
        np.add.at(place_counts, (prow[usable], pcol[usable]), 1)
        place_extra = (place_counts > 0) & ~fp
        missing = usable & ~covered
        print(f"Populated places: {len(places):,} read, {int(usable.sum()):,} in tiles that touch the boundary")
        print(
            f"  {int(covered.sum()):,} ({100 * covered.sum() / max(int(usable.sum()), 1):.1f}%) sit in footprint tiles; "
            f"{int(missing.sum()):,} do not, which adds {int(place_extra.sum()):,} tiles"
        )
        if missing.any():
            added = places[missing].copy()
            added["quadkey"] = quadkeys(px[missing], py[missing], zoom)
            added_path = out_dir / f"places_added_z{zoom}.csv"
            added.to_csv(added_path, index=False)
            print(f"  Those places are listed in {added_path.name}; check a few against recent imagery.")
        places_summary = {
            "read": int(len(places)),
            "in_boundary_tiles": int(usable.sum()),
            "in_footprint_tiles": int(covered.sum()),
            "tiles_added": int(place_extra.sum()),
        }

    buffer_ring = dilate(fp | place_extra, args.buffer_tiles) & grid.inside & ~fp & ~place_extra if args.buffer_tiles else np.zeros_like(fp)
    if args.buffer_tiles:
        print(f"Buffer of {args.buffer_tiles} tile(s): +{int(buffer_ring.sum()):,} tiles")

    ghsl = np.full(grid.shape, np.nan)
    wc = np.full(grid.shape, np.nan)
    builtup_extra = np.zeros_like(fp)
    builtup_split = None
    if args.builtup_csv:
        table = pd.read_csv(args.builtup_csv, on_bad_lines="skip")
        if "worldcover_m2" not in table.columns:
            raise SystemExit(
                f"{Path(args.builtup_csv).name} was made by an older version of this script, whose WorldCover areas were 20-25% too high "
                f"(its columns: {', '.join(map(str, table.columns))}). Delete it and run `builtup` again."
            )
        table = table.dropna().drop_duplicates(["x", "y"], keep="last")
        rows = table["y"].to_numpy(int) - grid.y0
        cols = table["x"].to_numpy(int) - grid.x0
        ok = (rows >= 0) & (rows < ny) & (cols >= 0) & (cols < nx)
        ghsl[rows[ok], cols[ok]] = table["ghsl_m2"].to_numpy(float)[ok]
        wc[rows[ok], cols[ok]] = table["worldcover_m2"].to_numpy(float)[ok]
        candidate = grid.inside & ~fp
        checked = candidate & ~np.isnan(ghsl)
        print(f"Built-up check covers {int(checked.sum()):,} of {int(candidate.sum()):,} tiles without footprints")
        if int(checked.sum()) < int(candidate.sum()):
            print("  (some candidate tiles are missing from the CSV; run `builtup` again to finish)")
        print("  Extra tiles for different thresholds (m2 of built-up land in the tile):")
        for low in (500, 1000, 2000, 5000):
            print(f"    GHSL >= {low:5d}: {int((checked & (ghsl >= low)).sum()):7,}      WorldCover >= {low:5d}: {int((checked & (wc >= low)).sum()):7,}")
        use_ghsl, use_wc = args.min_ghsl_m2 > 0, args.min_wc_m2 > 0  # A threshold of 0 leaves that source out
        free = checked & ~buffer_ring & ~place_extra
        by_ghsl = free & (ghsl >= args.min_ghsl_m2) if use_ghsl else np.zeros_like(fp)
        by_wc = free & (wc >= args.min_wc_m2) if use_wc else np.zeros_like(fp)
        builtup_extra = by_ghsl | by_wc
        rule = " or ".join(text for text, on in ((f"GHSL >= {args.min_ghsl_m2:g} m2", use_ghsl), (f"WorldCover >= {args.min_wc_m2:g} m2", use_wc)) if on)
        print(f"  Chosen ({rule or 'both sources are off'}): +{int(builtup_extra.sum()):,} tiles")
        if use_ghsl and use_wc:
            builtup_split = {"both": int((by_ghsl & by_wc).sum()), "ghsl_only": int((by_ghsl & ~by_wc).sum()), "worldcover_only": int((by_wc & ~by_ghsl).sum())}
            print(
                f"    {builtup_split['both']:,} pass both, {builtup_split['ghsl_only']:,} only GHSL, "
                f"{builtup_split['worldcover_only']:,} only WorldCover (which counts roads as built-up)"
            )

    final = fp | place_extra | buffer_ring | builtup_extra
    x, y = grid.xy(final)
    rows, cols = y - grid.y0, x - grid.x0
    source = np.where(
        fp[rows, cols], "footprint",
        np.where(place_extra[rows, cols], "place", np.where(buffer_ring[rows, cols], "buffer", "builtup")),
    )
    qk = quadkeys(x, y, zoom)
    west, south, east, north = tile_bounds(x, y, zoom)
    table = pd.DataFrame(
        {
            "quadkey": qk,
            "x": x,
            "y": y,
            "z": zoom,
            "parent_z12": [k[:12] for k in qk] if zoom >= 12 else qk,
            "source": source,
            "n_buildings": counts[rows, cols],
            "n_places": place_counts[rows, cols],
            "ghsl_m2": ghsl[rows, cols],
            "worldcover_m2": wc[rows, cols],
            "west": np.round(west, 7),
            "south": np.round(south, 7),
            "east": np.round(east, 7),
            "north": np.round(north, 7),
        }
    ).sort_values("quadkey")

    area = float(tile_area_km2(x, y, zoom).sum())
    csv_path = out_dir / TILES_FILE.format(zoom=zoom)
    table.to_csv(csv_path, index=False)

    meta_path = buildings.with_suffix(".json")
    summary = {
        "zoom": zoom,
        "tiles": int(len(table)),
        "tiles_by_source": table["source"].value_counts().to_dict(),
        "tile_area_km2": round(area),
        "boundary_area_km2": round(country_km2),
        "share_of_boundary_area": round(area / country_km2, 3),
        "min_buildings": args.min_buildings,
        "buffer_tiles": args.buffer_tiles,
        "places": places_summary,
        "min_ghsl_m2": args.min_ghsl_m2 if args.builtup_csv else None,
        "min_wc_m2": args.min_wc_m2 if args.builtup_csv else None,
        "builtup_split": builtup_split,
        "buildings_read": stats["rows"],
        "overture": json.loads(meta_path.read_text()) if meta_path.exists() else None,
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    (out_dir / f"tiles_z{zoom}_summary.json").write_text(json.dumps(summary, indent=2))

    if args.geojson:
        import geopandas as gpd
        import shapely

        boxes = shapely.box(table["west"], table["south"], table["east"], table["north"])
        gdf = gpd.GeoDataFrame(table.drop(columns=["west", "south", "east", "north"]), geometry=boxes, crs=4326)
        gdf.to_file(out_dir / f"tiles_z{zoom}.geojson", driver="GeoJSON")

    parents = table["parent_z12"].nunique()
    print(
        f"\nFinal list: {len(table):,} tiles ({table['source'].value_counts().to_dict()}), "
        f"{area:,.0f} km2 = {100 * area / country_km2:.0f}% of the boundary area"
        f"\nThe tiles sit in {parents:,} zoom-12 parents (median {int(table.groupby('parent_z12').size().median())} tiles each), a natural unit for Earth Engine tasks."
        f"\n  -> {csv_path}"
    )
    if args.geojson:
        print(f"Load tiles_z{zoom}.geojson in a GIS to see which areas are left out.")
    else:
        print("Add --geojson to also write the tiles for a GIS.")
    return 0


# Command: coverage


def parse_named_paths(items: list[str] | None, default: dict[str, Path]) -> dict[str, Path]:
    if not items:
        return default
    out = {}
    for item in items:
        name, _, path = item.partition("=")
        if not path:
            name, path = Path(item).parent.name.upper(), item
        out[name] = Path(path)
    return out


def read_point_frame(path: Path) -> pd.DataFrame:
    """Attributes plus lon and lat of every feature in a file (a polygon is reduced to a point inside it)."""
    import geopandas as gpd

    gdf = gpd.read_file(path)
    gdf = gdf[gdf.geometry.notna() & ~gdf.geometry.is_empty]
    if gdf.crs is not None and gdf.crs.to_epsg() != 4326:
        gdf = gdf.to_crs(4326)
    pts = gdf.geometry.representative_point()
    table = pd.DataFrame(gdf.drop(columns=gdf.geometry.name)).reset_index(drop=True)
    table["lon"] = pts.x.to_numpy()
    table["lat"] = pts.y.to_numpy()
    return table


def read_points(path: Path, damage_col: str | None, damage_values: list[str]):
    """Longitude and latitude of the labelled points, keeping only the wanted damage classes."""
    table = read_point_frame(path)
    if damage_col and damage_values:
        if damage_col not in table.columns:
            others = [c for c in table.columns if c not in ("lon", "lat")]
            raise SystemExit(
                f"{path.name} has no column '{damage_col}'. Columns found: {', '.join(others) or 'none'}. "
                "Pass the right one with --damage-col, or --damage-col '' to test every point."
            )
        wanted = [v.strip() for v in damage_values]
        try:
            keep = pd.to_numeric(table[damage_col], errors="coerce").isin([float(v) for v in wanted])
        except ValueError:
            keep = table[damage_col].astype(str).str.strip().str.lower().isin([v.lower() for v in wanted])
        if not keep.any():
            seen = table[damage_col].value_counts(dropna=False).head(10)
            print(f"  No point in {path.name} has {damage_col} in {wanted}. Values present: " + ", ".join(f"{k!r} ({v:,})" for k, v in seen.items()))
        table = table[keep]
    return table["lon"].to_numpy(), table["lat"].to_numpy()


def nearest_footprint_m(px: np.ndarray, py: np.ndarray, boxes: tuple, k: int = 8) -> np.ndarray:
    """Metres from each point to the nearest footprint box (0 inside a box), found among the k nearest box centres."""
    from scipy.spatial import cKDTree

    xmin, ymin, xmax, ymax = boxes
    if len(xmin) == 0:
        return np.full(len(px), np.inf)
    ky = 111_132.0
    kx = 111_320.0 * np.cos(np.radians(np.mean(py)))
    tree = cKDTree(np.column_stack([(xmin + xmax) / 2 * kx, (ymin + ymax) / 2 * ky]))
    k = min(k, len(xmin))
    _, idx = tree.query(np.column_stack([px * kx, py * ky]), k=k)
    idx = idx.reshape(len(px), k)
    dx = np.maximum.reduce([xmin[idx] * kx - px[:, None] * kx, np.zeros(idx.shape), px[:, None] * kx - xmax[idx] * kx])
    dy = np.maximum.reduce([ymin[idx] * ky - py[:, None] * ky, np.zeros(idx.shape), py[:, None] * ky - ymax[idx] * ky])
    return np.hypot(dx, dy).min(axis=1)


def neighbour_keys(keys: np.ndarray, zoom: int, rings: int = 1) -> np.ndarray:
    """Tile keys (x * 2^zoom + y) of the given tiles plus `rings` rings of neighbours."""
    n = 1 << zoom
    x, y = keys // n, keys % n
    out = [keys]
    for dx in range(-rings, rings + 1):
        for dy in range(-rings, rings + 1):
            if dx or dy:
                out.append((x + dx) * n + (y + dy))
    return np.unique(np.concatenate(out))


def cmd_coverage(args) -> int:
    zoom = args.zoom
    n = 1 << zoom
    tiles_path = Path(args.tiles) if args.tiles else default_dir() / TILES_FILE.format(zoom=zoom)
    if not tiles_path.exists():
        raise SystemExit(f"{tiles_path} not found. Run `tiles` first, or pass --tiles.")
    buildings = Path(args.buildings) if args.buildings else default_dir() / BUILDINGS_FILE
    if not buildings.exists():
        raise SystemExit(f"{buildings} not found. Run `download` first, or pass --buildings.")
    table = pd.read_csv(tiles_path)
    if int(table["z"].iloc[0]) != zoom:
        raise SystemExit(f"{tiles_path.name} is zoom {int(table['z'].iloc[0])} but --zoom is {zoom}")
    all_keys = (table["x"].to_numpy(np.int64) * n + table["y"].to_numpy(np.int64))
    fp_keys = all_keys[(table["source"] == "footprint").to_numpy()]
    ring_keys = neighbour_keys(fp_keys, zoom, 1)
    print(f"Tile list {tiles_path.name}: {len(table):,} tiles ({len(fp_keys):,} with footprints)")

    base = labels_dir()
    labels = parse_named_paths(args.labels, {"ALP": base / "alp" / "unosat_labels.geojson", "RAQ": base / "raq" / "unosat_labels.geojson"})
    aois = parse_named_paths(args.aoi, {"ALP": base / "alp" / "unosat_aoi.geojson", "RAQ": base / "raq" / "unosat_aoi.geojson"})
    values = [v for v in args.damage_values.split(",") if v]

    out_dir = Path(args.out_dir) if args.out_dir else default_dir()
    summary = {}
    for name, path in labels.items():
        if not path.exists():
            print(f"\n{name}: {path} not found, skipped")
            continue
        lon, lat = read_points(path, args.damage_col, values)
        if len(lon) == 0:
            print(f"\n{name}: no labelled points to test in {path.name}, skipped")
            continue
        x, y = lonlat_to_tile(lon, lat, zoom)
        keys = x * n + y
        in_fp, in_ring, in_all = np.isin(keys, fp_keys), np.isin(keys, ring_keys), np.isin(keys, all_keys)
        margin = 0.03
        window = (lon.min() - margin, lat.min() - margin, lon.max() + margin, lat.max() + margin)
        parts = [b for b in iter_bounds(buildings, window=window)]
        boxes = tuple(np.concatenate([p[i] for p in parts]) if parts else np.zeros(0) for i in range(4))
        dist = nearest_footprint_m(lon, lat, boxes)

        pct = lambda m: f"{100 * m.mean():6.2f}% ({int(m.sum()):,}/{len(m):,})"
        print(f"\n{name}: {len(lon):,} labelled points in {len(np.unique(keys)):,} tiles; {len(boxes[0]):,} footprints within {margin} degrees")
        print(f"  Points inside a footprint tile:      {pct(in_fp)}")
        print(f"  ... with one ring of neighbours:     {pct(in_ring)}")
        print(f"  ... in the final list:               {pct(in_all)}")
        near = {m: float((dist <= m).mean()) for m in (25, 50, 100, 250)}
        print("  Footprint within 25 / 50 / 100 / 250 m of the point: " + " / ".join(f"{100 * near[m]:.1f}%" for m in near))
        entry = {
            "points": int(len(lon)),
            "in_footprint_tile": float(in_fp.mean()),
            "in_footprint_tile_plus_ring": float(in_ring.mean()),
            "in_final_list": float(in_all.mean()),
            "footprint_within_m": {str(m): v for m, v in near.items()},
        }

        aoi_path = aois.get(name)
        if aoi_path is not None and aoi_path.exists():
            grid = make_grid(load_boundary(aoi_path), zoom)
            ax, ay = grid.xy(grid.inside)
            aoi_keys = ax * n + ay
            share = float(np.isin(aoi_keys, all_keys).mean())
            print(f"  Tiles of the UNOSAT assessed area in the list: {int(np.isin(aoi_keys, all_keys).sum()):,}/{len(aoi_keys):,} ({100 * share:.1f}%)")
            entry["aoi_tiles_in_list"] = share

        missed = ~in_all
        if missed.any():
            out_dir.mkdir(parents=True, exist_ok=True)
            miss_path = out_dir / f"coverage_missed_{name.lower()}.csv"
            pd.DataFrame({"lon": lon[missed], "lat": lat[missed], "quadkey": quadkeys(x[missed], y[missed], zoom), "footprint_m": dist[missed]}).to_csv(miss_path, index=False)
            worst = pd.Series(quadkeys(x[missed], y[missed], zoom)).value_counts().head(5)
            print(f"  {int(missed.sum()):,} points outside the list (listed in {miss_path.name}); busiest missed tiles: " + ", ".join(f"{k} ({v})" for k, v in worst.items()))
        summary[name] = entry

    if summary:
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "coverage_summary.json").write_text(json.dumps(summary, indent=2))
    print(
        "\nThe three tile lines say whether any damaged building would have been left out of the run. "
        "The 'footprint within' line says how many damaged buildings have no footprint left nearby (their own footprint is gone), "
        "which is the case the extra ring of tiles and the built-up top-up guard against."
    )
    return 0


# Command: selftest


def make_fake_overture(folder: Path, seed: int = 0) -> dict:
    """Two Parquet parts laid out like Overture's building files, with known contents."""
    import pyarrow as pa
    import pyarrow.parquet as pq
    import shapely

    rng = np.random.default_rng(seed)
    parts = folder / "theme=buildings" / "type=building"
    parts.mkdir(parents=True, exist_ok=True)
    centres = np.array([[37.16, 36.20], [39.01, 35.95], [36.29, 33.51], [36.0, 38.5], [44.0, 33.0]])
    boxes = []
    for i, (cx, cy) in enumerate(centres):
        m = 4000 if i < 3 else 500
        lon = cx + rng.normal(0, 0.05, m)
        lat = cy + rng.normal(0, 0.05, m)
        w = rng.uniform(0.00005, 0.0003, m)
        h = rng.uniform(0.00005, 0.0003, m)
        boxes.append(np.column_stack([lon - w, lat - h, lon + w, lat + h]))
    boxes = np.vstack(boxes).astype(np.float32)
    ids = [f"b{i:06d}" for i in range(len(boxes))]
    half = len(boxes) // 2
    for part, sl in enumerate((slice(0, half), slice(half, None))):
        b = boxes[sl]
        bbox = pa.StructArray.from_arrays(
            [pa.array(b[:, 0]), pa.array(b[:, 1]), pa.array(b[:, 2]), pa.array(b[:, 3])], names=list(BBOX_FIELDS)
        )
        geoms = shapely.to_wkb(shapely.box(b[:, 0], b[:, 1], b[:, 2], b[:, 3]))
        table = pa.table({"id": ids[sl], "bbox": bbox, "geometry": pa.array(list(geoms), type=pa.binary()), "class": ["house"] * len(b)})
        pq.write_table(table, parts / f"part-{part}.parquet", row_group_size=2000)
    return {"boxes": boxes.astype(np.float64), "dir": parts}


def cmd_selftest(args) -> int:
    import pyarrow as pa

    failures = []

    def check(label: str, ok: bool, detail: str = ""):
        print(f"  [{'ok' if ok else 'FAIL'}] {label}" + (f" ({detail})" if detail else ""))
        if not ok:
            failures.append(label)

    print("1. Packages")
    versions = {"pyarrow": pa.__version__, "numpy": np.__version__, "pandas": pd.__version__}
    for name in ("shapely", "geopandas", "pyproj", "scipy", "ee"):
        try:
            module = __import__(name)
            versions[name] = getattr(module, "__version__", "installed")
        except ImportError:
            versions[name] = "MISSING"
    print("  " + ", ".join(f"{k} {v}" for k, v in versions.items()))
    for name in ("shapely", "geopandas", "pyproj", "scipy"):
        check(f"{name} available", versions[name] != "MISSING")
    if versions["shapely"] != "MISSING":
        check("shapely is version 2 or later", int(versions["shapely"].split(".")[0]) >= 2, versions["shapely"])
    try:
        import pyarrow.fs as pafs

        pafs.S3FileSystem(anonymous=True, region=S3_REGION)
        check("pyarrow has S3 support", True)
    except Exception as exc:
        check("pyarrow has S3 support", False, str(exc)[:80])
    if versions["ee"] == "MISSING":
        print("  (earthengine-api is only needed for `builtup`)")

    print("\n2. Tile maths")
    check("quadkey of tile (3, 5) at zoom 3 is 213", quadkeys([3], [5], 3)[0] == "213")
    qx, qy, qz = quadkey_to_tile("1202102332221212")
    check("quadkey round trip", qz == 16 and quadkeys([qx], [qy], 16)[0] == "1202102332221212")
    rng = np.random.default_rng(1)
    lon = rng.uniform(35.5, 42.5, 20000)
    lat = rng.uniform(32.2, 37.4, 20000)
    x, y = lonlat_to_tile(lon, lat, DEFAULT_ZOOM)
    w, s, e, nth = tile_bounds(x, y, DEFAULT_ZOOM)
    check("every point lies inside its own tile", bool(np.all((lon >= w) & (lon < e) & (lat >= s) & (lat <= nth))))
    area = float(tile_area_km2([int(x[0])], [int(y[0])], DEFAULT_ZOOM)[0])
    check("zoom-14 tile near 35N is about 4 km2", 3.5 < float(tile_area_km2(*lonlat_to_tile([37.0], [35.0], 14), 14)[0]) < 4.5, f"{area:.2f} km2 here")
    try:
        import mercantile

        ok = True
        for i in range(2000):
            t = mercantile.tile(float(lon[i]), float(lat[i]), DEFAULT_ZOOM)
            b = mercantile.bounds(t)
            ok &= (t.x, t.y) == (int(x[i]), int(y[i])) and abs(b.west - w[i]) < 1e-9 and abs(b.north - nth[i]) < 1e-9
        check("agrees with mercantile on 2000 points", bool(ok))
    except ImportError:
        print("  (mercantile not installed; skipped the cross-check)")

    print("\n3. Counting buildings per tile")
    box = (np.array([37.1, 37.1]), np.array([36.2, 36.2]), np.array([37.1001, 37.2]), np.array([36.2001, 36.2001]))
    tx, ty, tc, st = tile_counts([box], 14)
    tx0, ty0 = lonlat_to_tile(37.1, 36.2, 14)
    first = int(tc[(tx == tx0) & (ty == ty0)][0])
    check("a box touching several tiles is counted in each", len(tx) > 1 and st["multi_tile"] == 1 and first == 2, f"{len(tx)} tiles, {first} here")
    mask = np.zeros((6, 6), bool)
    mask[2, 3] = True
    check("dilate grows one tile into a 3 x 3 block", int(dilate(mask, 1).sum()) == 9 and int(dilate(mask, 2).sum()) == 25)

    print("\n4. Overture-style files")
    with tempfile.TemporaryDirectory() as tmp:
        fake = make_fake_overture(Path(tmp))
        modes = {}
        for mode_name, cols in (("struct", ["bbox"]), ("wkb", ["geometry"])):
            import pyarrow.dataset as ds

            dataset = ds.dataset(str(fake["dir"]), format="parquet")
            got = [bounds_from_batch(b, mode_name) for b in dataset.to_batches(columns=cols)]
            modes[mode_name] = np.column_stack([np.concatenate([g[i] for g in got]) for i in range(4)])
        check("bbox struct matches the geometry bounds", np.allclose(modes["struct"], modes["wkb"], atol=1e-6))
        check("all boxes read back", len(modes["struct"]) == len(fake["boxes"]))

        out_file = Path(tmp) / "out" / BUILDINGS_FILE
        ns = argparse.Namespace(release=DEFAULT_RELEASE, bbox="36.5,35.5,40.0,37.0", boundary=None, extra_columns="class", out=str(out_file),
                                source_path=str(fake["dir"]), proxy=None, io_threads=4, batch_size=1000)
        code = cmd_download(ns)
        got = pd.read_parquet(out_file)
        b = fake["boxes"].astype(np.float32).astype(np.float64)
        want = (b[:, 2] >= 36.5) & (b[:, 0] <= 40.0) & (b[:, 3] >= 35.5) & (b[:, 1] <= 37.0)
        check("download keeps exactly the buildings in the box", code == 0 and len(got) == int(want.sum()), f"{len(got):,} rows")
        check("download writes flat columns plus extras", list(got.columns) == list(BBOX_FIELDS) + ["class"])
        check("download writes a metadata file", out_file.with_suffix(".json").exists())

        x, y, c, _ = tile_counts(iter_bounds(out_file), 14)
        check("tile counts from the downloaded file add up", int(c.sum()) >= len(got))

        lon_pts = np.array([37.16, 37.17, 39.01, 41.5])
        lat_pts = np.array([36.20, 36.21, 35.95, 33.0])
        boxes_near = tuple(np.concatenate([p[i] for p in iter_bounds(out_file, window=(36.5, 35.5, 40.0, 37.0))]) for i in range(4))
        d = nearest_footprint_m(lon_pts, lat_pts, boxes_near)
        mid = (boxes_near[0] + boxes_near[2]) / 2
        brute = []
        ky = 111_132.0
        kx = 111_320.0 * np.cos(np.radians(lat_pts.mean()))
        for px, py in zip(lon_pts, lat_pts):
            dx = np.maximum.reduce([boxes_near[0] * kx - px * kx, np.zeros(len(mid)), px * kx - boxes_near[2] * kx])
            dy = np.maximum.reduce([boxes_near[1] * ky - py * ky, np.zeros(len(mid)), py * ky - boxes_near[3] * ky])
            brute.append(np.hypot(dx, dy).min())
        check("nearest-footprint distances match a brute-force search (within 1 m where footprints are dense)", bool(np.all(np.abs(d[:3] - np.array(brute[:3])) < 1.0)))

    print("\n5. Where things are read and written")
    print(f"  Output folder: {default_dir().resolve()}")
    for city in ("alp", "raq"):
        for kind in ("labels", "aoi"):
            path = labels_dir() / city / f"unosat_{kind}.geojson"
            print(f"  [{'found' if path.exists() else 'MISSING'}] {path}")

    print("\nDone." if not failures else f"\n{len(failures)} check(s) failed: {failures}")
    return 1 if failures else 0


# Entry point


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("selftest", help="Offline checks")

    sp = sub.add_parser("download", help="Overture building boxes -> local Parquet")
    sp.add_argument("--release", default=DEFAULT_RELEASE, help=f"Overture release, or 'latest' (default {DEFAULT_RELEASE})")
    sp.add_argument("--bbox", help="west,south,east,north (default: a box round Syria, or round --boundary)")
    sp.add_argument("--boundary", help="Syria polygon file; the box is its extent plus 0.05 degrees")
    sp.add_argument("--out", help=f"Output Parquet (default: <data>/syria/footprints/{BUILDINGS_FILE})")
    sp.add_argument("--extra-columns", help="Comma-separated extra Overture columns to keep, e.g. class,subtype,height")
    sp.add_argument("--source-path", help="Read a local Overture Parquet file or folder instead of S3")
    sp.add_argument("--proxy", help="HTTP proxy URL for S3 (default: HTTPS_PROXY if set; 'none' to ignore it)")
    sp.add_argument("--io-threads", type=int, default=16)
    sp.add_argument("--batch-size", type=int, default=262_144)

    def common(sp):
        sp.add_argument("--boundary", help="Syria polygon file (GeoJSON, GeoPackage or shapefile)")
        sp.add_argument("--buildings", help=f"Footprint boxes (default: <data>/syria/footprints/{BUILDINGS_FILE})")
        sp.add_argument("--zoom", type=int, default=DEFAULT_ZOOM)
        sp.add_argument("--min-buildings", type=int, default=1, help="Footprints a tile needs to count as built-up (default 1)")

    sp = sub.add_parser("builtup", help="GHSL and WorldCover built-up area for tiles without footprints (Earth Engine)")
    common(sp)
    sp.add_argument("--project", default=DEFAULT_PROJECT)
    sp.add_argument("--epoch", type=int, default=2015, help="GHSL epoch (default 2015, the baseline year)")
    sp.add_argument("--wc-scale", type=int, default=20, help="Metres at which WorldCover is sampled (default 20)")
    sp.add_argument("--chunk", type=int, default=500, help="Tiles per Earth Engine request")
    sp.add_argument("--workers", type=int, default=4, help="Requests in flight at once")
    sp.add_argument("--out", help="Output CSV")
    sp.add_argument("--check", action="store_true", help="Only query four test tiles (Aleppo, Raqqa, Damascus, desert) and print the numbers")

    sp = sub.add_parser("tiles", help="Final tile list")
    common(sp)
    sp.add_argument("--buffer-tiles", type=int, default=0, help="Add this many rings of tiles round the footprint and place tiles (default 0)")
    sp.add_argument("--places", help="Populated-places file (points, for example syr_populatedplaces.geojson); the tile of every place is added")
    sp.add_argument("--builtup-csv", help="Output of `builtup`, to add tiles the footprints miss")
    sp.add_argument("--min-ghsl-m2", type=float, default=2000, help="GHSL built-up area that adds a tile (default 2000; 0 leaves GHSL out)")
    sp.add_argument("--min-wc-m2", type=float, default=2000, help="WorldCover built-up area that adds a tile (default 2000; 0 leaves WorldCover out). WorldCover counts roads as built-up")
    sp.add_argument("--out-dir")
    sp.add_argument("--geojson", action="store_true", help="Also write the tiles as GeoJSON for a GIS")

    sp = sub.add_parser("coverage", help="UNOSAT coverage test")
    sp.add_argument("--tiles", help="Tile list from `tiles`")
    sp.add_argument("--buildings", help="Footprint boxes from `download`")
    sp.add_argument("--zoom", type=int, default=DEFAULT_ZOOM)
    sp.add_argument("--labels", nargs="+", help="NAME=path pairs (default: Aleppo and Raqqa from test_sites/processed)")
    sp.add_argument("--aoi", nargs="+", help="NAME=path pairs for the assessed-area polygons")
    sp.add_argument("--damage-col", default="damage")
    sp.add_argument("--damage-values", default="1,2", help="Values of the damage column to keep (default 1,2)")
    sp.add_argument("--out-dir")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return {
        "selftest": cmd_selftest, "download": cmd_download, "builtup": cmd_builtup,
        "tiles": cmd_tiles, "coverage": cmd_coverage,
    }[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
