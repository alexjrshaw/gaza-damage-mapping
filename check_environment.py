"""
Environment check for gaza-damage-mapping.
Run before executing the pipeline to verify all required packages are installed.
Usage:
    python3 check_environment.py
"""
import importlib
import sys

REQUIRED = {
    "ee": "earthengine-api",
    "fiona": "fiona",
    "osgeo.gdal": "GDAL",
    "geemap": "geemap",
    "geopandas": "geopandas",
    "matplotlib": "matplotlib",
    "numpy": "numpy",
    "pandas": "pandas",
    "pyarrow": "pyarrow",
    "pydrive2": "PyDrive2",
    "pyproj": "pyproj",
    "rasterio": "rasterio",
    "rioxarray": "rioxarray",
    "sklearn": "scikit-learn",
    "scipy": "scipy",
    "shapely": "shapely",
    "xarray": "xarray",
}

print("Checking required packages...\n")
missing = []
for module, package in REQUIRED.items():
    try:
        importlib.import_module(module)
        print(f"  OK  {package}")
    except ImportError:
        print(f"  MISSING  {package}")
        missing.append(package)

print()
if missing:
    print(f"Missing {len(missing)} package(s). Install with:")
    print(f"  pip install {' '.join(missing)}")
    sys.exit(1)
else:
    print("All required packages found.")
    v = sys.version_info
    print(f"\nPython {v.major}.{v.minor}.{v.micro}", end=" ")
    print("(OK)" if v.minor >= 10 else "(recommend 3.10+)")
