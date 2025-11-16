
# NB: these are files containing reference data used in the analysis. These files ar provided in the 'reference provided' folder.
#     Please refer to README.md for more details.

from pathlib import Path

# Base data directory (relative to repo root)
DATA_DIR = Path(__file__).resolve().parent

reference_files = {
    "station codes": DATA_DIR / "reference provided" / "stations_ref_coordinates.json",
    "all dft categories": DATA_DIR / "reference provided" / "stations_ref_with_dft.json",
    "UK map 1km": DATA_DIR / "reference provided" / "UK map shapefile" / "1km" / "gb_1km.shp",
    "UK map 10km": DATA_DIR / "reference provided" / "UK map shapefile" / "10km" / "gb_10km.shp",
    "UK map 100km": DATA_DIR / "reference provided" / "UK map shapefile" / "100km" / "gb_100km.shp",
    "track lines": DATA_DIR / "reference provided" / "track lines" / "elrs.shp",
}
