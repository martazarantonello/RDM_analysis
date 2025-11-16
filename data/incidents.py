
# NB: these are files containing delay incident data, named according to the period selected for this initial analysis.
#     Please rename the year or month as needed to reflect the data you are using.
#     Please refer to README.md for more details on how to obtain these files.

# period: one financial/rail year

from pathlib import Path

# Base data directory (relative to repo root)
DATA_DIR = Path(__file__).resolve().parents[1] / "data"

incident_files = {
    "23-24 P12": DATA_DIR / "Transparency_23-24_P12.csv",
    "23-24 P13": DATA_DIR / "Transparency_23-24_P13.csv",
    "24-25 P01": DATA_DIR / "Transparency_24-25_P01.csv",
    "24-25 P02": DATA_DIR / "Transparency_24-25_P02.csv",
    "24-25 P03": DATA_DIR / "Transparency_24-25_P03.csv",
    "24-25 P04": DATA_DIR / "Transparency_24-25_P04.csv",
    "24-25 P05": DATA_DIR / "Transparency_24-25_P05.csv",
    "24-25 P06": DATA_DIR / "Transparency_24-25_P06.csv",
    "24-25 P07": DATA_DIR / "Transparency_24-25_P07.csv",
    "24-25 P08": DATA_DIR / "Transparency_24-25_P08.csv",
    "24-25 P09": DATA_DIR / "Transparency_24-25_P09.csv",
    "24-25 P10": DATA_DIR / "Transparency_24-25_P10.csv", # only data set combined with passenger loadings
}
