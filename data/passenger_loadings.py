
#  NB: these are files containing one company's (SWR) passenger loadings data, named according to the period selected for this initial analysis.
#     Please rename the year or month as needed to reflect the data you are using.
#     Please refer to README.md for more details on how to obtain these files.

from pathlib import Path

# Base data directory (relative to repo root)
DATA_DIR = Path(__file__).resolve().parents[1] / "data"

SWR_passenger_loadings_data_files = {
    "24-25 P10" : DATA_DIR / "SWR Passenger Loadings - RY25 P10.xlsx", # only RY25 P10 is used in the analysis as the delay data only covers until month. The P11 to P13 delay data is not yet available.
    "24-25 P11" : DATA_DIR / "SWR Passenger Loadings - RY25 P11.xlsx",
    "24-25 P12" : DATA_DIR / "SWR Passenger Loadings - RY25 P12.xlsx",
    "24-25 P13" : DATA_DIR / "SWR Passenger Loadings - RY25 P13.xlsx",
    "25-26 P01" : DATA_DIR / "SWR Passenger Loadings - RY26 P01.xlsx",
    "25-26 P02" : DATA_DIR / "SWR Passenger Loadings - RY26 P02.xlsx",
    "25-26 P03" : DATA_DIR / "SWR Passenger Loadings - RY26 P03.xlsx",
    "25-26 P04" : DATA_DIR / "SWR Passenger Loadings - RY26 P04.xlsx",
}