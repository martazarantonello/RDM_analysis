# IMPORTANT: please run schedule_cleaning.py first to generate the cleaned schedule pickle file.

# Base data directory (relative to repo root)
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parents[1] / "data"

schedule_data = {
   "toc full": DATA_DIR / "CIF_ALL_FULL_DAILY_toc-full.json.gz", # file directly downloaded from RDM, contains 5 parts (4th part is the schedule)
   "schedule" : DATA_DIR / "CIF_ALL_FULL_DAILY_toc-full_p4.pkl", # cleaned schedule file in pickle format (only 4th part - actual schedule)
}