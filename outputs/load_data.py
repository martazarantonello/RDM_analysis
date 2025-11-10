from pathlib import Path
import pandas as pd

def load_processed_data(base_dir: str = "processed_data") -> pd.DataFrame:
    """
    Load all .parquet files from the processed_data folder (recursively)
    into a single pandas DataFrame.

    Automatically tries both pyarrow and fastparquet engines.
    Adds STANOX (folder name) and DAY (file name) columns.
    """

    # Use pathlib for cleaner, portable paths
    base_path = Path(base_dir)
    if not base_path.exists():
        base_path = Path("..") / base_dir

    parquet_files = list(base_path.glob("*/*.parquet"))
    if not parquet_files:
        print("⚠️ No parquet files found.")
        return pd.DataFrame()

    dfs = []
    skipped = []

    for file in parquet_files:
        try:
            df = pd.read_parquet(file)
        except Exception:
            try:
                df = pd.read_parquet(file, engine="fastparquet")
            except Exception:
                skipped.append(file)
                continue

        df["STANOX"] = file.parent.name
        df["DAY"] = file.stem
        dfs.append(df)

    if dfs:
        all_data = pd.concat(dfs, ignore_index=True)
        print(f"✅ Loaded {len(all_data):,} rows from {len(dfs)} files. Skipped {len(skipped)}.")
    else:
        all_data = pd.DataFrame()
        print("⚠️ No data loaded.")

    return all_data
