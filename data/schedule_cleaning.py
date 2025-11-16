# SCHEDULE CLEANING ONLY - CODE TO CLEAN THE OVERALL TOC FULL FILE TO EXTRACT ONLY THE SCHEDULE PART AND SAVE IT AS A PICKLE FILE

from pathlib import Path
import pandas as pd
import gzip
import json
import sys
import os
import time

# Add parent directory to path to import schedule_data
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data.schedule import schedule_data


def clean_schedule():
    """
    Extract only the JsonScheduleV1 objects (schedule data) from the
    CIF_ALL_FULL_DAILY_toc-full.json.gz file and save as a pickle file.
    
    The file is a newline-delimited JSON (NDJSON) file containing 5 types:
    1. JsonTimetableV1 - Header/metadata
    2. TiplocV1 - Location codes
    3. JsonAssociationV1 - Train associations
    4. JsonScheduleV1 - Schedule data (THIS IS WHAT WE EXTRACT)
    5. EOF - End of file marker
    """
    print("Opening TOC full file...")
    file_path = schedule_data["toc full"]

    with gzip.open(file_path, 'rt', encoding='utf-8') as f:
        n_lines = sum(1 for _ in f)
    print(f"Total lines in file: {n_lines:,}")

    # Read the gzipped NDJSON file and extract only JsonScheduleV1 objects
    schedules = []
    line_count = 0
    print("Reading and filtering for JsonScheduleV1 objects...")
    st = time.time()
    with gzip.open(file_path, 'rt', encoding='utf-8') as f:
        for i, line in enumerate(f):
            line_count += 1
            
            try:
                obj = json.loads(line)
                
                # Check if this is a JsonScheduleV1 object
                if isinstance(obj, dict) and 'JsonScheduleV1' in obj:
                    schedules.append(obj['JsonScheduleV1'])
                    
                    # Print progress every 100,000 lines
                    if len(schedules) % 100000 == 0:
                        print(f"  Found {len(schedules):,} schedule objects (processed {i+1:,} lines: {(line_count)/n_lines:.2%})...")
                        
            except json.JSONDecodeError as e:
                print(f"Warning: Could not parse line {i+1}: {e}")
                continue
    
    print(f"Total JsonScheduleV1 objects found: {len(schedules):,}")
    et = time.time()
    print(f"Time taken to read and filter: {et - st:.2f} seconds")
    
    if len(schedules) == 0:
        raise ValueError("No JsonScheduleV1 objects found in the file!")
    
    # Convert to DataFrame
    print("Converting to DataFrame...")
    df_schedule = pd.DataFrame(schedules)

    print("First few rows of the schedule DataFrame:")
    print(df_schedule.head())
    
    print(f"DataFrame shape: {df_schedule.shape}")
    print(f"Columns: {list(df_schedule.columns)}")
    
    # Save as pickle file
    output_path = schedule_data["schedule"]
    print(f"\nSaving to {output_path}...")
    df_schedule.to_pickle(output_path)
    
    print("\nSchedule cleaning completed successfully!")
    print(f"Saved {len(schedules):,} schedule records")
    return df_schedule


if __name__ == "__main__":
    clean_schedule()