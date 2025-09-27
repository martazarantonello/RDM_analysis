# ALL UTILITY FUNCTIONS

# IMPORTS
import json
import pickle
import sys
import os
import pandas as pd
import copy
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from datetime import datetime, timedelta
import numpy as np

from data.schedule import schedule_data
from data.reference import reference_files

from data.incidents import incident_files
from data.variables import benchmark_stanox

# Add the parent directory to the Python path to access the input module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))



# FOR SCHEDULE PROCESSING

def load_schedule_data(st_code, schedule_data, reference_files):
    """
    Load all necessary data for schedule processing.
    
    Returns:
        tuple: (train_count, tiploc, schedule_data_loaded, stanox_ref, tiploc_to_stanox)
    """
    # Load the category A stations reference file (pandas DataFrame)
    stanox_ref_df = pd.read_pickle(reference_files["category A stations"])
    
    # Convert DataFrame to list of dictionaries for backward compatibility
    stanox_ref = stanox_ref_df.to_dict('records')

    # Find the TIPLOC for this STANOX
    tiploc = None
    
    # First try to find using DataFrame operations (more efficient)
    matching_rows = stanox_ref_df[stanox_ref_df['stanox'] == str(st_code)]
    if not matching_rows.empty:
        tiploc = matching_rows.iloc[0]['tiploc']
    else:
        # Fallback: try with different type conversions
        try:
            # Try as integer
            matching_rows = stanox_ref_df[stanox_ref_df['stanox'] == int(st_code)]
            if not matching_rows.empty:
                tiploc = matching_rows.iloc[0]['tiploc']
        except (ValueError, TypeError):
            pass
    
    if tiploc is None:
        print(f"Warning: STANOX {st_code} not found in reference data")
        return 0, None, None, stanox_ref, {}

    # Load the schedule file (Pickle format)
    with open(schedule_data["toc full"], "rb") as f:
        schedule_data_loaded = pickle.load(f)

    # Count trains for this TIPLOC
    train_count = 0
    for entry in schedule_data_loaded:
        json_sched = entry.get("JsonScheduleV1", {})
        sched_segment = json_sched.get("schedule_segment", {})
        sched_loc = sched_segment.get("schedule_location", [])
        tiploc_codes = {loc.get("tiploc_code") for loc in sched_loc}
        if tiploc in tiploc_codes:
            train_count += 1

    # Create tiploc_to_stanox mapping from DataFrame
    tiploc_to_stanox = dict(zip(stanox_ref_df['tiploc'], stanox_ref_df['stanox']))

    return train_count, tiploc, schedule_data_loaded, stanox_ref, tiploc_to_stanox


# This code processes schedule files for one station code (st_code).
# st_code, and therefore tiploc, needs to be defined

# Add the parent directory to the Python path to access the input module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def get_day_code_mapping():
    """
    Create a mapping for day codes used throughout the application.
    
    Returns:
        dict: Mapping from day indices to day codes (0=Monday, 1=Tuesday, ..., 6=Sunday)
    """
    return {
        0: "MO",  # Monday
        1: "TU",  # Tuesday  
        2: "WE",  # Wednesday
        3: "TH",  # Thursday
        4: "FR",  # Friday
        5: "SA",  # Saturday
        6: "SU"   # Sunday
    }


def extract_schedule_days_runs(schedule_entry):
    """
    Extract schedule_days_runs from a schedule entry.
    
    Args:
        schedule_entry: Schedule entry dictionary
        
    Returns:
        str: Binary string representing days the schedule runs, or None if not found
    """
    try:
        # schedule_days_runs is at the JsonScheduleV1 level, not in schedule_segment
        return schedule_entry['JsonScheduleV1'].get('schedule_days_runs')
    except (KeyError, TypeError):
        return None


def get_english_day_types_from_schedule(schedule_entry):
    """
    Convert schedule_days_runs to list of ENGLISH_DAY_TYPE values.
    
    Args:
        schedule_entry: Schedule entry dictionary
        
    Returns:
        list: List of ENGLISH_DAY_TYPE values that this schedule runs on
    """
    schedule_days_runs = extract_schedule_days_runs(schedule_entry)
    if not schedule_days_runs:
        return []
    
    day_type_mapping = get_day_code_mapping()
    english_day_types = []
    
    # Convert binary string to list of day types
    for i, bit in enumerate(schedule_days_runs):
        if bit == '1' and i < len(day_type_mapping):
            english_day_types.append(day_type_mapping[i])
    
    return english_day_types



# MAIN FUNCTION: Process schedule for a specific STANOX code

def process_schedule(st_code, schedule_data=None, reference_files=None, 
                    train_count=None, tiploc=None, schedule_data_loaded=None, 
                    stanox_ref=None, tiploc_to_stanox=None):
    """
    Generate a schedule timeline for all trains that match the specified STANOX code.
    OPTIMIZED VERSION - Accepts pre-loaded data to avoid reloading from files.

    Args:
        st_code (str): STANOX code to process.
        schedule_data (dict, optional): Dictionary containing schedule data file paths.
        reference_files (dict, optional): Dictionary containing reference file paths.
        train_count (int, optional): Expected simple count of number of trains (from pre-loaded data).
        tiploc (str, optional): TIPLOC code corresponding to st_code.
        schedule_data_loaded (list, optional): Pre-loaded schedule data.
        stanox_ref (dict, optional): Pre-loaded STANOX reference data.
        tiploc_to_stanox (dict, optional): Pre-loaded TIPLOC to STANOX mapping.

    Returns:
        list: Sorted schedule timeline.
    """
    # If pre-loaded data is not provided, load it using the centralized function
    if any(param is None for param in [train_count, tiploc, schedule_data_loaded, stanox_ref, tiploc_to_stanox]):
        print("Loading data from files (this may take a while)...")
        train_count, tiploc, schedule_data_loaded, stanox_ref, tiploc_to_stanox = load_schedule_data(
            st_code, schedule_data, reference_files
        )
    else:
        print("Using pre-loaded data (much faster!)")
    
    if tiploc is None:
        return []  # STANOX not found
    
    # Convert stanox_ref to a dict if it's a list
    if isinstance(stanox_ref, list):
        stanox_ref = {str(entry.get("stanox")): entry for entry in stanox_ref if "stanox" in entry}

    processed_schedule = []
    trains_processed = 0

    print(f"Processing {len(schedule_data_loaded)} schedule entries for TIPLOC: {tiploc}")
    print(f"Expected to find {train_count} matching trains")

    # Process schedules to find matching trains (limit to train_count)
    for idx, s in enumerate(schedule_data_loaded):
        # Progress indicator for large datasets
        if idx % 10000 == 0:
            print(f"Processed {idx}/{len(schedule_data_loaded)} entries, found {trains_processed} matching trains")
        
        # Early termination if we've found enough trains
        if trains_processed >= train_count:
            print(f"Reached target of {train_count} trains, stopping early at entry {idx}")
            break
            
        try:
            schedule_locations = s['JsonScheduleV1']['schedule_segment']['schedule_location']
        except (KeyError, TypeError):
            continue  # Skip malformed entries
        
        # Single pass through schedule locations to find all needed data
        relevant_location = None
        origin_location = None
        destination_location = None
        
        for loc in schedule_locations:
            if loc.get('tiploc_code') == tiploc:
                relevant_location = loc
            if loc.get('location_type') == 'LO':
                origin_location = loc
            if loc.get('location_type') == 'LT':
                destination_location = loc
            
            # Early exit if we found all locations
            if relevant_location and origin_location and destination_location:
                break

        # If this train doesn't stop at our station, skip it
        if relevant_location is None:
            continue

        # Apply day type filtering if specified - REMOVED (using new day-of-week matching)
        # if day_type_filter and not matches_day_type(s, day_type_filter):
        #     continue

        # Extract the departure or arrival time for the relevant TIPLOC
        # Some trains might only have departure OR arrival, so check both
        s_time = None
        if relevant_location.get('departure'):
            s_time = relevant_location['departure'][:4]  # Extract the departure time
        elif relevant_location.get('arrival'):
            s_time = relevant_location['arrival'][:4]  # Extract the arrival time
        
        # Skip if neither departure nor arrival time is available
        if s_time is None:
            continue

        # Translate PLANNED_ORIGIN_LOCATION_CODE and PLANNED_DEST_LOCATION_CODE from TIPLOC to STANOX
        origin_stanox = tiploc_to_stanox.get(origin_location.get("tiploc_code"), "Unknown") if origin_location else "Unknown"
        dest_stanox = tiploc_to_stanox.get(destination_location.get("tiploc_code"), "Unknown") if destination_location else "Unknown"
        

        try:
            train_service_code = s["JsonScheduleV1"]["schedule_segment"]["CIF_train_service_code"]
        except (KeyError, TypeError):
            continue  # Skip entries without service code

        # Get the day types this schedule runs on
        schedule_day_types = get_english_day_types_from_schedule(s)

        # Determine the role of our target station in this train's journey
        station_role = "Unknown"
        target_stanox = tiploc_to_stanox.get(tiploc, "Unknown")
        
        if origin_location and origin_location.get("tiploc_code") == tiploc:
            station_role = "Origin"
        elif destination_location and destination_location.get("tiploc_code") == tiploc:
            station_role = "Destination"
        elif relevant_location:
            station_role = "Intermediate"

        s_data = {
            "TRAIN_SERVICE_CODE": train_service_code,
            "PLANNED_ORIGIN_LOCATION_CODE": origin_stanox,
            "PLANNED_ORIGIN_GBTT_DATETIME": origin_location.get("departure", "Unknown") if origin_location else "Unknown",
            "PLANNED_DEST_LOCATION_CODE": dest_stanox,
            "PLANNED_DEST_GBTT_DATETIME": destination_location.get("arrival", "Unknown") if destination_location else "Unknown",
            "PLANNED_CALLS": s_time,
            "ACTUAL_CALLS": s_time,
            "PFPI_MINUTES": 0.0,
            "INCIDENT_REASON": None,  # Will be populated during delay matching
            "INCIDENT_NUMBER": None,  # Will be populated during delay matching
            "EVENT_TYPE": None,       # Will be populated during delay matching
            "SECTION_CODE": None,     # Will be populated during delay matching
            "DELAY_DAY": None,        # Will be populated during delay matching
            "EVENT_DATETIME": None,   # Will be populated during delay matching
            "INCIDENT_START_DATETIME": None,  # Will be populated during delay matching
            "ENGLISH_DAY_TYPE": schedule_day_types,  # Keep as list of day codes (e.g., [MO, TU])
            "STATION_ROLE": station_role,  # Added to track the role of our target station
            "DFT_CATEGORY": stanox_ref.get(st_code, {}).get("dft_category", None),
            "PLATFORM_COUNT": stanox_ref.get(st_code, {}).get("numeric_platform_count", None)
        }

        # Add every train - no duplicate filtering for weekly schedules
        # Each instance represents a train on a different day
        processed_schedule.append(s_data)
        trains_processed += 1

    # Clean up is no longer needed since we removed the static variable
    print(f"Completed processing. Found {len(processed_schedule)} trains (including all weekly instances)")
    
    # Debug: Count station roles
    role_counts = {}
    for train in processed_schedule:
        role = train.get("STATION_ROLE", "Unknown")
        role_counts[role] = role_counts.get(role, 0) + 1
    
    print(f"Station roles for {tiploc} (STANOX {tiploc_to_stanox.get(tiploc, 'Unknown')}):")
    for role, count in role_counts.items():
        print(f"  {role}: {count} trains")
    
    # Sort the timeline by planned calls
    processed_schedule.sort(key=lambda x: int(x["PLANNED_CALLS"]))

    return processed_schedule


def extract_day_of_week_from_delay(delay_entry):
    """
    Extract the day of the week from PLANNED_ORIGIN_WTT_DATETIME in delay data.
    
    Args:
        delay_entry: Delay entry dictionary
        
    Returns:
        str: Day of week code (MO, TU, WE, TH, FR, SA, SU) or None if parsing fails
    """
    try:
        # Get the datetime string from delay data
        datetime_str = delay_entry.get('PLANNED_ORIGIN_WTT_DATETIME')
        if not datetime_str:
            return None
            
        # Parse the datetime string (format: "01-APR-2024 08:51")
        dt = datetime.strptime(datetime_str, "%d-%b-%Y %H:%M")
        
        # Convert to day of week (0=Monday, 1=Tuesday, ..., 6=Sunday)
        weekday = dt.weekday()
        
        # Use the shared day code mapping
        day_mapping = get_day_code_mapping()
        return day_mapping.get(weekday)
        
    except (ValueError, AttributeError, KeyError):
        return None


def schedule_runs_on_day(schedule_entry, target_day):
    """
    Check if a schedule entry runs on a specific day of the week.
    
    Args:
        schedule_entry: Schedule entry dictionary (from processed schedule)
        target_day: Day code (MO, TU, WE, TH, FR, SA, SU)
        
    Returns:
        bool: True if the schedule runs on the target day
    """
    # Use ENGLISH_DAY_TYPE which now contains the list of day codes
    schedule_day_types = schedule_entry.get('ENGLISH_DAY_TYPE', [])
    return target_day in schedule_day_types

# PROCESS DELAYS

# This code processes delay files for one station code (st_code).
# st_code needs to be defined

def process_delays(incident_files, st_code, output_dir):
    """
    Processes delay files by converting them to vertical JSON, removing irrelevant columns, and filtering rows.

    Args:
        incident_files (dict): Dictionary with period names as keys and file paths as values.
        output_dir (str): Directory to save the converted JSON files.
        st_code (str): The station code to filter delays.

    Returns:
        dict: Dictionary with period names as keys and processed DataFrames as values.
    """
    columns_to_remove = [
        "ATTRIBUTION_STATUS", "INCIDENT_EQUIPMENT", "APPLICABLE_TIMETABLE_FLAG", "TRACTION_TYPE",
        "TRAILING_LOAD"
    ]
    
    processed_delays = {}
    
    # Ensure st_code is an integer for proper comparison with STANOX columns
    st_code_int = int(st_code)
    
    for period_name, file_path in incident_files.items():
        # Load the delay data
        delay_df = pd.read_csv(file_path)
        
        # Filter for rows where START_STANOX or END_STANOX equals st_code
        delay_df = delay_df[
            (delay_df["START_STANOX"] == st_code_int) | (delay_df["END_STANOX"] == st_code_int)
        ]
        
        # Convert the DataFrame to a vertical JSON file (one JSON object per line)
        # Use the period name for the JSON file instead of the original filename
        json_file_name = os.path.join(output_dir, f"{period_name}.json")
        delay_df.to_json(json_file_name, orient="records", lines=True)
        
        # Remove irrelevant columns
        delay_df.drop(columns=columns_to_remove, inplace=True, errors='ignore')
        
        # Keep all EVENT_TYPE values including "C" (cancellations) for analysis
        
        processed_delays[period_name] = delay_df
    
    return processed_delays


# MATCHING PROCESSED DELAYS AND SCHEDULE

# This code wants to match schedule data with delay data, adjusting the timeline based on delays.
# It processes schedule data, applies delays, and generates an adjusted timeline for further analysis.


def adjust_schedule_timeline(processed_schedule, processed_delays, st_code=None):
    """
    Adjust the schedule timeline based on delays and generate an updated timeline.
    PANDAS OPTIMIZED VERSION: Uses pandas DataFrames for ultra-fast matching operations.

    Args:
        processed_schedule (list): List of processed schedule dictionaries.
        processed_delays (list): List of delay records from all days.
        st_code (str, optional): The station code being analyzed to determine correct planned call times.

    Returns:
        list: Adjusted schedule timeline sorted by actual calls.
    """
    
    print(f"Using pandas for delay matching: {len(processed_schedule)} schedule entries and {len(processed_delays)} delay entries...")
    
    if not processed_schedule or not processed_delays:
        print("No schedule or delay data to process")
        return processed_schedule if processed_schedule else []
    
    # Convert to DataFrames for fast operations
    schedule_df = pd.DataFrame(processed_schedule)
    delays_df = pd.DataFrame(processed_delays)
    
    print(f"Created DataFrames: schedule ({len(schedule_df)} rows), delays ({len(delays_df)} rows)")
    
    # Ensure consistent data types for merge columns
    print("Standardizing data types...")
    schedule_df['TRAIN_SERVICE_CODE'] = schedule_df['TRAIN_SERVICE_CODE'].astype(str)
    schedule_df['PLANNED_ORIGIN_LOCATION_CODE'] = schedule_df['PLANNED_ORIGIN_LOCATION_CODE'].astype(str)
    schedule_df['PLANNED_DEST_LOCATION_CODE'] = schedule_df['PLANNED_DEST_LOCATION_CODE'].astype(str)
    schedule_df['PLANNED_ORIGIN_GBTT_DATETIME'] = schedule_df['PLANNED_ORIGIN_GBTT_DATETIME'].astype(str)
    schedule_df['PLANNED_DEST_GBTT_DATETIME'] = schedule_df['PLANNED_DEST_GBTT_DATETIME'].astype(str)
    
    delays_df['TRAIN_SERVICE_CODE'] = delays_df['TRAIN_SERVICE_CODE'].astype(str)
    delays_df['PLANNED_ORIGIN_LOCATION_CODE'] = delays_df['PLANNED_ORIGIN_LOCATION_CODE'].astype(str)
    delays_df['PLANNED_DEST_LOCATION_CODE'] = delays_df['PLANNED_DEST_LOCATION_CODE'].astype(str)
    
    # Clean delay data and extract day information
    print("Cleaning and preprocessing delay data...")
    
    # Filter delays with valid datetime strings
    mask = (delays_df['PLANNED_ORIGIN_GBTT_DATETIME'].astype(str).str.len() >= 5) & \
           (delays_df['PLANNED_DEST_GBTT_DATETIME'].astype(str).str.len() >= 5) & \
           (delays_df['EVENT_DATETIME'].astype(str).str.len() >= 5)
    
    delays_clean = delays_df[mask].copy()
    print(f"Filtered to {len(delays_clean)} delays with valid datetime strings")
    
    if delays_clean.empty:
        print("No valid delays found after filtering")
        return processed_schedule
    
    # Extract time components for matching
    delays_clean['origin_time'] = delays_clean['PLANNED_ORIGIN_GBTT_DATETIME'].astype(str).str[-5:].str.replace(":", "")
    delays_clean['dest_time'] = delays_clean['PLANNED_DEST_GBTT_DATETIME'].astype(str).str[-5:].str.replace(":", "")
    delays_clean['event_time'] = delays_clean['EVENT_DATETIME'].astype(str).str[-5:].str.replace(":", "")
    
    # Extract delay days efficiently
    print("Extracting delay days...")
    delay_days = []
    for _, delay in delays_clean.iterrows():
        day = extract_day_of_week_from_delay(delay.to_dict())
        delay_days.append(day)
    
    delays_clean['delay_day'] = delay_days
    delays_clean = delays_clean[delays_clean['delay_day'].notna()]
    
    print(f"Filtered to {len(delays_clean)} delays with valid day information")
    
    if delays_clean.empty:
        print("No delays with valid day information")
        return processed_schedule
    
    # Expand schedule entries for multi-day schedules
    print("Expanding multi-day schedules...")
    schedule_expanded = []
    for _, sched in schedule_df.iterrows():
        sched_dict = sched.to_dict()
        english_day_types = sched_dict.get('ENGLISH_DAY_TYPE', [])
        if not english_day_types:
            english_day_types = ['MO']  # Default fallback
        
        for day in english_day_types:
            sched_copy = sched_dict.copy()
            sched_copy['current_day'] = day
            schedule_expanded.append(sched_copy)
    
    schedule_expanded_df = pd.DataFrame(schedule_expanded)
    print(f"Expanded to {len(schedule_expanded_df)} schedule entries (including multi-day)")
    
    # Perform vectorized matching - Origin matches
    print("Performing origin-based matching...")
    origin_matches = schedule_expanded_df.merge(
        delays_clean,
        left_on=['TRAIN_SERVICE_CODE', 'PLANNED_ORIGIN_LOCATION_CODE', 'PLANNED_ORIGIN_GBTT_DATETIME', 'current_day'],
        right_on=['TRAIN_SERVICE_CODE', 'PLANNED_ORIGIN_LOCATION_CODE', 'origin_time', 'delay_day'],
        how='left',
        suffixes=('', '_delay')
    )
    
    # Perform vectorized matching - Destination matches
    print("Performing destination-based matching...")
    dest_matches = schedule_expanded_df.merge(
        delays_clean,
        left_on=['TRAIN_SERVICE_CODE', 'PLANNED_DEST_LOCATION_CODE', 'PLANNED_DEST_GBTT_DATETIME', 'current_day'],
        right_on=['TRAIN_SERVICE_CODE', 'PLANNED_DEST_LOCATION_CODE', 'dest_time', 'delay_day'],
        how='left',
        suffixes=('', '_delay')
    )
    
    # Combine matches (prefer origin matches, then destination matches)
    print("Combining matches...")
    combined_matches = origin_matches.copy()
    
    # Fill in destination matches where origin matches are missing
    dest_only_mask = combined_matches['PFPI_MINUTES'].isna() & dest_matches['PFPI_MINUTES'].notna()
    
    for col in ['PFPI_MINUTES', 'INCIDENT_REASON', 'INCIDENT_NUMBER', 'EVENT_TYPE', 'SECTION_CODE', 'EVENT_DATETIME', 'INCIDENT_START_DATETIME', 'event_time', 'delay_day']:
        if col in dest_matches.columns:
            combined_matches.loc[dest_only_mask, col] = dest_matches.loc[dest_only_mask, col]
    
    # Apply delays to matched entries
    print("Applying delays to matched entries...")
    matched_mask = combined_matches['PFPI_MINUTES'].notna()
    
    combined_matches.loc[matched_mask, 'ACTUAL_CALLS'] = combined_matches.loc[matched_mask, 'event_time']
    combined_matches.loc[matched_mask, 'DELAY_DAY'] = combined_matches.loc[matched_mask, 'delay_day']
    
    # For non-matched entries, set ACTUAL_CALLS to PLANNED_CALLS
    non_matched_mask = ~matched_mask
    combined_matches.loc[non_matched_mask, 'ACTUAL_CALLS'] = combined_matches.loc[non_matched_mask, 'PLANNED_CALLS']
    combined_matches.loc[non_matched_mask, 'PFPI_MINUTES'] = 0
    
    # Remove temporary columns and get core schedule columns
    schedule_columns = list(schedule_df.columns) + [
        'ACTUAL_CALLS', 'PFPI_MINUTES', 'INCIDENT_REASON', 'INCIDENT_NUMBER', 'EVENT_TYPE', 'SECTION_CODE', 'DELAY_DAY', 'EVENT_DATETIME', 'INCIDENT_START_DATETIME',
        'DFT_CATEGORY', 'PLATFORM_COUNT', 'STATION_ROLE'  # <-- Add STATION_ROLE
    ]
    result_df = combined_matches[schedule_columns].copy()

    print(f"Matched {matched_mask.sum()} schedule entries with delays")

    # Add unmatched delays as new entries - OPTIMIZED VERSION
    print("Adding unmatched delays (optimized)...")
    
    # Get matched delay info more efficiently
    matched_delays_info = combined_matches[matched_mask][['TRAIN_SERVICE_CODE', 'DELAY_DAY', 'PFPI_MINUTES']].copy()
    
    # Create a set of tuples for fast lookup of matched delays
    if not matched_delays_info.empty:
        matched_delay_tuples = set(
            zip(matched_delays_info['TRAIN_SERVICE_CODE'], 
                matched_delays_info['DELAY_DAY'], 
                matched_delays_info['PFPI_MINUTES'])
        )
    else:
        matched_delay_tuples = set()
    
    # Create matching tuples for all delays
    delays_clean['match_tuple'] = list(zip(
        delays_clean['TRAIN_SERVICE_CODE'], 
        delays_clean['delay_day'], 
        delays_clean['PFPI_MINUTES']
    ))
    
    # Filter unmatched delays using vectorized operations
    unmatched_mask = ~delays_clean['match_tuple'].isin(matched_delay_tuples)
    unmatched_delays = delays_clean[unmatched_mask].copy()
    
    print(f"Found {len(unmatched_delays)} unmatched delays to add")
    
    # Vectorized creation of unmatched entries
    if not unmatched_delays.empty:
        # Determine planned_calls vectorized way
        origin_mask = (st_code is not None) & (unmatched_delays['PLANNED_ORIGIN_LOCATION_CODE'].astype(str) == str(st_code))
        dest_mask = (st_code is not None) & (unmatched_delays['PLANNED_DEST_LOCATION_CODE'].astype(str) == str(st_code))
        
        unmatched_delays['planned_calls'] = unmatched_delays['dest_time']  # Default
        unmatched_delays.loc[origin_mask, 'planned_calls'] = unmatched_delays.loc[origin_mask, 'origin_time']
        unmatched_delays.loc[dest_mask, 'planned_calls'] = unmatched_delays.loc[dest_mask, 'dest_time']
        
        # Create the unmatched entries DataFrame
        unmatched_entries_df = pd.DataFrame({
            "TRAIN_SERVICE_CODE": unmatched_delays['TRAIN_SERVICE_CODE'],
            "PLANNED_ORIGIN_LOCATION_CODE": unmatched_delays['PLANNED_ORIGIN_LOCATION_CODE'],
            "PLANNED_ORIGIN_GBTT_DATETIME": unmatched_delays['origin_time'],
            "PLANNED_DEST_LOCATION_CODE": unmatched_delays['PLANNED_DEST_LOCATION_CODE'],
            "PLANNED_DEST_GBTT_DATETIME": unmatched_delays['dest_time'],
            "PLANNED_CALLS": unmatched_delays['planned_calls'],
            "ACTUAL_CALLS": unmatched_delays['event_time'],
            "PFPI_MINUTES": unmatched_delays['PFPI_MINUTES'].astype(float),
            "INCIDENT_REASON": unmatched_delays['INCIDENT_REASON'],
            "INCIDENT_NUMBER": unmatched_delays['INCIDENT_NUMBER'],
            "EVENT_TYPE": unmatched_delays['EVENT_TYPE'],
            "SECTION_CODE": unmatched_delays['SECTION_CODE'],
            "DELAY_DAY": unmatched_delays['delay_day'],
            "EVENT_DATETIME": unmatched_delays['EVENT_DATETIME'],
            "INCIDENT_START_DATETIME": unmatched_delays['INCIDENT_START_DATETIME'],
            "ENGLISH_DAY_TYPE": unmatched_delays['delay_day'].apply(lambda x: [x]),
            "DFT_CATEGORY": None,
            "PLATFORM_COUNT": None,
            "STATION_ROLE": None  # <-- Add STATION_ROLE for unmatched delays
        })
        unmatched_entries = unmatched_entries_df.to_dict('records')
    else:
        unmatched_entries = []
    
    print(f"Added {len(unmatched_entries)} unmatched delays as new entries")
    
    # Combine all results
    final_result = result_df.to_dict('records') + unmatched_entries
    
    # Sort by actual calls
    final_result.sort(key=lambda x: int(x["ACTUAL_CALLS"]) if str(x["ACTUAL_CALLS"]).isdigit() else 0)
    
    print(f"Timeline adjustment complete: {len(final_result)} total entries")
    return final_result


# COUNT PLATFORMS
# Add the parent directory to the path to import from input module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def count_platforms(st_code):
    """
    Count the number of unique platforms for a station given its STANOX code.
    Optimized to only process schedule entries that contain the target station.
    
    Args:
        st_code (str or int): The STANOX code of the station
        
    Returns:
        int: Number of unique platforms for the station, or 0 if station not found
        
    Raises:
        FileNotFoundError: If reference or schedule files cannot be found
        ValueError: If the station code is not found in the reference data
    """
    
    # Convert st_code to string for consistent comparison
    st_code = str(st_code)
    
    # Load reference data to map STANOX to TIPLOC
    try:
        with open(reference_files["category A stations"], 'rb') as f:
            reference_data = pickle.load(f)
    except FileNotFoundError:
        raise FileNotFoundError(f"Reference file not found: {reference_files['category A stations']}")
    
    # Find the TIPLOC code for the given STANOX
    tiploc_code = None
    station_name = None
    for entry in reference_data:
        if entry.get('stanox') and str(entry['stanox']) == st_code:
            tiploc_code = entry.get('tiploc')
            station_name = entry.get('description') or entry.get('name')
            break
    
    if tiploc_code is None:
        raise ValueError(f"Station with STANOX code {st_code} not found in reference data")
    
    print(f"Looking for platforms at {station_name} (STANOX: {st_code}, TIPLOC: {tiploc_code})")
    
    # Load schedule data
    try:
        with open(schedule_data["toc full"], 'rb') as f:
            schedule_entries = pickle.load(f)
    except FileNotFoundError:
        raise FileNotFoundError(f"Schedule file not found: {schedule_data['toc full']}")
    
    # Collect all unique platforms for this station (optimized processing)
    platforms = set()
    entries_processed = 0
    entries_with_station = 0
    
    print(f"Processing {len(schedule_entries):,} schedule entries...")
    
    for i, entry in enumerate(schedule_entries):
        # Print progress every 50,000 entries
        if i % 50000 == 0 and i > 0:
            print(f"  Processed {i:,} entries, found {len(platforms)} platforms so far...")
        
        try:
            schedule_locations = entry['JsonScheduleV1']['schedule_segment']['schedule_location']
            
            # Quick check: does this train route include our target station?
            station_found_in_route = False
            for location in schedule_locations:
                if location.get('tiploc_code') == tiploc_code:
                    station_found_in_route = True
                    platform = location.get('platform')
                    # Only add non-null platforms
                    if platform is not None:
                        platforms.add(platform)
            
            if station_found_in_route:
                entries_with_station += 1
                
        except KeyError as e:
            # Skip entries with missing keys
            continue
        
        entries_processed += 1
    
    print(f"Processing complete!")
    print(f"  Total entries processed: {entries_processed:,}")
    print(f"  Entries containing {station_name}: {entries_with_station:,}")
    print(f"  Unique platforms found: {len(platforms)}")
    print(f"  Platform identifiers: {sorted(list(platforms))}")
    
    return len(platforms)



# Example usage:
if __name__ == "__main__":
    # Test with different stations
    print("Testing count_platforms function...")
    
    test_stations = [
        ("42140", "Crewe"),
        ("65630", "Birmingham New Street"),
        ("72410", "London Euston")
    ]
    
    for stanox, name in test_stations:
        try:
            print(f"\n{'='*50}")
            print(f"Testing {name} (STANOX: {stanox})")
            print('='*50)
            platform_count = count_platforms(stanox)
            print(f"Result: {name} has {platform_count} unique platforms")
            
        except Exception as e:
            print(f"Error testing {name}: {e}")
            import traceback
            traceback.print_exc()


# UTILS FOR GRAPHS!!

# for aggregate view:

def aggregate_view(incident_number, date):
    """
    Enhanced version that returns detailed data for visualization and creates meaningful charts
    with fixed 24-hour timeline from midnight to 23:59 for easy day-to-day comparison.
    """
    import glob
    import pickle
    import pandas as pd
    from datetime import datetime
    
    # Load all processed data files
    processed_files = glob.glob('../processed_data/category_a_station_*.pkl')
    
    if not processed_files:
        print("No processed data files found. Please run the preprocessor first.")
        return None
    
    all_incidents = []
    
    # Load data from all processed files
    for file_path in processed_files:
        try:
            with open(file_path, 'rb') as f:
                station_data = pickle.load(f)
            
            if isinstance(station_data, pd.DataFrame):
                # Handle incident number matching
                try:
                    incident_float = float(incident_number)
                    incident_mask = (station_data['INCIDENT_NUMBER'] == incident_float)
                except (ValueError, TypeError):
                    incident_mask = (station_data['INCIDENT_NUMBER'].astype(str) == str(incident_number))
                
                # Filter by date
                if 'EVENT_DATETIME' in station_data.columns:
                    station_data['event_date'] = pd.to_datetime(station_data['EVENT_DATETIME'], 
                                                              format='%d-%b-%Y %H:%M', errors='coerce').dt.date
                    
                    # Parse input date
                    target_date = None
                    date_formats = ['%d-%b-%Y', '%d-%B-%Y', '%Y-%m-%d', '%m/%d/%Y']
                    for fmt in date_formats:
                        try:
                            target_date = datetime.strptime(date, fmt).date()
                            break
                        except ValueError:
                            continue
                    
                    if target_date is None:
                        continue
                        
                    date_mask = (station_data['event_date'] == target_date)
                    
                    # Get filtered data
                    filtered_data = station_data[incident_mask & date_mask]
                    
                    if len(filtered_data) > 0:
                        all_incidents.extend(filtered_data.to_dict('records'))
                        
        except Exception as e:
            print(f"Error processing file {file_path}: {e}")
            continue
    
    if not all_incidents:
        print(f"No incidents found for INCIDENT_NUMBER {incident_number} on {date}")
        return None
    
    # Convert to DataFrame for easier manipulation
    df = pd.DataFrame(all_incidents)
    
    # Parse datetime with time information
    df['full_datetime'] = pd.to_datetime(df['EVENT_DATETIME'], format='%d-%b-%Y %H:%M', errors='coerce')
    df = df.dropna(subset=['full_datetime']).sort_values('full_datetime')
    
    # Parse target date and INCIDENT_START_DATETIME 
    target_date = datetime.strptime(date, '%d-%b-%Y').date()
    
    # Parse INCIDENT_START_DATETIME but don't filter by it - just prepare for visualization
    if 'INCIDENT_START_DATETIME' in df.columns:
        df['incident_start_datetime'] = pd.to_datetime(df['INCIDENT_START_DATETIME'], format='%d-%b-%Y %H:%M', errors='coerce')
        
        # Create chart datetime for incident start times (use target_date for consistent timeline)
        df['start_time_only'] = df['incident_start_datetime'].dt.time
        df['start_chart_datetime'] = df['start_time_only'].apply(lambda t: datetime.combine(target_date, t) if pd.notna(t) else None)
    
    # Create a base date for the timeline (use the target date)
    base_datetime = datetime.combine(target_date, datetime.min.time())
    
    # Extract just the time component and create datetime objects for the same day
    df['time_only'] = df['full_datetime'].dt.time
    df['chart_datetime'] = df['time_only'].apply(lambda t: datetime.combine(target_date, t))
    df['hour'] = df['full_datetime'].dt.hour
    
    # Calculate summary stats
    total_delay_minutes = df['PFPI_MINUTES'].fillna(0).sum()
    total_cancellations = len(df[df['EVENT_TYPE'] == 'C'])
    
    # Create fixed 24-hour timeline
    start_time = base_datetime  # 00:00:00
    end_time = base_datetime + timedelta(days=1) - timedelta(minutes=1)  # 23:59:00
    
    # Create visualizations with three subplots
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(16, 15))
    fig.suptitle(f'Incident Analysis: {incident_number} on {date}', fontsize=16, fontweight='bold', fontfamily='Arial')
    
    # Helper function to add time period shading
    def add_time_shading(ax):
        # Use the fixed timeline variables directly
        fixed_start = base_datetime  # 00:00:00
        fixed_end = base_datetime + timedelta(days=1) - timedelta(minutes=1)  # 23:59:00
        
        morning = datetime.combine(target_date, datetime.strptime('06:00', '%H:%M').time())
        evening = datetime.combine(target_date, datetime.strptime('18:00', '%H:%M').time())
        night = datetime.combine(target_date, datetime.strptime('22:00', '%H:%M').time())
        
        ax.axvspan(fixed_start, morning, alpha=0.1, color='blue')
        ax.axvspan(morning, evening, alpha=0.1, color='yellow')
        ax.axvspan(evening, night, alpha=0.1, color='orange')
        ax.axvspan(night, fixed_end, alpha=0.1, color='purple')
    
    # Helper function to format x-axis
    def format_time_axis(ax):
        # Use the fixed timeline variables directly
        fixed_start = base_datetime  # 00:00:00
        fixed_end = base_datetime + timedelta(days=1) - timedelta(minutes=1)  # 23:59:00
        ax.set_xlim(fixed_start, fixed_end)
        ax.xaxis.set_major_locator(mdates.HourLocator(interval=2))
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
        ax.xaxis.set_minor_locator(mdates.HourLocator(interval=1))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, fontsize=16, fontfamily='Arial')
        ax.grid(True, alpha=0.3)
    
    # Chart 1: Hourly Delay Totals (24-Hour View)
    ax1.set_title('Hourly Delay Totals (24-Hour View)', fontsize=16, pad=20, fontfamily='Arial')
    
    # Filter out rows with no delays and group by hour
    delay_data = df[df['PFPI_MINUTES'] > 0].copy()
    
    # Always create the full 24-hour timeline
    hour_datetimes = []
    hour_values = []
    
    # Group by hour and sum delays for each hour (if any)
    hourly_delays = delay_data.groupby('hour')['PFPI_MINUTES'].sum() if len(delay_data) > 0 else pd.Series()
    
    for hour in range(24):  # 0 to 23 - always show all 24 hours
        hour_dt = datetime.combine(target_date, datetime.strptime(f'{hour:02d}:00', '%H:%M').time())
        hour_datetimes.append(hour_dt)
        hour_values.append(hourly_delays.get(hour, 0))  # 0 if no delays in that hour
    
    # Plot as bar chart - always plot all 24 hours
    bars = ax1.bar(hour_datetimes, hour_values, 
            width=timedelta(minutes=45), alpha=0.7, color='steelblue',
            label=f'Hourly Delay Totals')
    
    # Add value labels on top of bars for non-zero values
    max_val = max(hour_values) if max(hour_values) > 0 else 1
    for i, (dt, val) in enumerate(zip(hour_datetimes, hour_values)):
        if val > 0:
            ax1.text(dt, val + max_val * 0.01, f'{val:.0f}', 
                    ha='center', va='bottom', fontsize=16, fontweight='bold', fontfamily='Arial')
    
    # Add INCIDENT_START_DATETIME markers
    if 'start_chart_datetime' in df.columns:
        incident_start_times = df[df['start_chart_datetime'].notna()]['start_chart_datetime'].unique()
        for incident_start_time in incident_start_times:
            ax1.axvline(x=incident_start_time, color='red', linestyle='--', linewidth=3, alpha=0.9)
        
        # Add legend entry for incident start time
        if len(incident_start_times) > 0:
            ax1.axvline(x=incident_start_times[0], color='red', linestyle='--', linewidth=3, alpha=0.9, 
                       label='Incident Start Time')
    
    ax1.legend(fontsize=16)
    
    ax1.set_ylabel('Total Delay Minutes per Hour', fontsize=16, fontfamily='Arial')
    ax1.set_xlabel('Hour of Day', fontsize=16, fontfamily='Arial')
    ax1.tick_params(axis='y', labelsize=16)
    format_time_axis(ax1)
    add_time_shading(ax1)
    
    # Chart 2: Delay Severity Distribution
    ax2.set_title('Delay Severity Distribution (Count by Severity Range)', fontsize=16, pad=20, fontfamily='Arial')
    
    if len(delay_data) > 0:
        # Create severity ranges
        delay_values = delay_data['PFPI_MINUTES'].values
        
        # Define severity bins
        bins = [0, 5, 15, 30, 60, 120, float('inf')]
        labels = ['1-5 min\n(Minor)', '6-15 min\n(Moderate)', '16-30 min\n(Significant)', 
                 '31-60 min\n(Major)', '61-120 min\n(Severe)', '120+ min\n(Critical)']
        
        # Count delays in each severity range
        counts, _ = np.histogram(delay_values, bins=bins)
        
        # Create bar chart with color coding by severity
        colors = ['lightgreen', 'yellow', 'orange', 'red', 'darkred', 'purple']
        bars = ax2.bar(labels, counts, color=colors, alpha=0.7, edgecolor='black', linewidth=1)
        
        # Add count labels on top of bars
        for bar, count in zip(bars, counts):
            if count > 0:
                ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(counts) * 0.01,
                        f'{count}', ha='center', va='bottom', fontsize=16, fontweight='bold', fontfamily='Arial')
        
        ax2.set_ylabel('Number of Delay Events', fontsize=16, fontfamily='Arial')
        ax2.set_xlabel('Delay Severity Range', fontsize=16, fontfamily='Arial')
        ax2.tick_params(axis='both', labelsize=16)
        
        # Add total delay info
        total_events = len(delay_data)
        avg_delay = delay_values.mean()
        ax2.text(0.02, 0.98, f'Total Events: {total_events}\nAverage Delay: {avg_delay:.1f} min', 
                transform=ax2.transAxes, verticalalignment='top', fontsize=16, fontfamily='Arial',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    else:
        ax2.text(0.5, 0.5, 'No delay events found', transform=ax2.transAxes, 
                ha='center', va='center', fontsize=16, fontfamily='Arial')
    
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Chart 3: Event Timeline (Delays and Cancellations)
    ax3.set_title('Event Timeline: Delays and Cancellations (24-Hour View)', fontsize=16, pad=20, fontfamily='Arial')
    
    # Separate delays and cancellations
    delays = df[df['EVENT_TYPE'] != 'C'].copy()
    cancellations = df[df['EVENT_TYPE'] == 'C'].copy()
    
    # Plot delays as scatter plot
    if len(delays) > 0:
        ax3.scatter(delays['chart_datetime'], delays['PFPI_MINUTES'], 
                   s=60, alpha=0.7, color='blue', label=f'Delays ({len(delays)})')
    
    # Plot cancellations as red X marks
    if len(cancellations) > 0:
        ax3.scatter(cancellations['chart_datetime'], cancellations['PFPI_MINUTES'], 
                   s=100, marker='X', color='red', alpha=0.8, 
                   label=f'Cancellations ({len(cancellations)})')
        
        # Add text annotations for cancellations
        for _, row in cancellations.iterrows():
            ax3.annotate('CANCELLED', 
                       (row['chart_datetime'], row['PFPI_MINUTES']), 
                       xytext=(5, 5), textcoords='offset points',
                       fontsize=16, color='red', weight='bold', fontfamily='Arial')
    
    # Add INCIDENT_START_DATETIME markers
    if 'start_chart_datetime' in df.columns:
        incident_start_times = df[df['start_chart_datetime'].notna()]['start_chart_datetime'].unique()
        for incident_start_time in incident_start_times:
            ax3.axvline(x=incident_start_time, color='red', linestyle='--', linewidth=3, alpha=0.9)
        
        # Add legend entry for incident start time
        if len(incident_start_times) > 0:
            ax3.axvline(x=incident_start_times[0], color='red', linestyle='--', linewidth=3, alpha=0.9, 
                       label='Incident Start Time')
    
    # Always show legend if there are any elements
    handles, labels = ax3.get_legend_handles_labels()
    if handles:
        ax3.legend(fontsize=16)
    
    ax3.set_ylabel('Delay Minutes', fontsize=16, fontfamily='Arial')
    ax3.set_xlabel('Time of Day (24-Hour Timeline)', fontsize=16, fontfamily='Arial')
    ax3.tick_params(axis='y', labelsize=16)
    format_time_axis(ax3)
    add_time_shading(ax3)
    
    plt.tight_layout()
    plt.show()
    
    # Return summary data
    # Calculate peak delay event with cancellation status
    peak_delay_info = "N/A"
    if len(df[df['PFPI_MINUTES'] > 0]) > 0:
        peak_idx = df['PFPI_MINUTES'].idxmax()
        peak_delay = df.loc[peak_idx, 'PFPI_MINUTES']
        peak_time = df.loc[peak_idx, 'full_datetime'].strftime('%H:%M')
        peak_is_cancelled = df.loc[peak_idx, 'EVENT_TYPE'] == 'C'
        
        if peak_is_cancelled:
            peak_delay_info = f"{peak_delay:.1f} minutes at {peak_time} (CANCELLED SERVICE)"
        else:
            peak_delay_info = f"{peak_delay:.1f} minutes at {peak_time} (Regular Delay)"
    
    # Calculate separate peak delays for regular services and cancellations
    delays_only = df[df['EVENT_TYPE'] != 'C']  # Non-cancelled services
    cancellations_only = df[df['EVENT_TYPE'] == 'C']  # Cancelled services
    
    peak_regular_delay = "N/A"
    peak_cancellation_delay = "N/A"
    
    if len(delays_only[delays_only['PFPI_MINUTES'] > 0]) > 0:
        peak_reg_idx = delays_only['PFPI_MINUTES'].idxmax()
        peak_reg_delay = delays_only.loc[peak_reg_idx, 'PFPI_MINUTES']
        peak_reg_time = delays_only.loc[peak_reg_idx, 'full_datetime'].strftime('%H:%M')
        peak_regular_delay = f"{peak_reg_delay:.1f} minutes at {peak_reg_time}"
    
    if len(cancellations_only[cancellations_only['PFPI_MINUTES'] > 0]) > 0:
        peak_canc_idx = cancellations_only['PFPI_MINUTES'].idxmax()
        peak_canc_delay = cancellations_only.loc[peak_canc_idx, 'PFPI_MINUTES']
        peak_canc_time = cancellations_only.loc[peak_canc_idx, 'full_datetime'].strftime('%H:%M')
        peak_cancellation_delay = f"{peak_canc_delay:.1f} minutes at {peak_canc_time}"

    summary = {
        "Total Delay Minutes": total_delay_minutes,
        "Total Cancellations": total_cancellations,
        "Total Records Found": len(df),
        "Incident Number": incident_number,
        "Date": date,
        "Time Range": f"{df['full_datetime'].min().strftime('%H:%M')} - {df['full_datetime'].max().strftime('%H:%M')}" if len(df) > 0 else "N/A",
        "Peak Delay Event": peak_delay_info,
        "Peak Regular Delay": peak_regular_delay,
        "Peak Cancellation Delay": peak_cancellation_delay
    }
    
    return summary