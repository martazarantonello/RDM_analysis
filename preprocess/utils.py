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
    # Check if we have pre-loaded data (the critical components)
    if schedule_data_loaded is not None and stanox_ref is not None and tiploc_to_stanox is not None:
        print("Using pre-loaded data (much faster!)")
        
        # If train_count and tiploc are not provided, calculate them from pre-loaded data
        if train_count is None or tiploc is None:
            # Find TIPLOC for this STANOX
            tiploc = None
            train_count = 0
            
            # Convert stanox_ref to list format if it's a DataFrame
            if hasattr(stanox_ref, 'to_dict'):
                stanox_ref_list = stanox_ref.to_dict('records')
            else:
                stanox_ref_list = stanox_ref
                
            # Find TIPLOC for this STANOX
            for entry in stanox_ref_list:
                if str(entry.get("stanox")) == str(st_code):
                    tiploc = entry.get("tiploc") or entry.get("tiploc_code")
                    break
            
            if tiploc:
                # Count matching trains in pre-loaded schedule data
                for s in schedule_data_loaded:
                    # Navigate to the correct data path: JsonScheduleV1 > schedule_segment > schedule_location
                    json_schedule = s.get("JsonScheduleV1", {})
                    schedule_segment = json_schedule.get("schedule_segment", {})
                    sched_loc = schedule_segment.get("schedule_location", [])
                    tiploc_codes = {loc.get("tiploc_code") for loc in sched_loc}
                    if tiploc in tiploc_codes:
                        train_count += 1
    else:
        print("Loading data from files (this may take a while)...")
        train_count, tiploc, schedule_data_loaded, stanox_ref, tiploc_to_stanox = load_schedule_data(
            st_code, schedule_data, reference_files
        )
    
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


# =======================================================================================
# BATCH PROCESSING OPTIMIZATION FUNCTIONS 
# =======================================================================================
# These functions optimize batch processing by loading data once and reusing it

def load_schedule_data_once(schedule_data, reference_files):
    """
    Load schedule data once to avoid reloading for each station.
    
    Args:
        schedule_data (dict): Dictionary containing schedule data file paths
        reference_files (dict): Dictionary containing reference file paths
        
    Returns:
        tuple: (schedule_data_loaded, stanox_ref, tiploc_to_stanox)
    """
    try:
        # Load schedule data
        print("  Loading schedule pickle file...")
        schedule_data_loaded = pd.read_pickle(schedule_data["toc full"])
        
        # Load reference data
        print("  Loading reference data...")
        with open(reference_files["all dft categories"], 'r') as f:
            reference_data = json.load(f)
        # Convert to DataFrame format expected by process_schedule
        stanox_ref = pd.DataFrame(reference_data)
        
        # Create TIPLOC to STANOX mapping
        print("  Creating TIPLOC to STANOX mapping...")
        # Check if the reference data has the expected columns
        if 'tiploc' in stanox_ref.columns and 'stanox' in stanox_ref.columns:
            tiploc_to_stanox = dict(zip(stanox_ref['tiploc'], stanox_ref['stanox']))
        elif 'tiploc_code' in stanox_ref.columns and 'stanox' in stanox_ref.columns:
            tiploc_to_stanox = dict(zip(stanox_ref['tiploc_code'], stanox_ref['stanox']))
        else:
            print(f"  Warning: Expected TIPLOC columns not found. Available columns: {list(stanox_ref.columns)}")
            tiploc_to_stanox = {}
        
        return schedule_data_loaded, stanox_ref, tiploc_to_stanox
        
    except Exception as e:
        print(f"  Error loading schedule data: {e}")
        return None, None, None


def load_incident_data_once(incident_files):
    """
    Load all incident data once to avoid reloading for each station.
    
    Args:
        incident_files (dict): Dictionary with period names as keys and file paths as values
        
    Returns:
        dict: Dictionary with period names as keys and loaded DataFrames as values
    """
    incident_data_loaded = {}
    
    try:
        for period_name, file_path in incident_files.items():
            print(f"  Loading incident data for period: {period_name}")
            df = pd.read_csv(file_path)
            incident_data_loaded[period_name] = df
            
        return incident_data_loaded
        
    except Exception as e:
        print(f"  Error loading incident data: {e}")
        return {}


def process_delays_optimized(incident_data_loaded, st_code, output_dir=None):
    """
    Process delays using pre-loaded incident data to avoid file I/O.
    
    Args:
        incident_data_loaded (dict): Pre-loaded incident data by period
        st_code (str): The station code to filter delays
        output_dir (str, optional): Directory to save converted JSON files (not used in optimized mode)
        
    Returns:
        dict: Dictionary with period names as keys and processed DataFrames as values
    """
    columns_to_remove = [
        "ATTRIBUTION_STATUS", "INCIDENT_EQUIPMENT", "APPLICABLE_TIMETABLE_FLAG", "TRACTION_TYPE",
        "TRAILING_LOAD"
    ]
    
    processed_delays = {}
    
    # Ensure st_code is an integer for proper comparison with STANOX columns
    st_code_int = int(st_code)
    
    for period_name, df in incident_data_loaded.items():
        try:
            # Remove irrelevant columns
            df_cleaned = df.drop(columns=[col for col in columns_to_remove if col in df.columns], errors='ignore')
            
            # Filter rows where START_STANOX or END_STANOX matches st_code
            mask = (df_cleaned['START_STANOX'] == st_code_int) | (df_cleaned['END_STANOX'] == st_code_int)
            df_filtered = df_cleaned[mask]
            
            if not df_filtered.empty:
                processed_delays[period_name] = df_filtered
                print(f"    Processed {len(df_filtered)} delay entries for period {period_name}")
            
        except Exception as e:
            print(f"    Error processing period {period_name}: {e}")
            continue
    
    return processed_delays