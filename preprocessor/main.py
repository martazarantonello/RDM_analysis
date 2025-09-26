# this will be the overall "save processed data" function that was previously defined
# to run then: 
# python -m preprocessor.main 
# in terminal
# you have two options to choose processing mode:
#      1. Process a single station       
#      2. Process all Category A stations
#      Enter choice (1 or 2): HERE


"""
Save processed schedule and delay data to pickle files for all Category A stations.
NB: this is for all category A station NOT SUITABLE for other categories.
    Please make changes accordingly if you want to process cat B,C1 and C2.
This script processes schedule data, applies delays, and saves the results as pandas DataFrames
organized by day of the week for each Category A station.
"""

import json
import os
import pandas as pd
import pickle
from datetime import datetime
from preprocessor.utils import adjust_schedule_timeline
from preprocessor.utils import process_schedule, get_day_code_mapping
from preprocessor.utils import process_delays
from data.schedule import schedule_data
from data.reference import reference_files
from data.incidents import incident_files


def get_weekday_from_schedule_entry(entry):
    """
    Extract the primary weekday from a schedule entry for sorting purposes.
    
    Args:
        entry: Schedule entry dictionary
        
    Returns:
        int: Weekday index (0=Monday, 6=Sunday) for sorting
    """
    # If there's a delay day, use that for sorting
    if entry.get('DELAY_DAY'):
        day_mapping = get_day_code_mapping()
        # Reverse mapping: day code to index
        reverse_mapping = {v: k for k, v in day_mapping.items()}
        return reverse_mapping.get(entry['DELAY_DAY'], 0)
    
    # Otherwise, use the first day from ENGLISH_DAY_TYPE
    english_day_types = entry.get('ENGLISH_DAY_TYPE', [])
    if english_day_types:
        day_mapping = get_day_code_mapping()
        # Reverse mapping: day code to index
        reverse_mapping = {v: k for k, v in day_mapping.items()}
        return reverse_mapping.get(english_day_types[0], 0)
    
    return 0  # Default to Monday if no day information


def load_category_a_stations():
    """
    Load Category A stations from the reference pickle file.
    
    Returns:
        list: List of STANOX codes for Category A stations
    """
    try:
        df = pd.read_pickle(reference_files["category A stations"])
        stanox_codes = df['stanox'].tolist()
        # Convert to strings to ensure consistency
        stanox_codes = [str(code) for code in stanox_codes]
        return stanox_codes
    except Exception as e:
        print(f"Error loading Category A stations: {e}")
        return []


def save_processed_data_by_weekday_to_dataframe(st_code, output_dir="processed_data"):
    """
    Process schedule and delay data, then save to pandas DataFrame organized by weekday.
    
    Args:
        st_code (str): STANOX code to process
        output_dir (str): Directory to save output files
        
    Returns:
        dict: Dictionary containing processed data as pandas DataFrames organized by weekday
    """
    print(f"Processing data for STANOX: {st_code}")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Step 1: Process schedule data
    print("Step 1: Processing schedule data...")
    processed_schedule = process_schedule(st_code, schedule_data, reference_files)
    print(f"Processed {len(processed_schedule)} schedule entries")
    
    if not processed_schedule:
        print(f"No schedule data found for station {st_code}. Skipping...")
        return None
    
    # Step 2: Process delay data
    print("Step 2: Processing delay data...")
    temp_delays_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "temp_delays")
    os.makedirs(temp_delays_path, exist_ok=True)
    processed_delays_dict = process_delays(incident_files, st_code, temp_delays_path)
    
    # Convert dictionary of DataFrames to list of delay records
    processed_delays = []
    for period_name, df in processed_delays_dict.items():
        if hasattr(df, 'to_dict'):  # Check if it's a DataFrame
            processed_delays.extend(df.to_dict('records'))
    
    print(f"Processed {len(processed_delays)} delay entries from {len(processed_delays_dict)} periods")
    
    # Step 3: Adjust schedule timeline with delays
    print("Step 3: Adjusting schedule timeline with delays...")
    schedule_timeline_adjusted = adjust_schedule_timeline(processed_schedule, processed_delays, st_code)
    print(f"Adjusted timeline contains {len(schedule_timeline_adjusted)} entries")
    
    if not schedule_timeline_adjusted:
        print(f"No adjusted schedule data found for station {st_code}. Skipping...")
        return None
    
    # Step 4: Remove identical duplicates (but preserve legitimate multi-day/delay variations)
    print("Step 4: Removing identical duplicate entries...")
    seen = set()
    deduplicated = []
    for entry in schedule_timeline_adjusted:
        # Create a hashable representation that allows same train to have different delays on different days
        # Include delay-specific fields in the key to distinguish between delayed and non-delayed versions
        key_fields = []
        for k, v in sorted(entry.items()):
            if isinstance(v, (str, int, float, type(None))):
                key_fields.append((k, v))
            elif isinstance(v, list):
                # Convert lists to tuples for hashing
                key_fields.append((k, tuple(v)))
        
        entry_key = tuple(key_fields)
        
        if entry_key not in seen:
            seen.add(entry_key)
            deduplicated.append(entry)
    
    print(f"Removed {len(schedule_timeline_adjusted) - len(deduplicated)} identical duplicate entries")
    schedule_timeline_adjusted = deduplicated
    print(f"Deduplicated timeline contains {len(schedule_timeline_adjusted)} entries")
    
    # Step 5: Organize data by weekday - Handle multi-day schedules
    print("Step 5: Organizing data by weekday...")
    day_mapping = get_day_code_mapping()
    weekday_data = {day_code: [] for day_code in day_mapping.values()}
    
    for entry in schedule_timeline_adjusted:
        # Handle multi-day schedules: save to ALL days the train runs
        english_day_types = entry.get('ENGLISH_DAY_TYPE', [])
        
        if english_day_types:
            # Create a copy of the entry for each day it runs
            for day_code in english_day_types:
                if day_code in weekday_data:  # Ensure it's a valid day code
                    # Create a copy of the entry for this specific day
                    day_specific_entry = entry.copy()
                    # Add metadata to track which day this instance represents
                    day_specific_entry['DATASET_TYPE'] = 'MULTI_DAY' if len(english_day_types) > 1 else 'SINGLE_DAY'
                    day_specific_entry['WEEKDAY'] = day_code
                    weekday_data[day_code].append(day_specific_entry)
        else:
            # Fallback: if no ENGLISH_DAY_TYPE, use the old logic
            weekday_index = get_weekday_from_schedule_entry(entry)
            day_code = day_mapping[weekday_index]
            entry['DATASET_TYPE'] = 'FALLBACK'
            entry['WEEKDAY'] = day_code
            weekday_data[day_code].append(entry)
    
    # Step 6: Convert to pandas DataFrames
    print("Step 6: Converting to pandas DataFrames...")
    weekday_dataframes = {}
    
    for day_code, entries in weekday_data.items():
        if entries:  # Only create DataFrame if there are entries for this day
            # Sort entries by ACTUAL_CALLS time - handle NaN and non-numeric values
            def safe_sort_key(x):
                actual_calls = x.get("ACTUAL_CALLS")
                if actual_calls is None or actual_calls == "NA":
                    return 0
                if isinstance(actual_calls, (int, float)):
                    if pd.isna(actual_calls):
                        return 0
                    return int(actual_calls)
                if isinstance(actual_calls, str):
                    if actual_calls.isdigit():
                        return int(actual_calls)
                    return 0
                return 0
            
            entries.sort(key=safe_sort_key)
            
            # Convert to DataFrame
            df = pd.DataFrame(entries)
            weekday_dataframes[day_code] = df
            print(f"Created DataFrame for {day_code} with {len(df)} entries")
    
    return weekday_dataframes


def save_all_category_a_stations(output_dir="processed_data"):
    """
    Process and save data for all Category A stations as pickle files.
    
    Args:
        output_dir (str): Directory to save output files
        
    Returns:
        dict: Summary of processing results
    """
    print("Loading Category A stations...")
    category_a_stations = load_category_a_stations()
    
    if not category_a_stations:
        print("No Category A stations found. Exiting.")
        return None
    
    print(f"Found {len(category_a_stations)} Category A stations: {category_a_stations}")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Clean up existing files
    print("Cleaning up existing Category A station files...")
    import glob
    cleanup_patterns = [
        f"{output_dir}/category_a_station_*.pkl",
        f"{output_dir}/category_a_stations_summary.pkl"
    ]
    
    files_removed = 0
    for pattern in cleanup_patterns:
        for filepath in glob.glob(pattern):
            try:
                os.remove(filepath)
                files_removed += 1
                print(f"  Removed: {filepath}")
            except Exception as e:
                print(f"  Warning: Could not remove {filepath}: {e}")
    
    if files_removed > 0:
        print(f"Cleaned up {files_removed} existing files")
    
    # Process each Category A station
    results = {
        'successful_stations': [],
        'failed_stations': [],
        'total_entries_by_station': {},
        'files_created': []
    }
    
    for i, st_code in enumerate(category_a_stations, 1):
        print(f"\n{'='*60}")
        print(f"Processing station {i}/{len(category_a_stations)}: {st_code}")
        print(f"{'='*60}")
        
        try:
            # Process the station data
            station_data = save_processed_data_by_weekday_to_dataframe(st_code, output_dir)
            
            if station_data:
                # Save each weekday DataFrame as a separate pickle file
                total_entries = 0
                for day_code, df in station_data.items():
                    filename = f"{output_dir}/category_a_station_{st_code}_{day_code}.pkl"
                    df.to_pickle(filename)
                    results['files_created'].append(filename)
                    total_entries += len(df)
                    print(f"Saved {len(df)} entries for {day_code} to {filename}")
                
                results['successful_stations'].append(st_code)
                results['total_entries_by_station'][st_code] = total_entries
                
                print(f"Successfully processed station {st_code}: {total_entries} total entries")
            else:
                print(f"No data found for station {st_code}")
                results['failed_stations'].append(st_code)
                
        except Exception as e:
            print(f"Error processing station {st_code}: {str(e)}")
            results['failed_stations'].append(st_code)
            import traceback
            traceback.print_exc()
    
    # Save summary
    summary_file = f"{output_dir}/category_a_stations_summary.pkl"
    with open(summary_file, 'wb') as f:
        pickle.dump(results, f)
    
    print(f"\n{'='*60}")
    print("PROCESSING COMPLETE")
    print(f"{'='*60}")
    print(f"Successfully processed: {len(results['successful_stations'])} stations")
    print(f"Failed to process: {len(results['failed_stations'])} stations")
    print(f"Total files created: {len(results['files_created'])}")
    print(f"Summary saved to: {summary_file}")
    
    if results['successful_stations']:
        print(f"\nSuccessful stations: {results['successful_stations']}")
        print(f"\nTotal entries by station:")
        for station, count in results['total_entries_by_station'].items():
            print(f"  {station}: {count} entries")
    
    if results['failed_stations']:
        print(f"\nFailed stations: {results['failed_stations']}")
    
    return results


def main(st_code=None, process_all_category_a=False):
    """
    Main function to demonstrate the data processing and saving functionality.
    
    Args:
        st_code (str, optional): STANOX code to process. If not provided and process_all_category_a is False, will prompt for input.
        process_all_category_a (bool): If True, process all Category A stations instead of a single station.
    """
    if process_all_category_a:
        print("Processing all Category A stations...")
        try:
            result = save_all_category_a_stations()
            
            if result:
                print(f"\n" + "="*50)
                print("ALL CATEGORY A STATIONS PROCESSING COMPLETE")
                print("="*50)
                print(f"Successfully processed: {len(result['successful_stations'])} stations")
                print(f"Failed to process: {len(result['failed_stations'])} stations")
                print(f"Total files created: {len(result['files_created'])}")
                
                if result['successful_stations']:
                    print(f"\nSuccessful stations: {result['successful_stations']}")
                    
                if result['failed_stations']:
                    print(f"\nFailed stations: {result['failed_stations']}")
            else:
                print("No Category A stations were processed.")
                
        except Exception as e:
            print(f"Error processing Category A stations: {str(e)}")
            import traceback
            traceback.print_exc()
    else:
        # Process single station (original functionality)
        # Get STANOX code from parameter or user input
        if st_code is None:
            st_code = input("Enter STANOX code to process: ").strip()
            if not st_code:
                print("No STANOX code provided. Exiting.")
                return None
        
        try:
            result = save_processed_data_by_weekday_to_dataframe(st_code)
            
            if result:
                print(f"\n" + "="*50)
                print("PROCESSING COMPLETE")
                print("="*50)
                total_entries = sum(len(df) for df in result.values())
                print(f"Total entries processed: {total_entries}")
                print(f"DataFrames created: {len(result)}")
                
                print("\nDataFrames created:")
                for day_code, df in result.items():
                    print(f"  {day_code}: {len(df)} entries")
                    
                # Optionally save as pickle files
                save_choice = input("\nSave DataFrames as pickle files? (y/n): ").strip().lower()
                if save_choice == 'y':
                    output_dir = "processed_data"
                    os.makedirs(output_dir, exist_ok=True)
                    
                    for day_code, df in result.items():
                        filename = f"{output_dir}/single_station_{st_code}_{day_code}.pkl"
                        df.to_pickle(filename)
                        print(f"Saved {filename}")
                        
            else:
                print(f"No data found for station {st_code}")
                
        except Exception as e:
            print(f"Error processing data: {str(e)}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        if sys.argv[1] == "--all-category-a":
            # Process all Category A stations
            main(process_all_category_a=True)
        else:
            # Process specific station code
            main(st_code=sys.argv[1])
    else:
        # Interactive mode
        print("Choose processing mode:")
        print("1. Process a single station")
        print("2. Process all Category A stations")
        choice = input("Enter choice (1 or 2): ").strip()
        
        if choice == "2":
            main(process_all_category_a=True)
        else:
            main()
