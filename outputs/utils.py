# UTILS FOR GRAPHS!!
# 
# UPDATED: Both aggregate_view and incident_view functions now use the NEW parquet file structure:
# - File structure: processed_data/{station_id}/{day}.parquet (e.g., processed_data/11271/MO.parquet)
# - Uses fastparquet engine to handle parquet files with corruption issues
# - incident_view uses SMART LOADING: only loads files for the specific day of week (efficient)
# - aggregate_view loads ALL files for comprehensive analysis (less efficient but thorough)

import json
import pickle
import sys
import os
import pandas as pd
import copy
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from datetime import datetime, timedelta
import numpy as np
import glob

# Helper function to find processed_data directory
def find_processed_data_path():
    """
    Find the processed_data directory by checking multiple possible locations.
    Returns the path if found, None otherwise.
    """
    possible_paths = [
        '../processed_data',  # From outputs/ or demos/
        './processed_data',   # From root directory
        'processed_data',     # From root directory
        os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'processed_data')  # Absolute path
    ]
    
    for path in possible_paths:
        if os.path.exists(path):
            return path
    return None

# for aggregate view:

def aggregate_view(incident_number, date):
    """
    Enhanced version that works with the new parquet file structure in station folders
    and creates meaningful charts with fixed 24-hour timeline from midnight to 23:59 
    for easy day-to-day comparison.
    """
    
    # Find all parquet files in the new structure
    processed_base = find_processed_data_path()
    
    if processed_base is None:
        print("No processed_data directory found in any expected location. Please run the preprocessor first.")
        return None
    
    # Get all station directories
    station_dirs = [d for d in os.listdir(processed_base) 
                   if os.path.isdir(os.path.join(processed_base, d))]
    
    print(f"Found {len(station_dirs)} station directories")
    
    all_incidents = []
    files_processed = 0
    files_with_data = 0
    
    # Load data from all station directories
    for station_dir in station_dirs:
        station_path = os.path.join(processed_base, station_dir)
        
        # Get all parquet files in this station directory (MO, TU, WE, TH, FR, SA, SU)
        parquet_files = glob.glob(os.path.join(station_path, "*.parquet"))
        
        for file_path in parquet_files:
            files_processed += 1
            try:
                # Use fastparquet engine to read the file
                station_data = pd.read_parquet(file_path, engine='fastparquet')
                
                if isinstance(station_data, pd.DataFrame) and len(station_data) > 0:
                    # Handle incident number matching - check if INCIDENT_NUMBER column exists and has data
                    if 'INCIDENT_NUMBER' in station_data.columns:
                        # Remove rows where INCIDENT_NUMBER is null
                        station_data = station_data.dropna(subset=['INCIDENT_NUMBER'])
                        
                        if len(station_data) == 0:
                            continue
                        
                        # Handle incident number matching
                        try:
                            incident_float = float(incident_number)
                            incident_mask = (station_data['INCIDENT_NUMBER'] == incident_float)
                        except (ValueError, TypeError):
                            incident_mask = (station_data['INCIDENT_NUMBER'].astype(str) == str(incident_number))
                        
                        # Filter by date if EVENT_DATETIME exists
                        if 'EVENT_DATETIME' in station_data.columns:
                            # Remove rows where EVENT_DATETIME is null
                            station_data = station_data.dropna(subset=['EVENT_DATETIME'])
                            
                            if len(station_data) == 0:
                                continue
                            
                            # Convert EVENT_DATETIME to datetime and extract date
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
                                files_with_data += 1
                                all_incidents.extend(filtered_data.to_dict('records'))
                        
            except Exception as e:
                print(f"Error processing file {file_path}: {str(e)[:100]}...")
                continue
    
    print(f"Processed {files_processed} files, {files_with_data} files had matching data")
    
    if not all_incidents:
        print(f"No incidents found for INCIDENT_NUMBER {incident_number} on {date}")
        return None
    
    # Convert to DataFrame for easier manipulation
    df = pd.DataFrame(all_incidents)
    print(f"Total records found: {len(df)}")
    
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
    fig.suptitle(f'Incident Analysis: {incident_number} on {date}', fontsize=20, fontweight='bold')
    
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
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
        ax.grid(True, alpha=0.3)
    
    # Chart 1: Hourly Delay Totals (24-Hour View)
    
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
                    ha='center', va='bottom', fontsize=20, fontweight='bold')
    
    # Add INCIDENT_START_DATETIME markers
    if 'start_chart_datetime' in df.columns:
        incident_start_times = df[df['start_chart_datetime'].notna()]['start_chart_datetime'].unique()
        for incident_start_time in incident_start_times:
            ax1.axvline(x=incident_start_time, color='red', linestyle='--', linewidth=3, alpha=0.9)
        
        # Add legend entry for incident start time
        if len(incident_start_times) > 0:
            ax1.axvline(x=incident_start_times[0], color='red', linestyle='--', linewidth=3, alpha=0.9, 
                       label='Incident Start Time')
    
    ax1.legend()
    
    ax1.set_ylabel('Total Delay Minutes per Hour', fontsize=20)
    ax1.set_xlabel('Hour of Day', fontsize=12)
    format_time_axis(ax1)
    add_time_shading(ax1)
    
    # Chart 2: Delay Severity Distribution
    
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
                        f'{count}', ha='center', va='bottom', fontsize=20, fontweight='bold')
        
        ax2.set_ylabel('Number of Delay Events', fontsize=20)
        ax2.set_xlabel('Delay Severity Range', fontsize=20)
        
        # Add total delay info
        total_events = len(delay_data)
        avg_delay = delay_values.mean()
        ax2.text(0.02, 0.98, f'Total Events: {total_events}\nAverage Delay: {avg_delay:.1f} min', 
                transform=ax2.transAxes, verticalalignment='top', fontsize=20,
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    else:
        ax2.text(0.5, 0.5, 'No delay events found', transform=ax2.transAxes, 
                ha='center', va='center', fontsize=20)
    
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Chart 3: Event Timeline (Delays and Cancellations)
    
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
        
        # Text annotations for cancellations removed - legend shows cancellation info
    
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
        ax3.legend()
    
    ax3.set_ylabel('Delay Minutes', fontsize=20)
    ax3.set_xlabel('Time of Day (24-Hour Timeline)', fontsize=20)
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
        "Files Processed": files_processed,
        "Files with Data": files_with_data,
        "Incident Number": incident_number,
        "Date": date,
        "Time Range": f"{df['full_datetime'].min().strftime('%H:%M')} - {df['full_datetime'].max().strftime('%H:%M')}" if len(df) > 0 else "N/A",
        "Peak Delay Event": peak_delay_info,
        "Peak Regular Delay": peak_regular_delay,
        "Peak Cancellation Delay": peak_cancellation_delay
    }
    
    return summary


# for incident view:

def incident_view(incident_code, incident_date, analysis_date, analysis_hhmm, period_minutes):
    """
    Generate a detailed table showing each station affected by an incident with their calls and delays
    for a specific time period during the incident lifecycle.
    Shows trains that were shifted between time periods due to delays.
    
    Parameters:
    incident_code (int/float): The incident number to analyze
    incident_date (str): Incident date in 'DD-MMM-YYYY' format (used to locate the incident)
    analysis_date (str): Specific date to analyze in 'DD-MMM-YYYY' format
    analysis_hhmm (str): Start time for analysis in 'HHMM' format (e.g., '1830' for 18:30)
    period_minutes (int): Minutes from analysis start time to analyze
    
    Returns:
    tuple: (pandas.DataFrame, str, str) - Results table, incident start time string, and analysis period string
    """
    
    # Parse analysis time inputs
    try:
        analysis_datetime = datetime.strptime(f"{analysis_date} {analysis_hhmm[:2]}:{analysis_hhmm[2:]}", '%d-%b-%Y %H:%M')
    except ValueError:
        print(f"Error: Invalid analysis date/time format. Use 'DD-MMM-YYYY' for date and 'HHMM' for time.")
        return pd.DataFrame(), None, None
    
    analysis_end = analysis_datetime + timedelta(minutes=period_minutes)
    analysis_period_str = f"{analysis_datetime.strftime('%d-%b-%Y %H:%M')} to {analysis_end.strftime('%d-%b-%Y %H:%M')} ({period_minutes} min)"
    
    # Determine day of week from the analysis date (not incident date)
    day_mapping = {0: 'MO', 1: 'TU', 2: 'WE', 3: 'TH', 4: 'FR', 5: 'SA', 6: 'SU'}
    analysis_day_suffix = day_mapping[analysis_datetime.weekday()]
    
    # Load station files for the analysis day of week using NEW parquet structure
    processed_base = find_processed_data_path()
    
    if processed_base is None:
        print("No processed_data directory found in any expected location. Please run the preprocessor first.")
        return pd.DataFrame(), None, None
    
    # Get all station directories
    station_dirs = [d for d in os.listdir(processed_base) 
                   if os.path.isdir(os.path.join(processed_base, d))]
    
    # Build list of files to load (only for the specific day)
    target_files = []
    for station_dir in station_dirs:
        station_path = os.path.join(processed_base, station_dir)
        day_file = os.path.join(station_path, f"{analysis_day_suffix}.parquet")
        if os.path.exists(day_file):
            target_files.append((day_file, station_dir))
    
    print(f"Analyzing incident {incident_code} (started {incident_date})")
    print(f"Analysis period: {analysis_period_str}")
    print(f"Loading {len(target_files)} station files for {analysis_day_suffix}")
    
    # Find incident start time and collect data from each station
    incident_start_time = None
    incident_delay_day = None
    incident_section_code = None
    incident_reason = None
    station_results = []  # List to store results from each station
    
    for file_path, station_code in target_files:
        try:
            # Use fastparquet engine to read the file
            df = pd.read_parquet(file_path, engine='fastparquet')
            if not isinstance(df, pd.DataFrame):
                continue
                
            # Filter for the specific incident using the incident_date to locate it
            incident_data = df[df['INCIDENT_NUMBER'] == incident_code].copy()
            if incident_data.empty:
                continue
                
            incident_data['incident_date'] = incident_data['INCIDENT_START_DATETIME'].str.split(' ').str[0]
            incident_records = incident_data[incident_data['incident_date'] == incident_date].copy()
            if incident_records.empty:
                continue
                
            # Get incident start time (use first occurrence)
            if incident_start_time is None:
                incident_start_time = incident_records['INCIDENT_START_DATETIME'].iloc[0]
                incident_delay_day = incident_records['DELAY_DAY'].iloc[0]
                incident_section_code = incident_records['SECTION_CODE'].iloc[0]
                incident_reason = incident_records['INCIDENT_REASON'].iloc[0]
                
                # Validate that analysis time is after incident start
                incident_start_dt = datetime.strptime(incident_start_time, '%d-%b-%Y %H:%M')
                if analysis_datetime < incident_start_dt:
                    print(f"Warning: Analysis time ({analysis_datetime.strftime('%d-%b-%Y %H:%M')}) is before incident start ({incident_start_dt.strftime('%d-%b-%Y %H:%M')})")
            
            # For planned calls: Find what was originally scheduled for the analysis time period
            # Filter for same ENGLISH_DAY_TYPE but NO INCIDENT_NUMBER (non-delayed trains)
            planned_calls = 0
            if incident_delay_day and not pd.isna(incident_delay_day):
                non_delayed_data = df[
                    (df['INCIDENT_NUMBER'].isna()) &  # No incident number (non-delayed)
                    (df['ENGLISH_DAY_TYPE'].apply(lambda x: incident_delay_day in x if isinstance(x, list) else False))
                ].copy()
                
                if not non_delayed_data.empty:
                    # Use PLANNED_CALLS field for time filtering (format appears to be HHMM)
                    non_delayed_data = non_delayed_data[non_delayed_data['PLANNED_CALLS'].notna()].copy()
                    
                    if not non_delayed_data.empty:
                        # Convert PLANNED_CALLS to time format for comparison
                        def parse_time_string(time_str):
                            """Parse time string like '0001', '1230', '2359' to time object"""
                            try:
                                if isinstance(time_str, str) and len(time_str) >= 4:
                                    # Remove any trailing letters (like 'H') and take first 4 digits
                                    clean_time = ''.join(filter(str.isdigit, time_str))[:4]
                                    if len(clean_time) == 4:
                                        hour = int(clean_time[:2])
                                        minute = int(clean_time[2:])
                                        # Handle 24:XX format by converting to 00:XX
                                        if hour >= 24:
                                            hour = hour % 24
                                        return datetime.strptime(f"{hour:02d}:{minute:02d}", "%H:%M").time()
                                return None
                            except:
                                return None
                        
                        non_delayed_data['planned_time'] = non_delayed_data['PLANNED_CALLS'].apply(parse_time_string)
                        non_delayed_data = non_delayed_data[non_delayed_data['planned_time'].notna()].copy()
                        
                        if not non_delayed_data.empty:
                            # Get time window for filtering (use analysis time period)
                            analysis_start_time_only = analysis_datetime.time()
                            analysis_end_time_only = analysis_end.time()
                            
                            # Filter non-delayed trains to the analysis time period
                            if analysis_end_time_only > analysis_start_time_only:
                                # Same day period
                                period_non_delayed = non_delayed_data[
                                    (non_delayed_data['planned_time'] >= analysis_start_time_only) & 
                                    (non_delayed_data['planned_time'] <= analysis_end_time_only)
                                ].copy()
                            else:
                                # Period crosses midnight
                                period_non_delayed = non_delayed_data[
                                    (non_delayed_data['planned_time'] >= analysis_start_time_only) | 
                                    (non_delayed_data['planned_time'] <= analysis_end_time_only)
                                ].copy()
                            
                            # Count unique trains instead of total rows
                            planned_calls = period_non_delayed['TRAIN_SERVICE_CODE'].nunique()
            
            # For delayed trains analysis - now look at ALL incident data (not just from incident date)
            # Get all records for this incident regardless of date
            all_incident_data = df[df['INCIDENT_NUMBER'] == incident_code].copy()
            
            if not all_incident_data.empty:
                all_incident_data['EVENT_DATETIME_parsed'] = pd.to_datetime(all_incident_data['EVENT_DATETIME'], format='%d-%b-%Y %H:%M', errors='coerce')
                all_incident_data = all_incident_data[all_incident_data['EVENT_DATETIME_parsed'].notna()].copy()
                all_incident_data = all_incident_data[all_incident_data['PFPI_MINUTES'].notna()].copy()
                
                # Initialize variables
                delayed_trains_out = 0
                delay_minutes_out = []
                delayed_trains_in = 0
                delay_minutes_in = []
                
                if not all_incident_data.empty:
                    # Calculate original scheduled time = actual time - delay
                    all_incident_data['original_scheduled_time'] = (
                        all_incident_data['EVENT_DATETIME_parsed'] - 
                        pd.to_timedelta(all_incident_data['PFPI_MINUTES'], unit='minutes')
                    )
                    
                    # TRAINS ORIGINALLY SCHEDULED FOR THE ANALYSIS PERIOD
                    originally_scheduled_in_period = all_incident_data[
                        (all_incident_data['original_scheduled_time'] >= analysis_datetime) & 
                        (all_incident_data['original_scheduled_time'] <= analysis_end)
                    ].copy()
                    
                    if not originally_scheduled_in_period.empty:
                        # Trains that were delayed OUT of the analysis period
                        trains_delayed_out_of_period = originally_scheduled_in_period[
                            (originally_scheduled_in_period['EVENT_DATETIME_parsed'] > analysis_end)
                        ]
                        
                        # Count unique trains instead of total rows
                        delayed_trains_out = trains_delayed_out_of_period['TRAIN_SERVICE_CODE'].nunique()
                        delay_minutes_out = trains_delayed_out_of_period['PFPI_MINUTES'].dropna().tolist()
                    
                    # TRAINS DELAYED INTO THE ANALYSIS PERIOD (originally scheduled for earlier periods)
                    trains_delayed_into_period = all_incident_data[
                        (all_incident_data['original_scheduled_time'] < analysis_datetime) &  # Originally before analysis period
                        (all_incident_data['EVENT_DATETIME_parsed'] >= analysis_datetime) &   # Actually arrived in analysis period
                        (all_incident_data['EVENT_DATETIME_parsed'] <= analysis_end)
                    ]
                    
                    # Count unique trains instead of total rows
                    delayed_trains_in = trains_delayed_into_period['TRAIN_SERVICE_CODE'].nunique()
                    delay_minutes_in = trains_delayed_into_period['PFPI_MINUTES'].dropna().tolist()
                
                # ACTUAL_CALLS = PLANNED_CALLS - delayed out + delayed in
                actual_calls = planned_calls - delayed_trains_out + delayed_trains_in
                
                # Only include stations that have data in this period
                if (planned_calls > 0 or delayed_trains_out > 0 or delayed_trains_in > 0):
                    # Store results for this station
                    station_results.append({
                        'STATION_CODE': station_code,
                        'PLANNED_CALLS': planned_calls,
                        'ACTUAL_CALLS': actual_calls,
                        'DELAYED_TRAINS_OUT': delayed_trains_out,
                        'DELAY_MINUTES_OUT': delay_minutes_out,
                        'DELAYED_TRAINS_IN': delayed_trains_in,
                        'DELAY_MINUTES_IN': delay_minutes_in
                    })
                
        except Exception as e:
            continue
    
    # Handle case where no incident start time was found
    if incident_start_time is None:
        print(f"Incident {incident_code} not found on {incident_date}")
        return pd.DataFrame(), None, None
    
    # Get station name for the section code
    incident_station_name = None
    if incident_section_code:
        try:
            # Load station codes reference file
            from data.reference import reference_files
            import json  # Ensure json is available in this scope
            with open(reference_files["station codes"], 'r') as f:
                station_codes_data = json.load(f)
            
            # Find station matching the section code
            for record in station_codes_data:
                if isinstance(record, dict) and str(record.get('stanox')) == str(incident_section_code):
                    incident_station_name = record.get('description')
                    break
        except Exception:
            pass  # Continue without station name if lookup fails
    
    # Print incident details
    print(f"Incident Details:")
    print(f"  Section Code: {incident_section_code}" + (f" ({incident_station_name})" if incident_station_name else ""))
    print(f"  Incident Reason: {incident_reason}")
    print(f"  Started: {incident_start_time}")
    
    if not station_results:
        print(f"No station activity found during analysis period")
        return pd.DataFrame(), incident_start_time, analysis_period_str
    
    # Create DataFrame directly from the list of results
    result_df = pd.DataFrame(station_results)
    result_df = result_df.sort_values('STATION_CODE').reset_index(drop=True)
    
    # Return results, incident start time, and analysis period description
    return result_df, incident_start_time, analysis_period_str


# for incident view but html:

def incident_view_html(incident_code, incident_date, analysis_date, analysis_hhmm, period_minutes, interval_minutes=10, output_file=None):
    """
    Create a dynamic interactive HTML map showing PERIOD-SPECIFIC delay minutes over time
    for a specific incident analysis period on a geographic map with timeline controls.
    
    The analysis period is split into intervals, and each interval shows the total delay
    minutes for that specific interval (NOT cumulative). Delays can go up or down between intervals.
    
    Parameters:
    incident_code (int/float): The incident number to analyze
    incident_date (str): Date when incident started in 'DD-MMM-YYYY' format
    analysis_date (str): Specific date to analyze in 'DD-MMM-YYYY' format
    analysis_hhmm (str): Start time for analysis in 'HHMM' format (e.g., '1900')
    period_minutes (int): Total duration of analysis period in minutes
    interval_minutes (int): Duration of each interval in minutes (default: 10)
    output_file (str): Optional HTML file path to save
    
    Returns:
    str: HTML content of the interactive map
    """
    
    # Load station coordinates data from comprehensive JSON file
    try:
        from data.reference import reference_files
        import json
        
        file_path = reference_files["all dft categories"]
        print(f"Attempting to load station coordinates from: {file_path}")
        
        if not os.path.exists(file_path):
            print(f"File does not exist: {file_path}")
            print("Available reference files:", list(reference_files.keys()))
            return None
            
        with open(file_path, 'r') as f:
            stations_coords_data = json.load(f)
            
        print(f"Successfully loaded {len(stations_coords_data)} station records")
        
    except KeyError as e:
        print(f"Reference file key not found: {e}")
        print("Available reference files:", list(reference_files.keys()))
        return None
    except FileNotFoundError:
        print("Station coordinates file not available. Cannot create map.")
        return None
    except Exception as e:
        print(f"Error loading station coordinates: {e}")
        return None
    
    # Parse analysis parameters
    try:
        analysis_datetime = datetime.strptime(f"{analysis_date} {analysis_hhmm[:2]}:{analysis_hhmm[2:]}", '%d-%b-%Y %H:%M')
    except ValueError:
        print(f"Error: Invalid analysis date/time format.")
        return None
    
    analysis_end = analysis_datetime + timedelta(minutes=period_minutes)
    
    print(f"Creating dynamic HTML map for incident {incident_code}")
    print(f"Analysis period: {analysis_datetime.strftime('%d-%b-%Y %H:%M')} to {analysis_end.strftime('%d-%b-%Y %H:%M')}")
    print(f"Interval size: {interval_minutes} minutes")
    
    # Calculate number of intervals
    num_intervals = period_minutes // interval_minutes
    if period_minutes % interval_minutes != 0:
        num_intervals += 1  # Include partial interval at the end
    
    print(f"Total intervals: {num_intervals}")
    
    # Determine day of week for analysis date
    day_mapping = {0: 'MO', 1: 'TU', 2: 'WE', 3: 'TH', 4: 'FR', 5: 'SA', 6: 'SU'}
    analysis_day_suffix = day_mapping[analysis_datetime.weekday()]
    
    # Load station files for the analysis day using NEW parquet structure
    processed_base = find_processed_data_path()
    
    if processed_base is None:
        print("No processed_data directory found in any expected location. Please run the preprocessor first.")
        return None
    
    # Get all station directories
    station_dirs = [d for d in os.listdir(processed_base) 
                   if os.path.isdir(os.path.join(processed_base, d))]
    
    # Build list of files to load (only for the specific day)
    target_files = []
    for station_dir in station_dirs:
        station_path = os.path.join(processed_base, station_dir)
        day_file = os.path.join(station_path, f"{analysis_day_suffix}.parquet")
        if os.path.exists(day_file):
            target_files.append((day_file, station_dir))
    
    print(f"Loading {len(target_files)} station files for {analysis_day_suffix}")
    
    # Create station coordinates mapping from JSON data
    station_coords_map = {}
    for station in stations_coords_data:
        if isinstance(station, dict):
            station_id = str(station.get('stanox', ''))
            latitude = station.get('latitude')
            longitude = station.get('longitude')
            description = station.get('description', 'Unknown Station')
            
            # Check if coordinates are valid (coordinates may be strings that can be converted to float)
            if (station_id and 
                latitude is not None and longitude is not None and
                str(latitude).replace('.', '').replace('-', '').isdigit() and 
                str(longitude).replace('.', '').replace('-', '').isdigit()):
                try:
                    station_coords_map[station_id] = {
                        'name': description,
                        'lat': float(latitude),
                        'lon': float(longitude)
                    }
                except ValueError:
                    # Skip station if coordinate conversion fails
                    continue
    
    print(f"Found coordinates for {len(station_coords_map)} stations")
    
    # Collect delay timeline data for all stations
    station_timeline_data = {}  # {station_code: [(datetime, cumulative_delay_minutes)]}
    incident_section_code = None
    incident_reason = None
    incident_start_time = None
    
    # Debug: show first few station codes we're looking for
    debug_station_codes = [station_code for _, station_code in target_files[:5]]
    print(f"Sample station codes from processed data: {debug_station_codes}")
    debug_coord_keys = list(station_coords_map.keys())[:5]
    print(f"Sample station IDs from coordinates: {debug_coord_keys}")
    
    for file_path, station_code in target_files:
        if station_code not in station_coords_map:
            continue  # Skip stations without coordinates
            
        try:
            # Use fastparquet engine to read the file
            df = pd.read_parquet(file_path, engine='fastparquet')
            if not isinstance(df, pd.DataFrame):
                continue
                
            # Get all incident data for this station
            all_incident_data = df[df['INCIDENT_NUMBER'] == incident_code].copy()
            if all_incident_data.empty:
                continue
            
            # Extract incident information on first occurrence
            if incident_section_code is None and not all_incident_data.empty:
                # Get incident details from the first record that matches the incident date
                incident_records = all_incident_data[all_incident_data['INCIDENT_START_DATETIME'].str.contains(incident_date, na=False)]
                if not incident_records.empty:
                    incident_section_code = incident_records['SECTION_CODE'].iloc[0]
                    incident_reason = incident_records['INCIDENT_REASON'].iloc[0]
                    incident_start_time = incident_records['INCIDENT_START_DATETIME'].iloc[0]
            
            # Filter to events within our analysis period
            all_incident_data['EVENT_DATETIME_parsed'] = pd.to_datetime(all_incident_data['EVENT_DATETIME'], format='%d-%b-%Y %H:%M', errors='coerce')
            all_incident_data = all_incident_data[all_incident_data['EVENT_DATETIME_parsed'].notna()].copy()
            
            # Filter to analysis period
            period_data = all_incident_data[
                (all_incident_data['EVENT_DATETIME_parsed'] >= analysis_datetime) &
                (all_incident_data['EVENT_DATETIME_parsed'] <= analysis_end) &
                (all_incident_data['PFPI_MINUTES'].notna())
            ].copy()
            
            if period_data.empty:
                continue
            
            # Create timeline of INTERVAL-SPECIFIC delays for this station (NOT cumulative)
            # Group delays by the specified interval_minutes
            interval_delays = {}  # {interval_start_time: total_delay_for_interval}
            
            for _, row in period_data.iterrows():
                event_time = row['EVENT_DATETIME_parsed']
                delay = row['PFPI_MINUTES']
                
                # Calculate which interval this event belongs to
                minutes_from_start = int((event_time - analysis_datetime).total_seconds() / 60)
                interval_number = minutes_from_start // interval_minutes
                
                # Calculate the start time of this interval
                interval_start = analysis_datetime + timedelta(minutes=interval_number * interval_minutes)
                
                if interval_start not in interval_delays:
                    interval_delays[interval_start] = 0
                interval_delays[interval_start] += delay
            
            # Convert to timeline format - each timestamp shows total delays for that interval
            timeline = []
            for interval_start in sorted(interval_delays.keys()):
                interval_total_delay = interval_delays[interval_start]
                timeline.append((interval_start, interval_total_delay))
            
            if timeline:
                station_timeline_data[station_code] = timeline
                
        except Exception as e:
            continue
    
    if not station_timeline_data:
        print("No delay data found for the analysis period")
        return None
    
    print(f"Found delay data for {len(station_timeline_data)} stations")
    
    # Get incident location coordinates
    incident_lat = 54.5  # Default to center of UK
    incident_lon = -2.0
    incident_station_name = None
    
    if incident_section_code:
        try:
            # Load station codes reference file
            from data.reference import reference_files
            import json  # Ensure json is available in this scope
            with open(reference_files["station codes"], 'r') as f:
                station_codes_data = json.load(f)
            
            # Find station matching the section code
            for record in station_codes_data:
                if isinstance(record, dict) and str(record.get('stanox')) == str(incident_section_code):
                    if record.get('latitude') and record.get('longitude'):
                        incident_lat = float(record['latitude'])
                        incident_lon = float(record['longitude'])
                        incident_station_name = record.get('description')
                        print(f"Found incident location: {incident_station_name} ({incident_lat}, {incident_lon})")
                        break
        except Exception as e:
            print(f"Warning: Could not load incident location coordinates: {e}")
    
    # Create time steps for animation using the specified interval
    time_steps = []
    current_time = analysis_datetime
    while current_time < analysis_end:
        time_steps.append(current_time)
        current_time += timedelta(minutes=interval_minutes)
    
    # Build timeline data structure for JavaScript
    timeline_data = {}
    for step_time in time_steps:
        time_key = step_time.strftime('%H:%M')
        timeline_data[time_key] = {}
        
        # For each station, find the SPECIFIC delay for this exact interval (NOT cumulative)
        for station_code, station_timeline in station_timeline_data.items():
            interval_delay = 0
            # Find delays that occurred exactly in this interval
            for event_time, delay_amount in station_timeline:
                # Check if this event happened in this specific interval window
                time_window_start = step_time
                time_window_end = step_time + timedelta(minutes=interval_minutes)
                
                if time_window_start <= event_time < time_window_end:
                    interval_delay += delay_amount
            
            # Only include if there are delays in this specific interval
            if interval_delay > 0:
                timeline_data[time_key][station_code] = interval_delay
    
    # Prepare data for JavaScript
    import json
    station_coords_json = json.dumps(station_coords_map)
    timeline_data_json = json.dumps(timeline_data)
    time_steps_json = json.dumps([t.strftime('%H:%M') for t in time_steps])
    
    # Create HTML content
    html_content = f'''<!DOCTYPE html>
<html>
<head>
    <title>Incident {incident_code} - Period-Specific Delay Analysis</title>
    <meta charset="utf-8" />
    <link rel="stylesheet" href="https://unpkg.com/leaflet@1.7.1/dist/leaflet.css" />
    <script src="https://unpkg.com/leaflet@1.7.1/dist/leaflet.js"></script>
    <style>
        body {{ margin: 0; font-family: Arial, sans-serif; background: #f5f5f5; }}
        #map {{ height: 75vh; }}
        #controls {{ 
            height: 25vh; 
            padding: 20px; 
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
        }}
        .control-panel {{ 
            background: rgba(255,255,255,0.95); 
            padding: 15px; 
            margin: 10px 0; 
            border-radius: 10px; 
            box-shadow: 0 4px 15px rgba(0,0,0,0.2);
            color: #333;
        }}
        #timeline {{ width: 100%; margin: 10px 0; height: 8px; }}
        .time-display {{ 
            font-size: 24px; 
            font-weight: bold; 
            text-align: center; 
            color: #2c3e50;
            margin: 10px 0;
        }}
        .play-controls {{ text-align: center; margin: 15px 0; }}
        .btn {{ 
            background: #3498db; 
            color: white; 
            border: none; 
            padding: 10px 20px; 
            margin: 0 5px; 
            border-radius: 25px; 
            cursor: pointer; 
            font-size: 16px;
            transition: all 0.3s ease;
        }}
        .btn:hover {{ background: #2980b9; transform: translateY(-2px); }}
        .btn:active {{ transform: translateY(0); }}
        #station-info {{ 
            max-height: 120px; 
            overflow-y: auto; 
            margin-top: 10px;
            font-size: 14px;
        }}
        .station-delay {{ 
            display: inline-block; 
            margin: 2px 5px; 
            padding: 3px 8px; 
            border-radius: 15px; 
            font-size: 12px;
        }}
        .delay-low {{ background: #2ecc71; color: white; }}
        .delay-med {{ background: #f39c12; color: white; }}
        .delay-high {{ background: #e74c3c; color: white; }}
        .delay-extreme {{ background: #8e44ad; color: white; }}
        .legend {{ 
            background: rgba(255,255,255,0.95); 
            padding: 10px; 
            border-radius: 8px;
            margin: 10px 0;
            font-size: 12px;
            color: #333;
        }}
        .legend-item {{ 
            display: inline-block;
            margin: 3px 8px 3px 0; 
        }}
        .legend-color {{ 
            display: inline-block; 
            width: 15px; 
            height: 15px; 
            border-radius: 50%; 
            margin-right: 5px; 
            vertical-align: middle;
        }}
    </style>
</head>
<body>
    <!-- Map Container -->
    <div id="map">
    </div>
    
    <!-- Controls Container -->
    <div id="controls">
        <div class="control-panel">
            <h3 style="margin-top: 0;">Incident {incident_code} - {analysis_date}</h3>
            <p style="margin: 5px 0;">Analysis Period: {analysis_datetime.strftime('%H:%M')} - {analysis_end.strftime('%H:%M')} ({period_minutes} min total, {interval_minutes}-min intervals)</p>
            <p style="margin: 5px 0; font-weight: bold;">Section: {incident_section_code or 'N/A'}{' (' + incident_station_name + ')' if incident_station_name else ''} | Reason: {incident_reason or 'N/A'} | Started: {incident_start_time or 'N/A'}</p>
            
            <div class="time-display" id="current-time">{time_steps[0].strftime('%H:%M')}</div>
            
            <input type="range" id="timeline" min="0" max="{len(time_steps)-1}" value="0" step="1">
            
            <div class="play-controls">
                <button class="btn" onclick="playTimeline()">▶ Play</button>
                <button class="btn" onclick="pauseTimeline()">⏸ Pause</button>
                <button class="btn" onclick="resetTimeline()">⏮ Reset</button>
            </div>
            
            <div class="legend">
                <strong>Interval Delay Minutes:</strong>
                <div class="legend-item"><span class="legend-color" style="background: #2ecc71;"></span>1-15 min</div>
                <div class="legend-item"><span class="legend-color" style="background: #f39c12;"></span>16-45 min</div>
                <div class="legend-item"><span class="legend-color" style="background: #e74c3c;"></span>46-90 min</div>
                <div class="legend-item"><span class="legend-color" style="background: #8e44ad;"></span>90+ min</div>
            </div>
            
            <div id="station-info">No delays at this time</div>
        </div>
    </div>

    <script>
        // Initialize map
        var map = L.map('map').setView([54.5, -2.0], 6);
        L.tileLayer('https://{{s}}.tile.openstreetmap.org/{{z}}/{{x}}/{{y}}.png', {{
            attribution: '© OpenStreetMap contributors'
        }}).addTo(map);
        
        // Add incident location marker at actual coordinates
        var incidentMarker = L.marker([{incident_lat}, {incident_lon}], {{
            icon: L.divIcon({{
                html: '<div style="font-size: 24px; color: red; font-weight: bold;">✕</div>',
                className: 'incident-marker',
                iconSize: [30, 30],
                iconAnchor: [15, 15]
            }})
        }}).addTo(map);
        
        incidentMarker.bindPopup(`
            <strong>Incident Location</strong><br>
            Station: {incident_station_name or 'Unknown'}<br>
            Incident: {incident_code}<br>
            Section: {incident_section_code or 'N/A'}<br>
            Reason: {incident_reason or 'N/A'}<br>
            Started: {incident_start_time or 'N/A'}<br>
            <em>Coordinates: {incident_lat:.4f}, {incident_lon:.4f}</em>
        `);
        
        // Data
        var stationCoords = {station_coords_json};
        var timelineData = {timeline_data_json};
        var timeSteps = {time_steps_json};
        
        // State
        var currentIndex = 0;
        var isPlaying = false;
        var playInterval;
        var markers = {{}};
        
        // Color functions
        function getDelayColor(delayMinutes) {{
            if (delayMinutes >= 90) return '#8e44ad';      // Purple for extreme delays
            if (delayMinutes >= 46) return '#e74c3c';      // Red for high delays
            if (delayMinutes >= 16) return '#f39c12';      // Orange for medium delays
            return '#2ecc71';                              // Green for low delays
        }}
        
        // Update map for specific time index
        function updateMap(index) {{
            try {{
                currentIndex = index;
                var timeKey = timeSteps[index];
                var delays = timelineData[timeKey] || {{}};
                
                console.log('🕐 Updating map for time:', timeKey, 'with', Object.keys(delays).length, 'stations');
                
                // Clear existing markers
                Object.values(markers).forEach(m => map.removeLayer(m));
                markers = {{}};
                
                // Update time display
                document.getElementById('current-time').textContent = timeKey;
            
            // Create markers for stations with delays
            var stationInfoHtml = '';
            var totalSystemDelay = 0;
            var stationCount = 0;
            
            Object.entries(delays).forEach(([stationCode, delayMinutes]) => {{
                if (stationCoords[stationCode] && delayMinutes > 0) {{
                    var coords = stationCoords[stationCode];
                    
                    // Create circle marker (same size, different colors)
                    var marker = L.circleMarker([coords.lat, coords.lon], {{
                        radius: 8,  // Fixed size
                        fillColor: getDelayColor(delayMinutes),
                        color: '#000',
                        weight: 2,
                        opacity: 1,
                        fillOpacity: 0.8
                    }});
                    
                    // Calculate end time for 5-minute window
                    var timeparts = timeKey.split(':');
                    var hours = parseInt(timeparts[0]);
                    var minutes = parseInt(timeparts[1]);
                    var endMinutes = (minutes + 5) % 60;
                    var endHours = hours + Math.floor((minutes + 5) / 60);
                    var endTime = endHours.toString().padStart(2, '0') + ':' + endMinutes.toString().padStart(2, '0');
                    
                    // Calculate end time for interval window
                    var timeparts = timeKey.split(':');
                    var hours = parseInt(timeparts[0]);
                    var minutes = parseInt(timeparts[1]);
                    var endMinutes = (minutes + {interval_minutes}) % 60;
                    var endHours = hours + Math.floor((minutes + {interval_minutes}) / 60);
                    var endTime = endHours.toString().padStart(2, '0') + ':' + endMinutes.toString().padStart(2, '0');
                    
                    marker.bindPopup(`
                        <strong>${{coords.name}}</strong><br>
                        Station: ${{stationCode}}<br>
                        Interval Delay: ${{delayMinutes}} minutes<br>
                        Time Window: ${{timeKey}} - ${{endTime}} ({interval_minutes} min)
                    `);
                    
                    marker.addTo(map);
                    markers[stationCode] = marker;
                    
                    // Add to station info
                    var delayClass = delayMinutes >= 90 ? 'delay-extreme' : 
                                   delayMinutes >= 46 ? 'delay-high' : 
                                   delayMinutes >= 16 ? 'delay-med' : 'delay-low';
                    
                    stationInfoHtml += `<span class="station-delay ${{delayClass}}">${{coords.name.substring(0,15)}}: ${{delayMinutes}}min</span>`;
                    
                    totalSystemDelay += delayMinutes;
                    stationCount++;
                }}
            }});
            
            // Update station info panel
            if (stationInfoHtml) {{
                var avgDelay = (totalSystemDelay / stationCount).toFixed(1);
                stationInfoHtml = `<strong>Interval Impact:</strong> ${{stationCount}} stations, ${{totalSystemDelay}} minutes in this {interval_minutes}-min interval (avg: ${{avgDelay}}min)<br><br>${{stationInfoHtml}}`;
            }} else {{
                stationInfoHtml = 'No delays in this interval';
            }}
            
            document.getElementById('station-info').innerHTML = stationInfoHtml;
            }} catch (error) {{
                console.error('❌ Error updating map:', error);
                document.getElementById('station-info').innerHTML = 'Error updating map: ' + error.message;
            }}
        }}
        
        // Timeline controls
        document.getElementById('timeline').addEventListener('input', function(e) {{
            pauseTimeline();
            updateMap(parseInt(e.target.value));
        }});
        
        function playTimeline() {{
            if (!isPlaying) {{
                isPlaying = true;
                playInterval = setInterval(() => {{
                    if (currentIndex < timeSteps.length - 1) {{
                        currentIndex++;
                        document.getElementById('timeline').value = currentIndex;
                        updateMap(currentIndex);
                    }} else {{
                        pauseTimeline();
                    }}
                }}, 1000);  // 1 second per time step
            }}
        }}
        
        function pauseTimeline() {{
            isPlaying = false;
            if (playInterval) clearInterval(playInterval);
        }}
        
        function resetTimeline() {{
            pauseTimeline();
            currentIndex = 0;
            document.getElementById('timeline').value = 0;
            updateMap(0);
        }}
        
        // Initialize with error checking
        console.log(' Starting map initialization...');
        console.log('Map object:', map);
        console.log('Station coordinates loaded:', Object.keys(stationCoords).length);
        console.log('Time steps:', timeSteps.length);
        console.log('Timeline data:', Object.keys(timelineData).length, 'time points');
        
        if (timeSteps.length === 0) {{
            console.error('❌ No time steps available!');
            document.getElementById('station-info').innerHTML = 'No time data available';
        }} else if (Object.keys(stationCoords).length === 0) {{
            console.error('❌ No station coordinates available!');
            document.getElementById('station-info').innerHTML = 'No station coordinates available';
        }} else if (Object.keys(timelineData).length === 0) {{
            console.error('❌ No timeline data available!');
            document.getElementById('station-info').innerHTML = 'No delay data available for this period';
        }} else {{
            console.log('✅ All data loaded successfully, initializing map...');
            updateMap(0);
        }}
        
        console.log('🎬 Map initialization complete!');
    </script>
</body>
</html>'''
    
    # Save HTML file
    if output_file is None:
        safe_date = analysis_date.replace('-', '_')
        safe_time = analysis_hhmm
        output_file = f'incident_{incident_code}_{safe_date}_{safe_time}_progressive.html'
    
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        print(f" DYNAMIC DELAY MAP CREATED! ")
        print(f"File: {output_file}")
        print(f"Time steps: {len(time_steps)} ({interval_minutes}-minute intervals)")
        print(f"Stations mapped: {len([s for s in station_timeline_data.keys() if s in station_coords_map])}")
        print(f" Features: Play/Pause controls, Color-coded delays, Interval-specific timeline")
        print(" Open the HTML file in your browser to explore the dynamic timeline!")
        
    except Exception as e:
        print(f"Error saving file: {e}")
    
    return html_content
    num_intervals = period_minutes // interval_minutes
    if period_minutes % interval_minutes != 0:
        num_intervals += 1
    
    print(f"📊 Creating {num_intervals} time intervals of {interval_minutes} minutes each")
    
    # Load coordinates for ALL stations to show the complete network
    print("�️ Loading coordinates for ALL network stations...")
    coord_file_path = reference_files["all dft categories"]
    with open(coord_file_path, 'r') as f:
        stations_coords_data = json.load(f)
    
    # Create station coordinates mapping for stations with valid DFT categories only
    station_coords_map = {}
    valid_categories = {'A', 'B', 'C1', 'C2'}
    
    for station in stations_coords_data:
        if isinstance(station, dict):
            station_id = str(station.get('stanox', ''))
            dft_category = station.get('dft_category', '')
            latitude = station.get('latitude')
            longitude = station.get('longitude')
            description = station.get('description', 'Unknown Station')
            
            # Only include stations with valid DFT categories (A, B, C1, C2) and coordinates
            if (dft_category in valid_categories and 
                latitude is not None and longitude is not None and
                str(latitude).replace('.', '').replace('-', '').isdigit() and 
                str(longitude).replace('.', '').replace('-', '').isdigit()):
                try:
                    station_coords_map[station_id] = {
                        'name': description,
                        'lat': float(latitude),
                        'lon': float(longitude),
                        'category': dft_category
                    }
                except ValueError:
                    continue
    
    print(f"📍 Loaded coordinates for {len(station_coords_map)} network stations (A/B/C1/C2 categories)")
    
    # Find affected stations across all intervals for delay data
    print("🔍 Finding affected stations with delays...")
    all_affected_stations = set()
    
    for i in range(num_intervals):
        interval_start = analysis_datetime + timedelta(minutes=i * interval_minutes)
        interval_start_str = interval_start.strftime('%H%M')
        actual_interval_minutes = min(interval_minutes, period_minutes - (i * interval_minutes))
        
        # Get delay data for this interval to find affected stations
        temp_data, _, _ = incident_view(incident_code, incident_date, analysis_date, interval_start_str, actual_interval_minutes)
        if not temp_data.empty:
            affected_in_interval = temp_data[(temp_data['DELAYED_TRAINS_OUT'] > 0) | (temp_data['DELAYED_TRAINS_IN'] > 0)]
            for _, row in affected_in_interval.iterrows():
                all_affected_stations.add(str(row['STATION_CODE']))
    
    print(f"🚉 Found {len(all_affected_stations)} stations with delays across all intervals")
    
    # Get incident location information for the 'X' marker
    incident_location = None
    incident_section_code = None
    incident_station_name = None
    
    # Get incident section code by replicating the logic from incident_view function
    # Look through the processed data to find the incident section code
    processed_base = find_processed_data_path()
    
    if processed_base is not None:
        # Get day of week for analysis date
        day_mapping = {0: 'MO', 1: 'TU', 2: 'WE', 3: 'TH', 4: 'FR', 5: 'SA', 6: 'SU'}
        analysis_day_suffix = day_mapping[analysis_datetime.weekday()]
        
        # Get all station directories and look for the incident
        station_dirs = [d for d in os.listdir(processed_base) 
                       if os.path.isdir(os.path.join(processed_base, d))]
        
        for station_dir in station_dirs[:5]:  # Just check first few stations to find section code
            station_path = os.path.join(processed_base, station_dir)
            day_file = os.path.join(station_path, f"{analysis_day_suffix}.parquet")
            if os.path.exists(day_file):
                try:
                    df = pd.read_parquet(day_file, engine='fastparquet')
                    incident_data = df[df['INCIDENT_NUMBER'] == incident_code].copy()
                    if not incident_data.empty:
                        incident_data['incident_date'] = incident_data['INCIDENT_START_DATETIME'].str.split(' ').str[0]
                        incident_records = incident_data[incident_data['incident_date'] == incident_date].copy()
                        if not incident_records.empty:
                            incident_section_code = incident_records['SECTION_CODE'].iloc[0]
                            break
                except:
                    continue
    
    # Find coordinates for incident location using STANOX code(s)
    # Handle both single STANOX codes and colon-separated section codes (e.g., "04303:04730")
    incident_locations = []  # Store multiple locations if section contains multiple STANOX codes
    
    if incident_section_code:
        # Split section code by colon to handle cases like "04303:04730"
        stanox_codes = [code.strip() for code in str(incident_section_code).split(':')]
        
        for stanox_code in stanox_codes:
            for station in stations_coords_data:
                if isinstance(station, dict) and str(station.get('stanox')) == str(stanox_code):
                    latitude = station.get('latitude')
                    longitude = station.get('longitude')
                    if (latitude is not None and longitude is not None and
                        str(latitude).replace('.', '').replace('-', '').isdigit() and 
                        str(longitude).replace('.', '').replace('-', '').isdigit()):
                        try:
                            location_info = {
                                'lat': float(latitude),
                                'lon': float(longitude),
                                'name': station.get('description', 'Incident Location'),
                                'stanox': stanox_code
                            }
                            incident_locations.append(location_info)
                            station_name = station.get('description', 'Unknown')
                            print(f"📍 Found incident location: {station_name} ({stanox_code})")
                        except ValueError:
                            continue
                    break  # Found this STANOX code, move to next one
        
        # For backward compatibility, set primary incident_location to first found location
        if incident_locations:
            incident_location = incident_locations[0]
            incident_station_name = incident_location['name']
    
    # Create time intervals and get delay data for each interval using incident_view()
    time_intervals = []
    interval_heatmap_data = []  # Store heat map data for each interval
    
    for i in range(num_intervals):
        interval_start = analysis_datetime + timedelta(minutes=i * interval_minutes)
        interval_end = interval_start + timedelta(minutes=interval_minutes)
        
        # Calculate how many minutes this interval represents
        actual_interval_minutes = min(interval_minutes, period_minutes - (i * interval_minutes))
        
        time_intervals.append({
            'start': interval_start,
            'end': interval_end,
            'label': f"{interval_start.strftime('%H:%M')}-{interval_end.strftime('%H:%M')}"
        })
        
        # Get delay data for this specific interval using incident_view
        interval_start_str = interval_start.strftime('%H%M')
        interval_delay_data, _, _ = incident_view(
            incident_code, 
            incident_date, 
            analysis_date, 
            interval_start_str, 
            actual_interval_minutes
        )
        
        # Convert interval delay data to heat map points
        interval_points = []
        if not interval_delay_data.empty:
            for _, row in interval_delay_data.iterrows():
                station_code = str(row['STATION_CODE'])
                if station_code in station_coords_map:
                    coords = station_coords_map[station_code]
                    
                    # Calculate average delay minutes per station for this interval
                    total_delayed_trains = row['DELAYED_TRAINS_OUT'] + row['DELAYED_TRAINS_IN']
                    
                    if total_delayed_trains > 0:  # Only include stations with delays
                        # Get delay minutes (these are lists, so we need to sum them)
                        delay_minutes_out = row['DELAY_MINUTES_OUT'] if isinstance(row['DELAY_MINUTES_OUT'], list) else []
                        delay_minutes_in = row['DELAY_MINUTES_IN'] if isinstance(row['DELAY_MINUTES_IN'], list) else []
                        
                        # Calculate total delay minutes for this station in this interval
                        total_delay_minutes = sum(delay_minutes_out) + sum(delay_minutes_in)
                        
                        # Calculate average delay per train at this station
                        average_delay_minutes = total_delay_minutes / total_delayed_trains if total_delayed_trains > 0 else 0
                        
                        if average_delay_minutes > 0:  # Only include if there are actual delay minutes
                            interval_points.append({
                                'lat': coords['lat'],
                                'lng': coords['lon'],
                                'weight': average_delay_minutes,  # Now using average delay minutes
                                'name': coords['name'],
                                'delayed_trains': total_delayed_trains,
                                'total_delay_minutes': total_delay_minutes
                            })
        
        interval_heatmap_data.append(interval_points)
        print(f"  Interval {i+1}: {len(interval_points)} stations with delays")
    
    print(f"🚉 Processed {len(interval_heatmap_data)} time intervals for animation")
    
    # Generate filename if not provided
    if output_file is None:
        output_file = f'animated_heatmap_incident_{incident_code}_{analysis_date.replace("-", "_")}_{analysis_hhmm}_{period_minutes}min_interval{interval_minutes}min.html'
    
    # Create time step data for JavaScript
    time_steps_js = []
    for i, interval in enumerate(time_intervals):
        time_steps_js.append(f"'{interval['label']}'")
    
    # Create heat map data for JavaScript
    heatmap_data_js = []
    for interval_points in interval_heatmap_data:
        if interval_points:
            points_js = []
            for point in interval_points:
                points_js.append(f"[{point['lat']}, {point['lng']}, {point['weight']}]")
            heatmap_data_js.append(f"[{', '.join(points_js)}]")
        else:
            heatmap_data_js.append("[]")
    
    # Calculate global max weight for consistent scaling across all intervals
    max_weight = 0
    total_stations_affected = set()
    
    for interval_points in interval_heatmap_data:
        for point in interval_points:
            max_weight = max(max_weight, point['weight'])
            total_stations_affected.add(point['name'])
    
    if max_weight == 0:
        print("⚠️ No delay data found for heat map animation")
        return None
    
    print(f"🚉 Found {len(total_stations_affected)} unique stations affected across all intervals")
    print(f"📈 Maximum delay weight: {max_weight:.1f} minutes")
    
    # Generate filename if not provided
    if output_file is None:
        output_file = f'heatmap_incident_{incident_code}_{analysis_date.replace("-", "_")}_{analysis_hhmm}_{period_minutes}min.html'
    
    # Create JavaScript for ALL station markers (grey dots for unaffected stations)
    all_stations_js = []
    for station_id, coords in station_coords_map.items():
        all_stations_js.append(f"[{coords['lat']}, {coords['lon']}, '{coords['name']}', '{station_id}']")
    
    stations_js_array = "[" + ", ".join(all_stations_js) + "]"
    
    # Create incident marker JavaScript for all found incident locations
    incident_marker_js = ""
    if incident_locations:
        incident_marker_js = "// Add incident location markers with clean 'X' symbol\n"
        for i, location in enumerate(incident_locations):
            incident_marker_js += f'''
        var incidentMarker{i} = L.marker([{location['lat']}, {location['lon']}], {{
            icon: L.divIcon({{
                className: 'incident-marker-clean',
                html: '<div style="color: red; font-size: 20px; font-weight: bold; font-family: Arial, sans-serif;">✕</div>',
                iconSize: [20, 20],
                iconAnchor: [10, 10]
            }})
        }}).addTo(map);
        
        incidentMarker{i}.bindPopup('<b>Incident Location</b><br>Station: {location['name']}<br>STANOX: {location['stanox']}<br>Section Code: {incident_section_code}');
        '''
    
    # Create animated HTML with heat map timeline controls
    html_content = f'''
<!DOCTYPE html>
<html>
<head>
    <title>Animated Railway Delay Heat Map - Incident {incident_code}</title>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css" 
          integrity="sha256-p4NxAoJBhIIN+hmNHrzRCf9tD/miZyoHS5obTRR9BMY=" crossorigin="" />
    <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"
            integrity="sha256-20nQCchB9co0qIjJZRGuk2/Z9VM+kNiyxNV1lvTlZBo=" crossorigin=""></script>
    <script src="https://cdn.jsdelivr.net/npm/leaflet.heat@0.2.0/dist/leaflet-heat.min.js"></script>
    <style>
        body {{ margin: 0; padding: 0; font-family: Arial, sans-serif; }}
        #map {{ height: 100vh; width: 100vw; }}
        
        /* Enhanced heat map visibility */
        .leaflet-heatmap-layer {{
            opacity: 0.8 !important;
        }}
        
        /* Incident marker styling */
        .incident-marker {{
            z-index: 1000 !important;
        }}
        .incident-marker div {{
            box-shadow: 0 2px 6px rgba(0,0,0,0.5) !important;
        }}
        
        .controls-panel {{
            position: absolute;
            bottom: 20px;
            left: 20px;
            right: 20px;
            background: white;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 4px 15px rgba(0,0,0,0.3);
            z-index: 1000;
            display: flex;
            flex-direction: column;
            gap: 15px;
        }}
        .controls-top {{
            display: flex;
            justify-content: space-between;
            align-items: flex-start;
            gap: 20px;
        }}
        .info-section {{
            flex: 1;
            font-size: 14px;
        }}
        .legend-section {{
            flex: 1;
            font-size: 12px;
        }}
        .controls-bottom {{
            display: flex;
            flex-direction: column;
            gap: 10px;
        }}
        .time-controls {{
            display: flex;
            align-items: center;
            gap: 10px;
        }}
        .time-slider {{
            flex: 1;
            margin: 0;
        }}
        .play-button {{
            background: #007cba;
            color: white;
            border: none;
            padding: 10px 20px;
            border-radius: 5px;
            cursor: pointer;
        }}
        .play-button:hover {{
            background: #005a87;
        }}
        .current-time {{
            font-weight: bold;
            color: #007cba;
            margin: 0;
        }}
        .legend-title {{
            font-weight: bold;
            margin-bottom: 8px;
            color: #333;
        }}
        .legend-item {{
            display: flex;
            align-items: center;
            margin: 2px 0;
            font-size: 11px;
        }}
        .legend-color {{
            width: 16px;
            height: 16px;
            margin-right: 6px;
            border-radius: 2px;
            margin-right: 8px;
            border-radius: 3px;
        }}
        .title {{
            font-size: 16px;
            font-weight: bold;
            margin-bottom: 10px;
            color: #333;
        }}
    </style>
</head>
<body>
    <div id="map"></div>
    
    <div class="controls-panel">
        <div class="controls-top">
            <div class="info-section">
                <div style="font-size: 16px; font-weight: bold; margin-bottom: 10px; color: #333;">🔥 Railway Delay Heat Map - Incident {incident_code}</div>
                <div><strong>Period:</strong> {analysis_date} {analysis_hhmm[:2]}:{analysis_hhmm[2:]} ({period_minutes} min)</div>
                <div><strong>Intervals:</strong> {num_intervals} × {interval_minutes} min</div>
                <div><strong>Stations Affected:</strong> {len(total_stations_affected)} | <strong>Max Avg Delay:</strong> {max_weight:.1f} minutes</div>
                <div style="margin-top: 8px; font-size: 12px; color: #666;">
                    Shows average delay minutes per station per interval across the entire network
                </div>
            </div>
            <div class="legend-section">
                <div class="legend-title">Delay Intensity Legend</div>
                <div class="legend-item">
                    <div class="legend-color" style="background: #32CD32;"></div>
                    <span>Minor (1-5 min)</span>
                </div>
                <div class="legend-item">
                    <div class="legend-color" style="background: #FFD700;"></div>
                    <span>Moderate (6-15 min)</span>
                </div>
                <div class="legend-item">
                    <div class="legend-color" style="background: #FF8C00;"></div>
                    <span>Significant (16-30 min)</span>
                </div>
                <div class="legend-item">
                    <div class="legend-color" style="background: #FF0000;"></div>
                    <span>Major (31-60 min)</span>
                </div>
                <div class="legend-item">
                    <div class="legend-color" style="background: #8B0000;"></div>
                    <span>Severe (61-120 min)</span>
                </div>
                <div class="legend-item">
                    <div class="legend-color" style="background: #8A2BE2;"></div>
                    <span>Critical (120+ min)</span>
                </div>
                <div style="font-size: 10px; margin-top: 5px; color: #666;">
                    Red X = Incident Location
                </div>
            </div>
        </div>
        <div class="controls-bottom">
            <div class="current-time" id="currentTime">Time: Loading...</div>
            <div class="time-controls">
                <button id="playButton" class="play-button">▶ Play Animation</button>
                <input type="range" id="timeSlider" class="time-slider" min="0" max="{num_intervals-1}" value="0">
                <span id="intervalInfo">Interval 1 of {num_intervals}</span>
            </div>
        </div>
    </div>

    <script>
        // Initialize map centered on UK
        var map = L.map('map').setView([54.5, -2.5], 6);

        // Add OpenStreetMap tiles
        L.tileLayer('https://{{s}}.tile.openstreetmap.org/{{z}}/{{x}}/{{y}}.png', {{
            attribution: '© OpenStreetMap contributors'
        }}).addTo(map);

        // Time interval data for animation
        var timeSteps = [{', '.join(time_steps_js)}];
        var heatmapDataByInterval = [{', '.join(heatmap_data_js)}];
        var currentIndex = 0;
        var isPlaying = false;
        var playInterval;
        
        // Create heat layer (initially empty) with precise, less spread-out settings
        // Using same color grading as aggregate_view function's delay severity chart
        var heat = L.heatLayer([], {{
            radius: 18,    // Slightly larger radius for better visibility
            blur: 6,       // Less blur for sharper edges
            maxZoom: 17,
            max: 1.0,      // Normalize to 1.0 for better visibility
            minOpacity: 0.3,  // Minimum opacity to ensure visibility
            gradient: {{
                0.0: '#32CD32',       // Bright lime green for minor delays (1-5 min)
                0.17: '#FFD700',      // Bright gold for moderate delays (6-15 min)
                0.33: '#FF8C00',      // Bright dark orange for significant delays (16-30 min)
                0.50: '#FF0000',      // Bright red for major delays (31-60 min)
                0.67: '#8B0000',      // Dark red for severe delays (61-120 min)
                1.0: '#8A2BE2'        // Blue violet for critical delays (120+ min)
            }}
        }}).addTo(map);
        
        // Add ALL network stations as grey dots
        var allStations = {stations_js_array};
        var stationMarkers = [];
        
        allStations.forEach(function(station) {{
            var marker = L.circleMarker([station[0], station[1]], {{
                radius: 2,
                fillColor: '#808080',
                color: '#606060',
                weight: 1,
                opacity: 0.6,
                fillOpacity: 0.4
            }}).addTo(map);
            
            marker.bindPopup('<b>' + station[2] + '</b><br>Station ID: ' + station[3] + '<br>Status: No delays');
            stationMarkers.push(marker);
        }});
        
        console.log('Added', stationMarkers.length, 'network station markers');
        
        {incident_marker_js}
        
        // Function to update heat map for specific time interval
        function updateHeatMap(index) {{
            if (index >= 0 && index < heatmapDataByInterval.length) {{
                var data = heatmapDataByInterval[index];
                console.log('Updating heat map for interval', index, 'with', data.length, 'points:', data);
                heat.setLatLngs(data);
                document.getElementById('currentTime').textContent = 'Time: ' + timeSteps[index];
                var avgDelay = data.length > 0 ? (data.reduce((sum, point) => sum + point[2], 0) / data.length) : 0;
                document.getElementById('intervalInfo').textContent = 'Interval ' + (index + 1) + ' of ' + timeSteps.length + ' (' + data.length + ' stations, avg: ' + avgDelay.toFixed(1) + ' min)';
                document.getElementById('timeSlider').value = index;
                currentIndex = index;
            }}
        }}
        
        // Animation controls
        function playAnimation() {{
            if (isPlaying) {{
                clearInterval(playInterval);
                isPlaying = false;
                document.getElementById('playButton').textContent = '▶ Play Animation';
            }} else {{
                isPlaying = true;
                document.getElementById('playButton').textContent = '⏸ Pause Animation';
                playInterval = setInterval(function() {{
                    currentIndex++;
                    if (currentIndex >= timeSteps.length) {{
                        currentIndex = 0;
                    }}
                    updateHeatMap(currentIndex);
                }}, 1500); // Change interval every 1.5 seconds
            }}
        }}
        
        // Add discrete reference markers (small, subtle dots)
        var referenceMarkers = L.layerGroup();
        if (heatmapDataByInterval.length > 0 && heatmapDataByInterval[0].length > 0) {{
            heatmapDataByInterval[0].forEach(function(point, i) {{
                L.circleMarker([point[0], point[1]], {{
                    radius: 2,           // Much smaller
                    fillColor: '#333',   // Dark gray instead of red
                    color: '#666',       // Gray border
                    weight: 1,           // Thin border
                    opacity: 0.6,        // Semi-transparent
                    fillOpacity: 0.4     // More subtle
                }}).bindPopup('Station ' + (i+1) + '<br>Avg Delay: ' + point[2].toFixed(1) + ' min').addTo(referenceMarkers);
            }});
            referenceMarkers.addTo(map);
            console.log('Added', heatmapDataByInterval[0].length, 'discrete reference markers');
        }}
        
        // Check if required libraries are loaded
        console.log('🔍 Checking libraries...');
        console.log('Leaflet loaded:', typeof L !== 'undefined');
        console.log('Heat layer available:', typeof L !== 'undefined' && typeof L.heatLayer !== 'undefined');
        
        if (typeof L === 'undefined') {{
            document.getElementById('currentTime').textContent = 'Error: Leaflet library failed to load';
            return;
        }}
        
        if (typeof L.heatLayer === 'undefined') {{
            document.getElementById('currentTime').textContent = 'Error: Heat map plugin failed to load';
            return;
        }}
        
        // Initialize function to set up the heat map
        function initializeHeatMap() {{
            console.log('🔍 Initializing heat map...');
            console.log('Time steps:', timeSteps.length);
            console.log('Heatmap data intervals:', heatmapDataByInterval.length);
            console.log('All stations array length:', allStations.length);
            
            // Check if required elements exist
            var timeElement = document.getElementById('currentTime');
            var sliderElement = document.getElementById('timeSlider');
            var buttonElement = document.getElementById('playButton');
            
            console.log('DOM elements found:', {{
                currentTime: !!timeElement,
                timeSlider: !!sliderElement,
                playButton: !!buttonElement
            }});
            
            // If elements are not found, retry after a short delay
            if (!timeElement || !sliderElement || !buttonElement) {{
                console.log('⏳ DOM elements not ready, retrying in 100ms...');
                setTimeout(initializeHeatMap, 100);
                return;
            }}
        
        // Set up event listeners first
        if (buttonElement) {{
            buttonElement.addEventListener('click', playAnimation);
            console.log('✅ Play button event listener attached');
        }}
        
        if (sliderElement) {{
            sliderElement.addEventListener('input', function(e) {{
                updateHeatMap(parseInt(e.target.value));
                // Pause animation when user manually changes time
                if (isPlaying) {{
                    clearInterval(playInterval);
                    isPlaying = false;
                    if (buttonElement) buttonElement.textContent = '▶ Play Animation';
                }}
            }});
            console.log('✅ Slider event listener attached');
        }}
        
            // Initialize with first interval if everything is ready
            if (timeSteps.length > 0 && timeElement) {{
                console.log('✅ Initializing with first interval');
                updateHeatMap(0);
            }} else {{
                console.error('❌ Missing elements or no data:', {{
                    timeSteps: timeSteps.length,
                    timeElement: !!timeElement
                }});
                if (timeElement) timeElement.textContent = 'Error: No data available';
            }}
        }}
        
        // Start initialization - use multiple approaches to ensure it runs
        if (document.readyState === 'loading') {{
            document.addEventListener('DOMContentLoaded', initializeHeatMap);
        }} else {{
            // DOM is already loaded, start immediately
            setTimeout(initializeHeatMap, 50);
        }}
    </script>
</body>
</html>'''
    
    # Save the HTML file
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    print(f"🔥 Animated HTML heat map saved as: {output_file}")
    print(f"🌐 Open {output_file} in your browser to view the animated heat map with timeline controls")
    print(f"🎬 Use Play/Pause button and slider to control the temporal animation")
    
    return html_content


def incident_view_heatmap_html(incident_code, incident_date, analysis_date, analysis_hhmm, period_minutes, interval_minutes=10, output_file=None):
    """
    Create a dynamic interactive HTML heatmap showing railway network delays as continuous color clouds.
    Displays delay intensity as vibrant heatmap visualization with invisible clickable areas over 
    affected stations for detailed information popups.
    
    Parameters:
    incident_code (int/float): The incident number to analyze
    incident_date (str): Date when incident started in 'DD-MMM-YYYY' format
    analysis_date (str): Specific date to analyze in 'DD-MMM-YYYY' format
    analysis_hhmm (str): Start time for analysis in 'HHMM' format (e.g., '1900')
    period_minutes (int): Total duration of analysis period in minutes
    interval_minutes (int): Duration of each interval in minutes (default: 10)
    output_file (str): Optional HTML file path to save
    
    Returns:
    str: HTML content of the interactive heatmap
    """
    
    # Load station coordinates data from comprehensive JSON file
    try:
        from data.reference import reference_files
        import json
        
        file_path = reference_files["all dft categories"]
        print(f"Loading station coordinates from: {file_path}")
        
        if not os.path.exists(file_path):
            print(f"File does not exist: {file_path}")
            return None
            
        with open(file_path, 'r') as f:
            stations_coords_data = json.load(f)
            
        print(f"Successfully loaded {len(stations_coords_data)} station records")
        
    except Exception as e:
        print(f"Error loading station coordinates: {e}")
        return None
    
    
    # Parse analysis parameters
    try:
        analysis_datetime = datetime.strptime(f"{analysis_date} {analysis_hhmm[:2]}:{analysis_hhmm[2:]}", '%d-%b-%Y %H:%M')
    except ValueError:
        print(f"Error: Invalid analysis date/time format.")
        return None
    
    analysis_end = analysis_datetime + timedelta(minutes=period_minutes)
    
    print(f"Creating heatmap for incident {incident_code}")
    print(f"Analysis period: {analysis_datetime.strftime('%d-%b-%Y %H:%M')} to {analysis_end.strftime('%d-%b-%Y %H:%M')}")
    print(f"Interval size: {interval_minutes} minutes")
    
    # Calculate number of intervals
    num_intervals = period_minutes // interval_minutes
    if period_minutes % interval_minutes != 0:
        num_intervals += 1
    
    print(f"Total intervals: {num_intervals}")
    
    # Determine day of week for analysis date
    day_mapping = {0: 'MO', 1: 'TU', 2: 'WE', 3: 'TH', 4: 'FR', 5: 'SA', 6: 'SU'}
    analysis_day_suffix = day_mapping[analysis_datetime.weekday()]
    
    # Load station files for the analysis day using NEW parquet structure
    processed_base = find_processed_data_path()
    
    if processed_base is None:
        print("No processed_data directory found in any expected location. Please run the preprocessor first.")
        return None
    
    # Get all station directories
    station_dirs = [d for d in os.listdir(processed_base) 
                   if os.path.isdir(os.path.join(processed_base, d))]
    
    # Build list of files to load (only for the specific day)
    target_files = []
    for station_dir in station_dirs:
        station_path = os.path.join(processed_base, station_dir)
        day_file = os.path.join(station_path, f"{analysis_day_suffix}.parquet")
        if os.path.exists(day_file):
            target_files.append((day_file, station_dir))
    
    print(f"Loading {len(target_files)} station files for {analysis_day_suffix}")
    
    # Create station coordinates mapping from JSON data (DFT categories A, B, C1, C2 only)
    all_station_coords_map = {}
    valid_categories = {'A', 'B', 'C1', 'C2'}
    
    for station in stations_coords_data:
        if isinstance(station, dict):
            station_id = str(station.get('stanox', ''))
            dft_category = station.get('dft_category', '')
            latitude = station.get('latitude')
            longitude = station.get('longitude')
            description = station.get('description', 'Unknown Station')
            
            # Only include stations with valid DFT categories (A, B, C1, C2) and coordinates
            if (dft_category in valid_categories and 
                station_id and 
                latitude is not None and longitude is not None and
                str(latitude).replace('.', '').replace('-', '').isdigit() and 
                str(longitude).replace('.', '').replace('-', '').isdigit()):
                try:
                    all_station_coords_map[station_id] = {
                        'name': description,
                        'lat': float(latitude),
                        'lon': float(longitude),
                        'category': dft_category
                    }
                except ValueError:
                    continue
    
    print(f"Found coordinates for {len(all_station_coords_map)} stations (DFT categories A/B/C1/C2)")
    
    # Collect delay timeline data for affected stations (same logic as incident_view_html)
    station_timeline_data = {}
    incident_section_code = None
    incident_reason = None
    incident_start_time = None
    
    for file_path, station_code in target_files:
        if station_code not in all_station_coords_map:
            continue
            
        try:
            df = pd.read_parquet(file_path, engine='fastparquet')
            if not isinstance(df, pd.DataFrame):
                continue
                
            # Get all incident data for this station
            all_incident_data = df[df['INCIDENT_NUMBER'] == incident_code].copy()
            if all_incident_data.empty:
                continue
            
            # Extract incident information on first occurrence
            if incident_section_code is None and not all_incident_data.empty:
                incident_records = all_incident_data[all_incident_data['INCIDENT_START_DATETIME'].str.contains(incident_date, na=False)]
                if not incident_records.empty:
                    incident_section_code = incident_records['SECTION_CODE'].iloc[0]
                    incident_reason = incident_records['INCIDENT_REASON'].iloc[0]
                    incident_start_time = incident_records['INCIDENT_START_DATETIME'].iloc[0]
            
            # Filter to events within our analysis period
            all_incident_data['EVENT_DATETIME_parsed'] = pd.to_datetime(all_incident_data['EVENT_DATETIME'], format='%d-%b-%Y %H:%M', errors='coerce')
            all_incident_data = all_incident_data[all_incident_data['EVENT_DATETIME_parsed'].notna()].copy()
            
            # Filter to analysis period
            period_data = all_incident_data[
                (all_incident_data['EVENT_DATETIME_parsed'] >= analysis_datetime) &
                (all_incident_data['EVENT_DATETIME_parsed'] <= analysis_end) &
                (all_incident_data['PFPI_MINUTES'].notna())
            ].copy()
            
            if period_data.empty:
                continue
            
            # Create timeline of INTERVAL-SPECIFIC delays for this station
            interval_delays = {}
            
            for _, row in period_data.iterrows():
                event_time = row['EVENT_DATETIME_parsed']
                delay = row['PFPI_MINUTES']
                
                # Calculate which interval this event belongs to
                minutes_from_start = int((event_time - analysis_datetime).total_seconds() / 60)
                interval_number = minutes_from_start // interval_minutes
                
                # Calculate the start time of this interval
                interval_start = analysis_datetime + timedelta(minutes=interval_number * interval_minutes)
                
                if interval_start not in interval_delays:
                    interval_delays[interval_start] = 0
                interval_delays[interval_start] += delay
            
            # Convert to timeline format
            timeline = []
            for interval_start in sorted(interval_delays.keys()):
                interval_total_delay = interval_delays[interval_start]
                timeline.append((interval_start, interval_total_delay))
            
            if timeline:
                station_timeline_data[station_code] = timeline
                
        except Exception as e:
            continue
    
    print(f"Found delay data for {len(station_timeline_data)} affected stations")
    print(f"Incident section code: {incident_section_code}")
    print(f"Incident reason: {incident_reason}")
    print(f"Incident start time: {incident_start_time}")
    
    # Get incident location coordinates - handle multiple STANOX codes separated by colon
    incident_lat = 54.5  # Default to center of UK
    incident_lon = -2.0
    incident_station_name = None
    incident_locations = []  # Store multiple locations if section contains multiple STANOX codes
    
    if incident_section_code:
        try:
            from data.reference import reference_files
            with open(reference_files["station codes"], 'r') as f:
                station_codes_data = json.load(f)
            
            # Split section code by colon to handle cases like "04303:04730"
            stanox_codes = [code.strip() for code in str(incident_section_code).split(':')]
            print(f"Looking for STANOX codes: {stanox_codes}")
            
            for stanox_code in stanox_codes:
                print(f"Searching for STANOX: {stanox_code}")
                found_match = False
                for record in station_codes_data:
                    if isinstance(record, dict):
                        record_stanox = record.get('stanox')
                        # Try different formats: with/without leading zeros, as int vs string
                        if (str(record_stanox) == str(stanox_code) or 
                            str(record_stanox).zfill(5) == str(stanox_code).zfill(5) or
                            str(record_stanox).lstrip('0') == str(stanox_code).lstrip('0')):
                            found_match = True
                            if record.get('latitude') and record.get('longitude'):
                                location_info = {
                                    'lat': float(record['latitude']),
                                    'lon': float(record['longitude']),
                                    'name': record.get('description', 'Incident Location'),
                                    'stanox': stanox_code
                                }
                                incident_locations.append(location_info)
                                station_name = record.get('description', 'Unknown')
                                print(f"📍 Found incident location: {station_name} ({stanox_code})")
                            break  # Found this STANOX code, move to next one
                if not found_match:
                    print(f"❌ STANOX {stanox_code} not found in reference data")
            
            # For backward compatibility, set primary location to first found location
            if incident_locations:
                incident_lat = incident_locations[0]['lat']
                incident_lon = incident_locations[0]['lon']
                incident_station_name = incident_locations[0]['name']
                
        except Exception as e:
            print(f"Warning: Could not load incident location coordinates: {e}")
    
    # Create time steps for animation
    time_steps = []
    current_time = analysis_datetime
    while current_time < analysis_end:
        time_steps.append(current_time)
        current_time += timedelta(minutes=interval_minutes)
    
    # Build timeline data structure for JavaScript (include ALL stations, delays for affected ones)
    timeline_data = {}
    for step_time in time_steps:
        time_key = step_time.strftime('%H:%M')
        timeline_data[time_key] = {}
        
        # For each station in the entire network
        for station_code in all_station_coords_map.keys():
            if station_code in station_timeline_data:
                # Station has delays - calculate interval-specific delay
                station_timeline = station_timeline_data[station_code]
                interval_delay = 0
                
                for event_time, delay_amount in station_timeline:
                    time_window_start = step_time
                    time_window_end = step_time + timedelta(minutes=interval_minutes)
                    
                    if time_window_start <= event_time < time_window_end:
                        interval_delay += delay_amount
                
                timeline_data[time_key][station_code] = interval_delay
            else:
                # Station has no delays - set to 0
                timeline_data[time_key][station_code] = 0
    
    # Prepare data for JavaScript
    station_coords_json = json.dumps(all_station_coords_map)
    timeline_data_json = json.dumps(timeline_data)
    time_steps_json = json.dumps([t.strftime('%H:%M') for t in time_steps])
    
    # Create incident markers JavaScript
    incident_markers_js = ""
    if incident_locations:
        for i, loc in enumerate(incident_locations):
            incident_markers_js += f'''
            var incidentMarker{i} = L.marker([{loc['lat']}, {loc['lon']}], {{
                icon: L.divIcon({{
                    html: '<div style="font-size: 24px; color: red; font-weight: bold;">✕</div>',
                    className: 'incident-marker',
                    iconSize: [30, 30],
                    iconAnchor: [15, 15]
                }})
            }}).addTo(map);
            
            incidentMarker{i}.bindPopup(`
                <strong>Incident Location</strong><br>
                Station: {loc['name']}<br>  
                STANOX: {loc['stanox']}<br>
                Incident: {incident_code}<br>
                Section: {incident_section_code or 'N/A'}<br>
                Reason: {incident_reason or 'N/A'}<br>
                Started: {incident_start_time or 'N/A'}
            `);'''
    
    # Create HTML content with continuous heatmap visualization
    html_content = f'''<!DOCTYPE html>
<html>
<head>
    <title>Incident {incident_code} - Network Heatmap Analysis</title>
    <meta charset="utf-8" />
    <link rel="stylesheet" href="https://unpkg.com/leaflet@1.7.1/dist/leaflet.css" />
    <script src="https://unpkg.com/leaflet@1.7.1/dist/leaflet.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/leaflet.heat@0.2.0/dist/leaflet-heat.js"></script>
    <style>
        body {{ margin: 0; font-family: Arial, sans-serif; background: #f5f5f5; }}
        .gradient-legend {{
            display: flex;
            align-items: center;
            margin: 10px 0;
        }}
        .gradient-bar {{
            width: 200px;
            height: 20px;
            background: linear-gradient(to right, 
                rgba(0,255,0,1.0) 0%,        /* Bright green - minor delays (1-14 min) */
                rgba(255,255,0,1.0) 33%,     /* Bright yellow - medium delays (15-29 min) */
                rgba(255,165,0,1.0) 66%,     /* Bright orange - high delays (30-59 min) */
                rgba(255,0,0,1.0) 100%       /* Bright red - critical delays (60+ min) */
            );
            border: 1px solid #ccc;
            border-radius: 10px;
            margin: 0 10px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.2);
        }}
        .gradient-labels {{
            display: flex;
            justify-content: space-between;
            width: 200px;
            margin: 5px 10px 0 10px;
            font-size: 10px;
            color: #666;
        }}
        #map {{ height: 75vh; }}
        #controls {{ 
            height: 25vh; 
            padding: 20px; 
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
        }}
        .control-panel {{ 
            background: rgba(255,255,255,0.95); 
            padding: 15px; 
            margin: 10px 0; 
            border-radius: 10px; 
            box-shadow: 0 4px 15px rgba(0,0,0,0.2);
            color: #333;
        }}
        #timeline {{ width: 100%; margin: 10px 0; height: 8px; }}
        .time-display {{ 
            font-size: 24px; 
            font-weight: bold; 
            text-align: center; 
            color: #2c3e50;
            margin: 10px 0;
        }}
        .play-controls {{ text-align: center; margin: 15px 0; }}
        .btn {{ 
            background: #3498db; 
            color: white; 
            border: none; 
            padding: 10px 20px; 
            margin: 0 5px; 
            border-radius: 25px; 
            cursor: pointer; 
            font-size: 16px;
            transition: all 0.3s ease;
        }}
        .btn:hover {{ background: #2980b9; transform: translateY(-2px); }}
        .btn:active {{ transform: translateY(0); }}
        #station-info {{ 
            max-height: 120px; 
            overflow-y: auto; 
            margin-top: 10px;
            font-size: 14px;
        }}
        .station-delay {{ 
            display: inline-block; 
            margin: 2px 5px; 
            padding: 3px 8px; 
            border-radius: 15px; 
            font-size: 12px;
        }}
        .delay-low {{ background: rgba(0,255,0,1.0); color: black; }}
        .delay-med {{ background: rgba(255,255,0,1.0); color: black; }}
        .delay-high {{ background: rgba(255,165,0,1.0); color: white; }}
        .delay-extreme {{ background: rgba(255,0,0,1.0); color: white; }}
        .legend {{ 
            background: rgba(255,255,255,0.95); 
            padding: 10px; 
            border-radius: 8px;
            margin: 10px 0;
            font-size: 12px;
            color: #333;
        }}
        .legend-item {{ 
            display: inline-block;
            margin: 3px 8px 3px 0; 
        }}
        .legend-color {{ 
            display: inline-block; 
            width: 15px; 
            height: 15px; 
            border-radius: 50%; 
            margin-right: 5px; 
            vertical-align: middle;
        }}
        
        /* Cloud-like heatmap circle styles */
        .delay-circle {{
            filter: blur(3px) brightness(1.1);
            transition: filter 0.3s ease;
        }}
        .delay-circle:hover {{
            filter: blur(1px) brightness(1.3);
            /* Removed transform: scale() to prevent circle movement */
        }}
    </style>
</head>
<body>
    <!-- Map Container -->
    <div id="map">
    </div>
    
    <!-- Controls Container -->
    <div id="controls">
        <div class="control-panel">
            <h3 style="margin-top: 0;">Incident {incident_code} - Network Heatmap (A/B/C1/C2 Stations) - {analysis_date}</h3>
            <p style="margin: 5px 0;">Analysis Period: {analysis_datetime.strftime('%H:%M')} - {analysis_end.strftime('%H:%M')} ({period_minutes} min total, {interval_minutes}-min intervals)</p>
            <p style="margin: 5px 0; font-weight: bold;">Section: {incident_section_code or 'N/A'}{' (' + incident_station_name + ')' if incident_station_name else ''} | Reason: {incident_reason or 'N/A'} | Started: {incident_start_time or 'N/A'}</p>
            
            <div class="time-display" id="current-time">{time_steps[0].strftime('%H:%M')}</div>
            
            <input type="range" id="timeline" min="0" max="{len(time_steps)-1}" value="0" step="1">
            
            <div class="play-controls">
                <button class="btn" onclick="playTimeline()">▶ Play</button>
                <button class="btn" onclick="pauseTimeline()">⏸ Pause</button>
                <button class="btn" onclick="resetTimeline()">⏮ Reset</button>
            </div>
            
            <div class="legend">
                <strong>Custom Delay Visualization - Exact Color Matching:</strong>
                <div class="gradient-legend">
                    <span style="font-size: 10px; font-weight: bold;">Minor</span>
                    <div class="gradient-bar"></div>
                    <span style="font-size: 10px; font-weight: bold;">Critical</span>
                </div>
                <div class="gradient-labels">
                    <span>1min</span>
                    <span>15min</span>
                    <span>30min</span>
                    <span>60+min</span>
                </div>

            </div>
            
            <div id="station-info">Loading network heatmap...</div>
        </div>
    </div>

    <script>
        // Initialize map
        var map = L.map('map').setView([54.5, -2.0], 6);
        L.tileLayer('https://{{s}}.tile.openstreetmap.org/{{z}}/{{x}}/{{y}}.png', {{
            attribution: '© OpenStreetMap contributors'
        }}).addTo(map);
        
        // Add incident location markers for all found locations
        
        // Custom colored circles for precise color control (bypassing Leaflet.heat limitations)
        var delayCircles = L.layerGroup().addTo(map);
        
        // Color function that matches our legend exactly
        function getExactDelayColor(delayMinutes) {{
            if (delayMinutes >= 60) return '#ff0000';      // Bright red for critical delays (60+ min)
            if (delayMinutes >= 30) return '#ffa500';      // Bright orange for high delays (30-59 min)
            if (delayMinutes >= 15) return '#ffff00';      // Bright yellow for medium delays (15-29 min)
            return '#00ff00';                              // Bright green for minor delays (1-14 min)
        }}
        
        // Consistent circle size - same for all delays for uniform heatmap appearance
        function getCircleSize(delayMinutes) {{
            return 15;  // Fixed size for all delays to create uniform cloud-like heatmap
        }}
        
        // Data
        var stationCoords = {station_coords_json};
        var timelineData = {timeline_data_json};
        var timeSteps = {time_steps_json};
        
        // State
        var currentIndex = 0;
        var isPlaying = false;
        var playInterval;
        var markers = {{}};
        
        // Color functions for heatmap
        function getHeatmapColor(delayMinutes) {{
            if (delayMinutes === 0) return '#cccccc';      // Grey for no delays
            if (delayMinutes >= 60) return '#ff0000';      // Bright red for critical delays (60+ min) - matches legend
            if (delayMinutes >= 30) return '#ffa500';      // Bright orange for high delays (30-59 min) - matches legend
            if (delayMinutes >= 15) return '#ffff00';      // Bright yellow for medium delays (15-29 min) - matches legend
            return '#00ff00';                              // Bright green for minor delays (1-14 min) - matches legend
        }}
        
        // Update heatmap for specific time index
        function updateMap(index) {{
            try {{
                currentIndex = index;
                var timeKey = timeSteps[index];
                var delays = timelineData[timeKey] || {{}};
                
                console.log('🕐 Updating heatmap for time:', timeKey, 'with', Object.keys(delays).length, 'stations');
                
                // Clear existing station markers
                Object.values(markers).forEach(m => map.removeLayer(m));
                markers = {{}};
                
                // Update time display
                document.getElementById('current-time').textContent = timeKey;
                
                // Clear existing delay circles and create new ones with exact colors
                delayCircles.clearLayers();
                
                var stationInfoHtml = '';
                var totalDelayedStations = 0;
                var totalSystemDelay = 0;
                var delayedStations = [];
                
                // Create colored delay circles for each affected station
                Object.entries(stationCoords).forEach(([stationCode, coords]) => {{
                    var delayMinutes = delays[stationCode] || 0;
                    
                    // Create colored circles for stations with delays
                    if (delayMinutes > 0) {{
                        var color = getExactDelayColor(delayMinutes);
                        var size = getCircleSize(delayMinutes);
                        
                        // Debug logging for extreme delays
                        if (delayMinutes >= 120) {{
                            console.log(`🔥 EXTREME DELAY: Station ${{coords.name}} has ${{delayMinutes}} min delay → color ${{color}} size ${{size}} (should be bright red)`);
                        }}
                        
                        // Create cloud-like colored circle for the delay
                        var delayCircle = L.circleMarker([coords.lat, coords.lon], {{
                            radius: size,
                            fillColor: color,
                            color: color,
                            weight: 0,               // No border for cleaner cloud effect
                            opacity: 0.7,            // Slightly transparent for blending
                            fillOpacity: 0.5,        // More transparent for cloud-like appearance
                            className: 'delay-circle' // Apply CSS class for blur effects
                        }});
                        
                        // Add popup with detailed information
                        delayCircle.bindPopup(`
                            <strong>${{coords.name}}</strong><br>
                            Station: ${{stationCode}}<br>
                            Interval Delay: ${{delayMinutes}} minutes<br>
                            Time Window: ${{timeKey}} ({interval_minutes} min)
                        `);
                        
                        delayCircles.addLayer(delayCircle);
                    }}
                    
                    // Create invisible clickable markers only for affected stations (with delays)
                    if (delayMinutes > 0) {{
                        var marker = L.circleMarker([coords.lat, coords.lon], {{
                            radius: 8,               // Larger invisible area for easier clicking
                            fillColor: 'transparent', // Invisible fill
                            color: 'transparent',     // Invisible border
                            weight: 0,               // No border
                            opacity: 0,              // Completely invisible
                            fillOpacity: 0           // Completely invisible
                        }});
                        
                        // Add popup with station information
                        var popupContent = `
                            <strong>${{coords.name}}</strong><br>
                            Station: ${{stationCode}}<br>
                            Interval Delay: ${{delayMinutes}} minutes<br>
                            Time Window: ${{timeKey}} - ${{timeKey}} ({interval_minutes} min)
                        `;
                        
                        marker.bindPopup(popupContent);
                        marker.addTo(map);
                        markers[stationCode] = marker;
                    }}
                    
                    // Collect statistics for affected stations
                    if (delayMinutes > 0) {{
                        totalDelayedStations++;
                        totalSystemDelay += delayMinutes;
                        
                        var delayClass = delayMinutes >= 60 ? 'delay-extreme' : 
                                       delayMinutes >= 30 ? 'delay-high' : 
                                       delayMinutes >= 15 ? 'delay-med' : 'delay-low';
                        
                        delayedStations.push({{
                            name: coords.name.substring(0,15),
                            delay: delayMinutes,
                            class: delayClass
                        }});
                    }}
                }});
                
                // Sort delayed stations by delay amount (highest first)
                delayedStations.sort((a, b) => b.delay - a.delay);
                
                // Update station info panel
                if (totalDelayedStations > 0) {{
                    var avgDelay = (totalSystemDelay / totalDelayedStations).toFixed(1);
                    stationInfoHtml = `<strong>Network Impact:</strong> ${{totalDelayedStations}} stations affected, ${{totalSystemDelay}} total minutes in this {interval_minutes}-min interval (avg: ${{avgDelay}}min)<br><br>`;
                    
                    // Show top affected stations
                    var topStations = delayedStations.slice(0, 10); // Show top 10
                    topStations.forEach(station => {{
                        stationInfoHtml += `<span class="station-delay ${{station.class}}">${{station.name}}: ${{station.delay}}min</span>`;
                    }});
                    
                    if (delayedStations.length > 10) {{
                        stationInfoHtml += `<span style="margin-left: 10px; font-style: italic;">...and ${{delayedStations.length - 10}} more</span>`;
                    }}
                }} else {{
                    stationInfoHtml = `<strong>Network Status:</strong> No delays detected in this interval. ${{Object.keys(stationCoords).length}} stations (A/B/C1/C2 categories) monitored.`;
                }}
                
                document.getElementById('station-info').innerHTML = stationInfoHtml;
                
            }} catch (error) {{
                console.error('❌ Error updating heatmap:', error);
                document.getElementById('station-info').innerHTML = 'Error updating heatmap: ' + error.message;
            }}
        }}
        
        // Timeline controls (same as incident_view_html)
        document.getElementById('timeline').addEventListener('input', function(e) {{
            pauseTimeline();
            updateMap(parseInt(e.target.value));
        }});
        
        function playTimeline() {{
            if (!isPlaying) {{
                isPlaying = true;
                playInterval = setInterval(() => {{
                    if (currentIndex < timeSteps.length - 1) {{
                        currentIndex++;
                        document.getElementById('timeline').value = currentIndex;
                        updateMap(currentIndex);
                    }} else {{
                        pauseTimeline();
                    }}
                }}, 1500);  // Slightly slower for heatmap viewing
            }}
        }}
        
        function pauseTimeline() {{
            isPlaying = false;
            if (playInterval) clearInterval(playInterval);
        }}
        
        function resetTimeline() {{
            pauseTimeline();
            currentIndex = 0;
            document.getElementById('timeline').value = 0;
            updateMap(0);
        }}
        
        // Initialize heatmap with better error handling
        function initializeHeatmap() {{
            console.log('🌡️ Starting heatmap initialization...');
            console.log('Leaflet available:', typeof L !== 'undefined');
            console.log('Leaflet.heat available:', typeof L.heatLayer !== 'undefined');
            console.log('Station coordinates loaded:', Object.keys(stationCoords).length);
            console.log('Time steps:', timeSteps.length);
            console.log('Timeline data:', Object.keys(timelineData).length);
            
            if (typeof L === 'undefined') {{
                console.error('❌ Leaflet not loaded!');
                document.getElementById('station-info').innerHTML = 'Error: Leaflet library not loaded';
                return;
            }}
            
            if (typeof L.heatLayer === 'undefined') {{
                console.error('❌ Leaflet.heat plugin not loaded!');
                document.getElementById('station-info').innerHTML = 'Error: Leaflet.heat plugin not loaded';
                return;
            }}
            
            if (timeSteps.length === 0) {{
                console.error('❌ No time steps available!');
                document.getElementById('station-info').innerHTML = 'No time data available';
            }} else if (Object.keys(stationCoords).length === 0) {{
                console.error('❌ No station coordinates available!');
                document.getElementById('station-info').innerHTML = 'No station coordinates available';
            }} else {{
                console.log('✅ All data loaded successfully, initializing heatmap...');
                updateMap(0);
            }}
            
            // Add incident location markers
            {incident_markers_js}
            
            console.log('🎬 Heatmap initialization complete!');
        }}
        
        // Wait for all scripts to load before initializing
        if (document.readyState === 'loading') {{
            document.addEventListener('DOMContentLoaded', initializeHeatmap);
        }} else {{
            // DOM is already loaded, but wait a bit for scripts
            setTimeout(initializeHeatmap, 100);
        }}
    </script>
</body>
</html>'''
    
    # Save HTML file
    if output_file is None:
        safe_date = analysis_date.replace('-', '_')
        safe_time = analysis_hhmm
        output_file = f'heatmap_{incident_code}_{safe_date}_{safe_time}_period{period_minutes}min_interval{interval_minutes}min.html'
    
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        print(f"🌡️ CONTINUOUS HEATMAP CREATED! ")
        print(f"File: {output_file}")
        print(f"Time steps: {len(time_steps)} ({interval_minutes}-minute intervals)")
        print(f"Total stations mapped: {len(all_station_coords_map)} (DFT categories A/B/C1/C2)")
        print(f"Affected stations: {len(station_timeline_data)}")
        print(f"🔥 Features: Continuous delay heatmap with fading colors + grey station dots overlay")
        print("🌐 Open the HTML file in your browser to explore the continuous heatmap!")
        
    except Exception as e:
        print(f"Error saving file: {e}")
    
    return html_content


# all functions for the station view:

def station_view(incident_code, incident_date, station_id, interval_minutes=30):
    """
    Station analysis with corrected timing logic - simplified output.
    """
    
    # Convert incident date
    try:
        incident_datetime = datetime.strptime(incident_date, '%d-%b-%Y')
        incident_day_suffix = incident_datetime.strftime('%a').upper()[:2]
    except ValueError as e:
        print(f"Error parsing date: {e}")
        return None
    
    # Load data
    processed_base = '../processed_data'
    station_file = os.path.join(processed_base, station_id, f"{incident_day_suffix}.parquet")
    
    if not os.path.exists(station_file):
        print(f"File not found: {station_file}")
        return None
    
    try:
        station_data = pd.read_parquet(station_file, engine='fastparquet')
    except Exception as e:
        print(f"Error loading data: {e}")
        return None
    
    # Filter for trains with planned calls
    train_mask = station_data['PLANNED_CALLS'].notna()
    all_train_data = station_data[train_mask].copy()
    
    # Maximum delay deduplication
    if len(all_train_data) > 0:
        all_train_data['delay_numeric'] = pd.to_numeric(all_train_data['PFPI_MINUTES'], errors='coerce').fillna(0)
        all_train_data['dedup_priority'] = all_train_data['delay_numeric'] * 1000
        
        if 'ACTUAL_CALLS' in all_train_data.columns:
            all_train_data['dedup_priority'] += all_train_data['ACTUAL_CALLS'].notna().astype(int) * 100
        
        basic_dedup_cols = ['TRAIN_SERVICE_CODE', 'PLANNED_CALLS']
        basic_available = [col for col in basic_dedup_cols if col in all_train_data.columns]
        
        if len(basic_available) >= 2:
            all_train_data = all_train_data.sort_values(['delay_numeric', 'dedup_priority'], ascending=[False, False])
            all_train_data = all_train_data.drop_duplicates(subset=basic_available, keep='first')
            all_train_data = all_train_data.drop(['delay_numeric', 'dedup_priority'], axis=1)
    
    if len(all_train_data) == 0:
        return None
    
    # Process times
    def parse_time_simple(time_val, base_date):
        if pd.isna(time_val):
            return None
        try:
            time_str = str(int(time_val)).zfill(4)
            hour = int(time_str[:2])
            minute = int(time_str[2:])
            return base_date.replace(hour=hour, minute=minute, second=0, microsecond=0)
        except:
            return None
    
    # Parse times and apply corrected timing logic
    all_train_data['planned_dt'] = all_train_data['PLANNED_CALLS'].apply(
        lambda x: parse_time_simple(x, incident_datetime))
    all_train_data['original_actual_dt'] = all_train_data['ACTUAL_CALLS'].apply(
        lambda x: parse_time_simple(x, incident_datetime))
    all_train_data['delay_minutes'] = pd.to_numeric(all_train_data['PFPI_MINUTES'], errors='coerce').fillna(0)
    
    # Create corrected actual times
    corrected_actual_times = []
    for _, row in all_train_data.iterrows():
        planned_dt = row['planned_dt']
        original_actual_dt = row['original_actual_dt']
        delay_min = row['delay_minutes']
        
        if pd.isna(planned_dt):
            corrected_actual_times.append(None)
            continue
            
        if delay_min > 0:
            corrected_actual = planned_dt + timedelta(minutes=delay_min)
            corrected_actual_times.append(corrected_actual)
        elif delay_min == 0:
            corrected_actual_times.append(planned_dt)
        else:
            if pd.notna(original_actual_dt):
                corrected_actual_times.append(original_actual_dt)
            else:
                corrected_actual_times.append(planned_dt)
    
    all_train_data['effective_time'] = corrected_actual_times
    valid_data = all_train_data[all_train_data['effective_time'].notna()].copy()
    
    if len(valid_data) == 0:
        return None
    
    # Create time intervals
    day_start = valid_data['effective_time'].min()
    day_end = valid_data['effective_time'].max()
    current_time = day_start
    intervals = []
    
    while current_time < day_end:
        next_time = current_time + timedelta(minutes=interval_minutes)
        
        interval_trains = valid_data[
            (valid_data['effective_time'] >= current_time) & 
            (valid_data['effective_time'] < next_time)
        ]
        
        if len(interval_trains) > 0:
            arrival_trains = interval_trains[interval_trains['EVENT_TYPE'] != 'C']
            cancellation_trains = interval_trains[interval_trains['EVENT_TYPE'] == 'C']
            
            if len(arrival_trains) > 0 or len(cancellation_trains) > 0:
                if len(arrival_trains) > 0:
                    delay_values = arrival_trains['delay_minutes'].tolist()
                    ontime_arrivals = len([d for d in delay_values if d == 0.0])
                    delayed_arrivals = len([d for d in delay_values if d > 0.0])
                    delayed_minutes = [round(d, 1) for d in delay_values if d > 0.0]
                else:
                    ontime_arrivals = 0
                    delayed_arrivals = 0
                    delayed_minutes = []
                
                total_cancellations = len(cancellation_trains)
                
                intervals.append({
                    'time_period': f"{current_time.strftime('%H:%M')}-{next_time.strftime('%H:%M')}",
                    'ontime_arrival_count': ontime_arrivals,
                    'delayed_arrival_count': delayed_arrivals,
                    'cancellation_count': total_cancellations,
                    'delay_minutes': delayed_minutes
                })
        
        current_time = next_time
    
    # Create and return final output
    station_summary = pd.DataFrame(intervals)
    
    if len(station_summary) == 0:
        return None
    
    station_summary['delay_minutes'] = station_summary['delay_minutes'].apply(
        lambda x: x if isinstance(x, list) else []
    )
    
    final_output = station_summary[['time_period', 'ontime_arrival_count', 'delayed_arrival_count', 'cancellation_count', 'delay_minutes']].copy()
    
    # Print the output table
    print(f"\nStation {station_id} Analysis - {incident_date} (Incident {incident_code})")
    print("=" * 80)
    print(final_output.to_string(index=False))
    print("=" * 80)
    
    return final_output

print("station_view function ready!")


def plot_station_arrivals_violin(incident_code, incident_date, station_id, interval_minutes=60, num_platforms=12, figsize=(18, 12)):
    """
    Generate a normalized violin plot for train arrivals at a station during an incident.
    """
    
    import matplotlib.pyplot as plt
    import numpy as np
    from scipy import stats
    
    # Get station analysis data (without printing output)
    import io
    import sys
    
    # Temporarily redirect stdout to suppress station_view output
    old_stdout = sys.stdout
    sys.stdout = buffer = io.StringIO()
    
    try:
        summary_df = station_view(incident_code, incident_date, station_id, interval_minutes)
    finally:
        sys.stdout = old_stdout
    
    if summary_df is None or summary_df.empty:
        print("No data available for analysis")
        return None
    
    # Normalize arrival counts by platform count
    ontime_counts = summary_df['ontime_arrival_count'].values / num_platforms
    delayed_counts = summary_df['delayed_arrival_count'].values / num_platforms
    
    # Remove zero values for cleaner visualization
    non_zero_ontime = ontime_counts[ontime_counts > 0]
    non_zero_delayed = delayed_counts[delayed_counts > 0]
    
    if len(non_zero_ontime) == 0 and len(non_zero_delayed) == 0:
        print("No train arrivals found to plot")
        return None
        
    # Prepare combined data for violin plot
    all_ontime_data = list(non_zero_ontime) if len(non_zero_ontime) > 0 else []
    all_delayed_data = list(non_zero_delayed) if len(non_zero_delayed) > 0 else []
    
    combined_data = []
    colors_for_scatter = []
    
    if len(all_ontime_data) > 0:
        combined_data.extend(all_ontime_data)
        colors_for_scatter.extend(['green'] * len(all_ontime_data))
    
    if len(all_delayed_data) > 0:
        combined_data.extend(all_delayed_data)
        colors_for_scatter.extend(['red'] * len(all_delayed_data))
    
    # Create the violin plot
    fig, ax = plt.subplots(figsize=figsize)
    
    violin_parts = ax.violinplot([combined_data], positions=[0], showmeans=True, 
                                showmedians=True, showextrema=True, widths=0.8)
    
    # Customize violin appearance
    for pc in violin_parts['bodies']:
        pc.set_facecolor('lightblue')
        pc.set_alpha(0.6)
        pc.set_edgecolor('darkblue')
        pc.set_linewidth(2)
    
    violin_parts['cmeans'].set_color('black')
    violin_parts['cmeans'].set_linewidth(3)
    violin_parts['cmedians'].set_color('white')
    violin_parts['cmedians'].set_linewidth(4)
    violin_parts['cbars'].set_color('black')
    violin_parts['cmins'].set_color('black')
    violin_parts['cmaxes'].set_color('black')
    
    # Add scatter points with different colors
    np.random.seed(42)
    combined_array = np.array(combined_data)
    colors_array = np.array(colors_for_scatter)
    jitter_all = np.random.normal(0, 0.05, len(combined_data))
    
    if len(all_ontime_data) > 0:
        ontime_mask = colors_array == 'green'
        ax.scatter(jitter_all[ontime_mask], combined_array[ontime_mask], 
                  alpha=0.8, s=80, color='darkgreen', zorder=3, 
                  edgecolors='white', linewidth=1.5, 
                  label=f'On-time arrivals')
    
    if len(all_delayed_data) > 0:
        delayed_mask = colors_array == 'red'
        ax.scatter(jitter_all[delayed_mask], combined_array[delayed_mask], 
                  alpha=0.8, s=80, color='darkred', zorder=3, 
                  edgecolors='white', linewidth=1.5, 
                  label=f'Delayed arrivals')
    
    # Calculate density for x-axis scaling
    kde_combined = stats.gaussian_kde(combined_data, bw_method=0.3)
    y_min, y_max = min(combined_data), max(combined_data)
    y_range = np.linspace(y_min, y_max, 100)
    density_values = kde_combined(y_range)
    max_density = np.max(density_values)
    
    # Set axis properties with numerical density values
    violin_width = 0.4
    ax.set_xlim(-violin_width * 1.5, violin_width * 1.5)
    ax.set_ylim(0, max(combined_data) + 0.5)
    
    density_positions = [-violin_width, -violin_width/2, 0, violin_width/2, violin_width]
    density_values_at_positions = [max_density * abs(pos) / violin_width for pos in density_positions]
    density_labels = [f'{val:.3f}' for val in density_values_at_positions]
    
    ax.set_xticks(density_positions)
    ax.set_xticklabels(density_labels)
    
    # Set labels and title
    ax.set_xlabel('Probability Density (trains per platform per period)', fontsize=20)
    ax.set_ylabel(f'Train Arrivals per Platform per {interval_minutes}-minute Period', fontsize=20)
    
    # Calculate statistics
    mean_combined = np.mean(combined_data)
    median_combined = np.median(combined_data)
    std_combined = np.std(combined_data)
    
    # Finalize plot
    ax.grid(True, alpha=0.3, axis='y')
    ax.legend(loc='upper right', fontsize=20)
    plt.tight_layout()
    plt.show()
    
    # Return analysis results
    results = {
        'summary_df': summary_df,
        'combined_stats': {
            'mean': mean_combined,
            'median': median_combined,
            'std': std_combined,
            'range': (min(combined_data), max(combined_data)),
            'max_density': max_density,
            'total_periods': len(combined_data)
        },
        'ontime_stats': {
            'periods': len(all_ontime_data),
            'mean': np.mean(all_ontime_data) if len(all_ontime_data) > 0 else 0,
            'median': np.median(all_ontime_data) if len(all_ontime_data) > 0 else 0,
            'range': (min(all_ontime_data), max(all_ontime_data)) if len(all_ontime_data) > 0 else (0, 0),
            'total': sum(all_ontime_data) * num_platforms if len(all_ontime_data) > 0 else 0
        },
        'delayed_stats': {
            'periods': len(all_delayed_data),
            'mean': np.mean(all_delayed_data) if len(all_delayed_data) > 0 else 0,
            'median': np.median(all_delayed_data) if len(all_delayed_data) > 0 else 0,
            'range': (min(all_delayed_data), max(all_delayed_data)) if len(all_delayed_data) > 0 else (0, 0),
            'total': sum(all_delayed_data) * num_platforms if len(all_delayed_data) > 0 else 0
        },
        'parameters': {
            'incident_code': incident_code,
            'incident_date': incident_date,
            'station_id': station_id,
            'interval_minutes': interval_minutes,
            'num_platforms': num_platforms
        }
    }
    
    return results

print("plot_station_arrivals_violin() function ready!")


def plot_normal_operations_violin(incident_code, incident_date, station_id, interval_minutes=60, num_platforms=12, figsize=(18, 12)):
    """
    Generate a normalized violin plot for "normal" operations using identical data filtering
    as the incident analysis, but treating all trains as on-time.
    """
    
    import matplotlib.pyplot as plt
    import numpy as np
    from scipy import stats
    
    # Get the incident analysis data to ensure identical filtering (without printing output)
    import io
    import sys
    
    # Temporarily redirect stdout to suppress station_view output
    old_stdout = sys.stdout
    sys.stdout = buffer = io.StringIO()
    
    try:
        incident_summary = station_view(incident_code, incident_date, station_id, interval_minutes)
    finally:
        sys.stdout = old_stdout
    
    if incident_summary is None or incident_summary.empty:
        print("No incident data available for comparison")
        return None
    
    # Convert incident data to "normal operations" by treating all trains as on-time
    normal_intervals = []
    
    for _, row in incident_summary.iterrows():
        total_arrivals = row['ontime_arrival_count'] + row['delayed_arrival_count']
        
        if total_arrivals > 0:
            normal_intervals.append({
                'time_period': row['time_period'],
                'arrival_count': total_arrivals,
                'original_ontime': row['ontime_arrival_count'],
                'original_delayed': row['delayed_arrival_count']
            })
    
    corrected_normal_summary = pd.DataFrame(normal_intervals)
    
    if len(corrected_normal_summary) == 0:
        print("No corrected normal operations data to plot")
        return None
    
    # Normalize by platform count
    arrival_counts = corrected_normal_summary['arrival_count'].values / num_platforms
    non_zero_arrivals = arrival_counts[arrival_counts > 0]
    
    # Create the violin plot
    fig, ax = plt.subplots(figsize=figsize)
    
    violin_parts = ax.violinplot([non_zero_arrivals], positions=[0], showmeans=True, 
                                showmedians=True, showextrema=True, widths=0.8)
    
    # Customize violin appearance
    for pc in violin_parts['bodies']:
        pc.set_facecolor('lightgreen')
        pc.set_alpha(0.7)
        pc.set_edgecolor('darkgreen')
        pc.set_linewidth(2)
    
    violin_parts['cmeans'].set_color('black')
    violin_parts['cmeans'].set_linewidth(3)
    violin_parts['cmedians'].set_color('white')
    violin_parts['cmedians'].set_linewidth(4)
    violin_parts['cbars'].set_color('black')
    violin_parts['cmins'].set_color('black')
    violin_parts['cmaxes'].set_color('black')
    
    # Add scatter points (all green since all are treated as "on-time")
    np.random.seed(42)
    jitter_all = np.random.normal(0, 0.05, len(non_zero_arrivals))
    ax.scatter(jitter_all, non_zero_arrivals, alpha=0.8, s=80, color='darkgreen', 
              zorder=3, edgecolors='white', linewidth=1.5)
    
    # Calculate density for x-axis scaling
    kde_corrected = stats.gaussian_kde(non_zero_arrivals, bw_method=0.3)
    y_min_corr, y_max_corr = min(non_zero_arrivals), max(non_zero_arrivals)
    y_range_corr = np.linspace(y_min_corr, y_max_corr, 100)
    density_values_corr = kde_corrected(y_range_corr)
    max_density_corr = np.max(density_values_corr)
    
    # Set axis properties with numerical density values
    violin_width = 0.4
    ax.set_xlim(-violin_width * 1.5, violin_width * 1.5)
    ax.set_ylim(0, max(non_zero_arrivals) + 0.5)
    
    density_positions = [-violin_width, -violin_width/2, 0, violin_width/2, violin_width]
    density_values_at_positions = [max_density_corr * abs(pos) / violin_width for pos in density_positions]
    density_labels = [f'{val:.3f}' for val in density_values_at_positions]
    
    ax.set_xticks(density_positions)
    ax.set_xticklabels(density_labels)
    
    # Set labels and title
    ax.set_xlabel('Probability Density (trains per platform per period)', fontsize=20)
    ax.set_ylabel(f'Normalized Arrivals per Platform per {interval_minutes}-minute Period', fontsize=20)
    
    # Calculate statistics
    mean_corrected = np.mean(non_zero_arrivals)
    median_corrected = np.median(non_zero_arrivals)
    std_corrected = np.std(non_zero_arrivals)
    
    # Finalize plot
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.show()
    
    # Return results
    results = {
        'corrected_normal_summary_df': corrected_normal_summary,
        'corrected_normal_stats': {
            'mean': mean_corrected,
            'median': median_corrected,
            'std': std_corrected,
            'range': (min(non_zero_arrivals), max(non_zero_arrivals)),
            'max_density': max_density_corr,
            'active_periods': len(non_zero_arrivals),
            'total_trains': corrected_normal_summary['arrival_count'].sum()
        },
        'parameters': {
            'incident_code': incident_code,
            'incident_date': incident_date,
            'station_id': station_id,
            'interval_minutes': interval_minutes,
            'num_platforms': num_platforms,
            'note': 'Uses identical data filtering as incident analysis'
        }
    }
    
    return results

print("plot_normal_operations_violin() function ready!")


def plot_2d_kde_arrivals_delays(station_summary_df, station_id, incident_code, incident_date, num_platforms=12, figsize=(12, 8)):
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.stats import gaussian_kde
    from datetime import datetime
    
    if station_summary_df is None or station_summary_df.empty:
        print("No station summary data available for 2D KDE analysis")
        return None, None
    
    # Extract arrival counts and average delay data from station_view outputs
    arrival_counts = []
    delay_values = []
    
    for _, row in station_summary_df.iterrows():
        # Get total arrivals for this time period and normalize by platform count
        total_arrivals = row['ontime_arrival_count'] + row['delayed_arrival_count']
        normalized_arrivals = total_arrivals / num_platforms
        
        # Calculate average delay for this time period
        delays = row['delay_minutes']
        if isinstance(delays, list) and len(delays) > 0:
            # Calculate average delay for this period (including on-time trains as 0 delay)
            total_delay_minutes = sum(delays)
            delayed_trains = len(delays)
            ontime_trains = row['ontime_arrival_count']
            
            # Average delay across ALL trains in this period (delayed + on-time)
            if total_arrivals > 0:
                average_delay = total_delay_minutes / total_arrivals
                arrival_counts.append(normalized_arrivals)
                delay_values.append(average_delay)
        elif total_arrivals > 0:
            # Period with only on-time trains (average delay = 0)
            arrival_counts.append(normalized_arrivals)
            delay_values.append(0.0)
    
    if len(arrival_counts) < 10:
        print(f"Insufficient data points ({len(arrival_counts)}) for KDE analysis")
        return None, None
    
    x = np.array(arrival_counts)
    y = np.array(delay_values)
    
    xy = np.vstack([x, y])
    kde = gaussian_kde(xy)
    
    x_grid = np.linspace(x.min(), x.max(), 50)
    y_grid = np.linspace(y.min(), y.max(), 50)
    X, Y = np.meshgrid(x_grid, y_grid)
    positions = np.vstack([X.ravel(), Y.ravel()])
    Z = kde(positions).reshape(X.shape)
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Create custom colormap that starts with white for zero/low values
    import matplotlib.colors as mcolors
    import matplotlib.cm as cm
    
    # Create a custom colormap starting with white
    colors = ['white', '#440154', '#31688e', '#35b779', '#fde725']  # white -> viridis colors
    n_bins = 256
    custom_cmap = mcolors.LinearSegmentedColormap.from_list('custom_viridis', colors, N=n_bins)
    
    # Set the lowest density values to map to white explicitly
    Z_min, Z_max = Z.min(), Z.max()
    norm = mcolors.Normalize(vmin=Z_min, vmax=Z_max)
    
    contour = ax.contourf(X, Y, Z, levels=20, cmap=custom_cmap, norm=norm, alpha=0.7)
    ax.scatter(x, y, c='red', s=15, alpha=0.6, edgecolors='black', linewidth=0.5)
    
    ax.set_xlabel(f'Train Arrivals per Platform per Period')
    ax.set_ylabel('Average Delay Minutes')
    ax.grid(True, alpha=0.3)
    
    plt.colorbar(contour, ax=ax, label='Density')
    plt.tight_layout()
    plt.show()
    
    return fig, ax

# train view functions:

def train_view(all_data, origin_code, destination_code, input_date_str):
    """
    View all train journeys between an OD pair and check for incidents on a specific date.
    Corrects PLANNED_CALLS using ACTUAL_CALLS - PFPI_MINUTES.
    
    Parameters
    ----------
    all_data : pd.DataFrame
        Must contain:
        ['TRAIN_SERVICE_CODE', 'PLANNED_ORIGIN_LOCATION_CODE', 'PLANNED_ORIGIN_GBTT_DATETIME',
         'PLANNED_DEST_LOCATION_CODE', 'PLANNED_DEST_GBTT_DATETIME', 'PLANNED_CALLS', 'ACTUAL_CALLS',
         'PFPI_MINUTES', 'INCIDENT_REASON', 'INCIDENT_NUMBER', 'EVENT_TYPE', 'SECTION_CODE', 'DELAY_DAY',
         'EVENT_DATETIME', 'INCIDENT_START_DATETIME', 'ENGLISH_DAY_TYPE', 'STATION_ROLE', 'DFT_CATEGORY',
         'PLATFORM_COUNT', 'DATASET_TYPE', 'WEEKDAY', 'STANOX', 'DAY']
    origin_code : str or int
        Origin location code.
    destination_code : str or int
        Destination location code.
    input_date_str : str
        Date in the format 'DD-MMM-YYYY' (e.g., '13-JUN-2024').

    Returns
    -------
    pd.DataFrame or str
        DataFrame of matching incidents (with corrected PLANNED_CALLS) or a message.
    """

    # --- Ensure OD_PAIR exists ---
    if 'OD_PAIR' not in all_data.columns:
        all_data['OD_PAIR'] = (
            all_data['PLANNED_ORIGIN_LOCATION_CODE'].astype(str)
            + '_'
            + all_data['PLANNED_DEST_LOCATION_CODE'].astype(str)
        )

    # --- Define OD pair key and convert dates ---
    od_pair = f"{origin_code}_{destination_code}"
    input_date = pd.to_datetime(input_date_str, format='%d-%b-%Y', errors='coerce')
    all_data['INCIDENT_START_DATETIME'] = pd.to_datetime(
        all_data['INCIDENT_START_DATETIME'], errors='coerce'
    )

    # --- Check if OD pair exists ---
    if od_pair not in all_data['OD_PAIR'].unique():
        message = f"OD pair {od_pair} not found in dataset."
        print(message)
        return message

    # --- Subset for this OD pair ---
    trains_between = all_data[all_data['OD_PAIR'] == od_pair].copy()

    # --- Correct PLANNED_CALLS using ACTUAL_CALLS - PFPI_MINUTES ---
    trains_between['ACTUAL_CALLS_dt'] = pd.to_datetime(
        trains_between['ACTUAL_CALLS'], format='%H%M', errors='coerce'
    )
    trains_between['PFPI_MINUTES_num'] = pd.to_numeric(
        trains_between['PFPI_MINUTES'], errors='coerce'
    )

    trains_between['CORRECTED_PLANNED_CALLS_dt'] = (
        trains_between['ACTUAL_CALLS_dt']
        - pd.to_timedelta(trains_between['PFPI_MINUTES_num'].fillna(0), unit='m')
    )
    trains_between['PLANNED_CALLS'] = trains_between[
        'CORRECTED_PLANNED_CALLS_dt'
    ].dt.strftime('%H%M').fillna(trains_between['PLANNED_CALLS'])

    print(f" Train journeys between {origin_code} and {destination_code}: {len(trains_between)} records found.")
    print(f"Unique train service codes: {trains_between['TRAIN_SERVICE_CODE'].dropna().unique().tolist()}")


    # --- Filter incidents by date ---
    incidents_on_date = trains_between[
        trains_between['INCIDENT_START_DATETIME'].dt.date == input_date.date()
    ].copy()

    if incidents_on_date.empty:
        message = f" No incidents found for OD pair {od_pair} on {input_date_str}."
        print(message)
        return message
    else:
        message = f" {len(incidents_on_date)} incident(s) found for OD pair {od_pair} on {input_date_str}:"
        print(message)

        # --- Columns to show ---
        cols_to_show = [
            'TRAIN_SERVICE_CODE', 'PLANNED_ORIGIN_LOCATION_CODE', 'PLANNED_ORIGIN_GBTT_DATETIME',
            'PLANNED_DEST_LOCATION_CODE', 'PLANNED_DEST_GBTT_DATETIME', 'PLANNED_CALLS', 'ACTUAL_CALLS',
            'PFPI_MINUTES', 'INCIDENT_REASON', 'INCIDENT_NUMBER', 'EVENT_TYPE', 'SECTION_CODE', 'DELAY_DAY',
            'EVENT_DATETIME', 'INCIDENT_START_DATETIME', 'ENGLISH_DAY_TYPE', 'STATION_ROLE', 'DFT_CATEGORY',
            'PLATFORM_COUNT', 'DATASET_TYPE', 'WEEKDAY', 'STANOX', 'DAY'
        ]

        # Display filtered columns only (in your preferred order)
        display(incidents_on_date[cols_to_show])
        return incidents_on_date[cols_to_show]

def get_stanox_for_service(all_data, train_service_code, origin_code, destination_code):
    """
    For a given train service and OD pair, find the first train of the day (by earliest PLANNED_ORIGIN_GBTT_DATETIME),
    then order the unique STANOX codes for that train:
    - Use PLANNED_ORIGIN_GBTT_DATETIME for the origin,
    - PLANNED_DEST_GBTT_DATETIME for the destination,
    - PLANNED_CALLS for all other stations.
    Return the ordered list of STANOX codes for that train only, always including the origin as the first element.
    """
    import pandas as pd

    # --- Ensure OD_PAIR exists ---
    if 'OD_PAIR' not in all_data.columns:
        all_data['OD_PAIR'] = (
            all_data['PLANNED_ORIGIN_LOCATION_CODE'].astype(str)
            + '_' +
            all_data['PLANNED_DEST_LOCATION_CODE'].astype(str)
        )

    od_pair = f"{origin_code}_{destination_code}"

    # --- Filter dataset for this service and OD pair ---
    subset = all_data[
        (all_data['OD_PAIR'] == od_pair)
        & (all_data['TRAIN_SERVICE_CODE'].astype(str) == str(train_service_code))
    ].copy()

    if subset.empty:
        message = f"🚫 No records found for train service {train_service_code} on OD pair {od_pair}."
        print(message)
        return message

    # --- Find the first train of the day (earliest PLANNED_ORIGIN_GBTT_DATETIME) ---
    subset['PLANNED_ORIGIN_GBTT_DATETIME'] = pd.to_datetime(subset['PLANNED_ORIGIN_GBTT_DATETIME'], format='%H%M', errors='coerce')
    first_origin_time = subset['PLANNED_ORIGIN_GBTT_DATETIME'].min()
    first_train = subset[subset['PLANNED_ORIGIN_GBTT_DATETIME'] == first_origin_time].copy()

    # --- For this train, get all unique STANOX codes and their times ---
    def get_time(row):
        stanox = str(row['STANOX'])
        if stanox == str(origin_code) and pd.notna(row.get('PLANNED_ORIGIN_GBTT_DATETIME', None)):
            return pd.to_datetime(row['PLANNED_ORIGIN_GBTT_DATETIME'], format='%H%M', errors='coerce')
        elif stanox == str(destination_code) and pd.notna(row.get('PLANNED_DEST_GBTT_DATETIME', None)):
            return pd.to_datetime(row['PLANNED_DEST_GBTT_DATETIME'], format='%H%M', errors='coerce')
        elif pd.notna(row.get('PLANNED_CALLS', None)) and str(row['PLANNED_CALLS']).isdigit() and len(str(row['PLANNED_CALLS'])) == 4:
            # Use arbitrary date for time-only values to allow sorting
            return pd.to_datetime(str(row['PLANNED_CALLS']), format='%H%M', errors='coerce')
        else:
            return pd.NaT

    # Keep only unique STANOX for this train, keeping the earliest time for each
    first_train['ORDER_TIME'] = first_train.apply(get_time, axis=1)
    ordered = first_train.groupby('STANOX', as_index=False)['ORDER_TIME'].min()
    ordered = ordered.sort_values('ORDER_TIME')
    stanox_list = ordered['STANOX'].astype(str).tolist()

    # Always include origin_code as the first element, even if not present in the data
    origin_str = str(origin_code)
    if origin_str in stanox_list:
        stanox_list.remove(origin_str)
    stanox_list = [origin_str] + stanox_list

    print(f"✅ Ordered STANOX for first train of the day for service {train_service_code} on OD pair {od_pair} (origin always first):")
    print(stanox_list)
    return stanox_list

def map_train_journey_with_incidents(
    all_data, service_stanox, incident_results=None,
    stations_ref_path=r"C:\Users\39342\University of Glasgow\Ji-Eun Byun - MZ-JB\MSci (Research) 2024-25\reference data\stations_ref_with_dft.json",
    incident_color="purple", service_code=None, date_str=None
    ):
    """
    1. Map each unique incident (INCIDENT_NUMBER, INCIDENT_START_DATETIME) as a numbered marker (chronological index), popup shows incident number, datetime and reason.
    2. Map service_stanox as the journey sequence, connecting stations with lines.
    3. For each STANOX, sum all PFPI_MINUTES from provided incident result DataFrames and use color-grading for the marker. Station popups list the chronological incident indices (1,2,3...) that affected them.
    """
    import json
    import folium
    import pandas as pd

    # Load station reference
    with open(stations_ref_path, "r") as f:
        station_ref = json.load(f)

    # Build station coordinates for the service_stanox sequence only
    stanox_coords = []
    for s in service_stanox:
        s_str = str(int(float(s))) if isinstance(s, (int, float)) else str(s)
        match = next((item for item in station_ref if str(item.get("stanox", "")) == s_str), None)
        if match and 'latitude' in match and 'longitude' in match:
            try:
                lat = float(match['latitude'])
                lon = float(match['longitude'])
                stanox_coords.append((s_str, lat, lon))
            except Exception:
                continue

    if not stanox_coords:
        print("⚠️ No coordinates found for STANOX in this service.")
        return None

    # --- Folium map visualization ---
    mid_lat = sum([lat for _, lat, _ in stanox_coords]) / len(stanox_coords)
    mid_lon = sum([lon for _, _, lon in stanox_coords]) / len(stanox_coords)
    m = folium.Map(location=[mid_lat, mid_lon], zoom_start=8, tiles="CartoDB positron")

    # Add title if provided
    if service_code and date_str:
        title_html = f"<div style='position: fixed; top: 10px; left: 50%; transform: translateX(-50%); z-index:9999; font-size:18px; background: white; border:2px solid grey; border-radius:8px; padding: 10px;'><b>Train Service: {service_code}</b><br><b>Date: {date_str}</b></div>"
        m.get_root().html.add_child(folium.Element(title_html))

    # Draw lines between each consecutive station in stanox_coords
    for i in range(len(stanox_coords) - 1):
        start = stanox_coords[i][1], stanox_coords[i][2]
        end = stanox_coords[i+1][1], stanox_coords[i+1][2]
        folium.PolyLine([start, end], color="blue", weight=4, opacity=0.8).add_to(m)

    # Prepare list of DataFrames from incident_results
    dfs = []
    if incident_results:
        for res in incident_results:
            if isinstance(res, pd.DataFrame):
                dfs.append(res)

    delays_df = pd.concat(dfs, ignore_index=True) if dfs else None

    # Aggregate delays for each STANOX (sum of PFPI_MINUTES for all incidents and all records)
    stanox_delay = {}
    stanox_incidents = {}  # maps stanox -> list of incident id strings
    if delays_df is not None and 'STANOX' in delays_df.columns and 'PFPI_MINUTES' in delays_df.columns:
        delays_df['PFPI_MINUTES_num'] = pd.to_numeric(delays_df['PFPI_MINUTES'], errors='coerce').fillna(0)
        # Normalize INCIDENT_NUMBER to string for consistent display
        if 'INCIDENT_NUMBER' in delays_df.columns:
            def _norm_inc(x):
                try:
                    if pd.isna(x):
                        return None
                    xf = float(x)
                    if xf.is_integer():
                        return str(int(xf))
                    else:
                        return str(x)
                except Exception:
                    return str(x)
            delays_df['INCIDENT_NUMBER_str'] = delays_df['INCIDENT_NUMBER'].apply(_norm_inc)
        else:
            delays_df['INCIDENT_NUMBER_str'] = None

        for stanox, group in delays_df.groupby('STANOX'):
            total_delay = group['PFPI_MINUTES_num'].sum()
            stanox_delay[str(stanox)] = total_delay
            # collect unique incident numbers for this stanox
            if 'INCIDENT_NUMBER_str' in group.columns:
                unique_incs = sorted([str(v) for v in pd.unique(group['INCIDENT_NUMBER_str'].dropna())])
            else:
                unique_incs = []
            stanox_incidents[str(stanox)] = unique_incs

    # Build chronological ranking for unique incidents (1 = earliest start time)
    incident_rank = {}  # maps incident_id_str -> rank int
    if delays_df is not None and 'INCIDENT_NUMBER_str' in delays_df.columns and 'INCIDENT_START_DATETIME' in delays_df.columns:
        temp = delays_df[['INCIDENT_NUMBER_str', 'INCIDENT_START_DATETIME']].dropna(subset=['INCIDENT_NUMBER_str', 'INCIDENT_START_DATETIME']).drop_duplicates(subset=['INCIDENT_NUMBER_str']).copy()
        if not temp.empty:
            temp['INCIDENT_START_dt'] = pd.to_datetime(temp['INCIDENT_START_DATETIME'], errors='coerce')
            temp = temp.sort_values('INCIDENT_START_dt')
            temp = temp.reset_index(drop=True)
            temp['incident_rank'] = temp.index + 1
            incident_rank = dict(zip(temp['INCIDENT_NUMBER_str'].astype(str), temp['incident_rank'].astype(int)))

    # --- Color grading function to match incident_view_heatmap_html legend ---
    def get_color(delay):
        try:
            d = float(delay)
        except Exception:
            d = 0
        if d == 0:
            return "blue"
        if d <= 5:
            return '#32CD32'     # Minor (1-5 min) - Lime Green
        elif d <= 15:
            return '#FFD700'     # Moderate (6-15 min) - Gold
        elif d <= 30:
            return '#FF8C00'     # Significant (16-30 min) - Dark Orange
        elif d <= 60:
            return '#FF0000'     # Major (31-60 min) - Red
        elif d <= 120:
            return '#8B0000'     # Severe (61-120 min) - Dark Red
        else:
            return '#8A2BE2'     # Critical (120+ min) - Blue Violet

    # Map station markers with color-grading (no delay=blue, then severity colours)
    for stanox, lat, lon in stanox_coords:
        delay_val = stanox_delay.get(stanox, 0)
        color = get_color(delay_val)
        # Prepare ranked incident list for popup, truncate if long
        inc_list = stanox_incidents.get(stanox, [])
        if inc_list:
            # Map incident ids to ranks, use id fallback if rank not available
            inc_ranks = [str(incident_rank.get(str(i), i)) for i in inc_list]
            if len(inc_ranks) > 10:
                inc_display = ', '.join(inc_ranks[:10]) + f', ... (+{len(inc_ranks)-10} more)'
            else:
                inc_display = ', '.join(inc_ranks)
            incidents_html = f"<br><b>Incidents (by index):</b> {inc_display}"
        else:
            incidents_html = ''

        popup_html = f"<b>STANOX {stanox}</b><br>Total delay: {delay_val:.1f} min{incidents_html}"
        folium.CircleMarker(
            location=(lat, lon),
            radius=6,
            color=color,
            fill=True,
            fill_opacity=0.8,
            popup=folium.Popup(popup_html, max_width=400)
        ).add_to(m)

    # Map unique incidents aggregated by SECTION_CODE: create one numbered marker per location with all incident ranks
    incident_records = pd.concat(dfs, ignore_index=True) if dfs else None
    if incident_records is not None and 'INCIDENT_NUMBER' in incident_records.columns and 'INCIDENT_START_DATETIME' in incident_records.columns and 'SECTION_CODE' in incident_records.columns:
        # Normalize incident id string
        incident_records['INCIDENT_NUMBER_str'] = incident_records['INCIDENT_NUMBER'].apply(lambda x: (str(int(float(x))) if (pd.notna(x) and float(x).is_integer()) else str(x)))
        # Build unique incident list with parsed start times
        incident_unique = incident_records.drop_duplicates(subset=['INCIDENT_NUMBER_str', 'SECTION_CODE']).copy()
        incident_unique['INCIDENT_START_dt'] = pd.to_datetime(incident_unique['INCIDENT_START_DATETIME'], errors='coerce')
        # Compute incident durations
        incident_durations = {}
        for inc in incident_records['INCIDENT_NUMBER_str'].unique():
            subset = incident_records[incident_records['INCIDENT_NUMBER_str'] == inc]
            start = pd.to_datetime(subset['INCIDENT_START_DATETIME'].min(), format='%Y-%m-%d %H:%M:%S', errors='coerce')
            end = pd.to_datetime(subset['EVENT_DATETIME'].max(), format='%d-%b-%Y %H:%M', errors='coerce')
            # Ensure duration is not negative (handle data inconsistencies)
            duration = end - start
            duration = max(duration, pd.Timedelta(0))
            incident_durations[inc] = duration
        # For each SECTION_CODE, collect all incidents that occurred there
        section_map = {}
        for _, row in incident_unique.iterrows():
            section = str(row['SECTION_CODE'])
            inc_id = str(row['INCIDENT_NUMBER_str'])
            inc_num = row.get('INCIDENT_NUMBER')
            inc_time = row.get('INCIDENT_START_DATETIME')
            inc_reason = row.get('INCIDENT_REASON') if 'INCIDENT_REASON' in row.index else None
            rank = incident_rank.get(inc_id)
            entry = {
                'inc_id': inc_id,
                'inc_num': inc_num,
                'inc_time': inc_time,
                'inc_reason': inc_reason,
                'rank': rank if rank is not None else ''
            }
            section_map.setdefault(section, []).append(entry)

        # For each section, sort entries by rank (if available) then plot a single numbered marker showing combined ranks
        for section_code, entries in section_map.items():
            # sort by rank where possible
            entries_sorted = sorted(entries, key=lambda e: (e['rank'] if isinstance(e['rank'], int) else 999999))
            ranks = [str(e['rank']) if e['rank'] != '' else e['inc_id'] for e in entries_sorted]
            ranks_display = ','.join(ranks)
            # popup lists each incident number, datetime and reason
            popup_lines = []
            for e in entries_sorted:
                reason_text = e['inc_reason'] if e.get('inc_reason') else 'N/A'
                dur = incident_durations.get(e['inc_id'], 'N/A')
                popup_lines.append(f"Incident: {e['inc_num']} — {e['inc_time']} — Reason: {reason_text} — Duration: {dur}")
            popup_html = '<br>'.join(popup_lines)

            # find coords for this section_code
            match = next((item for item in station_ref if str(item.get("stanox", "")) == section_code), None)
            if match and 'latitude' in match and 'longitude' in match:
                lat = float(match['latitude']) + 0.0005  # Small offset to avoid overlap with station markers
                lon = float(match['longitude']) + 0.0005  # Small offset to avoid overlap with station markers
                # adjust DivIcon size based on text length
                size_px = max(28, min(80, 12 * len(ranks_display)))
                number_html = f"<div style='background:{incident_color};color:#fff;border-radius:50%;min-width:{size_px}px;height:{size_px}px;display:inline-flex;align-items:center;justify-content:center;font-weight:bold;border:2px solid #ffffff;padding:4px'>{ranks_display}</div>"
                folium.Marker(
                    location=(lat, lon),
                    icon=folium.DivIcon(html=number_html),
                    popup=folium.Popup(popup_html, max_width=400)
                ).add_to(m)

    # --- Add legend/key for color grading matching incident_view_heatmap_html ---
    legend_html = '''
     <div style="position: fixed; bottom: 100px; left: 50px; width: 260px; height: 170px; z-index:9999; font-size:14px; background: white; border:2px solid grey; border-radius:8px; padding: 10px;">
     <b>Delay Intensity Key</b><br>
     <i class="fa fa-circle" style="color:blue"></i> 0 min (No delay)<br>
     <i class="fa fa-circle" style="color:#32CD32"></i> 1-5 min (Minor)<br>
     <i class="fa fa-circle" style="color:#FFD700"></i> 6-15 min (Moderate)<br>
     <i class="fa fa-circle" style="color:#FF8C00"></i> 16-30 min (Significant)<br>
     <i class="fa fa-circle" style="color:#FF0000"></i> 31-60 min (Major)<br>
     <i class="fa fa-circle" style="color:#8B0000"></i> 61-120 min (Severe)<br>
     <i class="fa fa-circle" style="color:#8A2BE2"></i> 120+ min (Critical)<br>
     <br><b>Incident markers:</b> numbered by chronological order (1 = earliest). Multiple ranks at same location shown as comma-separated list.<br>
     </div>
     '''
    m.get_root().html.add_child(folium.Element(legend_html))

    folium.LayerControl().add_to(m)
    print("Map created for service journey and incidents with color-graded station markers and aggregated numbered incident markers.")
    return m

def train_view_2(all_data, service_stanox, service_code, stations_ref_path=r"C:\Users\39342\University of Glasgow\Ji-Eun Byun - MZ-JB\MSci (Research) 2024-25\reference data\stations_ref_with_dft.json"):
    """
    Compute reliability metrics for each station in the service_stanox list for a given train service code.

    Metrics now exclude PFPI_MINUTES == 0.0 when computing mean/variance and incident counts.
    OnTime% is computed on the original PFPI distribution (<=0) so it still reflects punctuality.

    Returns a DataFrame with columns: ServiceCode, StationName, MeanDelay, DelayVariance, OnTime%, IncidentCount
    """
    import json
    import pandas as pd
    import numpy as np

    # Load station reference
    try:
        with open(stations_ref_path, "r") as f:
            station_ref = json.load(f)
        stanox_to_name = {str(item.get("stanox", "")): (item.get("description") or item.get("name") or str(item.get("stanox",""))) for item in station_ref}
    except Exception:
        stanox_to_name = {}

    results = []

    for s in service_stanox:
        # Filter data for this STANOX and service code
        subset = all_data[
            (all_data['STANOX'] == str(s)) &
            (all_data['TRAIN_SERVICE_CODE'].astype(str) == str(service_code))
        ].copy()

        if subset.empty:
            continue

        # Convert PFPI_MINUTES to numeric (keep full series for on-time calc)
        pfpi_all = pd.to_numeric(subset['PFPI_MINUTES'], errors='coerce').dropna()
        # On-time percentage (<= 0 minutes)
        on_time_pct = (pfpi_all <= 0).sum() / len(pfpi_all) * 100 if len(pfpi_all) > 0 else np.nan

        # Exclude 0.0 delays for mean/variance and incident counting
        pfpi_pos = pfpi_all[pfpi_all > 0]
        mean_delay = pfpi_pos.mean() if len(pfpi_pos) > 0 else np.nan
        delay_variance = pfpi_pos.var() if len(pfpi_pos) > 0 else np.nan
        incident_count = len(pfpi_pos)

        station_name = stanox_to_name.get(str(s), f"{s}")

        results.append({
            'ServiceCode': service_code,
            'StationName': station_name,
            'MeanDelay': mean_delay,
            'DelayVariance': delay_variance,
            'OnTime%': on_time_pct,
            'IncidentCount': incident_count
        })

    return pd.DataFrame(results)

def plot_reliability_graphs(all_data, service_stanox, service_code, stations_ref_path=r"C:\Users\39342\University of Glasgow\Ji-Eun Byun - MZ-JB\MSci (Research) 2024-25\reference data\stations_ref_with_dft.json", cap_minutes=75):
    """
    Generate overlapping density (KDE) curves and cumulative distribution plots: Delay distribution per station (all curves overlapping, different colours), excluding delay==0.0 and capped at cap_minutes.
    """
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd

    # try to get station names
    try:
        import json
        with open(stations_ref_path, 'r') as f:
            station_ref = json.load(f)
        stanox_to_name = {str(item.get('stanox','')): item.get('description') or item.get('name') or str(item.get('stanox','')) for item in station_ref}
    except Exception:
        stanox_to_name = {}

    station_labels = []
    delay_data = []

    for s in service_stanox:
        subset = all_data[
            (all_data['STANOX'] == str(s)) &
            (all_data['TRAIN_SERVICE_CODE'].astype(str) == str(service_code))
        ].copy()

        if subset.empty:
            continue

        # convert pfpi and drop nan
        pfpi_all = pd.to_numeric(subset['PFPI_MINUTES'], errors='coerce').dropna()
        # exclude zeros for plotting/stats (per user request)
        pfpi_pos = pfpi_all[pfpi_all > 0]
        delays = pfpi_pos.values
        if len(delays) == 0:
            continue

        label = stanox_to_name.get(str(s), str(s))
        station_labels.append(label)
        delay_data.append(delays)

    if not station_labels:
        print('No data to plot.')
        return

    # Graph 1: Overlapping density plots (KDE) per station
    plt.figure(figsize=(10, 6))

    cmap = plt.get_cmap('tab10')
    n = len(delay_data)
    colors = [cmap(i % cmap.N) for i in range(n)]

    # Determine x range from percentiles but cap at cap_minutes
    all_vals = np.concatenate(delay_data)
    xmin = max(0, np.nanpercentile(all_vals, 1))
    xmax = np.nanpercentile(all_vals, 99)
    x_vals = np.linspace(xmin, xmax, 400)

    try:
        import seaborn as sns
        for i, delays in enumerate(delay_data):
            # clip delays to [0, cap_minutes]
            clipped = np.clip(delays, 0, cap_minutes)
            sns.kdeplot(clipped, bw_adjust=1, label=station_labels[i], color=colors[i], fill=False, clip=(0, cap_minutes))
    except Exception:
        try:
            from scipy.stats import gaussian_kde
            for i, delays in enumerate(delay_data):
                clipped = np.clip(delays, 0, cap_minutes)
                try:
                    kde = gaussian_kde(clipped)
                    y = kde(x_vals)
                except Exception:
                    y = np.zeros_like(x_vals)
                    hist_vals, bin_edges = np.histogram(clipped, bins=20, density=True)
                    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
                    y = np.interp(x_vals, bin_centers, hist_vals, left=0, right=0)
                plt.plot(x_vals, y, label=station_labels[i], color=colors[i])
        except Exception:
            for i, delays in enumerate(delay_data):
                clipped = np.clip(delays, 0, cap_minutes)
                hist_vals, bin_edges = np.histogram(clipped, bins=30, density=True)
                bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
                plt.plot(bin_centers, hist_vals, label=station_labels[i], color=colors[i])

    plt.xlim(0, cap_minutes)
    plt.xlabel('Delay (minutes)')
    plt.ylabel('PDF')
    plt.title(f'Delay Distribution per Station (overlapping KDEs, capped at {cap_minutes} min)')
    plt.legend(loc='upper right', fontsize='small')
    plt.grid(axis='y', linestyle='--', alpha=0.4)
    plt.tight_layout()
    plt.show()

    # Graph 2: Cumulative distribution plots per station
    plt.figure(figsize=(10, 6))

    for i, delays in enumerate(delay_data):
        clipped = np.clip(delays, 0, cap_minutes)
        sorted_delays = np.sort(clipped)
        y = np.arange(1, len(sorted_delays) + 1) / len(sorted_delays)
        plt.plot(sorted_delays, y, label=station_labels[i], color=colors[i])

    plt.xlim(0, cap_minutes)
    plt.xlabel('Delay (minutes)')
    plt.ylabel('CDF')
    plt.title(f'Cumulative Delay Distribution per Station (capped at {cap_minutes} min)')
    plt.legend(loc='lower right', fontsize='small')
    plt.grid(axis='y', linestyle='--', alpha=0.4)
    plt.tight_layout()
    plt.show()


# for time_view,

def create_time_view_html(date_str, all_data):
    """
    Create an HTML map showing affected stations for a given date, with markers sized by incident count and colored by total PFPI minutes.
    """
    import folium
    import json
    from data.reference import reference_files
    # --- Color grading function to match incident_view_heatmap_html legend ---
    def get_color(delay):
        try:
            d = float(delay)
        except Exception:
            d = 0
        if d == 0:
            return "blue"
        if d <= 5:
            return '#32CD32'     # Minor (1-5 min) - Lime Green
        elif d <= 15:
            return '#FFD700'     # Moderate (6-15 min) - Gold
        elif d <= 30:
            return '#FF8C00'     # Significant (16-30 min) - Dark Orange
        elif d <= 60:
            return '#FF0000'     # Major (31-60 min) - Red
        elif d <= 120:
            return '#8B0000'     # Severe (61-120 min) - Dark Red
        else:
            return '#8A2BE2'     # Critical (120+ min) - Blue Violet
    
    # Filter data for the specified date
    filtered_data = all_data[all_data['INCIDENT_START_DATETIME'].str.contains(date_str, na=False)]
    
    if filtered_data.empty:
        print(f"No data found for date {date_str}")
        return
    
    # Get unique affected STANOX codes
    affected_stanox = filtered_data['STANOX'].unique()
    
    # Count incidents per STANOX
    incident_counts = filtered_data.groupby('STANOX')['INCIDENT_NUMBER'].nunique()
    
    # Sum PFPI_MINUTES per STANOX
    total_pfpi = filtered_data.groupby('STANOX')['PFPI_MINUTES'].sum()
    
    # Load station coordinates
    try:
        with open(reference_files["all dft categories"], 'r') as f:
            stations_data = json.load(f)
        stanox_to_coords = {}
        for station in stations_data:
            if 'stanox' in station and 'latitude' in station and 'longitude' in station:
                stanox_to_coords[str(station['stanox'])] = [station['latitude'], station['longitude']]
    except Exception as e:
        print(f"Error loading coordinates: {e}")
        return
    
    # Create Folium map centered on UK
    m = folium.Map(location=[54.5, -2.5], zoom_start=6)
    
    # Add markers for each affected station, sized by incident count and colored by total PFPI
    for stanox in affected_stanox:
        stanox_str = str(stanox)
        if stanox_str in stanox_to_coords:
            lat, lon = stanox_to_coords[stanox_str]
            count = incident_counts.get(stanox, 0)
            count = int(count) if pd.notna(count) else 0
            total_delay = total_pfpi.get(stanox, 0)
            total_delay = float(total_delay) if pd.notna(total_delay) else 0.0
            color = get_color(total_delay)
            radius = int(5 + count * 2)  # Scale radius with incident count
            folium.CircleMarker(
                location=[float(lat), float(lon)],
                radius=radius,
                color=color,
                fill=True,
                fill_color=color,
                fill_opacity=0.7,
                popup=f"STANOX: {stanox_str}<br>Incidents: {count}<br>Total Delay: {total_delay:.1f} min"
            ).add_to(m)
        else:
            print(f"Coordinates not found for STANOX: {stanox_str}")
    
    # Add title
    title_html = f'''
    <div style="position: fixed; top: 10px; left: 50px; width: 350px; height: 50px; background-color: white; border:2px solid grey; z-index:9999; font-size:16px; padding: 7px; text-align: center;">
    <b>One-Day Delay Map for {date_str}</b>
    </div>
    '''
    m.get_root().html.add_child(folium.Element(title_html))
    
    # Add legend
    legend_html = '''
    <div style="position: fixed; bottom: 100px; left: 50px; width: 180px; height: 300px; background-color: white; border:2px solid grey; z-index:9999; font-size:14px; padding: 10px;">
    <p><b>Delay Legend (Total PFPI Minutes)</b></p>
    <p><span style="color:blue;">●</span> 0 min</p>
    <p><span style="color:#32CD32;">●</span> 1-5 min</p>
    <p><span style="color:#FFD700;">●</span> 6-15 min</p>
    <p><span style="color:#FF8C00;">●</span> 16-30 min</p>
    <p><span style="color:#FF0000;">●</span> 31-60 min</p>
    <p><span style="color:#8B0000;">●</span> 61-120 min</p>
    <p><span style="color:#8A2BE2;">●</span> 120+ min</p>
    </div>
    '''
    m.get_root().html.add_child(folium.Element(legend_html))
    
    # Save the map to HTML file
    output_file = f"time_view_{date_str.replace('-', '_')}.html"
    m.save(output_file)
    print(f"Map saved to {output_file}")