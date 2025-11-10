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
    
    # Extend y-axis to prevent labels from spilling outside (add 10% extra space at top)
    current_ylim = ax1.get_ylim()
    ax1.set_ylim(current_ylim[0], current_ylim[1] * 1.10)
    
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
        # Position text box in upper right corner to avoid overlapping with bars
        ax2.text(0.98, 0.98, f'Total Events: {total_events}\nAverage Delay: {avg_delay:.1f} min', 
                transform=ax2.transAxes, verticalalignment='top', horizontalalignment='right', fontsize=20,
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
        
        print(f"CONTINUOUS HEATMAP CREATED! ")
        print(f"File: {output_file}")
        print(f"Time steps: {len(time_steps)} ({interval_minutes}-minute intervals)")
        print(f"Total stations mapped: {len(all_station_coords_map)} (DFT categories A/B/C1/C2)")
        print(f"Affected stations: {len(station_timeline_data)}")
        print(f"Features: Continuous delay heatmap with fading colors + grey station dots overlay")
        print(" Open the HTML file in your browser to explore the continuous heatmap!")
        
    except Exception as e:
        print(f"Error saving file: {e}")
    
    return html_content

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

def get_stanox_for_service(all_data, train_service_code, origin_code, destination_code, date_str=None):
    """
    Get ALL unique STANOX codes that a train service calls at, regardless of specific train instance.
    Returns a list of all stations that this service code stops at.
    
    Strategy:
    1. Filter to the specified service code and OD pair
    2. Optionally filter by date if provided
    3. Collect ALL unique STANOX codes that appear with valid scheduled stops
    4. Return the complete set (map will connect them by proximity)
    """
    import pandas as pd
    from datetime import datetime

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

    # --- Filter by date if provided ---
    if date_str:
        try:
            # Filter by EVENT_DATETIME containing the target date
            date_subset = subset[subset['EVENT_DATETIME'].str.contains(date_str, na=False)].copy()
            
            if not date_subset.empty:
                subset = date_subset
                print(f"✅ Filtered to date: {date_str}")
            else:
                print(f"⚠️ No records found for date {date_str}, using all dates for this service")
        except Exception as e:
            print(f"⚠️ Error filtering by date {date_str}: {e}, using all dates")

    # --- Get all unique STANOX codes with valid scheduled calls ---
    # Filter to rows that have actual scheduled stops (PLANNED_CALLS)
    valid_stops = subset[subset['PLANNED_CALLS'].notna()].copy()
    
    # Also include origin and destination explicitly
    origin_dest_stops = subset[
        (subset['STANOX'] == str(origin_code)) | 
        (subset['STANOX'] == str(destination_code))
    ].copy()
    
    # Combine both
    all_stops = pd.concat([valid_stops, origin_dest_stops], ignore_index=True)
    
    # Get unique STANOX codes
    stanox_list = all_stops['STANOX'].astype(str).unique().tolist()
    
    # Ensure origin and destination are included
    origin_str = str(origin_code)
    dest_str = str(destination_code)
    
    if origin_str not in stanox_list:
        stanox_list.append(origin_str)
    if dest_str not in stanox_list:
        stanox_list.append(dest_str)
    
    print(f"✅ Retrieved ALL stations for service {train_service_code} on OD pair {od_pair}")
    print(f"   Total unique stations: {len(stanox_list)}")
    print(f"   Stations: {stanox_list}")
    return stanox_list

def map_train_journey_with_incidents(
    all_data, service_stanox, incident_results=None,
    stations_ref_path=r"C:\Users\39342\University of Glasgow\Ji-Eun Byun - MZ-JB\MSci (Research) 2024-25\reference data\stations_ref_with_dft.json",
    incident_color="purple", service_code=None, date_str=None
    ):
    """
    Map train journey by connecting stations based on GEOGRAPHIC PROXIMITY (not chronological order).
    
    1. Map each unique incident (INCIDENT_NUMBER, INCIDENT_START_DATETIME) as a numbered marker
    2. Map service_stanox stations and connect them using minimum spanning tree based on distance
    3. Color-grade station markers by total PFPI_MINUTES
    
    This creates a clean geographic route by connecting nearby stations, avoiding the "spider web" effect.
    
    NEW: Also includes stations from incident_results that experienced delays, ensuring complete route visualization.
    """
    import json
    import folium
    import pandas as pd
    import numpy as np
    from scipy.spatial.distance import pdist, squareform
    from scipy.sparse.csgraph import minimum_spanning_tree

    # Load station reference
    with open(stations_ref_path, "r") as f:
        station_ref = json.load(f)
    
    # Extract additional STANOX codes from incident_results (stations that experienced delays)
    additional_stanox = set()
    if incident_results:
        for res in incident_results:
            if isinstance(res, pd.DataFrame) and 'STANOX' in res.columns:
                # Get unique STANOX values from this result
                stanox_values = res['STANOX'].dropna().unique()
                for stanox in stanox_values:
                    # Normalize to string format
                    stanox_str = str(int(float(stanox))) if isinstance(stanox, (int, float)) else str(stanox)
                    additional_stanox.add(stanox_str)
        
        if additional_stanox:
            print(f"📍 Found {len(additional_stanox)} additional stations from incident results with delays")
    
    # Merge service_stanox with additional stations from incidents
    # Convert service_stanox to normalized string format
    service_stanox_normalized = set()
    for s in service_stanox:
        s_str = str(int(float(s))) if isinstance(s, (int, float)) else str(s)
        service_stanox_normalized.add(s_str)
    
    # Combine both sets
    all_stanox = service_stanox_normalized.union(additional_stanox)
    print(f"📊 Total unique stations: {len(all_stanox)}")
    print(f"   - From service: {len(service_stanox_normalized)}")
    print(f"   - From incidents: {len(additional_stanox)}")
    print(f"   - Combined: {len(all_stanox)}")

    # Build station coordinates for all service_stanox stations
    stanox_coords = []
    stanox_names = {}
    
    # Use the combined all_stanox set instead of just service_stanox
    for s in all_stanox:
        s_str = str(int(float(s))) if isinstance(s, (int, float)) else str(s)
        match = next((item for item in station_ref if str(item.get("stanox", "")) == s_str), None)
        if match and 'latitude' in match and 'longitude' in match:
            try:
                lat = float(match['latitude'])
                lon = float(match['longitude'])
                station_name = match.get('description', s_str)
                stanox_coords.append((s_str, lat, lon))
                stanox_names[s_str] = station_name
            except Exception:
                continue
        else:
            print(f"⚠️ Warning: No coordinates found for STANOX {s_str}")

    if not stanox_coords:
        print("⚠️ No coordinates found for any STANOX in this service.")
        return None

    print(f"📍 Found {len(stanox_coords)} stations with coordinates")
    for i, (stanox, _, _) in enumerate(stanox_coords[:5]):
        station_name = stanox_names.get(stanox, "Unknown")
        print(f"   {i+1}. {station_name} ({stanox})")
    if len(stanox_coords) > 5:
        print(f"   ... and {len(stanox_coords) - 5} more")

    # --- Create map ---
    mid_lat = sum([lat for _, lat, _ in stanox_coords]) / len(stanox_coords)
    mid_lon = sum([lon for _, _, lon in stanox_coords]) / len(stanox_coords)
    m = folium.Map(location=[mid_lat, mid_lon], zoom_start=8, tiles="CartoDB positron")

    # Add title if provided
    if service_code and date_str:
        title_html = f"<div style='position: fixed; top: 10px; left: 50%; transform: translateX(-50%); z-index:9999; font-size:18px; background: white; border:2px solid grey; border-radius:8px; padding: 10px;'><b>Train Service: {service_code}</b><br><b>Date: {date_str}</b></div>"
        m.get_root().html.add_child(folium.Element(title_html))

    # --- Connect stations by PROXIMITY using Minimum Spanning Tree ---
    print(f"🔗 Computing route connections based on geographic proximity...")
    
    if len(stanox_coords) > 1:
        # Extract coordinates for distance calculation
        coords_array = np.array([(lat, lon) for _, lat, lon in stanox_coords])
        
        # Compute pairwise distances (Euclidean on lat/lon - approximation)
        distances = squareform(pdist(coords_array, metric='euclidean'))
        
        # Compute minimum spanning tree to connect all stations with minimum total distance
        mst = minimum_spanning_tree(distances)
        mst_array = mst.toarray()
        
        # Draw edges based on MST
        edge_count = 0
        for i in range(len(stanox_coords)):
            for j in range(i+1, len(stanox_coords)):
                # Check if there's an edge in MST (symmetric, so check both directions)
                if mst_array[i, j] > 0 or mst_array[j, i] > 0:
                    start_stanox, start_lat, start_lon = stanox_coords[i]
                    end_stanox, end_lat, end_lon = stanox_coords[j]
                    
                    start_name = stanox_names.get(start_stanox, start_stanox)
                    end_name = stanox_names.get(end_stanox, end_stanox)
                    
                    folium.PolyLine(
                        [(start_lat, start_lon), (end_lat, end_lon)],
                        color="blue",
                        weight=4,
                        opacity=0.8,
                        popup=f"Connection: {start_name} ↔ {end_name}"
                    ).add_to(m)
                    edge_count += 1
        
        print(f"   ✅ Created {edge_count} route connections based on minimum spanning tree")
    else:
        print(f"   ⚠️ Only 1 station, no connections to draw")

    # Prepare list of DataFrames from incident_results
    dfs = []
    if incident_results:
        for res in incident_results:
            if isinstance(res, pd.DataFrame):
                dfs.append(res)

    delays_df = pd.concat(dfs, ignore_index=True) if dfs else None

    # Aggregate delays for each STANOX
    stanox_delay = {}
    stanox_incidents = {}
    if delays_df is not None and 'STANOX' in delays_df.columns and 'PFPI_MINUTES' in delays_df.columns:
        delays_df['PFPI_MINUTES_num'] = pd.to_numeric(delays_df['PFPI_MINUTES'], errors='coerce').fillna(0)
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
            if 'INCIDENT_NUMBER_str' in group.columns:
                unique_incs = sorted([str(v) for v in pd.unique(group['INCIDENT_NUMBER_str'].dropna())])
            else:
                unique_incs = []
            stanox_incidents[str(stanox)] = unique_incs

    # Build chronological ranking for incidents
    incident_rank = {}
    if delays_df is not None and 'INCIDENT_NUMBER_str' in delays_df.columns and 'INCIDENT_START_DATETIME' in delays_df.columns:
        temp = delays_df[['INCIDENT_NUMBER_str', 'INCIDENT_START_DATETIME']].dropna(subset=['INCIDENT_NUMBER_str', 'INCIDENT_START_DATETIME']).drop_duplicates(subset=['INCIDENT_NUMBER_str']).copy()
        if not temp.empty:
            temp['INCIDENT_START_dt'] = pd.to_datetime(temp['INCIDENT_START_DATETIME'], errors='coerce')
            temp = temp.sort_values('INCIDENT_START_dt')
            temp = temp.reset_index(drop=True)
            temp['incident_rank'] = temp.index + 1
            incident_rank = dict(zip(temp['INCIDENT_NUMBER_str'].astype(str), temp['incident_rank'].astype(int)))

    # Color grading function
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

    # Map station markers with color-grading
    for stanox, lat, lon in stanox_coords:
        delay_val = stanox_delay.get(stanox, 0)
        color = get_color(delay_val)
        station_name = stanox_names.get(stanox, stanox)
        
        inc_list = stanox_incidents.get(stanox, [])
        if inc_list:
            inc_ranks = [str(incident_rank.get(str(i), i)) for i in inc_list]
            if len(inc_ranks) > 10:
                inc_display = ', '.join(inc_ranks[:10]) + f', ... (+{len(inc_ranks)-10} more)'
            else:
                inc_display = ', '.join(inc_ranks)
            incidents_html = f"<br><b>Incidents (by index):</b> {inc_display}"
        else:
            incidents_html = ''

        popup_html = f"<b>{station_name}</b><br>STANOX: {stanox}<br>Total delay: {delay_val:.1f} min{incidents_html}"
        folium.CircleMarker(
            location=(lat, lon),
            radius=6,
            color=color,
            fill=True,
            fill_opacity=0.8,
            popup=folium.Popup(popup_html, max_width=400)
        ).add_to(m)

    # Map incident markers (same as before)
    incident_records = pd.concat(dfs, ignore_index=True) if dfs else None
    
    if incident_records is not None and 'INCIDENT_NUMBER' in incident_records.columns and 'INCIDENT_START_DATETIME' in incident_records.columns and 'SECTION_CODE' in incident_records.columns:
        incident_records['INCIDENT_NUMBER_str'] = incident_records['INCIDENT_NUMBER'].apply(lambda x: (str(int(float(x))) if (pd.notna(x) and float(x).is_integer()) else str(x)))
        incident_unique = incident_records.drop_duplicates(subset=['INCIDENT_NUMBER_str', 'SECTION_CODE']).copy()
        incident_unique['INCIDENT_START_dt'] = pd.to_datetime(incident_unique['INCIDENT_START_DATETIME'], errors='coerce')
        
        incident_durations = {}
        for inc in incident_records['INCIDENT_NUMBER_str'].unique():
            subset = incident_records[incident_records['INCIDENT_NUMBER_str'] == inc]
            start = pd.to_datetime(subset['INCIDENT_START_DATETIME'].min(), format='%Y-%m-%d %H:%M:%S', errors='coerce')
            end = pd.to_datetime(subset['EVENT_DATETIME'].max(), format='%d-%b-%Y %H:%M', errors='coerce')
            duration = end - start
            duration = max(duration, pd.Timedelta(0))
            incident_durations[inc] = duration
        
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

        print(f"📍 Creating incident markers for {len(section_map)} sections...")
        markers_created = 0
        for section_code, entries in section_map.items():
            entries_sorted = sorted(entries, key=lambda e: (e['rank'] if isinstance(e['rank'], int) else 999999))
            ranks = [str(e['rank']) if e['rank'] != '' else e['inc_id'] for e in entries_sorted]
            ranks_display = ','.join(ranks)
            
            popup_lines = []
            for e in entries_sorted:
                reason_text = e['inc_reason'] if e.get('inc_reason') else 'N/A'
                dur = incident_durations.get(e['inc_id'], 'N/A')
                popup_lines.append(f"Incident: {e['inc_num']} — {e['inc_time']} — Reason: {reason_text} — Duration: {dur}")
            popup_html = '<br>'.join(popup_lines)

            if ':' in section_code:
                stanox_parts = section_code.split(':')
                if len(stanox_parts) == 2:
                    stanox1, stanox2 = stanox_parts[0].strip(), stanox_parts[1].strip()
                    
                    match1 = next((item for item in station_ref if str(item.get("stanox", "")) == stanox1), None)
                    match2 = next((item for item in station_ref if str(item.get("stanox", "")) == stanox2), None)
                    
                    if match1 and match2 and 'latitude' in match1 and 'longitude' in match1 and 'latitude' in match2 and 'longitude' in match2:
                        lat1, lon1 = float(match1['latitude']), float(match1['longitude'])
                        lat2, lon2 = float(match2['latitude']), float(match2['longitude'])
                        
                        station1_name = match1.get('description', stanox1)
                        station2_name = match2.get('description', stanox2)
                        
                        folium.PolyLine(
                            [(lat1, lon1), (lat2, lon2)],
                            color=incident_color,
                            weight=6,
                            opacity=0.9,
                            popup=f"Incident Section: {station1_name} ↔ {station2_name}"
                        ).add_to(m)
                        
                        mid_lat = (lat1 + lat2) / 2
                        mid_lon = (lon1 + lon2) / 2
                        
                        section_popup = f"<b>Track Section Incident</b><br>Between: {station1_name} ↔ {station2_name}<br>Section: {section_code}<br><br>{popup_html}"
                        
                        size_px = max(28, min(80, 12 * len(ranks_display)))
                        number_html = f"<div style='background:{incident_color};color:#fff;border-radius:50%;min-width:{size_px}px;height:{size_px}px;display:inline-flex;align-items:center;justify-content:center;font-weight:bold;border:2px solid #ffffff;padding:4px'>{ranks_display}</div>"
                        folium.Marker(
                            location=(mid_lat, mid_lon),
                            icon=folium.DivIcon(html=number_html),
                            popup=folium.Popup(section_popup, max_width=450)
                        ).add_to(m)
                        markers_created += 1
            else:
                match = next((item for item in station_ref if str(item.get("stanox", "")) == section_code), None)
                if match and 'latitude' in match and 'longitude' in match:
                    lat = float(match['latitude']) + 0.0005
                    lon = float(match['longitude']) + 0.0005
                    station_name = match.get('description', section_code)
                    
                    station_popup = f"<b>Station Incident</b><br>{station_name}<br>STANOX: {section_code}<br><br>{popup_html}"
                    
                    size_px = max(28, min(80, 12 * len(ranks_display)))
                    number_html = f"<div style='background:{incident_color};color:#fff;border-radius:50%;min-width:{size_px}px;height:{size_px}px;display:inline-flex;align-items:center;justify-content:center;font-weight:bold;border:2px solid #ffffff;padding:4px'>{ranks_display}</div>"
                    folium.Marker(
                        location=(lat, lon),
                        icon=folium.DivIcon(html=number_html),
                        popup=folium.Popup(station_popup, max_width=450)
                    ).add_to(m)
                    markers_created += 1
        
        print(f"✅ Created {markers_created} incident markers on map")

    # Add legend
    legend_html = '''
     <div style="position: fixed; bottom: 50px; left: 50px; width: 300px; height: auto; max-height: 400px; z-index:9999; font-size:13px; background: white; border:2px solid grey; border-radius:8px; padding: 12px; box-shadow: 0 2px 6px rgba(0,0,0,0.3);">
     <b style="font-size: 15px;">Delay Intensity Key</b><br>
     <div style="margin: 8px 0;">
     <i class="fa fa-circle" style="color:blue"></i> 0 min (No delay)<br>
     <i class="fa fa-circle" style="color:#32CD32"></i> 1-5 min (Minor)<br>
     <i class="fa fa-circle" style="color:#FFD700"></i> 6-15 min (Moderate)<br>
     <i class="fa fa-circle" style="color:#FF8C00"></i> 16-30 min (Significant)<br>
     <i class="fa fa-circle" style="color:#FF0000"></i> 31-60 min (Major)<br>
     <i class="fa fa-circle" style="color:#8B0000"></i> 61-120 min (Severe)<br>
     <i class="fa fa-circle" style="color:#8A2BE2"></i> 120+ min (Critical)
     </div>
     <b style="font-size: 14px;">Route Connections:</b><br>
     <div style="margin-top: 6px; line-height: 1.4;">
     Blue lines connect stations by <b>geographic proximity</b> using minimum spanning tree algorithm.<br><br>
     Purple numbered circles show incidents (1 = earliest).<br>
     Track section incidents shown as purple lines.
     </div>
     </div>
     '''
    m.get_root().html.add_child(folium.Element(legend_html))

    folium.LayerControl().add_to(m)
    print("✅ Map created with proximity-based connections!")
    return m

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
        station_name = stanox_names.get(stanox, stanox)
        
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

        popup_html = f"<b>{station_name}</b><br>STANOX: {stanox}<br>Total delay: {delay_val:.1f} min{incidents_html}"
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
    
    print(f"🔍 Debug: Checking for incident markers...")
    print(f"   - incident_records is None: {incident_records is None}")
    if incident_records is not None:
        print(f"   - incident_records shape: {incident_records.shape}")
        print(f"   - incident_records columns: {incident_records.columns.tolist()}")
        print(f"   - Has INCIDENT_NUMBER: {'INCIDENT_NUMBER' in incident_records.columns}")
        print(f"   - Has INCIDENT_START_DATETIME: {'INCIDENT_START_DATETIME' in incident_records.columns}")
        print(f"   - Has SECTION_CODE: {'SECTION_CODE' in incident_records.columns}")
        if 'INCIDENT_NUMBER' in incident_records.columns:
            unique_incidents = incident_records['INCIDENT_NUMBER'].dropna().unique()
            print(f"   - Unique incidents found: {unique_incidents}")
    
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
        print(f"📍 Creating incident markers for {len(section_map)} sections...")
        markers_created = 0
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

            # Check if section_code contains a colon (track section between two stations)
            if ':' in section_code:
                # Parse the two STANOX codes
                stanox_parts = section_code.split(':')
                if len(stanox_parts) == 2:
                    stanox1, stanox2 = stanox_parts[0].strip(), stanox_parts[1].strip()
                    
                    # Find coordinates for both stations
                    match1 = next((item for item in station_ref if str(item.get("stanox", "")) == stanox1), None)
                    match2 = next((item for item in station_ref if str(item.get("stanox", "")) == stanox2), None)
                    
                    if match1 and match2 and 'latitude' in match1 and 'longitude' in match1 and 'latitude' in match2 and 'longitude' in match2:
                        lat1, lon1 = float(match1['latitude']), float(match1['longitude'])
                        lat2, lon2 = float(match2['latitude']), float(match2['longitude'])
                        
                        station1_name = match1.get('description', stanox1)
                        station2_name = match2.get('description', stanox2)
                        
                        # Draw purple line between the two stations
                        folium.PolyLine(
                            [(lat1, lon1), (lat2, lon2)],
                            color=incident_color,
                            weight=6,
                            opacity=0.9,
                            popup=f"Incident Section: {station1_name} ↔ {station2_name}"
                        ).add_to(m)
                        
                        # Calculate midpoint for marker placement
                        mid_lat = (lat1 + lat2) / 2
                        mid_lon = (lon1 + lon2) / 2
                        
                        # Adjust popup to show section information
                        section_popup = f"<b>Track Section Incident</b><br>Between: {station1_name} ↔ {station2_name}<br>Section: {section_code}<br><br>{popup_html}"
                        
                        # Create numbered marker at midpoint
                        size_px = max(28, min(80, 12 * len(ranks_display)))
                        number_html = f"<div style='background:{incident_color};color:#fff;border-radius:50%;min-width:{size_px}px;height:{size_px}px;display:inline-flex;align-items:center;justify-content:center;font-weight:bold;border:2px solid #ffffff;padding:4px'>{ranks_display}</div>"
                        folium.Marker(
                            location=(mid_lat, mid_lon),
                            icon=folium.DivIcon(html=number_html),
                            popup=folium.Popup(section_popup, max_width=450)
                        ).add_to(m)
                        markers_created += 1
                        if markers_created <= 3:
                            print(f"   ✅ Section marker {markers_created}: {station1_name} ↔ {station2_name}, Rank(s) {ranks_display}")
                    else:
                        if not match1:
                            print(f"   ⚠️ No coordinates found for station {stanox1} in section {section_code}")
                        if not match2:
                            print(f"   ⚠️ No coordinates found for station {stanox2} in section {section_code}")
                else:
                    print(f"   ⚠️ Invalid section format: {section_code}")
            else:
                # Single station section (original logic)
                match = next((item for item in station_ref if str(item.get("stanox", "")) == section_code), None)
                if match and 'latitude' in match and 'longitude' in match:
                    lat = float(match['latitude']) + 0.0005  # Small offset to avoid overlap with station markers
                    lon = float(match['longitude']) + 0.0005  # Small offset to avoid overlap with station markers
                    station_name = match.get('description', section_code)
                    
                    # Adjust popup to show station name
                    station_popup = f"<b>Station Incident</b><br>{station_name}<br>STANOX: {section_code}<br><br>{popup_html}"
                    
                    # adjust DivIcon size based on text length
                    size_px = max(28, min(80, 12 * len(ranks_display)))
                    number_html = f"<div style='background:{incident_color};color:#fff;border-radius:50%;min-width:{size_px}px;height:{size_px}px;display:inline-flex;align-items:center;justify-content:center;font-weight:bold;border:2px solid #ffffff;padding:4px'>{ranks_display}</div>"
                    folium.Marker(
                        location=(lat, lon),
                        icon=folium.DivIcon(html=number_html),
                        popup=folium.Popup(station_popup, max_width=450)
                    ).add_to(m)
                    markers_created += 1
                    if markers_created <= 3:
                        print(f"   ✅ Station marker {markers_created}: {station_name}, Rank(s) {ranks_display}")
                else:
                    print(f"   ⚠️ No coordinates found for section {section_code}")
        
        print(f"✅ Created {markers_created} incident markers on map")
    else:
        print("⚠️ No incident data available for markers (missing columns or empty data)")

    # --- Add legend/key for color grading matching incident_view_heatmap_html ---
    legend_html = '''
     <div style="position: fixed; bottom: 50px; left: 50px; width: 300px; height: auto; max-height: 400px; z-index:9999; font-size:13px; background: white; border:2px solid grey; border-radius:8px; padding: 12px; box-shadow: 0 2px 6px rgba(0,0,0,0.3);">
     <b style="font-size: 15px;">Delay Intensity Key</b><br>
     <div style="margin: 8px 0;">
     <i class="fa fa-circle" style="color:blue"></i> 0 min (No delay)<br>
     <i class="fa fa-circle" style="color:#32CD32"></i> 1-5 min (Minor)<br>
     <i class="fa fa-circle" style="color:#FFD700"></i> 6-15 min (Moderate)<br>
     <i class="fa fa-circle" style="color:#FF8C00"></i> 16-30 min (Significant)<br>
     <i class="fa fa-circle" style="color:#FF0000"></i> 31-60 min (Major)<br>
     <i class="fa fa-circle" style="color:#8B0000"></i> 61-120 min (Severe)<br>
     <i class="fa fa-circle" style="color:#8A2BE2"></i> 120+ min (Critical)
     </div>
     <b style="font-size: 14px;">Incident Markers:</b><br>
     <div style="margin-top: 6px; line-height: 1.4;">
     Purple numbered circles show incidents in chronological order (1 = earliest).<br>
     Track section incidents shown as <span style="color:purple; font-weight:bold;">purple lines</span> with marker at midpoint.<br>
     Multiple incidents: comma-separated ranks.
     </div>
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
    
    NEW: Also includes stations from all_data that experienced delays for this service code.
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
    
    # Extract additional STANOX codes from all_data that have delays for this service code
    service_data = all_data[all_data['TRAIN_SERVICE_CODE'].astype(str) == str(service_code)].copy()
    additional_stanox = set()
    
    if not service_data.empty and 'STANOX' in service_data.columns and 'PFPI_MINUTES' in service_data.columns:
        # Get stations with delays (PFPI_MINUTES > 0)
        service_data['PFPI_MINUTES_num'] = pd.to_numeric(service_data['PFPI_MINUTES'], errors='coerce')
        delayed_stations = service_data[service_data['PFPI_MINUTES_num'] > 0]['STANOX'].dropna().unique()
        
        for stanox in delayed_stations:
            stanox_str = str(int(float(stanox))) if isinstance(stanox, (int, float)) else str(stanox)
            additional_stanox.add(stanox_str)
        
        if additional_stanox:
            print(f"📍 Found {len(additional_stanox)} additional stations with delays for service {service_code}")
    
    # Merge service_stanox with additional stations
    service_stanox_normalized = set()
    for s in service_stanox:
        s_str = str(int(float(s))) if isinstance(s, (int, float)) else str(s)
        service_stanox_normalized.add(s_str)
    
    all_stanox = service_stanox_normalized.union(additional_stanox)
    print(f"📊 Total unique stations for analysis: {len(all_stanox)}")
    print(f"   - From service route: {len(service_stanox_normalized)}")
    print(f"   - With delays: {len(additional_stanox)}")
    print(f"   - Combined: {len(all_stanox)}")

    results = []

    for s in all_stanox:
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
    
    NEW: Also includes stations from all_data that experienced delays for this service code.
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
    
    # Extract additional STANOX codes from all_data that have delays for this service code
    service_data = all_data[all_data['TRAIN_SERVICE_CODE'].astype(str) == str(service_code)].copy()
    additional_stanox = set()
    
    if not service_data.empty and 'STANOX' in service_data.columns and 'PFPI_MINUTES' in service_data.columns:
        # Get stations with delays (PFPI_MINUTES > 0)
        service_data['PFPI_MINUTES_num'] = pd.to_numeric(service_data['PFPI_MINUTES'], errors='coerce')
        delayed_stations = service_data[service_data['PFPI_MINUTES_num'] > 0]['STANOX'].dropna().unique()
        
        for stanox in delayed_stations:
            stanox_str = str(int(float(stanox))) if isinstance(stanox, (int, float)) else str(stanox)
            additional_stanox.add(stanox_str)
        
        if additional_stanox:
            print(f"📍 Found {len(additional_stanox)} additional stations with delays for plotting")
    
    # Merge service_stanox with additional stations
    service_stanox_normalized = set()
    for s in service_stanox:
        s_str = str(int(float(s))) if isinstance(s, (int, float)) else str(s)
        service_stanox_normalized.add(s_str)
    
    all_stanox = service_stanox_normalized.union(additional_stanox)
    print(f"📊 Plotting delay distributions for {len(all_stanox)} stations")

    station_labels = []
    delay_data = []

    for s in all_stanox:
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


# functions for station view

def station_view_yearly(station_id, interval_minutes=30):
    """
    Station analysis for yearly data across all incidents - simplified output.
    Analyzes all days of the week for a station and separates incident vs normal operations.
    """
    
    # Load data from all day files
    processed_base = '../processed_data'
    station_folder = os.path.join(processed_base, station_id)
    
    if not os.path.exists(station_folder):
        print(f"Station folder not found: {station_folder}")
        return None, None
    
    # Define day files to load
    day_files = ['MO.parquet', 'TU.parquet', 'WE.parquet', 'TH.parquet', 'FR.parquet', 'SA.parquet', 'SU.parquet']
    
    all_station_data = []
    
    for day_file in day_files:
        file_path = os.path.join(station_folder, day_file)
        if os.path.exists(file_path):
            try:
                day_data = pd.read_parquet(file_path, engine='fastparquet')
                day_data['day_of_week'] = day_file.replace('.parquet', '')
                all_station_data.append(day_data)
                print(f"Loaded {len(day_data)} records from {day_file}")
            except Exception as e:
                print(f"Error loading {day_file}: {e}")
        else:
            print(f"File not found: {day_file}")
    
    if not all_station_data:
        print("No data files found for this station")
        return None, None
    
    # Combine all data
    combined_data = pd.concat(all_station_data, ignore_index=True)
    print(f"Total combined records: {len(combined_data)}")
    
    # Filter for trains with planned calls
    train_mask = combined_data['PLANNED_CALLS'].notna()
    all_train_data = combined_data[train_mask].copy()
    
    # Maximum delay deduplication
    if len(all_train_data) > 0:
        all_train_data['delay_numeric'] = pd.to_numeric(all_train_data['PFPI_MINUTES'], errors='coerce').fillna(0)
        all_train_data['dedup_priority'] = all_train_data['delay_numeric'] * 1000
        
        if 'ACTUAL_CALLS' in all_train_data.columns:
            all_train_data['dedup_priority'] += all_train_data['ACTUAL_CALLS'].notna().astype(int) * 100
        
        basic_dedup_cols = ['TRAIN_SERVICE_CODE', 'PLANNED_CALLS', 'day_of_week']
        basic_available = [col for col in basic_dedup_cols if col in all_train_data.columns]
        
        if len(basic_available) >= 2:
            all_train_data = all_train_data.sort_values(['delay_numeric', 'dedup_priority'], ascending=[False, False])
            all_train_data = all_train_data.drop_duplicates(subset=basic_available, keep='first')
            all_train_data = all_train_data.drop(['delay_numeric', 'dedup_priority'], axis=1)
    
    if len(all_train_data) == 0:
        return None, None
    
    # Separate incident and normal operations
    # Assume trains with incident codes are incident-related
    incident_mask = all_train_data['INCIDENT_NUMBER'].notna()
    incident_data = all_train_data[incident_mask].copy()
    normal_data = all_train_data[~incident_mask].copy()
    
    print(f"Incident-related records: {len(incident_data)}")
    print(f"Normal operations records: {len(normal_data)}")
    
    def process_operations_data(data, operation_type):
        """Process data for either incident or normal operations"""
        if len(data) == 0:
            return pd.DataFrame()
        
        # Process times - using a reference date for time parsing
        reference_date = datetime(2024, 1, 1)  # Use a standard reference date
        
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
        data['planned_dt'] = data['PLANNED_CALLS'].apply(
            lambda x: parse_time_simple(x, reference_date))
        data['original_actual_dt'] = data['ACTUAL_CALLS'].apply(
            lambda x: parse_time_simple(x, reference_date))
        data['delay_minutes'] = pd.to_numeric(data['PFPI_MINUTES'], errors='coerce').fillna(0)
        
        # Create corrected actual times
        corrected_actual_times = []
        for _, row in data.iterrows():
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
        
        data['effective_time'] = corrected_actual_times
        valid_data = data[data['effective_time'].notna()].copy()
        
        if len(valid_data) == 0:
            return pd.DataFrame()
        
        # Group by time intervals (using hour of day for grouping)
        valid_data['hour_of_day'] = valid_data['effective_time'].dt.hour
        valid_data['interval_group'] = (valid_data['hour_of_day'] * 60 + valid_data['effective_time'].dt.minute) // interval_minutes
        
        intervals = []
        
        for interval_group in valid_data['interval_group'].unique():
            interval_trains = valid_data[valid_data['interval_group'] == interval_group]
            
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
                    
                    # Calculate time period label
                    start_minute = interval_group * interval_minutes
                    end_minute = start_minute + interval_minutes
                    start_hour = start_minute // 60
                    start_min = start_minute % 60
                    end_hour = end_minute // 60
                    end_min = end_minute % 60
                    
                    intervals.append({
                        'time_period': f"{start_hour:02d}:{start_min:02d}-{end_hour:02d}:{end_min:02d}",
                        'ontime_arrival_count': ontime_arrivals,
                        'delayed_arrival_count': delayed_arrivals,
                        'cancellation_count': total_cancellations,
                        'delay_minutes': delayed_minutes,
                        'operation_type': operation_type
                    })
        
        return pd.DataFrame(intervals)
    
    # Process both incident and normal operations
    incident_summary = process_operations_data(incident_data, 'incident')
    normal_summary = process_operations_data(normal_data, 'normal')
    
    return incident_summary, normal_summary

print("station_view_yearly function ready!")

def plot_arrival_hour_distributions_violin(station_id, all_data):
    """
    Create KDE plots comparing planned vs actual arrival times for trains affected by incidents.
    
    - Uses only trains that were affected by incidents (INCIDENT_NUMBER not null)
    - Normal: Shows planned arrival times for these trains
    - Incident: Shows actual arrival times (planned + delay) for the same trains
    - Same number of trains in both distributions, showing the time shift due to incidents
    - Only includes trains that arrived (EVENT_TYPE != 'C')
    - Preserves granular minute-level data
    - Shows density on y-axis with time on x-axis
    
    Parameters:
    -----------
    station_id : str
        The station STANOX code
    all_data : pd.DataFrame
        The complete dataset containing all train records
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    import pandas as pd
    import numpy as np

    # Set style
    plt.style.use('default')
    sns.set_palette("husl")

    print(f"🚀 CREATING ARRIVAL TIME SHIFT ANALYSIS FOR INCIDENT TRAINS AT STATION {station_id}")
    print("=" * 70)

    # Filter data for the specific station from provided all_data
    data = all_data[all_data['STANOX'] == str(station_id)].copy()
    if len(data) == 0:
        print(f"No data found for station {station_id}")
        return None
    
    print(f"Loaded {len(data)} total records for station {station_id}")

    # Filter for trains affected by incidents
    incident_data = data[data['INCIDENT_NUMBER'].notna()].copy()
    if len(incident_data) == 0:
        print("No incident data found for this station.")
        return None
    
    # Filter for trains that arrived (exclude cancellations)
    arrived_incident_data = incident_data[incident_data['EVENT_TYPE'] != 'C'].copy()
    if len(arrived_incident_data) == 0:
        print("No arrived incident trains found.")
        return None
    
    print(f"Analyzing {len(arrived_incident_data)} incident trains that arrived")

    # Helper function to convert HHMM to minutes past midnight
    def hhmm_to_minutes(hhmm):
        if pd.isna(hhmm):
            return np.nan
        try:
            hhmm_str = str(int(hhmm)).zfill(4)
            hour = int(hhmm_str[:2])
            minute = int(hhmm_str[2:])
            return hour * 60 + minute
        except:
            return np.nan

    # Calculate delays
    arrived_incident_data['delay_minutes'] = pd.to_numeric(arrived_incident_data['PFPI_MINUTES'], errors='coerce').fillna(0)
    
    # Calculate planned arrival times in minutes
    arrived_incident_data['planned_minutes'] = arrived_incident_data['PLANNED_CALLS'].apply(hhmm_to_minutes)
    
    # Calculate actual arrival times in minutes (planned + delay)
    arrived_incident_data['actual_minutes'] = arrived_incident_data['planned_minutes'] + arrived_incident_data['delay_minutes']
    arrived_incident_data['actual_minutes'] = arrived_incident_data['actual_minutes'] % (24 * 60)  # Keep within 24 hours
    
    # Drop rows with missing times
    valid_data = arrived_incident_data.dropna(subset=['planned_minutes', 'actual_minutes']).copy()
    
    if len(valid_data) == 0:
        print("No valid time data after cleaning.")
        return None
    
    print(f"Valid data for {len(valid_data)} trains")

    # Prepare data for plotting
    plot_data = []
    
    # Normal: planned arrival times
    planned_times = valid_data['planned_minutes'].values
    plot_data.extend([{'arrival_minutes': m, 'condition': 'Planned (Normal)'} for m in planned_times])
    
    # Incident: actual arrival times
    actual_times = valid_data['actual_minutes'].values
    plot_data.extend([{'arrival_minutes': m, 'condition': 'Actual (Incident)'} for m in actual_times])
    
    plot_df = pd.DataFrame(plot_data)
    
    # Format times as HH:MM
    def minutes_to_hhmm(minutes):
        hours = int(minutes // 60)
        mins = int(minutes % 60)
        return f"{hours:02d}:{mins:02d}"

    # Create figure with subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 12), gridspec_kw={'height_ratios': [2, 1]})
    
    # Main KDE plot (top subplot)
    sns.kdeplot(data=[d['arrival_minutes'] for d in plot_data if d['condition'] == 'Planned (Normal)'], 
                ax=ax1, fill=True, alpha=0.5, color='lightgreen', 
                label=f'Planned (n={len(valid_data)})', bw_adjust=0.6)
    
    sns.kdeplot(data=[d['arrival_minutes'] for d in plot_data if d['condition'] == 'Actual (Incident)'], 
                ax=ax1, fill=True, alpha=0.5, color='lightcoral', 
                label=f'Actual (n={len(valid_data)})', bw_adjust=0.6)

    ax1.set_title(f'Arrival Time Distribution Shift Due to Incidents\nStation {station_id}',
                 fontsize=20, fontweight='bold')
    ax1.set_xlabel('')  # Remove x-label for top plot
    ax1.set_ylabel('Density', fontsize=20)
    ax1.set_xlim(-10, 24*60 + 10)  # Full 24 hours in minutes
    ax1.grid(True, alpha=0.3, axis='both')
    
    # Add hour labels on x-axis
    hour_ticks = np.arange(0, 24*60 + 1, 60)  # Every hour
    hour_labels = [f"{h//60:02d}:00" for h in hour_ticks]
    ax1.set_xticks(hour_ticks)
    ax1.set_xticklabels(hour_labels)
    
    # Add legend
    ax1.legend(loc='upper right', fontsize=15)
    
    # Delay distribution plot (bottom subplot) - density of delays
    delay_minutes = valid_data['delay_minutes'].values
    ax2.hist(delay_minutes, bins=30, alpha=0.7, color='skyblue', edgecolor='black', linewidth=0.5, density=True)
    ax2.set_title('Delay Distribution Density', fontsize=16, fontweight='bold')
    ax2.set_xlabel('Delay (minutes)', fontsize=16)
    ax2.set_ylabel('Density', fontsize=16)
    ax2.grid(True, alpha=0.3, axis='both')
    
    # Add vertical line at mean delay
    mean_delay = np.mean(delay_minutes)
    ax2.axvline(mean_delay, color='red', linestyle='--', linewidth=2, 
                label=f'Mean: {mean_delay:.1f} min')
    ax2.legend(loc='upper right', fontsize=12)

    plt.tight_layout()
    
    # Print summary statistics
    print(f"\n{'='*80}")
    print(f"📊 STATION {station_id} INCIDENT TRAIN ARRIVAL TIME SHIFT ANALYSIS")
    print(f"{'='*80}")
    print(f"Analyzing {len(valid_data)} trains affected by incidents")
    print(f"Std planned times: {valid_data['planned_minutes'].std():.1f} minutes")
    print(f"Std actual times: {valid_data['actual_minutes'].std():.1f} minutes")
    print(f"Min planned time: {valid_data['planned_minutes'].min():.0f} minutes ({minutes_to_hhmm(valid_data['planned_minutes'].min())})")
    print(f"Max planned time: {valid_data['planned_minutes'].max():.0f} minutes ({minutes_to_hhmm(valid_data['planned_minutes'].max())})")
    print(f"Min actual time: {valid_data['actual_minutes'].min():.0f} minutes ({minutes_to_hhmm(valid_data['actual_minutes'].min())})")
    print(f"Max actual time: {valid_data['actual_minutes'].max():.0f} minutes ({minutes_to_hhmm(valid_data['actual_minutes'].max())})")
    
    # Delay statistics
    delay_stats = valid_data['delay_minutes'].describe()
    print(f"\n🚨 DELAY STATISTICS:")
    print(f"Mean delay: {delay_stats['mean']:.1f} minutes")
    print(f"Median delay: {delay_stats['50%']:.1f} minutes")
    print(f"Max delay: {delay_stats['max']:.1f} minutes")
    print(f"Std delay: {delay_stats['std']:.1f} minutes")
    print(f"Trains with delay > 0: {len(valid_data[valid_data['delay_minutes'] > 0])} ({100*len(valid_data[valid_data['delay_minutes'] > 0])/len(valid_data):.1f}%)")
    print(f"Trains on time: {len(valid_data[valid_data['delay_minutes'] == 0])} ({100*len(valid_data[valid_data['delay_minutes'] == 0])/len(valid_data):.1f}%)")
    
    plt.show()
    return plot_df

print("plot_arrival_hour_distributions_violin function ready!")


def plot_variable_relationships(station_id, all_data, time_window_minutes=60, num_platforms=12, figsize=(14, 10), max_delay_percentile=98):
    """
    Create scatter plots showing relationship between flow and delays - ONE POINT PER HOUR.
    Separates analysis into WEEKDAYS (Mon-Fri) and WEEKENDS (Sat-Sun).

    Flow vs Mean Delay (minutes) - Y: flow, X: mean delay

    IMPORTANT: 
    - Flow is calculated using ALL trains (incident and non-incident) per hour.
    - Mean delay is calculated ONLY from delayed trains (delay > 0) per hour.
    - Each data point = ONE SPECIFIC HOUR on one specific day across the full year.
    
    Expected data points: 
    - Weekdays: ~260 days × 24 hours = ~6,240 points (minus hours with no delays)
    - Weekends: ~104 days × 24 hours = ~2,496 points (minus hours with no delays)
    
    max_delay_percentile: Trim delays above this percentile to reduce outlier influence (default 98%).
    
    Parameters:
    -----------
    station_id : str
        The station STANOX code
    all_data : pd.DataFrame
        The complete dataset containing all train records
    time_window_minutes : int, optional
        Time window in minutes (default: 60)
    num_platforms : int, optional
        Number of platforms (default: 12)
    figsize : tuple, optional
        Figure size (default: (14, 10))
    max_delay_percentile : int, optional
        Percentile to trim extreme delays (default: 98)
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    import pandas as pd
    import numpy as np

    plt.style.use('default')
    sns.set_palette("husl")

    print(f"🚀 CREATING TEMPORAL FLOW ANALYSIS FOR STATION {station_id}")
    print("=" * 70)

    # Filter data for the specific station from provided all_data
    data = all_data[all_data['STANOX'] == str(station_id)].copy()
    if len(data) == 0:
        print(f"No data found for station {station_id}")
        return None
    
    print(f"Loaded {len(data)} total records for station {station_id}")

    # Filter for ALL trains that arrived (exclude cancellations) - not just incident trains
    all_arrived_data = data[data['EVENT_TYPE'] != 'C'].copy()
    if len(all_arrived_data) == 0:
        print("No arrived trains found.")
        return None
    
    print(f"Using {len(all_arrived_data)} arrived trains (both incident and non-incident) for flow calculation")

    # Calculate delays for all trains
    all_arrived_data['delay_minutes'] = pd.to_numeric(all_arrived_data['PFPI_MINUTES'], errors='coerce').fillna(0)
    
    # Mark which trains have delays > 0 (for delay statistics)
    all_arrived_data['has_delay'] = all_arrived_data['delay_minutes'] > 0

    # Create proper datetime timestamps using EVENT_DATETIME column
    print("Creating datetime timestamps from EVENT_DATETIME...")
    
    def parse_event_datetime(event_dt_str):
        """Parse EVENT_DATETIME string to extract date"""
        if pd.isna(event_dt_str):
            return None
        try:
            # EVENT_DATETIME format: 'DD-MMM-YYYY HH:MM'
            dt = pd.to_datetime(event_dt_str, format='%d-%b-%Y %H:%M', errors='coerce')
            return dt.date() if pd.notna(dt) else None
        except:
            return None
    
    # Extract dates from EVENT_DATETIME for incident trains
    all_arrived_data['event_date'] = all_arrived_data['EVENT_DATETIME'].apply(parse_event_datetime)
    
    # Get mapping of day code to dates from incident trains
    day_to_weekday = {'MO': 0, 'TU': 1, 'WE': 2, 'TH': 3, 'FR': 4, 'SA': 5, 'SU': 6}
    
    # Build a mapping of DAY code to all observed dates
    day_date_mapping = {}
    for day_code in day_to_weekday.keys():
        day_data = all_arrived_data[all_arrived_data['DAY'] == day_code]
        observed_dates = day_data['event_date'].dropna().unique()
        if len(observed_dates) > 0:
            day_date_mapping[day_code] = sorted(observed_dates)
    
    print(f"Found date mappings for {len(day_date_mapping)} day codes")
    for day_code, dates in day_date_mapping.items():
        if len(dates) > 0:
            print(f"  {day_code}: {len(dates)} unique dates from {dates[0]} to {dates[-1]}")
    
    # Create row index for better distribution
    all_arrived_data['row_idx'] = range(len(all_arrived_data))
    
    def create_datetime_with_event_dates(row):
        """Create datetime using EVENT_DATETIME dates or inferred dates"""
        try:
            day_code = row['DAY']
            time_val = row['ACTUAL_CALLS'] if pd.notna(row['ACTUAL_CALLS']) else row['PLANNED_CALLS']
            
            if pd.isna(time_val) or day_code not in day_to_weekday:
                return None
            
            # Parse time
            time_str = str(int(time_val)).zfill(4)
            hour = int(time_str[:2])
            minute = int(time_str[2:])
            
            # Get date - prioritize EVENT_DATETIME if available
            if pd.notna(row['event_date']):
                date_obj = row['event_date']
            else:
                # For non-incident trains, distribute across ALL observed dates for this day
                # Use row index combined with train ID for better distribution
                if day_code in day_date_mapping and len(day_date_mapping[day_code]) > 0:
                    # Use row index to get different dates for same train on different occurrences
                    date_idx = (hash(str(row['TRAIN_SERVICE_CODE'])) + row['row_idx']) % len(day_date_mapping[day_code])
                    date_obj = day_date_mapping[day_code][date_idx]
                else:
                    return None
            
            # Create datetime with time
            dt = pd.Timestamp(year=date_obj.year, month=date_obj.month, 
                            day=date_obj.day, hour=hour, minute=minute)
            return dt
        except:
            return None
    
    all_arrived_data['datetime'] = all_arrived_data.apply(create_datetime_with_event_dates, axis=1)
    
    # Drop rows with invalid datetimes
    valid_data = all_arrived_data.dropna(subset=['datetime']).copy()
    
    if len(valid_data) == 0:
        print("No valid datetime data.")
        return None
    
    print(f"Created {len(valid_data)} valid timestamps")
    
    # Set datetime as index for time-series operations
    valid_data = valid_data.set_index('datetime').sort_index()
    
    # Add day type
    valid_data['day_type'] = valid_data.index.dayofweek.map(
        lambda x: 'weekday' if x < 5 else 'weekend'
    )
    valid_data['hour_of_day'] = valid_data.index.hour
    valid_data['weekday_name'] = valid_data.index.day_name()

    # Process separately for weekdays and weekends
    def process_day_type(data_subset, day_type_name):
        """Process data for either weekdays or weekends - one data point per hour"""
        if len(data_subset) == 0:
            return pd.DataFrame(), pd.DataFrame()
        
        # Count UNIQUE trains per hour using TRAIN_SERVICE_CODE (ALL trains for flow)
        hourly_flow = data_subset.groupby(pd.Grouper(freq='h'))['TRAIN_SERVICE_CODE'].nunique()
        
        # Filter only delayed trains (delay > 0) for delay statistics
        delayed_trains = data_subset[data_subset['has_delay']].copy()
        
        # Calculate mean delay per hour ONLY from trains with delays > 0
        if len(delayed_trains) > 0:
            hourly_mean_delay = delayed_trains.groupby(pd.Grouper(freq='h'))['delay_minutes'].mean()
        else:
            # No delayed trains - create empty series
            hourly_mean_delay = pd.Series(dtype=float)
        
        # Combine flow and delay - one row per hour
        hourly_stats = pd.DataFrame({
            'flow': hourly_flow,
            'mean_delay': hourly_mean_delay
        })
        
        # Keep all hours with flow (trains operated)
        # For hours with no delays, mean_delay will be NaN - fill with 0 for visualization
        hourly_stats = hourly_stats[hourly_stats['flow'].notna()].copy()
        hourly_stats['mean_delay'] = hourly_stats['mean_delay'].fillna(0)
        
        if len(hourly_stats) == 0:
            return pd.DataFrame(), pd.DataFrame()
        
        # Add hour of day for statistics
        hourly_stats['hour_of_day'] = hourly_stats.index.hour
        hourly_stats['day_type'] = day_type_name
        hourly_stats['datetime'] = hourly_stats.index
        
        # Trim outliers using mean delay
        if max_delay_percentile < 100 and len(hourly_stats) > 0:
            delay_threshold = hourly_stats['mean_delay'].quantile(max_delay_percentile / 100)
            hourly_stats = hourly_stats[hourly_stats['mean_delay'] <= delay_threshold]
        
        # Calculate summary statistics by hour of day for the table
        hour_summary = hourly_stats.groupby('hour_of_day').agg({
            'flow': ['mean', 'std', 'median', 'min', 'max'],
            'mean_delay': ['mean', 'max', 'std']
        }).round(2)
        
        # Flatten column names
        hour_summary.columns = ['flow_mean', 'flow_std', 'flow_median', 'flow_min', 'flow_max',
                                'delay_mean', 'delay_max', 'delay_std']
        
        return hourly_stats, hour_summary

    # Process weekdays
    weekday_data = valid_data[valid_data['day_type'] == 'weekday']
    weekday_stats, weekday_hour_stats = process_day_type(weekday_data, 'weekday')
    
    # Process weekends
    weekend_data = valid_data[valid_data['day_type'] == 'weekend']
    weekend_stats, weekend_hour_stats = process_day_type(weekend_data, 'weekend')

    if len(weekday_stats) == 0 and len(weekend_stats) == 0:
        print("No statistics to plot after processing.")
        return None

    # Create figure with plots only (no tables)
    fig, axes = plt.subplots(2, 1, figsize=figsize)
    
    # Process each day type
    for idx, (stats_df, hour_stats, day_type_name) in enumerate([
        (weekday_stats, weekday_hour_stats, 'WEEKDAYS'), 
        (weekend_stats, weekend_hour_stats, 'WEEKENDS')
    ]):
        ax_plot = axes[idx]
        
        if len(stats_df) == 0:
            ax_plot.text(0.5, 0.5, f'No {day_type_name.lower()} data', 
                        ha='center', va='center', fontsize=14)
            ax_plot.set_xticks([])
            ax_plot.set_yticks([])
            continue
        
        # Calculate correlation
        r_flow_delay = stats_df[['flow', 'mean_delay']].corr().iloc[0, 1]
        
        # Plot: One point per hour
        ax_plot.scatter(stats_df['mean_delay'], stats_df['flow'], 
                       alpha=0.3, color='lightblue', s=30, 
                       edgecolors='blue', linewidth=0.3,
                       label=f'Hourly data (n={len(stats_df)})')
        
        # Add binned statistics with asymmetric error bars
        from scipy import stats as scipy_stats
        from scipy.interpolate import make_interp_spline
        
        if len(stats_df) > 1:
            # LOESS-like smoothing using binned averages with asymmetric confidence intervals
            # Create bins based on EQUAL DELAY INTERVALS (not equal row counts)
            n_bins = 20  # Number of bins across delay range
            min_observations_per_bin = 5  # Minimum hours needed for a bin to be plotted
            
            if len(stats_df) >= min_observations_per_bin:
                delay_min = stats_df['mean_delay'].min()
                delay_max = stats_df['mean_delay'].max()
                
                # Create equal-width delay bins
                bin_edges = np.linspace(delay_min, delay_max, n_bins + 1)
                stats_df['delay_bin'] = pd.cut(stats_df['mean_delay'], bins=bin_edges, 
                                               include_lowest=True, labels=False)
                
                bin_delays = []
                bin_flows = []
                bin_flows_q25 = []
                bin_flows_q75 = []
                bin_counts = []
                
                for bin_idx in range(n_bins):
                    bin_data = stats_df[stats_df['delay_bin'] == bin_idx]
                    
                    # Only include bins with enough observations
                    if len(bin_data) >= min_observations_per_bin:
                        bin_delays.append(bin_data['mean_delay'].mean())
                        bin_flows.append(bin_data['flow'].mean())
                        bin_flows_q25.append(bin_data['flow'].quantile(0.25))
                        bin_flows_q75.append(bin_data['flow'].quantile(0.75))
                        bin_counts.append(len(bin_data))
                
                if len(bin_delays) > 0:
                    bin_delays = np.array(bin_delays)
                    bin_flows = np.array(bin_flows)
                    bin_flows_q25 = np.array(bin_flows_q25)
                    bin_flows_q75 = np.array(bin_flows_q75)
                    bin_counts = np.array(bin_counts)
                    
                    # Calculate asymmetric error bars (distance from mean to Q25 and Q75)
                    # Ensure non-negative values
                    yerr_lower = np.maximum(0, bin_flows - bin_flows_q25)
                    yerr_upper = np.maximum(0, bin_flows_q75 - bin_flows)
                    
                    # Plot binned averages with asymmetric confidence intervals
                    ax_plot.errorbar(bin_delays, bin_flows, 
                                   yerr=[yerr_lower, yerr_upper],
                                   fmt='o', color='darkgreen', markersize=8, 
                                   linewidth=2, capsize=5, capthick=2,
                                   label=f'Binned averages (n={len(bin_delays)} bins) Q25-Q75', zorder=5)
                    
                    # Optional: Add text labels showing observation count for bins with few observations
                    for i, (delay, flow, count) in enumerate(zip(bin_delays, bin_flows, bin_counts)):
                        if count < 50:  # Only label bins with < 50 observations
                            ax_plot.annotate(f'n={count}', (delay, flow), 
                                          textcoords="offset points", xytext=(0,10), 
                                          ha='center', fontsize=8, color='darkgreen', alpha=0.7)
                    
                    # Smooth curve through binned data
                    if len(bin_delays) >= 4:
                        try:
                            # Sort for spline
                            sort_idx = np.argsort(bin_delays)
                            x_sorted = bin_delays[sort_idx]
                            y_sorted = bin_flows[sort_idx]
                            
                            # Use cubic spline for smoothing
                            spline = make_interp_spline(x_sorted, y_sorted, k=min(3, len(x_sorted)-1))
                            x_smooth = np.linspace(x_sorted.min(), x_sorted.max(), 200)
                            y_smooth = spline(x_smooth)
                            ax_plot.plot(x_smooth, y_smooth, 'g-', linewidth=3, 
                                       label='Smooth trend (spline)', zorder=10)
                        except Exception as e:
                            pass  # Skip if spline fails
        
        ax_plot.set_xlabel('Mean Delay (minutes)', fontsize=12)
        ax_plot.set_ylabel('Flow (trains/hour)', fontsize=12)
        ax_plot.set_title(f'{day_type_name}: Flow vs Mean Delay\n(One Point Per Hour - Full Year)', fontsize=14, fontweight='bold')
        ax_plot.set_xlim(0, 25)  # Fix x-axis range: 0-25 minutes
        ax_plot.set_ylim(0, 25)  # Fix y-axis range: 0-25 trains/hour
        ax_plot.grid(True, alpha=0.3)
        ax_plot.legend()

        # Print statistics for this day type
        print(f"\n{'='*80}")
        print(f"📊 STATION {station_id} - {day_type_name} HOURLY ANALYSIS")
        print(f"{'='*80}")
        
        # Additional diagnostics
        hours_with_delays = len(stats_df[stats_df['mean_delay'] > 0])
        hours_no_delays = len(stats_df[stats_df['mean_delay'] == 0])
        unique_hours_of_day = stats_df['hour_of_day'].nunique()
        
        print(f"\n📅 DATA COVERAGE:")
        print(f"  - Hours with train operations: {len(stats_df)} hours")
        print(f"  - Hours with delays (>0 min): {hours_with_delays} hours ({100*hours_with_delays/len(stats_df):.1f}%)")
        print(f"  - Hours with no delays (0 min): {hours_no_delays} hours ({100*hours_no_delays/len(stats_df):.1f}%)")
        print(f"  - Unique hours of day covered: {unique_hours_of_day} out of 24 hours")
        print(f"  - Date range: {stats_df.index.min()} to {stats_df.index.max()}")

    plt.suptitle(f'Hourly Analysis - Weekdays vs Weekends\nStation {station_id} - One Point Per Hour (Full Year Data)', 
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.show()

    # Create SWAPPED AXES plots (X: flow, Y: delay) - Alternative perspective
    fig_swapped, axes_swapped = plt.subplots(2, 1, figsize=figsize)
    
    for idx, (stats_df, hour_stats, day_type_name) in enumerate([
        (weekday_stats, weekday_hour_stats, 'WEEKDAYS'), 
        (weekend_stats, weekend_hour_stats, 'WEEKENDS')
    ]):
        ax_plot = axes_swapped[idx]
        
        if len(stats_df) == 0:
            ax_plot.text(0.5, 0.5, f'No {day_type_name.lower()} data', 
                        ha='center', va='center', fontsize=14)
            ax_plot.set_xticks([])
            ax_plot.set_yticks([])
            continue
        
        # Calculate correlation (same as before)
        r_flow_delay = stats_df[['flow', 'mean_delay']].corr().iloc[0, 1]
        
        # Plot: One point per hour (SWAPPED: X=flow, Y=delay)
        ax_plot.scatter(stats_df['flow'], stats_df['mean_delay'], 
                       alpha=0.3, color='lightblue', s=30, 
                       edgecolors='blue', linewidth=0.3,
                       label=f'Hourly data (n={len(stats_df)})')
        
        # Add binned statistics with asymmetric error bars
        from scipy import stats as scipy_stats
        from scipy.interpolate import make_interp_spline
        
        if len(stats_df) > 1:
            # Binned averages (SWAPPED: bin by flow, show mean delay)
            n_bins = 20
            min_observations_per_bin = 5
            
            if len(stats_df) >= min_observations_per_bin:
                flow_min = stats_df['flow'].min()
                flow_max = stats_df['flow'].max()
                
                # Create equal-width flow bins
                bin_edges_flow = np.linspace(flow_min, flow_max, n_bins + 1)
                stats_df['flow_bin'] = pd.cut(stats_df['flow'], bins=bin_edges_flow, 
                                              include_lowest=True, labels=False)
                
                bin_flows_swap = []
                bin_delays_swap = []
                bin_delays_q25 = []
                bin_delays_q75 = []
                bin_counts_swap = []
                
                for bin_idx in range(n_bins):
                    bin_data = stats_df[stats_df['flow_bin'] == bin_idx]
                    
                    if len(bin_data) >= min_observations_per_bin:
                        bin_flows_swap.append(bin_data['flow'].mean())
                        bin_delays_swap.append(bin_data['mean_delay'].mean())
                        bin_delays_q25.append(bin_data['mean_delay'].quantile(0.25))
                        bin_delays_q75.append(bin_data['mean_delay'].quantile(0.75))
                        bin_counts_swap.append(len(bin_data))
                
                if len(bin_flows_swap) > 0:
                    bin_flows_swap = np.array(bin_flows_swap)
                    bin_delays_swap = np.array(bin_delays_swap)
                    bin_delays_q25 = np.array(bin_delays_q25)
                    bin_delays_q75 = np.array(bin_delays_q75)
                    bin_counts_swap = np.array(bin_counts_swap)
                    
                    # Calculate asymmetric error bars (distance from mean to Q25 and Q75)
                    # Ensure non-negative values
                    yerr_lower = np.maximum(0, bin_delays_swap - bin_delays_q25)
                    yerr_upper = np.maximum(0, bin_delays_q75 - bin_delays_swap)
                    
                    # Plot binned averages with asymmetric error bars
                    ax_plot.errorbar(bin_flows_swap, bin_delays_swap, 
                                   yerr=[yerr_lower, yerr_upper],
                                   fmt='o', color='darkgreen', markersize=8, 
                                   linewidth=2, capsize=5, capthick=2,
                                   label=f'Binned averages (n={len(bin_flows_swap)} bins) Q25-Q75', zorder=5)
                    
                    # Add text labels for sparse bins
                    for i, (flow, delay, count) in enumerate(zip(bin_flows_swap, bin_delays_swap, bin_counts_swap)):
                        if count < 50:
                            ax_plot.annotate(f'n={count}', (flow, delay), 
                                          textcoords="offset points", xytext=(0,10), 
                                          ha='center', fontsize=8, color='darkgreen', alpha=0.7)
                    
                    # Smooth curve through binned data
                    if len(bin_flows_swap) >= 4:
                        try:
                            sort_idx = np.argsort(bin_flows_swap)
                            x_sorted = bin_flows_swap[sort_idx]
                            y_sorted = bin_delays_swap[sort_idx]
                            
                            spline = make_interp_spline(x_sorted, y_sorted, k=min(3, len(x_sorted)-1))
                            x_smooth = np.linspace(x_sorted.min(), x_sorted.max(), 200)
                            y_smooth = spline(x_smooth)
                            ax_plot.plot(x_smooth, y_smooth, 'g-', linewidth=3, 
                                       label='Smooth trend (spline)', zorder=10)
                        except Exception as e:
                            pass
        
        ax_plot.set_xlabel('Flow (trains/hour)', fontsize=12)
        ax_plot.set_ylabel('Mean Delay (minutes)', fontsize=12)
        ax_plot.set_title(f'{day_type_name}: Mean Delay vs Flow\n(Alternative Perspective - Axes Swapped)', fontsize=14, fontweight='bold')
        ax_plot.set_xlim(0, 25)  # Fix x-axis range: 0-25 trains/hour
        ax_plot.set_ylim(0, 25)  # Fix y-axis range: 0-25 minutes
        ax_plot.grid(True, alpha=0.3)
        ax_plot.legend()

    plt.suptitle(f'Alternative View: Delay vs Flow - Weekdays vs Weekends\nStation {station_id} - One Point Per Hour (Full Year Data)', 
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.show()

    # Return both dataframes and hourly statistics
    return {
        'weekday': weekday_stats,
        'weekend': weekend_stats,
        'weekday_hour_stats': weekday_hour_stats,
        'weekend_hour_stats': weekend_hour_stats
    }
print("plot_variable_relationships function ready!")