# all functions for the station view NOT USED:

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