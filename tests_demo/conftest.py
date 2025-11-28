"""
Pytest configuration file with fixtures for demo function tests.
This file is automatically loaded by pytest and provides fixtures available to all tests.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
import os

# Add the parent directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


@pytest.fixture
def sample_station_data():
    """Fixture providing sample station data for testing visualization functions."""
    # Create sample train data for a station over a few days
    np.random.seed(42)
    
    dates = pd.date_range('2024-01-01', periods=3, freq='D')
    data = []
    
    for date in dates:
        day_code = ['MO', 'TU', 'WE'][date.dayofweek]
        # Create 20 trains per day
        for hour in range(6, 23):  # 6 AM to 10 PM
            for train_num in range(2):  # 2 trains per hour
                train_id = f"TRAIN{date.strftime('%Y%m%d')}{hour:02d}{train_num}"
                arrival_time = hour + (train_num * 0.5)
                delay = max(0, np.random.normal(5, 3))  # Mean 5 min, std 3 min
                
                data.append({
                    'TRAIN_SERVICE_CODE': train_id,
                    'STANOX': '32000',  # Manchester Piccadilly
                    'DAY': day_code,
                    'ACTUAL_CALLS': int((arrival_time % 1) * 60 + (arrival_time // 1) * 100),
                    'PLANNED_CALLS': int((arrival_time % 1) * 60 + (arrival_time // 1) * 100),
                    'PFPI_MINUTES': delay,
                    'EVENT_TYPE': 'A',
                    'EVENT_DATETIME': f"{date.strftime('%d-%b-%Y')} {int(arrival_time):02d}:{int((arrival_time % 1) * 60):02d}",
                    'INCIDENT_NUMBER': f'INC{train_num}' if delay > 10 else None,
                    'INCIDENT_REASON': 'Signal failure' if delay > 10 else None,
                    'INCIDENT_START_DATETIME': f"{date.strftime('%d-%b-%Y')} {int(arrival_time):02d}:00" if delay > 10 else None,
                    'PLANNED_ORIGIN_LOCATION_CODE': '32000',
                    'PLANNED_DEST_LOCATION_CODE': '33087',
                    'delay_minutes': delay,
                    'PLANNED_ORIGIN_GBTT_DATETIME': f"{date.strftime('%d-%b-%Y')} {int(arrival_time):02d}:00",
                    'PLANNED_DEST_GBTT_DATETIME': f"{date.strftime('%d-%b-%Y')} {int(arrival_time)+1:02d}:00",
                    'SECTION_CODE': 'SC001',
                    'DELAY_DAY': date.strftime('%d-%b-%Y'),
                    'ENGLISH_DAY_TYPE': 'Weekday' if date.dayofweek < 5 else 'Weekend',
                    'STATION_ROLE': 'Major Hub',
                    'DFT_CATEGORY': 'A',
                    'PLATFORM_COUNT': 14,
                    'DATASET_TYPE': 'SCHEDULE',
                    'WEEKDAY': date.dayofweek
                })
    
    return pd.DataFrame(data)


@pytest.fixture
def sample_all_data():
    """Fixture providing comprehensive dataset with multiple stations."""
    np.random.seed(42)
    
    stations = [
        ('32000', 14),  # Manchester Piccadilly, 14 platforms
        ('33087', 5),   # Manchester Oxford Road, 5 platforms
        ('65630', 11)   # Birmingham New Street, 11 platforms
    ]
    
    dates = pd.date_range('2024-01-01', periods=7, freq='D')
    data = []
    
    for station_id, num_platforms in stations:
        for date in dates:
            day_code = ['MO', 'TU', 'WE', 'TH', 'FR', 'SA', 'SU'][date.dayofweek]
            # More trains on weekdays
            trains_per_hour = 3 if date.dayofweek < 5 else 2
            
            for hour in range(5, 24):  # 5 AM to 11 PM
                for train_num in range(trains_per_hour):
                    train_id = f"{station_id}_{date.strftime('%Y%m%d')}_{hour:02d}_{train_num}"
                    arrival_min = train_num * (60 // trains_per_hour)
                    
                    # Simulate varying delays based on time of day
                    if 7 <= hour <= 9 or 17 <= hour <= 19:  # Peak hours
                        delay = max(0, np.random.normal(8, 5))
                    else:
                        delay = max(0, np.random.normal(3, 2))
                    
                    data.append({
                        'TRAIN_SERVICE_CODE': train_id,
                        'STANOX': station_id,
                        'DAY': day_code,
                        'ACTUAL_CALLS': hour * 100 + arrival_min,
                        'PLANNED_CALLS': hour * 100 + arrival_min,
                        'PFPI_MINUTES': delay,
                        'EVENT_TYPE': 'A',
                        'EVENT_DATETIME': f"{date.strftime('%d-%b-%Y')} {hour:02d}:{arrival_min:02d}",
                        'INCIDENT_NUMBER': f'INC{train_num}' if delay > 15 else None,
                        'INCIDENT_REASON': 'Congestion' if delay > 15 else None,
                        'INCIDENT_START_DATETIME': f"{date.strftime('%d-%b-%Y')} {hour:02d}:00" if delay > 15 else None,
                        'PLATFORM': str((train_num % num_platforms) + 1),
                        'PLANNED_ORIGIN_LOCATION_CODE': station_id,
                        'PLANNED_DEST_LOCATION_CODE': stations[(stations.index((station_id, num_platforms)) + 1) % len(stations)][0],
                        'delay_minutes': delay,
                        'PLANNED_ORIGIN_GBTT_DATETIME': f"{date.strftime('%d-%b-%Y')} {hour:02d}:00",
                        'PLANNED_DEST_GBTT_DATETIME': f"{date.strftime('%d-%b-%Y')} {hour+1:02d}:00",
                        'SECTION_CODE': 'SC001',
                        'DELAY_DAY': date.strftime('%d-%b-%Y'),
                        'ENGLISH_DAY_TYPE': 'Weekday' if date.dayofweek < 5 else 'Weekend',
                        'STATION_ROLE': 'Major Hub',
                        'DFT_CATEGORY': 'A',
                        'PLATFORM_COUNT': num_platforms,
                        'DATASET_TYPE': 'SCHEDULE',
                        'WEEKDAY': date.dayofweek
                    })
    
    return pd.DataFrame(data)


@pytest.fixture
def empty_station_data():
    """Fixture providing empty DataFrame with correct schema."""
    return pd.DataFrame(columns=[
        'TRAIN_SERVICE_CODE', 'STANOX', 'DAY', 'ACTUAL_CALLS', 'PLANNED_CALLS',
        'PFPI_MINUTES', 'EVENT_TYPE', 'EVENT_DATETIME', 'INCIDENT_NUMBER',
        'INCIDENT_START_DATETIME', 'PLANNED_ORIGIN_LOCATION_CODE', 
        'PLANNED_DEST_LOCATION_CODE', 'delay_minutes', 'PLATFORM'
    ])


@pytest.fixture
def station_with_cancellations():
    """Fixture providing station data with some cancelled trains."""
    np.random.seed(42)
    
    data = []
    for hour in range(8, 18):
        for train_num in range(3):
            train_id = f"TRAIN{hour:02d}{train_num}"
            event_type = 'C' if train_num == 2 and hour % 3 == 0 else 'A'  # Some cancellations
            
            data.append({
                'TRAIN_SERVICE_CODE': train_id,
                'STANOX': '32000',
                'DAY': 'MO',
                'ACTUAL_CALLS': hour * 100 + (train_num * 20),
                'PLANNED_CALLS': hour * 100 + (train_num * 20),
                'PFPI_MINUTES': np.random.normal(5, 2) if event_type == 'A' else 0,
                'EVENT_TYPE': event_type,
                'EVENT_DATETIME': f"01-JAN-2024 {hour:02d}:{train_num * 20:02d}",
                'INCIDENT_NUMBER': None,
                'INCIDENT_START_DATETIME': None,
                'PLANNED_ORIGIN_LOCATION_CODE': '32000',
                'PLANNED_DEST_LOCATION_CODE': '33087',
                'delay_minutes': np.random.normal(5, 2) if event_type == 'A' else 0,
                'PLATFORM': str((train_num % 14) + 1)
            })
    
    return pd.DataFrame(data)


@pytest.fixture
def mock_matplotlib():
    """Mock matplotlib to prevent actual plot rendering during tests."""
    import matplotlib.pyplot as plt
    from unittest.mock import Mock, patch
    
    with patch('matplotlib.pyplot.show'), \
         patch('matplotlib.pyplot.savefig'), \
         patch('matplotlib.pyplot.close'):
        yield plt


@pytest.fixture
def expected_flow_calculation():
    """Fixture providing expected flow calculation results."""
    return {
        'weekday': {
            'mean_flow': 2.5,  # trains per hour
            'max_flow': 4.0,
            'min_flow': 1.0,
            'std_flow': 0.8
        },
        'weekend': {
            'mean_flow': 1.8,
            'max_flow': 3.0,
            'min_flow': 0.5,
            'std_flow': 0.6
        }
    }


@pytest.fixture
def expected_delay_statistics():
    """Fixture providing expected delay statistics."""
    return {
        'mean_delay': 5.2,
        'median_delay': 4.8,
        'q25_delay': 2.1,
        'q75_delay': 7.5,
        'max_delay': 18.3,
        'delayed_trains_pct': 78.5  # Percentage of trains with delay > 0
    }


@pytest.fixture
def normalized_metrics():
    """Fixture providing expected normalized metrics."""
    return {
        'num_platforms': 12,
        'normalized_flow': 0.208,  # 2.5 trains/hour / 12 platforms
        'normalized_trains_in_system': 0.417,  # 5 trains / 12 platforms
        'utilization': 0.417  # Same as trains in system
    }


@pytest.fixture
def sample_binned_data():
    """Fixture providing sample binned statistics for testing."""
    return pd.DataFrame({
        'flow_bin': [0, 1, 2, 3, 4],
        'mean_delay': [2.5, 4.2, 6.8, 9.1, 12.3],
        'q25_delay': [1.0, 2.0, 3.5, 5.0, 7.0],
        'q75_delay': [4.0, 6.0, 9.0, 12.0, 16.0],
        'count': [15, 23, 31, 19, 12]
    })


@pytest.fixture
def temp_output_dir(tmp_path):
    """Fixture providing a temporary output directory for plot files."""
    output_dir = tmp_path / "test_plots"
    output_dir.mkdir()
    return str(output_dir)


@pytest.fixture
def mock_valid_station_ids():
    """Fixture providing valid station STANOX codes for testing."""
    return {
        '32000': 'Manchester Piccadilly',
        '33087': 'Manchester Oxford Road',
        '65630': 'Birmingham New Street',
        '51531': 'Barking',
        '88403': 'Cannon Street'
    }
