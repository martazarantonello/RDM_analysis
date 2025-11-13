"""
Pytest configuration file with fixtures and shared test setup.
This file is automatically loaded by pytest and provides fixtures available to all tests.
These are data used for my testing (expected outcomes).
"""

import pytest
import pandas as pd
import os
import sys
from datetime import datetime

# Add the parent directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


@pytest.fixture
def sample_schedule_entry():
    """Fixture providing a sample schedule entry for testing."""
    return {
        'JsonScheduleV1': {
            'schedule_days_runs': '1111100',  # Monday-Friday
            'schedule_segment': {
                'signalling_id': '1T95',
                'CIF_train_category': 'XX',
                'CIF_headcode': 'TEST',
                'CIF_course_indicator': '1',
                'CIF_train_service_code': 'TEST123',
                'CIF_business_sector': 'XX',
                'CIF_power_type': 'EMU',
                'CIF_timing_load': '377',
                'CIF_speed': '100',
                'CIF_operating_characteristics': None,
                'CIF_train_class': 'B',
                'CIF_sleepers': 'N',
                'CIF_reservations': 'N',
                'CIF_connection_indicator': None,
                'CIF_catering_code': None,
                'CIF_service_branding': None,
                'schedule_location': [
                    {
                        'location_type': 'LO',
                        'record_identity': 'LO',
                        'tiploc_code': 'ORIGIN',
                        'tiploc_instance': None,
                        'departure': '0800',
                        'public_departure': '0800',
                        'platform': '1',
                        'line': '1',
                        'engineering_allowance': None,
                        'pathing_allowance': None,
                        'performance_allowance': None
                    },
                    {
                        'location_type': 'LI',
                        'record_identity': 'LI',
                        'tiploc_code': 'MIDDLE',
                        'tiploc_instance': None,
                        'departure': '0830',
                        'public_departure': '0830',
                        'platform': '2',
                        'line': '2',
                        'engineering_allowance': None,
                        'pathing_allowance': None,
                        'performance_allowance': None,
                        'arrival': '0828',
                        'public_arrival': '0828',
                        'pass': None,
                        'path': None
                    },
                    {
                        'location_type': 'LT',
                        'record_identity': 'LT',
                        'tiploc_code': 'DEST',
                        'tiploc_instance': None,
                        'platform': '3',
                        'arrival': '0900',
                        'public_arrival': '0900',
                        'path': None
                    }
                ]
            }
        }
    }


@pytest.fixture
def sample_delay_entry():
    """Fixture providing a sample delay entry for testing."""
    return {
        'TRAIN_SERVICE_CODE': 'TEST123',
        'PLANNED_ORIGIN_LOCATION_CODE': '12345',
        'PLANNED_ORIGIN_GBTT_DATETIME': '01-JAN-2024 08:00',
        'PLANNED_ORIGIN_WTT_DATETIME': '01-JAN-2024 08:00',
        'PLANNED_DEST_LOCATION_CODE': '67890',
        'PLANNED_DEST_GBTT_DATETIME': '01-JAN-2024 09:00',
        'EVENT_DATETIME': '01-JAN-2024 08:15',
        'PFPI_MINUTES': 15.0,
        'INCIDENT_REASON': 'Signal failure',
        'INCIDENT_NUMBER': 'INC001',
        'EVENT_TYPE': 'D',
        'SECTION_CODE': 'SEC01',
        'START_STANOX': 12345,
        'END_STANOX': 67890
    }


@pytest.fixture
def sample_processed_schedule():
    """Fixture providing a sample processed schedule list."""
    return [
        {
            "TRAIN_SERVICE_CODE": "TRAIN001",
            "PLANNED_ORIGIN_LOCATION_CODE": "12345",
            "PLANNED_ORIGIN_GBTT_DATETIME": "0800",
            "PLANNED_DEST_LOCATION_CODE": "67890",
            "PLANNED_DEST_GBTT_DATETIME": "0900",
            "PLANNED_CALLS": "0830",
            "ACTUAL_CALLS": "0830",
            "PFPI_MINUTES": 0.0,
            "INCIDENT_REASON": None,
            "INCIDENT_NUMBER": None,
            "EVENT_TYPE": None,
            "SECTION_CODE": None,
            "DELAY_DAY": None,
            "EVENT_DATETIME": None,
            "INCIDENT_START_DATETIME": None,
            "ENGLISH_DAY_TYPE": ["MO", "TU", "WE", "TH", "FR"],
            "STATION_ROLE": "Intermediate",
            "DFT_CATEGORY": "A",
            "PLATFORM_COUNT": 4
        }
    ]


@pytest.fixture
def sample_stanox_ref():
    """Fixture providing sample STANOX reference data."""
    return pd.DataFrame([
        {
            'stanox': '12345',
            'tiploc': 'TESTST',
            'description': 'Test Station',
            'dft_category': 'A',
            'numeric_platform_count': 4
        },
        {
            'stanox': '67890',
            'tiploc': 'DEST',
            'description': 'Destination Station',
            'dft_category': 'B',
            'numeric_platform_count': 2
        }
    ])


@pytest.fixture
def sample_tiploc_to_stanox():
    """Fixture providing TIPLOC to STANOX mapping."""
    return {
        'TESTST': '12345',
        'ORIGIN': '11111',
        'MIDDLE': '12345',
        'DEST': '67890'
    }


@pytest.fixture
def temp_output_dir(tmp_path):
    """Fixture providing a temporary output directory."""
    return str(tmp_path / "test_output")


@pytest.fixture
def mock_incident_files(tmp_path):
    """Fixture providing mock incident files."""
    # Create temporary CSV files for testing
    period1_data = pd.DataFrame([
        {
            'TRAIN_SERVICE_CODE': 'TEST001',
            'START_STANOX': 12345,
            'END_STANOX': 67890,
            'PFPI_MINUTES': 10.0,
            'INCIDENT_REASON': 'Test delay',
            'ATTRIBUTION_STATUS': 'Final',
            'INCIDENT_EQUIPMENT': 'Signal',
            'APPLICABLE_TIMETABLE_FLAG': 'Y',
            'TRACTION_TYPE': 'EMU',
            'TRAILING_LOAD': 8
        }
    ])
    
    period1_file = tmp_path / "period1.csv"
    period1_data.to_csv(period1_file, index=False)
    
    return {
        'Period 1': str(period1_file)
    }
