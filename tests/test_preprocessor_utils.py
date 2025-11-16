"""
Unit tests for preprocessor/utils.py functions.

This module tests the utility functions used in the preprocessing pipeline,
including day code mapping, schedule processing, and delay matching.
"""

import pytest
import pandas as pd
from datetime import datetime
from preprocess.utils import (
    get_day_code_mapping,
    extract_schedule_days_runs,
    get_english_day_types_from_schedule,
    extract_day_of_week_from_delay,
    schedule_runs_on_day
)


class TestDayCodeMapping:
    """Test cases for day code mapping functions."""
    
    def test_get_day_code_mapping_returns_dict(self):
        """Test that get_day_code_mapping returns a dictionary."""
        result = get_day_code_mapping()
        assert isinstance(result, dict)
    
    def test_get_day_code_mapping_has_seven_days(self):
        """Test that the mapping contains all 7 days of the week."""
        result = get_day_code_mapping()
        assert len(result) == 7
    
    def test_get_day_code_mapping_values(self):
        """Test that the mapping contains correct day codes."""
        result = get_day_code_mapping()
        expected = {
            0: "MO",
            1: "TU",
            2: "WE",
            3: "TH",
            4: "FR",
            5: "SA",
            6: "SU"
        }
        assert result == expected
    
    def test_monday_is_zero(self):
        """Test that Monday is index 0."""
        result = get_day_code_mapping()
        assert result[0] == "MO"
    
    def test_sunday_is_six(self):
        """Test that Sunday is index 6."""
        result = get_day_code_mapping()
        assert result[6] == "SU"


class TestScheduleDaysExtraction:
    """Test cases for extracting schedule days from entries."""
    
    def test_extract_schedule_days_runs_valid_entry(self, sample_schedule_entry):
        """Test extracting schedule_days_runs from a valid entry."""
        result = extract_schedule_days_runs(sample_schedule_entry)
        assert result == '1111100'
    
    def test_extract_schedule_days_runs_missing_key(self):
        """Test handling of missing schedule_days_runs key."""
        entry = {'JsonScheduleV1': {}}
        result = extract_schedule_days_runs(entry)
        assert result is None
    
    def test_extract_schedule_days_runs_malformed_entry(self):
        """Test handling of malformed schedule entry."""
        entry = {}
        result = extract_schedule_days_runs(entry)
        assert result is None
    
    def test_get_english_day_types_weekdays(self, sample_schedule_entry):
        """Test converting binary schedule to weekday codes."""
        result = get_english_day_types_from_schedule(sample_schedule_entry)
        assert result == ["MO", "TU", "WE", "TH", "FR"]
    
    def test_get_english_day_types_all_days(self):
        """Test schedule that runs every day."""
        entry = {
            'JsonScheduleV1': {
                'schedule_days_runs': '1111111'
            }
        }
        result = get_english_day_types_from_schedule(entry)
        assert result == ["MO", "TU", "WE", "TH", "FR", "SA", "SU"]
    
    def test_get_english_day_types_weekends_only(self):
        """Test schedule that runs only on weekends."""
        entry = {
            'JsonScheduleV1': {
                'schedule_days_runs': '0000011'
            }
        }
        result = get_english_day_types_from_schedule(entry)
        assert result == ["SA", "SU"]
    
    def test_get_english_day_types_no_days(self):
        """Test schedule that doesn't run."""
        entry = {
            'JsonScheduleV1': {
                'schedule_days_runs': '0000000'
            }
        }
        result = get_english_day_types_from_schedule(entry)
        assert result == []
    
    def test_get_english_day_types_empty_entry(self):
        """Test with empty entry."""
        result = get_english_day_types_from_schedule({})
        assert result == []


class TestDelayDayExtraction:
    """Test cases for extracting day of week from delay entries."""
    
    def test_extract_day_monday(self):
        """Test extracting Monday from delay entry."""
        delay_entry = {
            'PLANNED_ORIGIN_WTT_DATETIME': '01-JAN-2024 08:00'  # Monday
        }
        result = extract_day_of_week_from_delay(delay_entry)
        assert result == "MO"
    
    def test_extract_day_friday(self):
        """Test extracting Friday from delay entry."""
        delay_entry = {
            'PLANNED_ORIGIN_WTT_DATETIME': '05-JAN-2024 08:00'  # Friday
        }
        result = extract_day_of_week_from_delay(delay_entry)
        assert result == "FR"
    
    def test_extract_day_sunday(self):
        """Test extracting Sunday from delay entry."""
        delay_entry = {
            'PLANNED_ORIGIN_WTT_DATETIME': '07-JAN-2024 08:00'  # Sunday
        }
        result = extract_day_of_week_from_delay(delay_entry)
        assert result == "SU"
    
    def test_extract_day_missing_field(self):
        """Test handling of missing datetime field."""
        delay_entry = {}
        result = extract_day_of_week_from_delay(delay_entry)
        assert result is None
    
    def test_extract_day_invalid_format(self):
        """Test handling of invalid datetime format."""
        delay_entry = {
            'PLANNED_ORIGIN_WTT_DATETIME': 'invalid-date'
        }
        result = extract_day_of_week_from_delay(delay_entry)
        assert result is None
    
    def test_extract_day_none_value(self):
        """Test handling of None value."""
        delay_entry = {
            'PLANNED_ORIGIN_WTT_DATETIME': None
        }
        result = extract_day_of_week_from_delay(delay_entry)
        assert result is None


class TestScheduleRunsOnDay:
    """Test cases for checking if schedule runs on a specific day."""
    
    def test_schedule_runs_on_monday(self):
        """Test checking if schedule runs on Monday."""
        schedule = {
            'ENGLISH_DAY_TYPE': ['MO', 'TU', 'WE', 'TH', 'FR']
        }
        assert schedule_runs_on_day(schedule, 'MO') is True
    
    def test_schedule_not_runs_on_saturday(self):
        """Test checking if weekday schedule doesn't run on Saturday."""
        schedule = {
            'ENGLISH_DAY_TYPE': ['MO', 'TU', 'WE', 'TH', 'FR']
        }
        assert schedule_runs_on_day(schedule, 'SA') is False
    
    def test_schedule_runs_all_days(self):
        """Test schedule that runs every day."""
        schedule = {
            'ENGLISH_DAY_TYPE': ['MO', 'TU', 'WE', 'TH', 'FR', 'SA', 'SU']
        }
        for day in ['MO', 'TU', 'WE', 'TH', 'FR', 'SA', 'SU']:
            assert schedule_runs_on_day(schedule, day) is True
    
    def test_schedule_empty_days(self):
        """Test schedule with no running days."""
        schedule = {
            'ENGLISH_DAY_TYPE': []
        }
        assert schedule_runs_on_day(schedule, 'MO') is False
    
    def test_schedule_missing_field(self):
        """Test schedule with missing ENGLISH_DAY_TYPE field."""
        schedule = {}
        assert schedule_runs_on_day(schedule, 'MO') is False


class TestProcessDelays:
    """Test cases for process_delays function."""
    
    def test_process_delays_filters_by_stanox(self, mock_incident_files, tmp_path):
        """Test that process_delays filters by STANOX code."""
        from preprocessor.utils import process_delays
        import os
        
        # Create the output directory
        temp_output_dir = tmp_path / "test_output"
        os.makedirs(temp_output_dir, exist_ok=True)
        
        result = process_delays(mock_incident_files, '12345', str(temp_output_dir))
        
        assert 'Period 1' in result
        assert isinstance(result['Period 1'], pd.DataFrame)
        assert len(result['Period 1']) > 0
    
    def test_process_delays_removes_columns(self, mock_incident_files, tmp_path):
        """Test that process_delays removes specified columns."""
        from preprocessor.utils import process_delays
        import os
        
        # Create the output directory
        temp_output_dir = tmp_path / "test_output"
        os.makedirs(temp_output_dir, exist_ok=True)
        
        result = process_delays(mock_incident_files, '12345', str(temp_output_dir))
        df = result['Period 1']
        
        # Check that unwanted columns are removed
        unwanted_cols = ['ATTRIBUTION_STATUS', 'INCIDENT_EQUIPMENT', 
                        'APPLICABLE_TIMETABLE_FLAG', 'TRACTION_TYPE', 'TRAILING_LOAD']
        for col in unwanted_cols:
            assert col not in df.columns
    
    def test_process_delays_no_matching_stanox(self, mock_incident_files, tmp_path):
        """Test process_delays with no matching STANOX."""
        from preprocessor.utils import process_delays
        import os
        
        # Create the output directory
        temp_output_dir = tmp_path / "test_output"
        os.makedirs(temp_output_dir, exist_ok=True)
        
        result = process_delays(mock_incident_files, '99999', str(temp_output_dir))
        
        # Should still return dictionary with empty or filtered DataFrames
        assert isinstance(result, dict)


# Parametrized tests for better coverage
@pytest.mark.parametrize("day_index,expected_code", [
    (0, "MO"),
    (1, "TU"),
    (2, "WE"),
    (3, "TH"),
    (4, "FR"),
    (5, "SA"),
    (6, "SU"),
])
def test_day_mapping_parametrized(day_index, expected_code):
    """Parametrized test for all day mappings."""
    mapping = get_day_code_mapping()
    assert mapping[day_index] == expected_code


@pytest.mark.parametrize("binary_string,expected_days", [
    ('1111100', ['MO', 'TU', 'WE', 'TH', 'FR']),
    ('0000011', ['SA', 'SU']),
    ('1111111', ['MO', 'TU', 'WE', 'TH', 'FR', 'SA', 'SU']),
    ('0000000', []),
    ('1000000', ['MO']),
    ('0000001', ['SU']),
])
def test_binary_to_days_parametrized(binary_string, expected_days):
    """Parametrized test for binary string conversion."""
    entry = {'JsonScheduleV1': {'schedule_days_runs': binary_string}}
    result = get_english_day_types_from_schedule(entry)
    assert result == expected_days
