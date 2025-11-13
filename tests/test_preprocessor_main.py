"""
Unit tests for preprocessor/preprocessor.py main functions.

This module tests the main preprocessing functions including:
- Station loading
- Data processing and saving
- Category-based processing
"""

import pytest
import pandas as pd
import os
from unittest.mock import Mock, patch, MagicMock
from preprocessor.preprocessor import (
    get_weekday_from_schedule_entry,
    load_stations,
    save_processed_data_by_weekday_to_dataframe
)


class TestGetWeekdayFromScheduleEntry:
    """Test cases for get_weekday_from_schedule_entry function."""
    
    def test_weekday_from_delay_day(self):
        """Test extracting weekday when DELAY_DAY is present."""
        entry = {'DELAY_DAY': 'MO'}
        result = get_weekday_from_schedule_entry(entry)
        assert result == 0  # Monday
    
    def test_weekday_from_delay_day_friday(self):
        """Test extracting Friday from DELAY_DAY."""
        entry = {'DELAY_DAY': 'FR'}
        result = get_weekday_from_schedule_entry(entry)
        assert result == 4  # Friday
    
    def test_weekday_from_english_day_type(self):
        """Test extracting weekday from ENGLISH_DAY_TYPE when no DELAY_DAY."""
        entry = {'ENGLISH_DAY_TYPE': ['TU', 'WE', 'TH']}
        result = get_weekday_from_schedule_entry(entry)
        assert result == 1  # Tuesday (first in list)
    
    def test_weekday_from_english_day_type_multiple(self):
        """Test that it returns the first day from ENGLISH_DAY_TYPE."""
        entry = {'ENGLISH_DAY_TYPE': ['WE', 'TH', 'FR']}
        result = get_weekday_from_schedule_entry(entry)
        assert result == 2  # Wednesday (first in list)
    
    def test_weekday_empty_entry(self):
        """Test with empty entry - should default to 0."""
        entry = {}
        result = get_weekday_from_schedule_entry(entry)
        assert result == 0  # Default to Monday
    
    def test_weekday_invalid_day_code(self):
        """Test with invalid day code - should default to 0."""
        entry = {'DELAY_DAY': 'INVALID'}
        result = get_weekday_from_schedule_entry(entry)
        assert result == 0  # Default to Monday
    
    def test_weekday_empty_english_day_type(self):
        """Test with empty ENGLISH_DAY_TYPE list."""
        entry = {'ENGLISH_DAY_TYPE': []}
        result = get_weekday_from_schedule_entry(entry)
        assert result == 0  # Default to Monday


class TestLoadStations:
    """Test cases for load_stations function."""
    
    @patch('builtins.open', create=True)
    @patch('json.load')
    def test_load_stations_category_a(self, mock_json_load, mock_open):
        """Test loading Category A stations."""
        # Mock the JSON data
        mock_json_load.return_value = [
            {'stanox': '12345', 'dft_category': 'A'},
            {'stanox': '67890', 'dft_category': 'A'},
            {'stanox': '11111', 'dft_category': 'B'}
        ]
        
        result = load_stations(category='A')
        
        assert len(result) == 2
        assert '12345' in result
        assert '67890' in result
        assert '11111' not in result
    
    @patch('builtins.open', create=True)
    @patch('json.load')
    def test_load_stations_category_b(self, mock_json_load, mock_open):
        """Test loading Category B stations."""
        mock_json_load.return_value = [
            {'stanox': '12345', 'dft_category': 'A'},
            {'stanox': '67890', 'dft_category': 'B'},
            {'stanox': '11111', 'dft_category': 'B'}
        ]
        
        result = load_stations(category='B')
        
        assert len(result) == 2
        assert '67890' in result
        assert '11111' in result
        assert '12345' not in result
    
    @patch('builtins.open', create=True)
    @patch('json.load')
    def test_load_stations_all_categories(self, mock_json_load, mock_open):
        """Test loading all stations with categories."""
        mock_json_load.return_value = [
            {'stanox': '12345', 'dft_category': 'A'},
            {'stanox': '67890', 'dft_category': 'B'},
            {'stanox': '11111', 'dft_category': ''}  # Empty category - should be excluded
        ]
        
        result = load_stations(category=None)
        
        assert len(result) == 2
        assert '12345' in result
        assert '67890' in result
        assert '11111' not in result  # Excluded due to empty category
    
    @patch('builtins.open', create=True)
    @patch('json.load')
    def test_load_stations_missing_stanox(self, mock_json_load, mock_open):
        """Test handling of entries without STANOX."""
        mock_json_load.return_value = [
            {'stanox': '12345', 'dft_category': 'A'},
            {'dft_category': 'A'},  # Missing STANOX
        ]
        
        result = load_stations(category='A')
        
        assert len(result) == 1
        assert '12345' in result
    
    @patch('builtins.open', side_effect=FileNotFoundError)
    def test_load_stations_file_not_found(self, mock_open):
        """Test handling of missing reference file."""
        result = load_stations(category='A')
        
        assert result == []
    
    @patch('builtins.open', create=True)
    @patch('json.load')
    def test_load_stations_excludes_no_category(self, mock_json_load, mock_open):
        """Test that stations without category are excluded when loading all."""
        mock_json_load.return_value = [
            {'stanox': '12345', 'dft_category': 'A'},
            {'stanox': '67890'},  # No category field
            {'stanox': '11111', 'dft_category': None},  # None category
        ]
        
        result = load_stations(category=None)
        
        assert len(result) == 1
        assert '12345' in result


class TestSaveProcessedDataByWeekday:
    """Test cases for save_processed_data_by_weekday_to_dataframe function."""
    
    @patch('preprocessor.preprocessor.process_schedule')
    @patch('preprocessor.preprocessor.process_delays_optimized')
    @patch('preprocessor.preprocessor.adjust_schedule_timeline')
    def test_save_processed_data_returns_dict(
        self, mock_adjust, mock_delays, mock_schedule
    ):
        """Test that function returns a dictionary of DataFrames."""
        # Mock the schedule processing
        mock_schedule.return_value = [
            {
                'TRAIN_SERVICE_CODE': 'TEST001',
                'PLANNED_CALLS': '0800',
                'ACTUAL_CALLS': '0800',
                'ENGLISH_DAY_TYPE': ['MO']
            }
        ]
        
        # Mock the delay processing
        mock_delays.return_value = {}
        
        # Mock the timeline adjustment
        mock_adjust.return_value = [
            {
                'TRAIN_SERVICE_CODE': 'TEST001',
                'PLANNED_CALLS': '0800',
                'ACTUAL_CALLS': '0800',
                'ENGLISH_DAY_TYPE': ['MO']
            }
        ]
        
        # Create pre-loaded mock data
        schedule_data = pd.DataFrame([])
        stanox_ref = pd.DataFrame([{'stanox': '12345', 'tiploc': 'TEST'}])
        tiploc_to_stanox = {'TEST': '12345'}
        incident_data = {}
        
        result = save_processed_data_by_weekday_to_dataframe(
            '12345',
            schedule_data_loaded=schedule_data,
            stanox_ref=stanox_ref,
            tiploc_to_stanox=tiploc_to_stanox,
            incident_data_loaded=incident_data
        )
        
        assert isinstance(result, dict)
        assert 'MO' in result
        assert isinstance(result['MO'], pd.DataFrame)
    
    @patch('preprocessor.preprocessor.process_schedule')
    def test_save_processed_data_no_schedule_returns_none(self, mock_schedule):
        """Test that function returns None when no schedule data found."""
        mock_schedule.return_value = []
        
        result = save_processed_data_by_weekday_to_dataframe(
            '99999',
            schedule_data_loaded=pd.DataFrame([]),
            stanox_ref=pd.DataFrame([]),
            tiploc_to_stanox={},
            incident_data_loaded={}
        )
        
        assert result is None
    
    @patch('preprocessor.preprocessor.process_schedule')
    @patch('preprocessor.preprocessor.process_delays_optimized')
    @patch('preprocessor.preprocessor.adjust_schedule_timeline')
    def test_save_processed_data_multiple_days(
        self, mock_adjust, mock_delays, mock_schedule
    ):
        """Test processing schedule that runs on multiple days."""
        # Mock data for a train running Monday-Friday
        mock_schedule.return_value = [
            {
                'TRAIN_SERVICE_CODE': 'TEST001',
                'PLANNED_CALLS': '0800',
                'ACTUAL_CALLS': '0800',
                'ENGLISH_DAY_TYPE': ['MO', 'TU', 'WE', 'TH', 'FR']
            }
        ]
        
        mock_delays.return_value = {}
        mock_adjust.return_value = mock_schedule.return_value
        
        result = save_processed_data_by_weekday_to_dataframe(
            '12345',
            schedule_data_loaded=pd.DataFrame([]),
            stanox_ref=pd.DataFrame([{'stanox': '12345', 'tiploc': 'TEST'}]),
            tiploc_to_stanox={'TEST': '12345'},
            incident_data_loaded={}
        )
        
        # Should have entries for all weekdays
        assert 'MO' in result
        assert 'TU' in result
        assert 'WE' in result
        assert 'TH' in result
        assert 'FR' in result
        # Each day should have one entry
        assert len(result['MO']) == 1
        assert len(result['FR']) == 1


# Integration-like tests (with mocking to avoid file I/O)
class TestDataProcessingFlow:
    """Test the flow of data processing."""
    
    @patch('preprocessor.preprocessor.os.makedirs')
    @patch('preprocessor.preprocessor.process_schedule')
    @patch('preprocessor.preprocessor.process_delays_optimized')
    @patch('preprocessor.preprocessor.adjust_schedule_timeline')
    def test_full_processing_flow(
        self, mock_adjust, mock_delays, mock_schedule, mock_makedirs
    ):
        """Test the complete data processing flow."""
        # Setup mock returns
        mock_schedule.return_value = [
            {
                'TRAIN_SERVICE_CODE': 'TEST001',
                'PLANNED_CALLS': '0800',
                'ACTUAL_CALLS': '0800',
                'ENGLISH_DAY_TYPE': ['MO', 'TU']
            }
        ]
        
        mock_delays.return_value = {}
        mock_adjust.return_value = mock_schedule.return_value
        
        # Run the processing
        result = save_processed_data_by_weekday_to_dataframe(
            '12345',
            schedule_data_loaded=pd.DataFrame([]),
            stanox_ref=pd.DataFrame([{'stanox': '12345', 'tiploc': 'TEST'}]),
            tiploc_to_stanox={'TEST': '12345'},
            incident_data_loaded={}
        )
        
        # Verify results
        assert result is not None
        assert isinstance(result, dict)
        assert len(result) >= 2  # At least MO and TU
        
        # Verify each DataFrame has the expected structure
        for day_code, df in result.items():
            assert isinstance(df, pd.DataFrame)
            assert len(df) > 0
            assert 'TRAIN_SERVICE_CODE' in df.columns
            assert 'WEEKDAY' in df.columns
            assert df['WEEKDAY'].iloc[0] == day_code


# Parametrized tests
@pytest.mark.parametrize("delay_day,expected_index", [
    ('MO', 0),
    ('TU', 1),
    ('WE', 2),
    ('TH', 3),
    ('FR', 4),
    ('SA', 5),
    ('SU', 6),
])
def test_weekday_extraction_parametrized(delay_day, expected_index):
    """Parametrized test for weekday extraction."""
    entry = {'DELAY_DAY': delay_day}
    result = get_weekday_from_schedule_entry(entry)
    assert result == expected_index
