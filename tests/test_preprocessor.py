"""
Unit tests for preprocess/preprocessor.py main functions.

This module tests the main preprocessing functions including:
- Station loading
- Data processing and saving
- Category-based processing
- Save functions (for optimization validation)
"""

import pytest
import pandas as pd
import os
import json
import shutil
from unittest.mock import Mock, patch, MagicMock, mock_open
from preprocess.preprocessor import (
    get_weekday_from_schedule_entry,
    load_stations,
    save_processed_data_by_weekday_to_dataframe,
    save_stations_by_category
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
    
    @patch('preprocess.preprocessor.process_schedule')
    @patch('preprocess.preprocessor.process_delays_optimized')
    @patch('preprocess.preprocessor.adjust_schedule_timeline')
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
    
    @patch('preprocess.preprocessor.process_schedule')
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
    
    @patch('preprocess.preprocessor.process_schedule')
    @patch('preprocess.preprocessor.process_delays_optimized')
    @patch('preprocess.preprocessor.adjust_schedule_timeline')
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
    
    @patch('preprocess.preprocessor.os.makedirs')
    @patch('preprocess.preprocessor.process_schedule')
    @patch('preprocess.preprocessor.process_delays_optimized')
    @patch('preprocess.preprocessor.adjust_schedule_timeline')
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


# ============================================================================
# SAVE FUNCTIONS TESTS - For optimization validation
# ============================================================================


class TestSaveProcessedDataByWeekdayAdvanced:
    """Additional comprehensive test cases for save_processed_data_by_weekday_to_dataframe."""
    
    @patch('preprocess.preprocessor.process_schedule')
    @patch('preprocess.preprocessor.process_delays_optimized')
    @patch('preprocess.preprocessor.adjust_schedule_timeline')
    def test_deduplication_removes_identical_entries(
        self, mock_adjust, mock_delays, mock_schedule
    ):
        """Test that deduplication removes truly identical entries."""
        # Create schedule with duplicate entries
        duplicate_entry = {
            'TRAIN_SERVICE_CODE': 'TEST001',
            'PLANNED_CALLS': '0800',
            'ACTUAL_CALLS': '0800',
            'ENGLISH_DAY_TYPE': ['MO'],
            'PFPI_MINUTES': 0.0,
            'INCIDENT_REASON': None,
            'INCIDENT_NUMBER': None,
            'EVENT_TYPE': None,
            'SECTION_CODE': None,
            'DELAY_DAY': None,
            'EVENT_DATETIME': None,
            'INCIDENT_START_DATETIME': None,
            'PLANNED_ORIGIN_LOCATION_CODE': '12345',
            'PLANNED_ORIGIN_GBTT_DATETIME': '0800',
            'PLANNED_DEST_LOCATION_CODE': '67890',
            'PLANNED_DEST_GBTT_DATETIME': '0900',
            'STATION_ROLE': 'Origin',
            'DFT_CATEGORY': 'A',
            'PLATFORM_COUNT': 4
        }
        
        mock_schedule.return_value = [duplicate_entry]
        mock_delays.return_value = {}
        # Return 3 identical entries
        mock_adjust.return_value = [duplicate_entry.copy(), duplicate_entry.copy(), duplicate_entry.copy()]
        
        result = save_processed_data_by_weekday_to_dataframe(
            '12345',
            schedule_data_loaded=pd.DataFrame([]),
            stanox_ref=pd.DataFrame([{'stanox': '12345', 'tiploc': 'TEST'}]),
            tiploc_to_stanox={'TEST': '12345'},
            incident_data_loaded={}
        )
        
        # Should have only 1 entry after deduplication
        assert len(result['MO']) == 1
    
    @patch('preprocess.preprocessor.process_schedule')
    @patch('preprocess.preprocessor.process_delays_optimized')
    @patch('preprocess.preprocessor.adjust_schedule_timeline')
    def test_dataframes_have_correct_columns(
        self, mock_adjust, mock_delays, mock_schedule
    ):
        """Test that output DataFrames have all expected columns."""
        mock_schedule.return_value = [
            {
                'TRAIN_SERVICE_CODE': 'TEST001',
                'PLANNED_CALLS': '0800',
                'ACTUAL_CALLS': '0800',
                'ENGLISH_DAY_TYPE': ['MO'],
                'PFPI_MINUTES': 0.0,
                'INCIDENT_REASON': None,
                'INCIDENT_NUMBER': None,
                'EVENT_TYPE': None,
                'SECTION_CODE': None,
                'DELAY_DAY': None,
                'EVENT_DATETIME': None,
                'INCIDENT_START_DATETIME': None,
                'PLANNED_ORIGIN_LOCATION_CODE': '12345',
                'PLANNED_ORIGIN_GBTT_DATETIME': '0800',
                'PLANNED_DEST_LOCATION_CODE': '67890',
                'PLANNED_DEST_GBTT_DATETIME': '0900',
                'STATION_ROLE': 'Origin',
                'DFT_CATEGORY': 'A',
                'PLATFORM_COUNT': 4
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
        
        df = result['MO']
        
        # Check that key columns exist
        expected_columns = [
            'TRAIN_SERVICE_CODE',
            'PLANNED_CALLS',
            'ACTUAL_CALLS',
            'PFPI_MINUTES',
            'ENGLISH_DAY_TYPE',
            'STATION_ROLE',
            'DFT_CATEGORY',
            'PLATFORM_COUNT',
            'DATASET_TYPE',  # Added by the function
            'WEEKDAY'  # Added by the function
        ]
        
        for col in expected_columns:
            assert col in df.columns, f"Missing column: {col}"
    
    @patch('preprocess.preprocessor.process_schedule')
    @patch('preprocess.preprocessor.process_delays_optimized')
    @patch('preprocess.preprocessor.adjust_schedule_timeline')
    def test_entries_are_sorted_by_actual_calls(
        self, mock_adjust, mock_delays, mock_schedule
    ):
        """Test that entries in each DataFrame are sorted by ACTUAL_CALLS time."""
        mock_schedule.return_value = [
            {
                'TRAIN_SERVICE_CODE': 'TRAIN_LATE',
                'PLANNED_CALLS': '1200',
                'ACTUAL_CALLS': '1200',
                'ENGLISH_DAY_TYPE': ['MO'],
                'PFPI_MINUTES': 0.0,
                'INCIDENT_REASON': None,
                'INCIDENT_NUMBER': None,
                'EVENT_TYPE': None,
                'SECTION_CODE': None,
                'DELAY_DAY': None,
                'EVENT_DATETIME': None,
                'INCIDENT_START_DATETIME': None,
                'PLANNED_ORIGIN_LOCATION_CODE': '12345',
                'PLANNED_ORIGIN_GBTT_DATETIME': '1200',
                'PLANNED_DEST_LOCATION_CODE': '67890',
                'PLANNED_DEST_GBTT_DATETIME': '1300',
                'STATION_ROLE': 'Origin',
                'DFT_CATEGORY': 'A',
                'PLATFORM_COUNT': 4
            }
        ]
        
        mock_delays.return_value = {}
        
        # Return unsorted entries
        mock_adjust.return_value = [
            {
                'TRAIN_SERVICE_CODE': 'TRAIN_LATE',
                'PLANNED_CALLS': '1200',
                'ACTUAL_CALLS': '1200',
                'ENGLISH_DAY_TYPE': ['MO'],
                'PFPI_MINUTES': 0.0,
                'INCIDENT_REASON': None,
                'INCIDENT_NUMBER': None,
                'EVENT_TYPE': None,
                'SECTION_CODE': None,
                'DELAY_DAY': None,
                'EVENT_DATETIME': None,
                'INCIDENT_START_DATETIME': None,
                'PLANNED_ORIGIN_LOCATION_CODE': '12345',
                'PLANNED_ORIGIN_GBTT_DATETIME': '1200',
                'PLANNED_DEST_LOCATION_CODE': '67890',
                'PLANNED_DEST_GBTT_DATETIME': '1300',
                'STATION_ROLE': 'Origin',
                'DFT_CATEGORY': 'A',
                'PLATFORM_COUNT': 4
            },
            {
                'TRAIN_SERVICE_CODE': 'TRAIN_EARLY',
                'PLANNED_CALLS': '0600',
                'ACTUAL_CALLS': '0600',
                'ENGLISH_DAY_TYPE': ['MO'],
                'PFPI_MINUTES': 0.0,
                'INCIDENT_REASON': None,
                'INCIDENT_NUMBER': None,
                'EVENT_TYPE': None,
                'SECTION_CODE': None,
                'DELAY_DAY': None,
                'EVENT_DATETIME': None,
                'INCIDENT_START_DATETIME': None,
                'PLANNED_ORIGIN_LOCATION_CODE': '12345',
                'PLANNED_ORIGIN_GBTT_DATETIME': '0600',
                'PLANNED_DEST_LOCATION_CODE': '67890',
                'PLANNED_DEST_GBTT_DATETIME': '0700',
                'STATION_ROLE': 'Origin',
                'DFT_CATEGORY': 'A',
                'PLATFORM_COUNT': 4
            },
            {
                'TRAIN_SERVICE_CODE': 'TRAIN_MID',
                'PLANNED_CALLS': '0900',
                'ACTUAL_CALLS': '0900',
                'ENGLISH_DAY_TYPE': ['MO'],
                'PFPI_MINUTES': 0.0,
                'INCIDENT_REASON': None,
                'INCIDENT_NUMBER': None,
                'EVENT_TYPE': None,
                'SECTION_CODE': None,
                'DELAY_DAY': None,
                'EVENT_DATETIME': None,
                'INCIDENT_START_DATETIME': None,
                'PLANNED_ORIGIN_LOCATION_CODE': '12345',
                'PLANNED_ORIGIN_GBTT_DATETIME': '0900',
                'PLANNED_DEST_LOCATION_CODE': '67890',
                'PLANNED_DEST_GBTT_DATETIME': '1000',
                'STATION_ROLE': 'Origin',
                'DFT_CATEGORY': 'A',
                'PLATFORM_COUNT': 4
            }
        ]
        
        result = save_processed_data_by_weekday_to_dataframe(
            '12345',
            schedule_data_loaded=pd.DataFrame([]),
            stanox_ref=pd.DataFrame([{'stanox': '12345', 'tiploc': 'TEST'}]),
            tiploc_to_stanox={'TEST': '12345'},
            incident_data_loaded={}
        )
        
        df = result['MO']
        
        # Check that entries are sorted
        actual_calls = df['ACTUAL_CALLS'].tolist()
        # Convert to integers for comparison
        actual_calls_int = [int(x) if str(x).isdigit() else 0 for x in actual_calls]
        
        assert actual_calls_int == sorted(actual_calls_int), "Entries should be sorted by ACTUAL_CALLS"


class TestSaveStationsByCategory:
    """Test cases for save_stations_by_category function."""
    
    @patch('preprocess.preprocessor.load_stations')
    @patch('preprocess.preprocessor.load_schedule_data_once')
    @patch('preprocess.preprocessor.load_incident_data_once')
    @patch('preprocess.preprocessor.save_processed_data_by_weekday_to_dataframe')
    def test_returns_summary_dict(
        self, mock_save_func, mock_incident, mock_schedule, mock_load_stations, tmp_path
    ):
        """Test that function returns a summary dictionary with expected structure."""
        # Mock station loading
        mock_load_stations.return_value = ['12345', '67890']
        
        # Mock data loading
        mock_schedule.return_value = (pd.DataFrame([]), pd.DataFrame([]), {})
        mock_incident.return_value = {}
        
        # Mock successful processing
        mock_save_func.return_value = {
            'MO': pd.DataFrame([{'test': 'data'}]),
            'TU': pd.DataFrame([{'test': 'data'}])
        }
        
        output_dir = str(tmp_path / "test_output")
        result = save_stations_by_category(category='A', output_dir=output_dir)
        
        assert isinstance(result, dict)
        assert 'category' in result
        assert 'successful_stations' in result
        assert 'failed_stations' in result
        assert 'total_entries_by_station' in result
        assert 'files_created' in result
        
        assert result['category'] == 'A'
        assert len(result['successful_stations']) == 2
        assert len(result['failed_stations']) == 0
    
    @patch('preprocess.preprocessor.load_stations')
    def test_returns_none_when_no_stations_found(self, mock_load_stations):
        """Test that function returns None when no stations are found."""
        mock_load_stations.return_value = []
        
        result = save_stations_by_category(category='Z')
        
        assert result is None
    
    @patch('preprocess.preprocessor.load_stations')
    @patch('preprocess.preprocessor.load_schedule_data_once')
    @patch('preprocess.preprocessor.load_incident_data_once')
    @patch('preprocess.preprocessor.save_processed_data_by_weekday_to_dataframe')
    def test_handles_processing_failures_gracefully(
        self, mock_save_func, mock_incident, mock_schedule, mock_load_stations, tmp_path
    ):
        """Test that function handles failures for individual stations."""
        mock_load_stations.return_value = ['12345', '67890', '99999']
        
        mock_schedule.return_value = (pd.DataFrame([]), pd.DataFrame([]), {})
        mock_incident.return_value = {}
        
        # First station succeeds, second fails, third returns None
        mock_save_func.side_effect = [
            {'MO': pd.DataFrame([{'test': 'data'}])},  # Success
            Exception("Processing error"),  # Failure
            None  # No data
        ]
        
        output_dir = str(tmp_path / "test_output")
        result = save_stations_by_category(category='A', output_dir=output_dir)
        
        assert len(result['successful_stations']) == 1
        assert len(result['failed_stations']) == 2
        assert '12345' in result['successful_stations']
        assert '67890' in result['failed_stations']
        assert '99999' in result['failed_stations']
    
    @patch('preprocess.preprocessor.load_stations')
    @patch('preprocess.preprocessor.load_schedule_data_once')
    @patch('preprocess.preprocessor.load_incident_data_once')
    @patch('preprocess.preprocessor.save_processed_data_by_weekday_to_dataframe')
    def test_creates_parquet_files_for_each_day(
        self, mock_save_func, mock_incident, mock_schedule, mock_load_stations, tmp_path
    ):
        """Test that parquet files are created for each day of the week."""
        mock_load_stations.return_value = ['12345']
        
        mock_schedule.return_value = (pd.DataFrame([]), pd.DataFrame([]), {})
        mock_incident.return_value = {}
        
        # Return data for multiple days
        mock_save_func.return_value = {
            'MO': pd.DataFrame([{'TRAIN_SERVICE_CODE': 'TEST001'}]),
            'TU': pd.DataFrame([{'TRAIN_SERVICE_CODE': 'TEST002'}]),
            'WE': pd.DataFrame([{'TRAIN_SERVICE_CODE': 'TEST003'}])
        }
        
        output_dir = str(tmp_path / "test_output")
        result = save_stations_by_category(category='A', output_dir=output_dir)
        
        # Check that files were created
        station_folder = os.path.join(output_dir, '12345')
        assert os.path.exists(station_folder)
        
        # Check for parquet files
        assert os.path.exists(os.path.join(station_folder, 'MO.parquet'))
        assert os.path.exists(os.path.join(station_folder, 'TU.parquet'))
        assert os.path.exists(os.path.join(station_folder, 'WE.parquet'))
        
        # Verify files are readable
        df_mo = pd.read_parquet(os.path.join(station_folder, 'MO.parquet'))
        assert len(df_mo) == 1
        assert df_mo['TRAIN_SERVICE_CODE'].iloc[0] == 'TEST001'
    
    @patch('preprocess.preprocessor.load_stations')
    @patch('preprocess.preprocessor.load_schedule_data_once')
    @patch('preprocess.preprocessor.load_incident_data_once')
    @patch('preprocess.preprocessor.save_processed_data_by_weekday_to_dataframe')
    def test_cleans_up_existing_station_folders(
        self, mock_save_func, mock_incident, mock_schedule, mock_load_stations, tmp_path
    ):
        """Test that existing station folders are removed before processing."""
        mock_load_stations.return_value = ['12345']
        
        mock_schedule.return_value = (pd.DataFrame([]), pd.DataFrame([]), {})
        mock_incident.return_value = {}
        
        mock_save_func.return_value = {
            'MO': pd.DataFrame([{'test': 'new_data'}])
        }
        
        output_dir = str(tmp_path / "test_output")
        station_folder = os.path.join(output_dir, '12345')
        
        # Create existing folder with old file
        os.makedirs(station_folder, exist_ok=True)
        old_file = os.path.join(station_folder, 'old_file.txt')
        with open(old_file, 'w') as f:
            f.write("old data")
        
        # Run the function
        result = save_stations_by_category(category='A', output_dir=output_dir)
        
        # Old file should be gone
        assert not os.path.exists(old_file)
        
        # New file should exist
        assert os.path.exists(os.path.join(station_folder, 'MO.parquet'))


# ============================================================================
# BASELINE COMPARISON TESTS
# These tests should be run BEFORE and AFTER optimization to verify correctness
# ============================================================================

class TestBaselineComparison:
    """
    Tests to establish baseline outputs before optimization.
    
    Run these tests before optimization and save the outputs.
    After optimization, run again and compare outputs to ensure correctness.
    """
    
    @pytest.fixture
    def create_baseline_data(self, tmp_path):
        """Create baseline test data for comparison."""
        # This would use real data files if available
        # For now, we'll use mock data
        pass
    
    def test_save_baseline_output(self, tmp_path):
        """
        Save baseline output to file for comparison after optimization.
        
        Usage:
        1. Run this test BEFORE optimization
        2. Implement optimizations
        3. Run comparison test to verify outputs match
        """
        # This test would process a small subset of real data
        # and save the output for comparison
        # Implementation depends on having access to real data files
        pass
    
    def test_compare_with_baseline(self, tmp_path):
        """
        Compare current output with saved baseline.
        
        This test should fail if the optimization changes the output.
        """
        # Load baseline output
        # Run optimized function
        # Compare outputs (should be identical except for column order)
        pass

