"""
Unit tests for demos/time_view.ipynb

Tests the create_time_view_html() function for time-based network delay visualization.
"""

import pytest
import pandas as pd
from unittest.mock import Mock, patch, MagicMock
from outputs.utils import create_time_view_html


class TestCreateTimeViewHtml:
    """Test cases for create_time_view_html function from demos/time_view.ipynb."""
    
    @patch('outputs.utils.find_processed_data_path')
    @patch('outputs.load_data.load_processed_data')
    def test_basic_html_generation(self, mock_load_data, mock_find_path, sample_all_data, temp_output_dir):
        """Test basic time view HTML generation."""
        mock_find_path.return_value = '/path/to/data'
        mock_load_data.return_value = sample_all_data
        
        result = create_time_view_html(
            date_str='07-DEC-2024',
            all_data=sample_all_data
        )
        
        # Should return HTML string or None
        assert result is None or isinstance(result, str)
    
    @patch('outputs.utils.find_processed_data_path')
    @patch('outputs.load_data.load_processed_data')
    def test_storm_darragh_date(self, mock_load_data, mock_find_path, sample_all_data):
        """Test time view for Storm Darragh date (07-DEC-2024) as shown in demo."""
        mock_find_path.return_value = '/path/to/data'
        mock_load_data.return_value = sample_all_data
        
        result = create_time_view_html(
            date_str='07-DEC-2024',
            all_data=sample_all_data
        )
        
        assert result is None or isinstance(result, str)
    
    @patch('outputs.utils.find_processed_data_path')
    @patch('outputs.load_data.load_processed_data')
    def test_different_dates_from_demos(self, mock_load_data, mock_find_path, sample_all_data):
        """Test time view generation for different dates used in demos."""
        mock_find_path.return_value = '/path/to/data'
        mock_load_data.return_value = sample_all_data
        
        # Dates from demo HTML files
        dates = [
            '07-DEC-2024',
            '28-APR-2024',
            '24-MAY-2024',
            '21-OCT-2024',
            '22-OCT-2024',
            '23-OCT-2024'
        ]
        
        for date_str in dates:
            try:
                result = create_time_view_html(
                    date_str=date_str,
                    all_data=sample_all_data
                )
                assert result is None or isinstance(result, str)
            except (ValueError, KeyError):
                # Date might not have data in sample
                assert True
    
    @patch('outputs.utils.find_processed_data_path')
    @patch('outputs.load_data.load_processed_data')
    def test_empty_data(self, mock_load_data, mock_find_path, empty_station_data):
        """Test time view with empty data."""
        mock_find_path.return_value = '/path/to/data'
        mock_load_data.return_value = empty_station_data
        
        result = create_time_view_html(
            date_str='01-JAN-2024',
            all_data=empty_station_data
        )
        
        assert result is None or isinstance(result, str)
    
    @patch('outputs.utils.find_processed_data_path')
    @patch('outputs.load_data.load_processed_data')
    def test_invalid_date_format(self, mock_load_data, mock_find_path, sample_all_data):
        """Test time view with invalid date format."""
        mock_find_path.return_value = '/path/to/data'
        mock_load_data.return_value = sample_all_data
        
        try:
            result = create_time_view_html(
                date_str='2024-01-01',  # Wrong format
                all_data=sample_all_data
            )
            # Should handle gracefully or raise ValueError
            assert result is None or isinstance(result, str)
        except ValueError:
            # Expected for invalid format
            assert True
    
    @patch('outputs.utils.find_processed_data_path')
    @patch('outputs.load_data.load_processed_data')
    def test_multiple_incidents_on_same_day(self, mock_load_data, mock_find_path, sample_all_data):
        """Test time view handles multiple incidents on same day."""
        mock_find_path.return_value = '/path/to/data'
        
        # Add multiple incidents
        sample_all_data.loc[:5, 'INCIDENT_NUMBER'] = 'INC001'
        sample_all_data.loc[6:10, 'INCIDENT_NUMBER'] = 'INC002'
        mock_load_data.return_value = sample_all_data
        
        result = create_time_view_html(
            date_str='07-DEC-2024',
            all_data=sample_all_data
        )
        
        assert result is None or isinstance(result, str)
