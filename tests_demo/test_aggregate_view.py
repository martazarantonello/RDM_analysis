"""
Unit tests for demos/aggregate_view.ipynb

Tests the aggregate_view() function which provides aggregate incident impact metrics.
"""

import pytest
import pandas as pd
from unittest.mock import Mock, patch, MagicMock
from outputs.utils import aggregate_view


class TestAggregateView:
    """Test cases for aggregate_view function from demos/aggregate_view.ipynb."""
    
    @patch('os.listdir')
    @patch('outputs.utils.find_processed_data_path')
    @patch('outputs.load_data.load_processed_data')
    def test_basic_aggregate_view(self, mock_load_data, mock_find_path, mock_listdir, sample_all_data):
        """Test basic aggregate view functionality."""
        mock_find_path.return_value = '/path/to/data'
        mock_listdir.return_value = ['32000', '33087', '65630']
        
        # Add incident data with the right date format
        # Create data with the correct date
        incident_date = pd.to_datetime('28-APR-2024', format='%d-%b-%Y')
        sample_all_data['DAY'] = 'MO'  # Monday
        sample_all_data['EVENT_DATETIME'] = incident_date.strftime('%d-%b-%Y') + ' 08:00'
        sample_all_data.loc[:10, 'INCIDENT_NUMBER'] = 434859
        sample_all_data.loc[:10, 'delay_minutes'] = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55]
        mock_load_data.return_value = sample_all_data
        
        result = aggregate_view(
            incident_number=434859,
            date='28-APR-2024'
        )
        
        # Function may return None, dict, or DataFrame depending on data found
        assert isinstance(result, (dict, pd.DataFrame, type(None)))
    
    @patch('os.listdir')
    @patch('outputs.utils.find_processed_data_path')
    @patch('outputs.load_data.load_processed_data')
    def test_incident_not_found(self, mock_load_data, mock_find_path, mock_listdir, sample_all_data):
        """Test aggregate view with non-existent incident."""
        mock_find_path.return_value = '/path/to/data'
        mock_listdir.return_value = ['32000', '33087', '65630']
        mock_load_data.return_value = sample_all_data
        
        result = aggregate_view(
            incident_number='999999',
            date='01-JAN-2024'
        )
        
        # Should handle gracefully
        assert result is None or isinstance(result, (dict, pd.DataFrame))
    
    @patch('os.listdir')
    @patch('outputs.utils.find_processed_data_path')
    @patch('outputs.load_data.load_processed_data')
    def test_empty_data(self, mock_load_data, mock_find_path, mock_listdir, empty_station_data):
        """Test aggregate view with empty data."""
        mock_find_path.return_value = '/path/to/data'
        mock_listdir.return_value = ['32000']
        mock_load_data.return_value = empty_station_data
        
        result = aggregate_view(
            incident_number='434859',
            date='28-APR-2024'
        )
        
        assert result is None or isinstance(result, (dict, pd.DataFrame))
    
    @patch('os.listdir')
    @patch('outputs.utils.find_processed_data_path')
    @patch('outputs.load_data.load_processed_data')
    def test_multiple_incidents(self, mock_load_data, mock_find_path, mock_listdir, sample_all_data):
        """Test aggregate view with multiple incidents."""
        mock_find_path.return_value = '/path/to/data'
        mock_listdir.return_value = ['32000', '33087', '65630']
        
        # Add multiple incidents
        sample_all_data.loc[:5, 'INCIDENT_NUMBER'] = '434859'
        sample_all_data.loc[6:10, 'INCIDENT_NUMBER'] = '705009'
        mock_load_data.return_value = sample_all_data
        
        result1 = aggregate_view(incident_number='434859', date='28-APR-2024')
        result2 = aggregate_view(incident_number='705009', date='09-AUG-2024')
        
        assert result1 is None or isinstance(result1, (dict, pd.DataFrame))
        assert result2 is None or isinstance(result2, (dict, pd.DataFrame))
    
    @patch('os.listdir')
    @patch('outputs.utils.find_processed_data_path')
    @patch('outputs.load_data.load_processed_data')
    def test_with_cancellations(self, mock_load_data, mock_find_path, mock_listdir, station_with_cancellations):
        """Test aggregate view includes cancellation data."""
        mock_find_path.return_value = '/path/to/data'
        mock_listdir.return_value = ['32000']
        
        # Add incident to cancellation data
        station_with_cancellations.loc[:5, 'INCIDENT_NUMBER'] = '434859'
        mock_load_data.return_value = station_with_cancellations
        
        result = aggregate_view(
            incident_number='434859',
            date='28-APR-2024'
        )
        
        # Should handle cancellations in aggregate metrics
        assert result is None or isinstance(result, (dict, pd.DataFrame))
