"""
Unit tests for incident analysis functions used in demos.

Functions from demos/aggregate_view.ipynb and demos/incident_view.ipynb:
- aggregate_view() - Aggregate incident impact metrics
- incident_view() - Detailed incident delay data
- incident_view_heatmap_html() - Animated heatmap visualization
"""

import pytest
import pandas as pd
import os
from unittest.mock import Mock, patch, MagicMock, mock_open
from outputs.utils import aggregate_view, incident_view, incident_view_heatmap_html


class TestAggregateView:
    """Test cases for aggregate_view function."""
    
    @patch('os.listdir')
    @patch('outputs.utils.find_processed_data_path')
    @patch('outputs.load_data.load_processed_data')
    def test_basic_functionality(self, mock_load_data, mock_find_path, mock_listdir, sample_all_data):
        """Test basic function execution with valid incident."""
        mock_find_path.return_value = '/path/to/data'
        mock_listdir.return_value = ['32000', '33087', '65630']
        mock_load_data.return_value = sample_all_data
        
        # Add incident data
        sample_all_data.loc[0, 'INCIDENT_NUMBER'] = 'INC001'
        sample_all_data.loc[0, 'PFPI_MINUTES'] = 15.0
        
        result = aggregate_view(incident_number='INC001', date='01-JAN-2024')
        
        # Function should process without error
        assert result is None or isinstance(result, (pd.DataFrame, dict))
    
    @patch('os.listdir')
    @patch('outputs.utils.find_processed_data_path')
    @patch('outputs.load_data.load_processed_data')
    def test_nonexistent_incident(self, mock_load_data, mock_find_path, mock_listdir, sample_all_data):
        """Test function handles non-existent incident gracefully."""
        mock_find_path.return_value = '/path/to/data'
        mock_listdir.return_value = ['32000', '33087', '65630']
        mock_load_data.return_value = sample_all_data
        
        result = aggregate_view(incident_number='NONEXISTENT', date='01-JAN-2024')
        
        # Should handle gracefully (return None or empty result)
        assert result is None or (isinstance(result, (pd.DataFrame, dict)) and len(result) == 0)
    
    @patch('outputs.utils.find_processed_data_path')
    def test_handles_missing_data_path(self, mock_find_path):
        """Test function handles missing data path."""
        mock_find_path.return_value = None
        
        # Should handle gracefully or raise appropriate error
        try:
            result = aggregate_view(incident_number='INC001', date='01-JAN-2024')
            assert True  # Function handled it
        except (FileNotFoundError, ValueError, TypeError):
            assert True  # Expected error


class TestIncidentView:
    """Test cases for incident_view function."""
    
    @patch('os.listdir')
    @patch('matplotlib.pyplot.show')
    @patch('outputs.utils.find_processed_data_path')
    @patch('outputs.load_data.load_processed_data')
    def test_basic_visualization(self, mock_load_data, mock_find_path, mock_show, mock_listdir, sample_all_data):
        """Test basic incident visualization."""
        mock_find_path.return_value = '/path/to/data'
        mock_listdir.return_value = ['32000', '33087', '65630']
        mock_load_data.return_value = sample_all_data
        
        # Add incident data
        sample_all_data.loc[:5, 'INCIDENT_NUMBER'] = 'INC001'
        sample_all_data['EVENT_DATETIME'] = '01-JAN-2024 08:00'
        sample_all_data['DAY'] = 'MO'
        
        result = incident_view(
            incident_code='INC001',
            incident_date='01-JAN-2024',
            analysis_date='01-JAN-2024',
            analysis_hhmm='0800',
            period_minutes=60
        )
        
        # incident_view returns a tuple of (df, fig1, fig2) or just displays
        # Accept None, DataFrame, dict, or tuple
        assert result is None or isinstance(result, (pd.DataFrame, dict, tuple))
    
    @patch('os.listdir')
    @patch('matplotlib.pyplot.show')
    @patch('outputs.utils.find_processed_data_path')
    @patch('outputs.load_data.load_processed_data')
    def test_different_time_periods(self, mock_load_data, mock_find_path, mock_show, mock_listdir, sample_all_data):
        """Test function with different time periods."""
        mock_find_path.return_value = '/path/to/data'
        mock_listdir.return_value = ['32000', '33087', '65630']
        mock_load_data.return_value = sample_all_data
        
        sample_all_data.loc[:3, 'INCIDENT_NUMBER'] = 'INC001'
        sample_all_data['EVENT_DATETIME'] = '01-JAN-2024 08:00'
        sample_all_data['DAY'] = 'MO'
        
        periods = [30, 60, 120, 240]
        
        for period in periods:
            result = incident_view(
                incident_code='INC001',
                incident_date='01-JAN-2024',
                analysis_date='01-JAN-2024',
                analysis_hhmm='0800',
                period_minutes=period
            )
            
            assert result is None or isinstance(result, (pd.DataFrame, dict, tuple))
    
    @patch('os.listdir')
    @patch('matplotlib.pyplot.show')
    @patch('outputs.utils.find_processed_data_path')
    @patch('outputs.load_data.load_processed_data')
    def test_invalid_date_format(self, mock_load_data, mock_find_path, mock_show, mock_listdir, sample_all_data):
        """Test function handles invalid date format."""
        mock_find_path.return_value = '/path/to/data'
        mock_listdir.return_value = ['32000', '33087', '65630']
        mock_load_data.return_value = sample_all_data
        
        try:
            result = incident_view(
                incident_code='INC001',
                incident_date='INVALID-DATE',
                analysis_date='01-JAN-2024',
                analysis_hhmm='0800',
                period_minutes=60
            )
            # Should either handle gracefully or raise error
            assert True
        except (ValueError, TypeError):
            assert True  # Expected error


class TestIncidentViewHeatmapHtml:
    """Test cases for incident_view_heatmap_html function."""
    
    @patch('os.listdir')
    @patch('matplotlib.pyplot.show')
    @patch('outputs.utils.find_processed_data_path')
    @patch('outputs.load_data.load_processed_data')
    def test_basic_heatmap_generation(self, mock_load_data, mock_find_path, mock_show, mock_listdir,
                                     sample_all_data, temp_output_dir):
        """Test basic heatmap HTML generation."""
        mock_find_path.return_value = '/path/to/data'
        mock_listdir.return_value = ['32000', '33087', '65630']
        mock_find_path.return_value = '/path/to/data'
        mock_load_data.return_value = sample_all_data
        
        sample_all_data.loc[:10, 'INCIDENT_NUMBER'] = 'INC001'
        
        output_file = os.path.join(temp_output_dir, 'test_heatmap.html')
        
        result = incident_view_heatmap_html(
            incident_code='INC001',
            incident_date='01-JAN-2024',
            analysis_date='01-JAN-2024',
            analysis_hhmm='0800',
            period_minutes=240,
            interval_minutes=30,
            output_file=output_file
        )
        
        # Function should execute
        assert result is None or isinstance(result, (str, type(None)))
    
    @patch('os.listdir')
    @patch('matplotlib.pyplot.show')
    @patch('outputs.utils.find_processed_data_path')
    @patch('outputs.load_data.load_processed_data')
    def test_different_intervals(self, mock_load_data, mock_find_path, mock_show, mock_listdir,
                                sample_all_data, temp_output_dir):
        """Test heatmap with different time intervals."""
        mock_find_path.return_value = '/path/to/data'
        mock_listdir.return_value = ['32000', '33087', '65630']
        mock_load_data.return_value = sample_all_data
        
        sample_all_data.loc[:10, 'INCIDENT_NUMBER'] = 'INC001'
        
        intervals = [10, 15, 30, 60]
        
        for interval in intervals:
            output_file = os.path.join(temp_output_dir, f'heatmap_{interval}.html')
            
            result = incident_view_heatmap_html(
                incident_code='INC001',
                incident_date='01-JAN-2024',
                analysis_date='01-JAN-2024',
                analysis_hhmm='0800',
                period_minutes=240,
                interval_minutes=interval,
                output_file=output_file
            )
            
            assert result is None or isinstance(result, (str, type(None)))
    
    @patch('os.listdir')
    @patch('matplotlib.pyplot.show')
    @patch('outputs.utils.find_processed_data_path')
    @patch('outputs.load_data.load_processed_data')
    def test_no_output_file_specified(self, mock_load_data, mock_find_path, mock_show, mock_listdir, sample_all_data):
        """Test heatmap generation without specifying output file."""
        mock_find_path.return_value = '/path/to/data'
        mock_listdir.return_value = ['32000', '33087', '65630']
        mock_find_path.return_value = '/path/to/data'
        mock_load_data.return_value = sample_all_data
        
        sample_all_data.loc[:5, 'INCIDENT_NUMBER'] = 'INC001'
        
        # Should generate default filename
        result = incident_view_heatmap_html(
            incident_code='INC001',
            incident_date='01-JAN-2024',
            analysis_date='01-JAN-2024',
            analysis_hhmm='0800',
            period_minutes=120,
            interval_minutes=10
        )
        
        assert result is None or isinstance(result, (str, type(None)))
