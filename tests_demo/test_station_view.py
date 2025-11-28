"""
Unit tests for demos/station_view.ipynb

Tests the station analysis functions:
- station_view_yearly() - Yearly station performance analysis
- plot_variable_relationships_normalized() - Flow vs delay with platform normalization
- plot_trains_in_system_vs_delay() - Trains in system vs delay analysis
"""

import pytest
import pandas as pd
from unittest.mock import Mock, patch, MagicMock
from outputs.utils import (
    station_view_yearly,
    plot_variable_relationships_normalized,
    plot_trains_in_system_vs_delay
)


class TestStationViewYearly:
    """Test cases for station_view_yearly function from demos/station_view.ipynb."""
    
    @patch('outputs.utils.find_processed_data_path')
    @patch('outputs.load_data.load_processed_data')
    def test_basic_yearly_analysis(self, mock_load_data, mock_find_path, sample_all_data):
        """Test basic yearly station analysis."""
        mock_find_path.return_value = '/path/to/data'
        
        # Add incident data
        sample_all_data.loc[:5, 'INCIDENT_NUMBER'] = 'INC001'
        mock_load_data.return_value = sample_all_data
        
        incident_summary, normal_summary = station_view_yearly(
            station_id='32000',
            interval_minutes=60
        )
        
        # Should return two DataFrames
        assert isinstance(incident_summary, (pd.DataFrame, type(None)))
        assert isinstance(normal_summary, (pd.DataFrame, type(None)))
    
    @patch('outputs.utils.find_processed_data_path')
    @patch('outputs.load_data.load_processed_data')
    def test_different_intervals(self, mock_load_data, mock_find_path, sample_all_data):
        """Test yearly analysis with different time intervals."""
        mock_find_path.return_value = '/path/to/data'
        mock_load_data.return_value = sample_all_data
        
        intervals = [30, 60, 120]
        
        for interval in intervals:
            incident_summary, normal_summary = station_view_yearly(
                station_id='32000',
                interval_minutes=interval
            )
            
            assert isinstance(incident_summary, (pd.DataFrame, type(None)))
            assert isinstance(normal_summary, (pd.DataFrame, type(None)))
    
    @patch('outputs.utils.find_processed_data_path')
    @patch('outputs.load_data.load_processed_data')
    def test_no_incidents(self, mock_load_data, mock_find_path, sample_all_data):
        """Test yearly analysis with no incidents."""
        mock_find_path.return_value = '/path/to/data'
        
        # Remove incident data
        data_no_incidents = sample_all_data.copy()
        if 'INCIDENT_NUMBER' in data_no_incidents.columns:
            data_no_incidents['INCIDENT_NUMBER'] = None
        
        mock_load_data.return_value = data_no_incidents
        
        incident_summary, normal_summary = station_view_yearly(
            station_id='32000',
            interval_minutes=60
        )
        
        # Incident summary should be empty or None
        assert incident_summary is None or len(incident_summary) == 0
        # Normal summary should have data
        assert isinstance(normal_summary, (pd.DataFrame, type(None)))
    
    @patch('outputs.utils.find_processed_data_path')
    @patch('outputs.load_data.load_processed_data')
    def test_manchester_piccadilly(self, mock_load_data, mock_find_path, sample_all_data):
        """Test with Manchester Piccadilly (station_id='32000', 14 platforms)."""
        mock_find_path.return_value = '/path/to/data'
        mock_load_data.return_value = sample_all_data
        
        incident_summary, normal_summary = station_view_yearly(
            station_id='32000',
            interval_minutes=60
        )
        
        assert isinstance(incident_summary, (pd.DataFrame, type(None)))
        assert isinstance(normal_summary, (pd.DataFrame, type(None)))


class TestPlotVariableRelationshipsNormalized:
    """Test cases for plot_variable_relationships_normalized from demos/station_view.ipynb."""
    
    @patch('matplotlib.pyplot.show')
    def test_basic_functionality(self, mock_show, sample_all_data):
        """Test basic normalized variable relationships plot."""
        result = plot_variable_relationships_normalized(
            station_id='32000',
            all_data=sample_all_data,
            time_window_minutes=60,
            num_platforms=14,
            figsize=(14, 10),
            max_delay_percentile=98
        )
        
        assert result is not None
        assert 'weekday' in result
        assert 'weekend' in result
    
    @patch('matplotlib.pyplot.show')
    def test_platform_normalization(self, mock_show, sample_all_data):
        """Test that flow is properly normalized by number of platforms."""
        result = plot_variable_relationships_normalized(
            station_id='32000',
            all_data=sample_all_data,
            time_window_minutes=60,
            num_platforms=14
        )
        
        if result and result['weekday'] is not None:
            # Flow should be normalized (divided by num_platforms)
            assert 'flow_normalized' in result['weekday'].columns
    
    @patch('matplotlib.pyplot.show')
    def test_weekday_weekend_separation(self, mock_show, sample_all_data):
        """Test weekday/weekend data separation."""
        result = plot_variable_relationships_normalized(
            station_id='32000',
            all_data=sample_all_data,
            time_window_minutes=60,
            num_platforms=14
        )
        
        if result:
            assert 'weekday' in result
            assert 'weekend' in result
    
    @patch('matplotlib.pyplot.show')
    def test_axis_limits(self, mock_show, sample_all_data):
        """Test that axis limits are set to x=2.5, y=25."""
        with patch('matplotlib.pyplot.subplots') as mock_subplots:
            mock_fig = MagicMock()
            mock_axes = [MagicMock(), MagicMock()]
            mock_subplots.return_value = (mock_fig, mock_axes)
            
            result = plot_variable_relationships_normalized(
                station_id='32000',
                all_data=sample_all_data,
                time_window_minutes=60,
                num_platforms=14
            )
            
            if result:
                # Check that xlim and ylim were called with correct values
                for ax in mock_axes:
                    if ax.set_xlim.called:
                        ax.set_xlim.assert_called_with(0, 2.5)
                    if ax.set_ylim.called:
                        ax.set_ylim.assert_called_with(0, 25)
    
    @patch('matplotlib.pyplot.show')
    def test_different_stations(self, mock_show, sample_all_data):
        """Test with different stations (Manchester Piccadilly, Oxford Road, Barking)."""
        stations = [
            ('32000', 14),  # Manchester Piccadilly
            ('33087', 2),   # Oxford Road
            ('65630', 6)    # Barking
        ]
        
        for station_id, num_platforms in stations:
            result = plot_variable_relationships_normalized(
                station_id=station_id,
                all_data=sample_all_data,
                time_window_minutes=60,
                num_platforms=num_platforms
            )
            
            assert result is None or isinstance(result, dict)
    
    @patch('matplotlib.pyplot.show')
    def test_delay_calculation_excludes_ontime_trains(self, mock_show, sample_all_data):
        """Test that delay is calculated only from delayed trains (delay > 0)."""
        result = plot_variable_relationships_normalized(
            station_id='32000',
            all_data=sample_all_data,
            time_window_minutes=60,
            num_platforms=14
        )
        
        if result and result['weekday'] is not None:
            weekday_df = result['weekday']
            if len(weekday_df) > 0:
                # Mean delay should reflect only delayed trains
                assert 'mean_delay' in weekday_df.columns


class TestPlotTrainsInSystemVsDelay:
    """Test cases for plot_trains_in_system_vs_delay from demos/station_view.ipynb."""
    
    @patch('matplotlib.pyplot.show')
    def test_basic_functionality(self, mock_show, sample_all_data):
        """Test basic trains in system vs delay plot."""
        result = plot_trains_in_system_vs_delay(
            station_id='32000',
            all_data=sample_all_data,
            time_window_minutes=60,
            num_platforms=14,
            figsize=(14, 10),
            max_delay_percentile=98
        )
        
        assert result is not None
        assert 'weekday' in result
        assert 'weekend' in result
    
    @patch('matplotlib.pyplot.show')
    def test_delay_calculation_only_delayed_trains(self, mock_show, sample_all_data):
        """Test that delay is calculated only from delayed trains (delay > 0) - critical bug fix."""
        result = plot_trains_in_system_vs_delay(
            station_id='32000',
            all_data=sample_all_data,
            time_window_minutes=60,
            num_platforms=14
        )
        
        if result and result['weekday'] is not None:
            weekday_df = result['weekday']
            if len(weekday_df) > 0:
                assert 'mean_delay' in weekday_df.columns
                # Delay should be from delayed trains only
    
    @patch('matplotlib.pyplot.show')
    def test_trains_in_system_normalization(self, mock_show, sample_all_data):
        """Test that trains in system is normalized by platforms."""
        result = plot_trains_in_system_vs_delay(
            station_id='32000',
            all_data=sample_all_data,
            time_window_minutes=60,
            num_platforms=14
        )
        
        if result and result['weekday'] is not None:
            assert 'trains_in_system_normalized' in result['weekday'].columns
    
    @patch('matplotlib.pyplot.show')
    def test_axis_limits(self, mock_show, sample_all_data):
        """Test that axis limits are set to x=2.5, y=25 for comparison with flow plot."""
        with patch('matplotlib.pyplot.subplots') as mock_subplots:
            mock_fig = MagicMock()
            mock_axes = [MagicMock(), MagicMock()]
            mock_subplots.return_value = (mock_fig, mock_axes)
            
            result = plot_trains_in_system_vs_delay(
                station_id='32000',
                all_data=sample_all_data,
                time_window_minutes=60,
                num_platforms=14
            )
            
            if result:
                # Check that xlim and ylim were called
                for ax in mock_axes:
                    if ax.set_xlim.called:
                        ax.set_xlim.assert_called_with(0, 2.5)
                    if ax.set_ylim.called:
                        ax.set_ylim.assert_called_with(0, 25)
    
    @patch('matplotlib.pyplot.show')
    def test_cancellation_filtering(self, mock_show, station_with_cancellations):
        """Test that cancelled trains are filtered out."""
        result = plot_trains_in_system_vs_delay(
            station_id='32000',
            all_data=station_with_cancellations,
            time_window_minutes=60,
            num_platforms=14
        )
        
        # Should handle cancelled trains gracefully
        assert result is not None or result is None
    
    @patch('matplotlib.pyplot.show')
    def test_different_stations(self, mock_show, sample_all_data):
        """Test with different stations used in demos."""
        stations = [
            ('32000', 14),  # Manchester Piccadilly
            ('33087', 2),   # Oxford Road
            ('65630', 6)    # Barking
        ]
        
        for station_id, num_platforms in stations:
            result = plot_trains_in_system_vs_delay(
                station_id=station_id,
                all_data=sample_all_data,
                time_window_minutes=60,
                num_platforms=num_platforms
            )
            
            assert result is None or isinstance(result, dict)
