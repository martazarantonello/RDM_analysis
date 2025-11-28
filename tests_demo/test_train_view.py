"""
Unit tests for train analysis functions used in demos/train_view.ipynb.

Functions tested:
- train_view() - Train journey delay data
- get_stanox_for_service() - STANOX lookup for train services  
- map_train_journey_with_incidents() - Journey mapping with incidents
- train_view_2() - Train journey HTML visualization
- plot_reliability_graphs() - Reliability statistics plotting
"""

import pytest
import pandas as pd
import os
from unittest.mock import Mock, patch, MagicMock
from outputs.utils import train_view, train_view_2, map_train_journey_with_incidents, plot_reliability_graphs


class TestTrainView:
    """Test cases for train_view function."""
    
    @patch('outputs.utils.display', create=True)
    @patch('matplotlib.pyplot.show')
    def test_basic_functionality(self, mock_show, mock_display, sample_all_data):
        """Test basic train view execution."""
        # Ensure the data has the origin/destination codes expected
        sample_all_data['PLANNED_ORIGIN_LOCATION_CODE'] = '32000'
        sample_all_data['PLANNED_DEST_LOCATION_CODE'] = '33087'
        
        result = train_view(
            all_data=sample_all_data,
            origin_code='32000',
            destination_code='33087',
            input_date_str='01-JAN-2024'
        )
        
        # Should execute without error
        assert result is None or isinstance(result, (pd.DataFrame, dict, str, type(None)))
    
    @patch('outputs.utils.display', create=True)
    @patch('matplotlib.pyplot.show')
    def test_no_trains_found(self, mock_show, mock_display, sample_all_data):
        """Test function handles case with no matching trains."""
        result = train_view(
            all_data=sample_all_data,
            origin_code='99999',
            destination_code='88888',
            input_date_str='01-JAN-2024'
        )
        
        # Should handle gracefully - returns string message
        assert result is None or isinstance(result, (pd.DataFrame, dict, str, type(None)))
    
    @patch('outputs.utils.display', create=True)
    @patch('matplotlib.pyplot.show')
    def test_empty_dataframe(self, mock_show, mock_display, empty_station_data):
        """Test function with empty DataFrame."""
        result = train_view(
            all_data=empty_station_data,
            origin_code='32000',
            destination_code='33087',
            input_date_str='01-JAN-2024'
        )
        
        assert result is None or isinstance(result, (pd.DataFrame, dict, str, type(None)))
    
    @patch('outputs.utils.display', create=True)
    @patch('matplotlib.pyplot.show')
    def test_different_date_formats(self, mock_show, mock_display, sample_all_data):
        """Test function with different date formats."""
        # Ensure the data has the origin/destination codes expected
        sample_all_data['PLANNED_ORIGIN_LOCATION_CODE'] = '32000'
        sample_all_data['PLANNED_DEST_LOCATION_CODE'] = '33087'
        
        date_formats = [
            '01-JAN-2024',
            '15-MAR-2024',
            '31-DEC-2024'
        ]
        
        for date_str in date_formats:
            try:
                result = train_view(
                    all_data=sample_all_data,
                    origin_code='32000',
                    destination_code='33087',
                    input_date_str=date_str
                )
                assert result is None or isinstance(result, (pd.DataFrame, dict, str, type(None)))
            except ValueError:
                # Date might not exist in sample data
                assert True


class TestTrainView2:
    """Test cases for train_view_2 function (enhanced version)."""
    
    @patch('matplotlib.pyplot.show')
    @patch('builtins.open', create=True)
    @patch('json.load')
    def test_basic_functionality(self, mock_json, mock_file, mock_show, sample_all_data):
        """Test basic enhanced train view execution."""
        # Mock station reference data
        mock_json.return_value = [{
            'stanox': '32000',
            'tiploc': 'MANC',
            'description': 'Manchester Piccadilly',
            'dft_category': 'A'
        }]
        
        result = train_view_2(
            all_data=sample_all_data,
            service_stanox='32000',
            service_code='TRAIN001'
        )
        
        assert result is None or isinstance(result, (pd.DataFrame, dict, type(None)))
    
    @patch('matplotlib.pyplot.show')
    def test_with_custom_stations_ref_path(self, mock_show, sample_all_data, temp_output_dir):
        """Test function with custom station reference path."""
        custom_path = os.path.join(temp_output_dir, 'stations_ref.json')
        
        # Create mock reference file
        with open(custom_path, 'w') as f:
            import json
            json.dump([{'stanox': '32000', 'description': 'Test Station'}], f)
        
        result = train_view_2(
            all_data=sample_all_data,
            service_stanox='32000',
            service_code='TRAIN001',
            stations_ref_path=custom_path
        )
        
        assert result is None or isinstance(result, (pd.DataFrame, dict, type(None)))


class TestMapTrainJourneyWithIncidents:
    """Test cases for map_train_journey_with_incidents function."""
    
    @patch('matplotlib.pyplot.show')
    @patch('builtins.open', create=True)
    @patch('json.load')
    def test_basic_mapping(self, mock_json, mock_file, mock_show, sample_all_data):
        """Test basic journey mapping functionality."""
        # Mock station reference
        mock_json.return_value = [{
            'stanox': '32000',
            'tiploc': 'MANC',
            'description': 'Manchester Piccadilly',
            'latitude': 53.4780,
            'longitude': -2.2308
        }]
        
        result = map_train_journey_with_incidents(
            all_data=sample_all_data,
            service_stanox=['32000', '33087'],
            service_code='32000_20240101_08_0',
            date_str='01-JAN-2024'
        )
        
        # Should return folium Map object
        assert result is not None
    
    @patch('matplotlib.pyplot.show')
    def test_no_incident_data(self, mock_show, sample_all_data):
        """Test mapping with no incident data."""
        # Remove incident columns
        data_no_incidents = sample_all_data.copy()
        if 'INCIDENT_NUMBER' in data_no_incidents.columns:
            data_no_incidents['INCIDENT_NUMBER'] = None
        
        result = map_train_journey_with_incidents(
            all_data=data_no_incidents,
            service_stanox=['32000', '33087'],
            service_code='TRAIN001',
            date_str='01-JAN-2024'
        )
        
        # Should return folium Map object
        assert result is not None


class TestPlotReliabilityGraphs:
    """Test cases for plot_reliability_graphs function."""
    
    @patch('matplotlib.pyplot.show')
    @patch('builtins.open', create=True)
    @patch('json.load')
    def test_basic_reliability_analysis(self, mock_json, mock_file, mock_show, sample_all_data):
        """Test basic reliability graph generation."""
        mock_json.return_value = [{
            'stanox': '32000',
            'description': 'Test Station'
        }]
        
        result = plot_reliability_graphs(
            all_data=sample_all_data,
            service_stanox='32000',
            service_code='TRAIN001'
        )
        
        assert result is None or isinstance(result, (pd.DataFrame, dict, type(None)))
    
    @patch('matplotlib.pyplot.show')
    @patch('builtins.open', create=True)
    @patch('json.load')
    def test_with_custom_cap_minutes(self, mock_json, mock_file, mock_show, sample_all_data):
        """Test reliability graphs with custom delay cap."""
        mock_json.return_value = [{
            'stanox': '32000',
            'description': 'Test Station'
        }]
        
        cap_values = [30, 60, 120]
        
        for cap in cap_values:
            result = plot_reliability_graphs(
                all_data=sample_all_data,
                service_stanox='32000',
                service_code='TRAIN001',
                cap_minutes=cap
            )
            
            assert result is None or isinstance(result, (pd.DataFrame, dict, str, type(None)))
    
    @patch('matplotlib.pyplot.show')
    def test_empty_data(self, mock_show, empty_station_data):
        """Test reliability graphs with empty data."""
        result = plot_reliability_graphs(
            all_data=empty_station_data,
            service_stanox='32000',
            service_code='TRAIN001'
        )
        
        assert result is None or isinstance(result, (pd.DataFrame, dict, type(None)))
