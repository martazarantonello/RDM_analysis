"""
Baseline Comparison Utility

This script helps establish a baseline for the save functions BEFORE optimization
and compare outputs AFTER optimization to ensure correctness.

Usage:
    1. Before optimization:
       python tests/baseline_comparison.py save --stations 12345,67890

    2. After optimization:
       python tests/baseline_comparison.py compare --stations 12345,67890

    3. View differences:
       python tests/baseline_comparison.py diff --stations 12345,67890
"""

import os
import sys
import json
import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from preprocess.preprocessor import save_processed_data_by_weekday_to_dataframe, save_stations_by_category
from preprocess.utils import load_schedule_data_once, load_incident_data_once
from data.schedule import schedule_data
from data.reference import reference_files
from data.incidents import incident_files


class BaselineManager:
    """Manage baseline outputs for comparison."""
    
    def __init__(self, baseline_dir="tests/baselines"):
        self.baseline_dir = Path(baseline_dir)
        self.baseline_dir.mkdir(exist_ok=True, parents=True)
    
    def save_baseline(self, station_codes, optimized=False):
        """
        Save baseline output for specified stations.
        
        Args:
            station_codes: List of STANOX codes to process
            optimized: If True, save to 'optimized' subdirectory
        """
        output_subdir = "optimized" if optimized else "original"
        output_dir = self.baseline_dir / output_subdir
        output_dir.mkdir(exist_ok=True, parents=True)
        
        print(f"Saving baseline for {len(station_codes)} stations to {output_dir}")
        print("="*60)
        
        # Pre-load data once
        print("Pre-loading schedule and incident data...")
        schedule_data_loaded, stanox_ref, tiploc_to_stanox = load_schedule_data_once(
            schedule_data, reference_files
        )
        incident_data_loaded = load_incident_data_once(incident_files)
        
        results = {}
        
        for i, st_code in enumerate(station_codes, 1):
            print(f"\nProcessing station {i}/{len(station_codes)}: {st_code}")
            
            try:
                # Process the station
                start_time = datetime.now()
                
                station_data = save_processed_data_by_weekday_to_dataframe(
                    st_code,
                    schedule_data_loaded=schedule_data_loaded,
                    stanox_ref=stanox_ref,
                    tiploc_to_stanox=tiploc_to_stanox,
                    incident_data_loaded=incident_data_loaded
                )
                
                end_time = datetime.now()
                processing_time = (end_time - start_time).total_seconds()
                
                if station_data:
                    # Save each day's DataFrame
                    station_folder = output_dir / st_code
                    station_folder.mkdir(exist_ok=True, parents=True)
                    
                    day_counts = {}
                    for day_code, df in station_data.items():
                        # Save as parquet
                        parquet_file = station_folder / f"{day_code}.parquet"
                        df.to_parquet(parquet_file, index=False)
                        
                        # Also save as CSV for easier inspection
                        csv_file = station_folder / f"{day_code}.csv"
                        df.to_csv(csv_file, index=False)
                        
                        day_counts[day_code] = len(df)
                    
                    # Save metadata
                    metadata = {
                        'station_code': st_code,
                        'processing_time_seconds': processing_time,
                        'day_counts': day_counts,
                        'total_entries': sum(day_counts.values()),
                        'timestamp': datetime.now().isoformat(),
                        'optimized': optimized
                    }
                    
                    with open(station_folder / 'metadata.json', 'w') as f:
                        json.dump(metadata, f, indent=2)
                    
                    results[st_code] = {
                        'status': 'success',
                        'processing_time': processing_time,
                        'total_entries': sum(day_counts.values())
                    }
                    
                    print(f"  ✓ Saved {sum(day_counts.values())} total entries")
                    print(f"  ⏱ Processing time: {processing_time:.2f} seconds")
                else:
                    results[st_code] = {'status': 'no_data'}
                    print(f"  ⚠ No data found for station {st_code}")
                    
            except Exception as e:
                results[st_code] = {'status': 'error', 'error': str(e)}
                print(f"  ✗ Error: {str(e)}")
        
        # Save overall summary
        summary_file = output_dir / 'summary.json'
        with open(summary_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        print("\n" + "="*60)
        print("Baseline save complete!")
        print(f"Results saved to: {output_dir}")
        
        # Print summary
        successful = sum(1 for r in results.values() if r['status'] == 'success')
        print(f"\nSuccessful: {successful}/{len(station_codes)}")
        
        if successful > 0:
            total_time = sum(r['processing_time'] for r in results.values() if r['status'] == 'success')
            avg_time = total_time / successful
            print(f"Average processing time: {avg_time:.2f} seconds per station")
            print(f"Total processing time: {total_time:.2f} seconds")
        
        return results
    
    def compare_baselines(self, station_codes):
        """
        Compare original and optimized baselines.
        
        Args:
            station_codes: List of STANOX codes to compare
            
        Returns:
            dict: Comparison results
        """
        original_dir = self.baseline_dir / "original"
        optimized_dir = self.baseline_dir / "optimized"
        
        if not original_dir.exists():
            print("Error: Original baseline not found. Run 'save' command first.")
            return None
        
        if not optimized_dir.exists():
            print("Error: Optimized baseline not found. Run 'save --optimized' command first.")
            return None
        
        print("Comparing baselines...")
        print("="*60)
        
        comparison_results = {}
        
        for st_code in station_codes:
            print(f"\nComparing station: {st_code}")
            
            original_folder = original_dir / st_code
            optimized_folder = optimized_dir / st_code
            
            if not original_folder.exists():
                print(f"  ⚠ Original baseline missing for {st_code}")
                comparison_results[st_code] = {'status': 'missing_original'}
                continue
            
            if not optimized_folder.exists():
                print(f"  ⚠ Optimized baseline missing for {st_code}")
                comparison_results[st_code] = {'status': 'missing_optimized'}
                continue
            
            # Load metadata
            with open(original_folder / 'metadata.json', 'r') as f:
                original_meta = json.load(f)
            
            with open(optimized_folder / 'metadata.json', 'r') as f:
                optimized_meta = json.load(f)
            
            # Compare processing times
            original_time = original_meta['processing_time_seconds']
            optimized_time = optimized_meta['processing_time_seconds']
            speedup = original_time / optimized_time if optimized_time > 0 else 0
            
            print(f"  Processing time:")
            print(f"    Original:  {original_time:.2f}s")
            print(f"    Optimized: {optimized_time:.2f}s")
            print(f"    Speedup:   {speedup:.2f}x")
            
            # Compare data
            day_comparisons = {}
            all_days_match = True
            
            for day_code in original_meta['day_counts'].keys():
                original_file = original_folder / f"{day_code}.parquet"
                optimized_file = optimized_folder / f"{day_code}.parquet"
                
                if not original_file.exists() or not optimized_file.exists():
                    day_comparisons[day_code] = {'status': 'missing_file'}
                    all_days_match = False
                    continue
                
                # Load and compare DataFrames
                df_original = pd.read_parquet(original_file)
                df_optimized = pd.read_parquet(optimized_file)
                
                # Sort both by same columns for fair comparison
                sort_cols = ['TRAIN_SERVICE_CODE', 'PLANNED_CALLS']
                if all(col in df_original.columns for col in sort_cols):
                    df_original = df_original.sort_values(sort_cols).reset_index(drop=True)
                if all(col in df_optimized.columns for col in sort_cols):
                    df_optimized = df_optimized.sort_values(sort_cols).reset_index(drop=True)
                
                # Compare
                differences = []
                
                # Check row counts
                if len(df_original) != len(df_optimized):
                    differences.append(f"Row count mismatch: {len(df_original)} vs {len(df_optimized)}")
                
                # Check columns
                original_cols = set(df_original.columns)
                optimized_cols = set(df_optimized.columns)
                
                if original_cols != optimized_cols:
                    missing = original_cols - optimized_cols
                    extra = optimized_cols - original_cols
                    if missing:
                        differences.append(f"Missing columns: {missing}")
                    if extra:
                        differences.append(f"Extra columns: {extra}")
                
                # Check data values (for common columns)
                common_cols = original_cols & optimized_cols
                for col in common_cols:
                    try:
                        # Use pandas comparison with NaN handling
                        if not df_original[col].equals(df_optimized[col]):
                            # Check if differences are just floating point precision
                            if df_original[col].dtype in [np.float64, np.float32]:
                                if not np.allclose(df_original[col], df_optimized[col], 
                                                  rtol=1e-5, atol=1e-8, equal_nan=True):
                                    differences.append(f"Column '{col}' has different values")
                            else:
                                differences.append(f"Column '{col}' has different values")
                    except Exception as e:
                        differences.append(f"Error comparing column '{col}': {str(e)}")
                
                if differences:
                    day_comparisons[day_code] = {
                        'status': 'mismatch',
                        'differences': differences
                    }
                    all_days_match = False
                    print(f"  ✗ {day_code}: Data mismatch")
                    for diff in differences:
                        print(f"      - {diff}")
                else:
                    day_comparisons[day_code] = {'status': 'match'}
                    print(f"  ✓ {day_code}: Data matches")
            
            comparison_results[st_code] = {
                'status': 'complete',
                'data_matches': all_days_match,
                'original_time': original_time,
                'optimized_time': optimized_time,
                'speedup': speedup,
                'day_comparisons': day_comparisons
            }
        
        # Save comparison results
        comparison_file = self.baseline_dir / 'comparison_results.json'
        with open(comparison_file, 'w') as f:
            json.dump(comparison_results, f, indent=2)
        
        print("\n" + "="*60)
        print("Comparison complete!")
        print(f"Results saved to: {comparison_file}")
        
        # Print summary
        print("\nSummary:")
        all_match = all(r.get('data_matches', False) for r in comparison_results.values() 
                       if r.get('status') == 'complete')
        
        if all_match:
            print("  ✓ All data matches between original and optimized!")
        else:
            print("  ✗ Some data mismatches found. Review details above.")
        
        # Calculate average speedup
        speedups = [r['speedup'] for r in comparison_results.values() 
                   if r.get('status') == 'complete']
        if speedups:
            avg_speedup = sum(speedups) / len(speedups)
            print(f"\n  Average speedup: {avg_speedup:.2f}x")
        
        return comparison_results


def main():
    parser = argparse.ArgumentParser(description='Baseline comparison utility for save functions')
    parser.add_argument('command', choices=['save', 'compare', 'diff'],
                       help='Command to run: save, compare, or diff')
    parser.add_argument('--stations', type=str, required=True,
                       help='Comma-separated list of STANOX codes to process')
    parser.add_argument('--optimized', action='store_true',
                       help='Use optimized version (for save command)')
    
    args = parser.parse_args()
    
    # Parse station codes
    station_codes = [s.strip() for s in args.stations.split(',')]
    
    manager = BaselineManager()
    
    if args.command == 'save':
        manager.save_baseline(station_codes, optimized=args.optimized)
    
    elif args.command == 'compare':
        manager.compare_baselines(station_codes)
    
    elif args.command == 'diff':
        # Show detailed differences
        print("Detailed diff view - Not implemented yet")
        print("Use CSV files in baseline folders for manual inspection")


if __name__ == '__main__':
    main()
