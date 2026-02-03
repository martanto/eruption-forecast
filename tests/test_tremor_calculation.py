#!/usr/bin/env python
"""Test script for tremor calculation refactoring.

This script tests the refactored tremor calculation module with real SDS data
from the OJN station for 3 days (2025-01-01 to 2025-01-03).

Tests include:
- Tremor calculation (RSAM + DSAR)
- Daily tremor plotting
- Output validation

Author: Refactoring Phase 1
Date: 2026-02-03
"""

import os
import sys
from datetime import datetime, timedelta
from pathlib import Path

# Add src to path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

from eruption_forecast.tremor.calculate_tremor import CalculateTremor
from eruption_forecast.plot import plot_tremor
from eruption_forecast.logger import logger
import pandas as pd


def plot_daily_tremor(
    df: pd.DataFrame,
    figures_dir: str,
    start_date: str,
    end_date: str,
    station: str,
    dpi: int = 150,
) -> int:
    """Plot daily tremor data for each day in the date range.

    Creates individual plots for each day showing all tremor metrics
    in 10-minute resolution. This is useful for detailed daily analysis.

    Args:
        df (pd.DataFrame): Tremor DataFrame with DatetimeIndex
        figures_dir (str): Directory to save figures
        start_date (str): Start date (YYYY-MM-DD)
        end_date (str): End date (YYYY-MM-DD)
        station (str): Station identifier (NSLC format)
        dpi (int, optional): Plot resolution. Defaults to 150.

    Returns:
        int: Number of daily plots created

    Examples:
        >>> plot_daily_tremor(df, "figures/", "2025-01-01", "2025-01-03", "VG.OJN.00.EHZ")
        3
    """
    print("  Creating daily tremor plots...")
    print()

    # Create daily plots subdirectory
    daily_dir = os.path.join(figures_dir, "daily")
    os.makedirs(daily_dir, exist_ok=True)

    # Parse dates
    start = datetime.strptime(start_date, "%Y-%m-%d")
    end = datetime.strptime(end_date, "%Y-%m-%d")

    # Generate date range
    current_date = start
    plots_created = 0

    while current_date <= end:
        date_str = current_date.strftime("%Y-%m-%d")

        # Filter data for this specific day
        day_start = current_date.replace(hour=0, minute=0, second=0)
        day_end = current_date.replace(hour=23, minute=59, second=59)

        # Select data for this day
        daily_df = df[(df.index >= day_start) & (df.index <= day_end)]

        if len(daily_df) > 0:
            # Create filename
            filename = f"{station}_{date_str}.png"

            # Plot using the existing plot_tremor function
            plot_tremor(
                df=daily_df,
                interval=2,  # 2-hour intervals for x-axis
                interval_unit="hours",
                filename=filename,
                figure_dir=daily_dir,
                title=f"{station} - {date_str}",
                overwrite=True,
                dpi=dpi,
                verbose=False,
            )

            plots_created += 1
            print(f"    [OK] {date_str}: {len(daily_df)} samples -> {filename}")
        else:
            print(f"    [SKIP] {date_str}: No data available")

        # Move to next day
        current_date += timedelta(days=1)

    print()
    print(f"  Daily plots saved to: {daily_dir}")
    return plots_created


def test_tremor_calculation():
    """Test tremor calculation with 3 days of real data."""

    print("=" * 80)
    print("TREMOR CALCULATION TEST - Refactored Module")
    print("=" * 80)
    print()

    # Configuration
    sds_dir = r"D:\Data\OJN"
    station = "OJN"
    channel = "EHZ"
    network = "VG"
    location = "00"
    start_date = "2025-01-01"
    end_date = "2025-01-03"
    output_dir = "output_test"

    print(f"Configuration:")
    print(f"  SDS Directory: {sds_dir}")
    print(f"  Station: {station}")
    print(f"  Channel: {channel}")
    print(f"  Network: {network}")
    print(f"  Location: {location}")
    print(f"  Date Range: {start_date} to {end_date}")
    print(f"  Output Directory: {output_dir}")
    print()

    # Validate SDS directory exists
    if not os.path.exists(sds_dir):
        print(f"[ERROR] SDS directory not found: {sds_dir}")
        return False

    print(f"[OK] SDS directory exists: {sds_dir}")
    print()

    try:
        print("-" * 80)
        print("STEP 1: Initialize CalculateTremor")
        print("-" * 80)

        tremor_calc = CalculateTremor(
            start_date=start_date,
            end_date=end_date,
            station=station,
            channel=channel,
            network=network,
            location=location,
            output_dir=output_dir,
            n_jobs=1,  # Single process for easier debugging
            verbose=True,
            debug=False,
            overwrite=True,
            cleanup_tmp_dir=False,  # Keep tmp files for inspection
            plot_tmp=False,
            save_plot=True,
        )

        print(f"[OK] CalculateTremor initialized")
        print(f"   NSLC: {tremor_calc.nslc}")
        print(f"   Frequency Bands: {tremor_calc.freq_bands_alias}")
        print(f"   Methods: {tremor_calc.methods}")
        print(f"   Total Days: {tremor_calc.n_days}")
        print()

        print("-" * 80)
        print("STEP 2: Set Data Source (SDS)")
        print("-" * 80)

        tremor_calc.from_sds(sds_dir=sds_dir)
        print(f"[OK] Data source set to SDS")
        print(f"   SDS Path: {tremor_calc.sds.sds_dir}")
        print()

        print("-" * 80)
        print("STEP 3: Run Tremor Calculation")
        print("-" * 80)

        result = tremor_calc.run()

        print()
        print(f"[OK] Tremor calculation completed successfully!")
        print(f"   Output CSV: {result.csv}")
        print()

        print("-" * 80)
        print("STEP 4: Validate Results")
        print("-" * 80)

        df = result.df

        # Basic validation
        print(f"DataFrame Shape: {df.shape}")
        print(f"  Rows: {len(df)}")
        print(f"  Columns: {len(df.columns)}")
        print()

        print("Columns:")
        for col in df.columns:
            print(f"  - {col}")
        print()

        print(f"Date Range:")
        print(f"  Start: {df.index[0]}")
        print(f"  End: {df.index[-1]}")
        print()

        print(f"First 5 rows:")
        print(df.head())
        print()

        print(f"Statistics:")
        print(df.describe())
        print()

        # Check for NaN values
        nan_counts = df.isna().sum()
        total_nans = nan_counts.sum()
        print(f"NaN Values: {total_nans}")
        if total_nans > 0:
            print("  NaN counts per column:")
            for col, count in nan_counts[nan_counts > 0].items():
                print(f"    {col}: {count}")
        print()

        # Check file system
        print("-" * 80)
        print("STEP 5: Check Output Files")
        print("-" * 80)

        station_dir = result.station_dir
        tremor_dir = result.tremor_dir
        tmp_dir = result.tmp_dir

        print(f"Station Directory: {station_dir}")
        print(f"  Exists: {os.path.exists(station_dir)}")

        print(f"Tremor Directory: {tremor_dir}")
        print(f"  Exists: {os.path.exists(tremor_dir)}")

        print(f"Temporary Directory: {tmp_dir}")
        print(f"  Exists: {os.path.exists(tmp_dir)}")

        if os.path.exists(tmp_dir):
            tmp_files = os.listdir(tmp_dir)
            print(f"  Temporary files ({len(tmp_files)}):")
            for f in tmp_files:
                print(f"    - {f}")

        print()

        # Check CSV file
        if os.path.exists(result.csv):
            file_size = os.path.getsize(result.csv)
            print(f"[OK] Output CSV exists: {result.csv}")
            print(f"   File size: {file_size:,} bytes ({file_size/1024:.2f} KB)")
        else:
            print(f"[FAIL] Output CSV not found: {result.csv}")
            return False

        print()

        # Check plots
        if result.save_plot:
            figures_dir = result.figures_dir
            print(f"Figures Directory: {figures_dir}")
            if os.path.exists(figures_dir):
                figures = [f for f in os.listdir(figures_dir) if f.endswith('.png')]
                print(f"  Plot files ({len(figures)}):")
                for f in figures:
                    print(f"    - {f}")
            print()

        print("=" * 80)
        print("STEP 6: Plot Daily Tremor Data")
        print("=" * 80)

        # Test daily plotting functionality
        daily_plots_created = plot_daily_tremor(
            df=df,
            figures_dir=figures_dir,
            start_date=start_date,
            end_date=end_date,
            station=tremor_calc.nslc
        )

        print(f"[OK] Created {daily_plots_created} daily tremor plots")
        print()

        print("=" * 80)
        print("[OK] ALL TESTS PASSED!")
        print("=" * 80)
        print()
        print(f"Summary:")
        print(f"  - Calculated tremor for {tremor_calc.n_days} days")
        print(f"  - Generated {len(df)} time windows (10-minute intervals)")
        print(f"  - Computed {len(df.columns)} tremor metrics")
        print(f"  - Created {daily_plots_created} daily plots")
        print(f"  - Output saved to: {result.csv}")
        print()

        return True

    except Exception as e:
        print()
        print("=" * 80)
        print("[FAIL] TEST FAILED!")
        print("=" * 80)
        print(f"Error: {type(e).__name__}")
        print(f"Message: {e}")
        print()

        import traceback
        print("Traceback:")
        traceback.print_exc()

        return False


if __name__ == "__main__":
    success = test_tremor_calculation()
    sys.exit(0 if success else 1)
