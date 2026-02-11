# Standard library imports
import os
import sys
from datetime import datetime

# Third party imports
import numpy as np
from obspy import Stream, Trace, UTCDateTime

# Add src to path to import modules
sys.path.append(os.path.join(os.getcwd(), 'src'))

# Project imports
from eruption_forecast.dsar import DSAR


def create_dummy_stream(sampling_rate=100.0, duration=600):
    """Create a dummy stream with 10 minutes of data + 1 second"""
    npts = int(sampling_rate * duration)
    t = np.linspace(0, duration, npts)

    # Simple sine wave
    data = np.sin(2 * np.pi * 1.0 * t)

    stats = {
        'network': 'TEST',
        'station': 'TEST',
        'location': '00',
        'channel': 'BHZ',
        'npts': npts,
        'sampling_rate': sampling_rate,
        'mseed': {'dataquality': 'D'},
        'starttime': UTCDateTime(datetime(2023, 1, 1, 0, 0, 0))
    }

    trace = Trace(data=data, header=stats)
    return Stream(traces=[trace])

def verify_dsar_calculation():
    print("Verifying DSAR calculation...")

    # Create two streams
    # Stream 1: Amplitude 1.0 -> Mean(|sin|) approx 0.6366
    stream1 = create_dummy_stream()

    # Stream 2: Amplitude 2.0 -> Mean(|2*sin|) approx 1.2732
    stream2 = create_dummy_stream()
    stream2[0].data = stream2[0].data * 2.0

    dsar_calculator = DSAR(verbose=True)

    result = dsar_calculator.calculate(
        stream1=stream1,
        stream2=stream2,
        window_duration_minutes=10
    )

    print("\nResult Series:")
    print(result)

    # Expected Ratio: Mean(|Stream1|) / Mean(|Stream2|) = 1 / 2 = 0.5
    # The first window (index 0) should be close to 0.5

    first_window_value = result.iloc[0]
    print(f"\nCalculated Value: {first_window_value}")

    assert np.isclose(first_window_value, 0.5, rtol=1e-2), \
        f"Expected 0.5, got {first_window_value}"

    print("\n✅ Verification Successful: DSAR ratio is correct!")

if __name__ == "__main__":
    verify_dsar_calculation()
