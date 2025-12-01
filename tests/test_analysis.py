"""
Tests for analysis functions.
"""
import pytest
import pandas as pd
import numpy as np
from hydroburn.analysis import (
    extract_annual_maxima,
    compute_fdc,
    extract_events,
)

@pytest.fixture
def sample_analysis_df():
    """Create a sample DataFrame for analysis tests."""
    dates = pd.to_datetime(pd.date_range(start="2020-01-01", periods=3 * 365, freq="D"))
    # Simple sine wave to simulate seasonal flow with some noise and a spike
    flow = 10 + 5 * np.sin(np.arange(len(dates)) * 2 * np.pi / 365) + np.random.rand(len(dates))
    flow[100] = 50 # Add a peak for event extraction
    df = pd.DataFrame({'discharge': flow}, index=dates)
    return df

def test_extract_annual_maxima(sample_analysis_df):
    """Test extraction of annual maxima."""
    maxima = extract_annual_maxima(sample_analysis_df)
    assert len(maxima) > 0
    assert 'max_discharge' in maxima.columns

def test_compute_fdc(sample_analysis_df):
    """Test flow duration curve computation."""
    exceedance, discharge = compute_fdc(sample_analysis_df['discharge'])
    assert len(exceedance) == len(discharge)
    assert np.all(np.diff(discharge) <= 0) # Should be sorted descending

def test_extract_events(sample_analysis_df):
    """Test storm event extraction."""
    from hydroburn.preprocess import separate_baseflow
    df = separate_baseflow(sample_analysis_df)
    events = extract_events(df, area_km2=100)
    assert len(events) > 0
    assert events[0].peak_discharge > 40
