"""
Tests for preprocessing functions.
"""
import pytest
import pandas as pd
import numpy as np
from hydroburn.preprocess import (
    cfs_to_m3s,
    separate_baseflow,
    quality_control,
)

@pytest.fixture
def sample_df():
    """Create a sample DataFrame for preprocessing tests."""
    dates = pd.to_datetime(pd.date_range(start="2022-01-01", periods=10, freq="D"))
    discharge = np.array([5, 8, 15, 12, 9, 7, np.nan, 10, 11, 6], dtype=float)
    df = pd.DataFrame({'discharge': discharge}, index=dates)
    return df

def test_cfs_to_m3s():
    """Test unit conversion."""
    assert cfs_to_m3s(1) == pytest.approx(0.028316847)

def test_baseflow_separation(sample_df):
    """Test baseflow separation."""
    df = separate_baseflow(sample_df.dropna())
    assert 'baseflow' in df.columns
    assert 'quickflow' in df.columns
    assert (df['baseflow'] <= df['discharge']).all()

def test_quality_control(sample_df):
    """Test quality control and gap filling."""
    df, report = quality_control(sample_df, max_gap_days=2)
    assert 'gaps_filled' in report
    assert df['discharge'].isna().sum() == 0
