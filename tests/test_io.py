"""
Tests for I/O operations.
"""
import pytest
import pandas as pd
import geopandas as gpd
from shapely.geometry import Polygon
from hydroburn.io import load_catchment, compute_catchment_area, load_streamflow

@pytest.fixture
def sample_catchment(tmp_path):
    """Create a sample catchment shapefile."""
    p = Polygon([(0, 0), (1, 1), (1, 0)])
    gdf = gpd.GeoDataFrame({"data": [1]}, geometry=[p], crs="EPSG:4326")
    filepath = tmp_path / "test_catchment.shp"
    gdf.to_file(filepath)
    return str(filepath)

@pytest.fixture
def sample_streamflow(tmp_path):
    """Create a sample streamflow CSV."""
    dates = pd.to_datetime(['2022-01-01', '2022-01-02', '2022-01-03'])
    data = {'datetime': dates, '00060_Mean': [10, 12, 15]}
    df = pd.DataFrame(data)
    filepath = tmp_path / "test_streamflow.csv"
    df.to_csv(filepath, index=False)
    return str(filepath)

def test_load_catchment(sample_catchment):
    """Test loading a catchment shapefile."""
    gdf = load_catchment(sample_catchment)
    assert isinstance(gdf, gpd.GeoDataFrame)
    assert not gdf.empty

def test_compute_catchment_area(sample_catchment):
    """Test catchment area computation."""
    gdf = load_catchment(sample_catchment)
    area = compute_catchment_area(gdf)
    assert area > 0

def test_load_streamflow(sample_streamflow):
    """Test loading a streamflow CSV."""
    df = load_streamflow(sample_streamflow)
    assert isinstance(df, pd.DataFrame)
    assert isinstance(df.index, pd.DatetimeIndex)
    assert 'discharge' in df.columns
    assert len(df) == 3
