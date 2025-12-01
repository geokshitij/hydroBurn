"""
Tests for visualization functions.
"""
import pytest
from pathlib import Path
import pandas as pd
import numpy as np
import geopandas as gpd
from hydroburn.config import HydroBurnConfig
from hydroburn.visualization import (
    plot_hydrograph,
    plot_baseflow_index,
    plot_fdc_comparison,
    plot_annual_maxima,
    plot_flood_frequency,
    plot_event_boxplots,
    map_catchment,
)

# Helper to create a dummy config for tests
@pytest.fixture
def test_config(tmp_path):
    """Creates a dummy config and output directories."""
    output_dir = tmp_path / "output"
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True)
    
    return HydroBurnConfig(
        station_id="08390500",
        fire_date=pd.Timestamp("2020-08-01"),
        output_dir=str(output_dir),
        catchment_file="/path/to/dummy.shp",
        streamflow_file="/path/to/dummy.csv",
    )

# Helper to create sample data
@pytest.fixture
def sample_data():
    """Creates sample data for plotting."""
    dates = pd.to_datetime(pd.date_range("2019-01-01", "2021-12-31", freq="D"))
    flow = pd.Series(abs(np.random.randn(len(dates))) * 10, index=dates, name="flow_m3s")
    baseflow = flow * 0.5
    baseflow.name = "baseflow_m3s"
    df = pd.DataFrame({"flow_m3s": flow, "baseflow_m3s": baseflow})
    df["period"] = "pre-fire"
    df.loc[df.index >= "2020-08-01", "period"] = "post-fire"
    return df

@pytest.fixture
def sample_catchment_gdf(tmp_path):
    """Creates a sample GeoDataFrame for the catchment map."""
    from shapely.geometry import Polygon
    p = Polygon([(0, 0), (1, 1), (1, 0)])
    gdf = gpd.GeoDataFrame([1], geometry=[p], crs="EPSG:4326")
    return gdf

def test_plot_hydrograph(test_config, sample_data):
    """Test that hydrograph plot is created."""
    filepath = Path(test_config.figures_dir) / "hydrograph.png"
    plot_hydrograph(sample_data, test_config)
    assert filepath.exists()
    assert filepath.stat().st_size > 0

def test_plot_baseflow_index(test_config, sample_data):
    """Test that baseflow plot is created."""
    filepath = Path(test_config.figures_dir) / "baseflow_separation.png"
    
    # Create dummy annual_bfi data
    annual_bfi = pd.DataFrame({
        'year': [2019, 2020, 2021],
        'bfi': [0.5, 0.4, 0.6]
    })
    
    plot_baseflow_index(annual_bfi, fire_year=test_config.fire_date.year, output_path=str(filepath))
    assert filepath.exists()
    assert filepath.stat().st_size > 0

def test_plot_fdc_comparison(test_config, sample_data):
    """Test that FDC plot is created."""
    filepath = Path(test_config.figures_dir) / "fdc_comparison.png"
    plot_fdc_comparison(sample_data, test_config)
    assert filepath.exists()
    assert filepath.stat().st_size > 0

def test_map_catchment(test_config, sample_catchment_gdf):
    """Test that catchment map is created."""
    filepath = Path(test_config.figures_dir) / "catchment_map.png"
    map_catchment(sample_catchment_gdf, test_config)
    assert filepath.exists()
    assert filepath.stat().st_size > 0
