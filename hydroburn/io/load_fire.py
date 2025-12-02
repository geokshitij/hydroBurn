"""
Module for loading and processing fire history data.
"""
import geopandas as gpd
import pandas as pd
from typing import List

def load_fire_history(fire_history_file: str, catchment_gdf: gpd.GeoDataFrame) -> pd.DataFrame:
    """
    Loads fire history data, clips it to the catchment, and calculates annual burned area.

    Parameters
    ----------
    fire_history_file : str
        Path to the fire history shapefile.
    catchment_gdf : gpd.GeoDataFrame
        GeoDataFrame of the catchment boundary.

    Returns
    -------
    pd.DataFrame
        A DataFrame with 'year' and 'burned_area_km2'.
    """
    # Load fire data
    fire_gdf = gpd.read_file(fire_history_file)

    # Ensure CRS match
    if fire_gdf.crs != catchment_gdf.crs:
        fire_gdf = fire_gdf.to_crs(catchment_gdf.crs)

    # Clip fire points to the catchment polygon
    fires_in_catchment = gpd.clip(fire_gdf, catchment_gdf)

    if fires_in_catchment.empty:
        return pd.DataFrame(columns=['year', 'burned_area_km2'])

    # Convert ACQ_DATE to datetime
    fires_in_catchment['ACQ_DATE'] = pd.to_datetime(fires_in_catchment['ACQ_DATE'])
    fires_in_catchment['year'] = fires_in_catchment['ACQ_DATE'].dt.year

    # This is an approximation: it assumes each fire point represents a certain area.
    # A more accurate method would use fire perimeter polygons.
    # For FIRMS data, each point is a 1km pixel.
    pixel_area_km2 = 1.0
    
    # Calculate burned area per year
    # Also get the earliest acquisition date for each year to serve as the fire start date
    annual_stats = fires_in_catchment.groupby('year').agg({
        'ACQ_DATE': 'min',
        'geometry': 'count'  # Count points
    }).rename(columns={'ACQ_DATE': 'fire_date', 'geometry': 'point_count'})
    
    annual_stats['burned_area_km2'] = annual_stats['point_count'] * pixel_area_km2
    annual_stats = annual_stats.reset_index()[['year', 'burned_area_km2', 'fire_date']]
    
    return annual_stats
