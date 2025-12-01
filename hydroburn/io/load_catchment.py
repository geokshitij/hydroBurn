"""
Catchment loading and area calculation module.

Loads catchment boundaries from shapefiles or GeoJSON and computes
drainage area using appropriate projections.
"""

import geopandas as gpd
import numpy as np
from pathlib import Path
from typing import Tuple, Optional
import warnings


def load_catchment(filepath: str) -> gpd.GeoDataFrame:
    """
    Load catchment boundary from shapefile or GeoJSON.
    
    Parameters
    ----------
    filepath : str
        Path to shapefile (.shp) or GeoJSON (.geojson, .json)
    
    Returns
    -------
    geopandas.GeoDataFrame
        Catchment boundary with geometry
    
    Raises
    ------
    FileNotFoundError
        If file does not exist
    ValueError
        If file format is not supported
    """
    path = Path(filepath)
    
    if not path.exists():
        # Try with different extensions for shapefiles
        if path.suffix == '.shp':
            raise FileNotFoundError(f"Shapefile not found: {filepath}")
        # Check if it's a shapefile without extension
        shp_path = path.with_suffix('.shp')
        if shp_path.exists():
            path = shp_path
        else:
            raise FileNotFoundError(f"Catchment file not found: {filepath}")
    
    # Load based on extension
    suffix = path.suffix.lower()
    if suffix in ['.shp']:
        gdf = gpd.read_file(path)
    elif suffix in ['.geojson', '.json']:
        gdf = gpd.read_file(path)
    else:
        # Try to read anyway (geopandas is flexible)
        try:
            gdf = gpd.read_file(path)
        except Exception as e:
            raise ValueError(f"Unsupported file format: {suffix}. Error: {e}")
    
    # Validate geometry
    if gdf.empty:
        raise ValueError("Catchment file contains no features")
    
    if not all(gdf.geometry.is_valid):
        warnings.warn("Some geometries are invalid. Attempting to fix...")
        gdf.geometry = gdf.geometry.buffer(0)
    
    return gdf


def get_utm_crs(gdf: gpd.GeoDataFrame) -> str:
    """
    Determine appropriate UTM CRS for the catchment.
    
    Parameters
    ----------
    gdf : geopandas.GeoDataFrame
        Catchment in any CRS
    
    Returns
    -------
    str
        PROJ4 string for appropriate UTM zone
    """
    # Ensure WGS84 for centroid calculation
    if gdf.crs is None:
        warnings.warn("No CRS defined. Assuming WGS84 (EPSG:4326)")
        gdf = gdf.set_crs("EPSG:4326")
    elif gdf.crs.to_epsg() != 4326:
        gdf = gdf.to_crs("EPSG:4326")
    
    # Get centroid
    centroid = gdf.geometry.union_all().centroid
    lon, lat = centroid.x, centroid.y
    
    # Calculate UTM zone
    utm_zone = int((lon + 180) / 6) + 1
    hemisphere = "north" if lat >= 0 else "south"
    
    # Return PROJ4 string
    return f"+proj=utm +zone={utm_zone} +{hemisphere} +datum=WGS84 +units=m +no_defs"


def compute_catchment_area(gdf: gpd.GeoDataFrame) -> float:
    """
    Compute catchment area in km².
    
    Uses UTM projection centered on catchment for accurate area calculation.
    
    Parameters
    ----------
    gdf : geopandas.GeoDataFrame
        Catchment boundary
    
    Returns
    -------
    float
        Catchment area in km²
    """
    # Get appropriate UTM CRS
    utm_crs = get_utm_crs(gdf)
    
    # Reproject to UTM
    gdf_utm = gdf.to_crs(utm_crs)
    
    # Compute area in m², convert to km²
    area_m2 = gdf_utm.geometry.area.sum()
    area_km2 = area_m2 / 1e6
    
    return area_km2


def get_catchment_centroid(gdf: gpd.GeoDataFrame) -> Tuple[float, float]:
    """
    Get catchment centroid in WGS84 coordinates.
    
    Parameters
    ----------
    gdf : geopandas.GeoDataFrame
        Catchment boundary
    
    Returns
    -------
    tuple
        (longitude, latitude) of centroid
    """
    # Ensure WGS84
    if gdf.crs is None:
        gdf = gdf.set_crs("EPSG:4326")
    elif gdf.crs.to_epsg() != 4326:
        gdf = gdf.to_crs("EPSG:4326")
    
    centroid = gdf.geometry.union_all().centroid
    return (centroid.x, centroid.y)


def get_catchment_bounds(gdf: gpd.GeoDataFrame) -> Tuple[float, float, float, float]:
    """
    Get catchment bounding box in WGS84 coordinates.
    
    Parameters
    ----------
    gdf : geopandas.GeoDataFrame
        Catchment boundary
    
    Returns
    -------
    tuple
        (minx, miny, maxx, maxy) bounding box
    """
    # Ensure WGS84
    if gdf.crs is None:
        gdf = gdf.set_crs("EPSG:4326")
    elif gdf.crs.to_epsg() != 4326:
        gdf = gdf.to_crs("EPSG:4326")
    
    return tuple(gdf.total_bounds)


def load_catchment_with_area(filepath: str) -> Tuple[gpd.GeoDataFrame, float]:
    """
    Convenience function to load catchment and compute area.
    
    Parameters
    ----------
    filepath : str
        Path to catchment file
    
    Returns
    -------
    tuple
        (GeoDataFrame, area_km2)
    """
    gdf = load_catchment(filepath)
    area_km2 = compute_catchment_area(gdf)
    return gdf, area_km2
