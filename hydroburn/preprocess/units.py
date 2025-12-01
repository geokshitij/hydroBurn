"""
Unit conversion module for streamflow data.

Provides functions for converting between different discharge units
and computing runoff depth from discharge and catchment area.
"""

import numpy as np
import pandas as pd
from typing import Union

# Conversion constants
CFS_TO_M3S = 0.028316847  # 1 ft³/s = 0.028316847 m³/s
M3S_TO_CFS = 1 / CFS_TO_M3S
MM_PER_M = 1000
M2_PER_KM2 = 1e6
SECONDS_PER_DAY = 86400


def cfs_to_m3s(Q_cfs: Union[float, np.ndarray, pd.Series]) -> Union[float, np.ndarray, pd.Series]:
    """
    Convert cubic feet per second to cubic meters per second.
    
    Parameters
    ----------
    Q_cfs : float, array, or Series
        Discharge in ft³/s (cfs)
    
    Returns
    -------
    same type as input
        Discharge in m³/s
    """
    return Q_cfs * CFS_TO_M3S


def m3s_to_cfs(Q_m3s: Union[float, np.ndarray, pd.Series]) -> Union[float, np.ndarray, pd.Series]:
    """
    Convert cubic meters per second to cubic feet per second.
    
    Parameters
    ----------
    Q_m3s : float, array, or Series
        Discharge in m³/s
    
    Returns
    -------
    same type as input
        Discharge in ft³/s (cfs)
    """
    return Q_m3s * M3S_TO_CFS


def discharge_to_depth(
    Q_m3s: Union[float, np.ndarray, pd.Series],
    area_km2: float,
    dt_seconds: float = SECONDS_PER_DAY
) -> Union[float, np.ndarray, pd.Series]:
    """
    Convert discharge to runoff depth.
    
    This converts volumetric flow rate to an equivalent depth over the
    catchment area, useful for comparing runoff between catchments of
    different sizes and for water balance calculations.
    
    Parameters
    ----------
    Q_m3s : float, array, or Series
        Discharge in m³/s
    area_km2 : float
        Catchment area in km²
    dt_seconds : float
        Time step in seconds (default: 86400 for daily data)
    
    Returns
    -------
    same type as input
        Runoff depth in mm per time step
    
    Notes
    -----
    Calculation:
        volume (m³) = Q (m³/s) × dt (s)
        depth (m) = volume (m³) / area (m²)
        depth (mm) = depth (m) × 1000
    
    For daily data with dt=86400s:
        depth_mm = Q_m3s × 86400 / (area_km2 × 1e6) × 1000
                 = Q_m3s × 86.4 / area_km2
    """
    area_m2 = area_km2 * M2_PER_KM2
    volume_m3 = Q_m3s * dt_seconds
    depth_m = volume_m3 / area_m2
    depth_mm = depth_m * MM_PER_M
    return depth_mm


def depth_to_discharge(
    depth_mm: Union[float, np.ndarray, pd.Series],
    area_km2: float,
    dt_seconds: float = SECONDS_PER_DAY
) -> Union[float, np.ndarray, pd.Series]:
    """
    Convert runoff depth to discharge.
    
    Inverse of discharge_to_depth().
    
    Parameters
    ----------
    depth_mm : float, array, or Series
        Runoff depth in mm per time step
    area_km2 : float
        Catchment area in km²
    dt_seconds : float
        Time step in seconds (default: 86400 for daily data)
    
    Returns
    -------
    same type as input
        Discharge in m³/s
    """
    area_m2 = area_km2 * M2_PER_KM2
    depth_m = depth_mm / MM_PER_M
    volume_m3 = depth_m * area_m2
    Q_m3s = volume_m3 / dt_seconds
    return Q_m3s


def convert_discharge_units(
    df: pd.DataFrame,
    from_units: str,
    to_units: str,
    area_km2: float = None,
    discharge_col: str = "discharge"
) -> pd.DataFrame:
    """
    Convert discharge column to different units.
    
    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame with discharge column
    from_units : str
        Source units: 'cfs', 'm3s', or 'mm_day'
    to_units : str
        Target units: 'cfs', 'm3s', or 'mm_day'
    area_km2 : float, optional
        Catchment area in km² (required for mm_day conversions)
    discharge_col : str
        Name of discharge column
    
    Returns
    -------
    pandas.DataFrame
        DataFrame with converted discharge values
    """
    df = df.copy()
    Q = df[discharge_col].values
    
    # First convert to m³/s
    if from_units.lower() in ['cfs', 'ft3/s', 'ft3s']:
        Q_m3s = cfs_to_m3s(Q)
    elif from_units.lower() in ['m3s', 'm3/s', 'cms']:
        Q_m3s = Q
    elif from_units.lower() in ['mm_day', 'mm/day', 'mmday']:
        if area_km2 is None:
            raise ValueError("area_km2 required for mm/day conversion")
        Q_m3s = depth_to_discharge(Q, area_km2)
    else:
        raise ValueError(f"Unknown source units: {from_units}")
    
    # Then convert to target units
    if to_units.lower() in ['cfs', 'ft3/s', 'ft3s']:
        Q_out = m3s_to_cfs(Q_m3s)
    elif to_units.lower() in ['m3s', 'm3/s', 'cms']:
        Q_out = Q_m3s
    elif to_units.lower() in ['mm_day', 'mm/day', 'mmday']:
        if area_km2 is None:
            raise ValueError("area_km2 required for mm/day conversion")
        Q_out = discharge_to_depth(Q_m3s, area_km2)
    else:
        raise ValueError(f"Unknown target units: {to_units}")
    
    df[discharge_col] = Q_out
    return df


def compute_annual_runoff(
    df: pd.DataFrame,
    area_km2: float,
    discharge_col: str = "discharge",
    from_units: str = "m3s"
) -> pd.DataFrame:
    """
    Compute annual runoff totals in mm.
    
    Parameters
    ----------
    df : pandas.DataFrame
        Daily streamflow data with datetime index
    area_km2 : float
        Catchment area in km²
    discharge_col : str
        Name of discharge column
    from_units : str
        Units of discharge column ('cfs' or 'm3s')
    
    Returns
    -------
    pandas.DataFrame
        Annual runoff totals with columns: year, runoff_mm
    """
    df = df.copy()
    
    # Convert to m³/s if needed
    if from_units.lower() in ['cfs', 'ft3/s']:
        df[discharge_col] = cfs_to_m3s(df[discharge_col])
    
    # Convert to mm/day
    df['runoff_mm'] = discharge_to_depth(df[discharge_col], area_km2)
    
    # Sum by year
    annual = df.groupby(df.index.year)['runoff_mm'].sum().reset_index()
    annual.columns = ['year', 'runoff_mm']
    
    return annual
