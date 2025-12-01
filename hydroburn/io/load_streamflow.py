"""
Streamflow data loading module.

Loads and parses USGS streamflow CSV files with proper datetime handling
and timezone conversion.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, Tuple, Dict
from datetime import datetime
import warnings


def load_streamflow(
    filepath: str,
    datetime_column: str = "datetime",
    discharge_column: str = "00060_Mean",
    timezone: str = "America/Denver",
    quality_column: Optional[str] = "00060_Mean_cd",
) -> pd.DataFrame:
    """
    Load streamflow data from CSV file.
    
    Handles USGS NWIS format with datetime parsing and timezone conversion.
    
    Parameters
    ----------
    filepath : str
        Path to CSV file
    datetime_column : str
        Name of datetime column
    discharge_column : str
        Name of discharge column
    timezone : str
        Target timezone (e.g., 'America/Denver' for MST/MDT)
    quality_column : str, optional
        Name of quality code column (if present)
    
    Returns
    -------
    pandas.DataFrame
        DataFrame with columns:
        - datetime: timezone-aware datetime index
        - discharge: discharge values
        - quality: quality codes (if available)
    
    Raises
    ------
    FileNotFoundError
        If file does not exist
    ValueError
        If required columns are missing
    """
    path = Path(filepath)
    
    if not path.exists():
        raise FileNotFoundError(f"Streamflow file not found: {filepath}")
    
    # Read CSV
    df = pd.read_csv(filepath)
    
    # Validate required columns
    if datetime_column not in df.columns:
        raise ValueError(f"Datetime column '{datetime_column}' not found. "
                        f"Available columns: {list(df.columns)}")
    
    if discharge_column not in df.columns:
        raise ValueError(f"Discharge column '{discharge_column}' not found. "
                        f"Available columns: {list(df.columns)}")
    
    # Parse datetime
    df[datetime_column] = pd.to_datetime(df[datetime_column])
    
    # Handle timezone
    if df[datetime_column].dt.tz is None:
        # Assume UTC if no timezone info (USGS data is often in UTC)
        df[datetime_column] = df[datetime_column].dt.tz_localize('UTC')
    
    # Convert to target timezone
    df[datetime_column] = df[datetime_column].dt.tz_convert(timezone)
    
    # Create clean output DataFrame
    result = pd.DataFrame({
        'datetime': df[datetime_column],
        'discharge': pd.to_numeric(df[discharge_column], errors='coerce'),
    })
    
    # Add quality codes if available
    if quality_column and quality_column in df.columns:
        result['quality'] = df[quality_column]
    
    # Set datetime as index
    result = result.set_index('datetime').sort_index()
    
    # Remove duplicate indices if any
    if result.index.duplicated().any():
        warnings.warn(f"Found {result.index.duplicated().sum()} duplicate timestamps. Keeping first.")
        result = result[~result.index.duplicated(keep='first')]
    
    return result


def get_streamflow_info(df: pd.DataFrame) -> Dict:
    """
    Get summary information about streamflow data.
    
    Parameters
    ----------
    df : pandas.DataFrame
        Streamflow DataFrame from load_streamflow()
    
    Returns
    -------
    dict
        Dictionary with data summary statistics
    """
    discharge = df['discharge']
    
    info = {
        'start_date': df.index.min(),
        'end_date': df.index.max(),
        'n_records': len(df),
        'n_missing': discharge.isna().sum(),
        'pct_missing': discharge.isna().sum() / len(df) * 100,
        'n_zero': (discharge == 0).sum(),
        'n_negative': (discharge < 0).sum(),
        'min': discharge.min(),
        'max': discharge.max(),
        'mean': discharge.mean(),
        'median': discharge.median(),
        'std': discharge.std(),
        'record_years': (df.index.max() - df.index.min()).days / 365.25,
    }
    
    # Check temporal resolution
    if len(df) > 1:
        dt = df.index.to_series().diff().median()
        info['temporal_resolution'] = str(dt)
        info['is_daily'] = dt == pd.Timedelta(days=1)
    
    # Quality code breakdown if available
    if 'quality' in df.columns:
        info['quality_codes'] = df['quality'].value_counts().to_dict()
    
    return info


def split_pre_post(
    df: pd.DataFrame, 
    fire_date: str,
    buffer_days: int = 0
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split streamflow data into pre-fire and post-fire periods.
    
    Parameters
    ----------
    df : pandas.DataFrame
        Streamflow DataFrame with datetime index
    fire_date : str
        Fire date in 'YYYY-MM-DD' format
    buffer_days : int
        Days to exclude around fire date (symmetric buffer)
    
    Returns
    -------
    tuple
        (pre_fire_df, post_fire_df)
    """
    fire_dt = pd.Timestamp(fire_date)
    
    # Make timezone-aware if needed
    if df.index.tz is not None and fire_dt.tz is None:
        fire_dt = fire_dt.tz_localize(df.index.tz)
    
    # Apply buffer
    pre_end = fire_dt - pd.Timedelta(days=buffer_days)
    post_start = fire_dt + pd.Timedelta(days=buffer_days)
    
    pre = df[df.index < pre_end].copy()
    post = df[df.index >= post_start].copy()
    
    return pre, post


def get_water_year(dt: pd.Timestamp) -> int:
    """
    Get water year for a given date.
    
    Water year starts October 1. For example:
    - Oct 1, 2023 → WY 2024
    - Sep 30, 2024 → WY 2024
    
    Parameters
    ----------
    dt : pandas.Timestamp
        Date
    
    Returns
    -------
    int
        Water year
    """
    if dt.month >= 10:
        return dt.year + 1
    return dt.year


def add_water_year(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add water year column to DataFrame.
    
    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame with datetime index
    
    Returns
    -------
    pandas.DataFrame
        DataFrame with 'water_year' column added
    """
    df = df.copy()
    df['water_year'] = df.index.map(get_water_year)
    return df
