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
    datetime_column: Optional[str] = None,
    discharge_column: Optional[str] = None,
    timezone: str = "UTC",
    quality_column: Optional[str] = None,
) -> pd.DataFrame:
    """
    Load streamflow data from CSV file.
    
    Handles USGS NWIS format with datetime parsing and timezone conversion.
    
    Parameters
    ----------
    filepath : str
        Path to CSV file
    datetime_column : str, optional
        Name of datetime column (if not specified, attempts to auto-detect)
    discharge_column : str, optional
        Name of discharge column (if not specified, attempts to auto-detect)
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
    
    # Auto-detect columns if not specified
    if datetime_column is None:
        possible_dt_cols = ['datetime', 'DATETIME', 'timestamp', 'Timestamp', 'Date']
        for col in possible_dt_cols:
            if col in df.columns:
                datetime_column = col
                break
        if datetime_column is None:
            raise ValueError(f"Could not auto-detect datetime column. Please specify. Available: {list(df.columns)}")

    if discharge_column is None:
        possible_q_cols = ['00060_Mean', 'discharge', 'flow', 'Flow', 'Discharge']
        for col in possible_q_cols:
            if col in df.columns:
                discharge_column = col
                break
        if discharge_column is None:
            raise ValueError(f"Could not auto-detect discharge column. Please specify. Available: {list(df.columns)}")

    # Rename to standard names
    rename_dict = {
        datetime_column: "datetime",
        discharge_column: "discharge"
    }
    if quality_column and quality_column in df.columns:
        rename_dict[quality_column] = "quality"
    
    df = df.rename(columns=rename_dict)

    # Parse datetime
    df["datetime"] = pd.to_datetime(df["datetime"])
    
    # Handle timezone
    if df["datetime"].dt.tz is None:
        # Assume UTC if no timezone info (USGS data is often in UTC)
        df["datetime"] = df["datetime"].dt.tz_localize('UTC')
    
    df = df.set_index("datetime")
    df = df.tz_convert(timezone)
    
    # Convert discharge to numeric, coercing errors
    df['discharge'] = pd.to_numeric(df['discharge'], errors='coerce')
    
    # Select final columns
    final_cols = ["discharge"]
    if "quality" in df.columns:
        final_cols.append("quality")
        
    return df[final_cols]


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
