"""
Quality control module for streamflow data.

Provides functions for gap detection, gap filling, and data validation.
"""

import pandas as pd
import numpy as np
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass
import warnings


@dataclass
class GapInfo:
    """Information about a gap in the time series."""
    start: pd.Timestamp
    end: pd.Timestamp
    duration_days: float
    
    @property
    def duration(self) -> pd.Timedelta:
        return self.end - self.start


def detect_gaps(
    df: pd.DataFrame,
    expected_freq: str = "D",
    discharge_col: str = "discharge"
) -> List[GapInfo]:
    """
    Detect gaps in streamflow time series.
    
    Parameters
    ----------
    df : pandas.DataFrame
        Streamflow data with datetime index
    expected_freq : str
        Expected frequency ('D' for daily, 'H' for hourly)
    discharge_col : str
        Name of discharge column
    
    Returns
    -------
    list of GapInfo
        List of detected gaps
    """
    gaps = []
    
    # Check for missing timestamps
    if not isinstance(expected_freq, pd.Timedelta):
        expected_dt = pd.Timedelta(f"1{expected_freq}")
    else:
        expected_dt = expected_freq
    time_diffs = df.index.to_series().diff()
    
    # Find gaps in timestamps
    gap_mask = time_diffs > expected_dt * 1.5  # Allow 50% tolerance
    gap_indices = df.index[gap_mask]
    
    for idx in gap_indices:
        # Get the timestamp before this gap
        loc = df.index.get_loc(idx)
        if loc > 0:
            start = df.index[loc - 1]
            end = idx
            duration = (end - start).days
            gaps.append(GapInfo(start=start, end=end, duration_days=duration))
    
    # Also check for NaN values
    nan_mask = df[discharge_col].isna()
    if nan_mask.any():
        # Group consecutive NaNs
        nan_groups = (nan_mask != nan_mask.shift()).cumsum()
        for group_id in nan_groups[nan_mask].unique():
            group_idx = df.index[nan_groups == group_id]
            if len(group_idx) > 0:
                start = group_idx.min()
                end = group_idx.max()
                duration = (end - start).days + 1
                gaps.append(GapInfo(start=start, end=end, duration_days=duration))
    
    # Sort by start date and remove duplicates
    gaps = sorted(gaps, key=lambda x: x.start)
    
    return gaps


def fill_gaps(
    df: pd.DataFrame,
    max_gap_days: int = 2,
    method: str = "linear",
    discharge_col: str = "discharge"
) -> Tuple[pd.DataFrame, List[GapInfo]]:
    """
    Fill short gaps in streamflow data.
    
    Parameters
    ----------
    df : pandas.DataFrame
        Streamflow data with datetime index
    max_gap_days : int
        Maximum gap length to fill (days)
    method : str
        Interpolation method ('linear', 'nearest', 'zero')
    discharge_col : str
        Name of discharge column
    
    Returns
    -------
    tuple
        (filled_df, list of filled gaps)
    """
    df = df.copy()
    filled_gaps = []
    
    # Get gaps before filling
    gaps_before = detect_gaps(df, discharge_col=discharge_col)
    
    # Only fill gaps up to max_gap_days
    short_gaps = [g for g in gaps_before if g.duration_days <= max_gap_days]
    
    # Interpolate NaN values
    if df[discharge_col].isna().any():
        # Create mask for values that will be filled
        was_nan = df[discharge_col].isna()
        
        # Interpolate
        df[discharge_col] = df[discharge_col].interpolate(
            method=method,
            limit=max_gap_days,
            limit_direction='both'
        )
        
        # Track which gaps were filled
        filled_mask = was_nan & ~df[discharge_col].isna()
        if filled_mask.any():
            filled_gaps = short_gaps
    
    # Ensure no negative values after interpolation
    df[discharge_col] = df[discharge_col].clip(lower=0)
    
    return df, filled_gaps


def quality_control(
    df: pd.DataFrame,
    max_gap_days: int = 2,
    flag_negative: bool = True,
    flag_zeros: bool = False,
    discharge_col: str = "discharge"
) -> Tuple[pd.DataFrame, Dict]:
    """
    Perform comprehensive quality control on streamflow data.
    
    Parameters
    ----------
    df : pandas.DataFrame
        Streamflow data with datetime index
    max_gap_days : int
        Maximum gap length to fill
    flag_negative : bool
        Whether to flag negative values as issues
    flag_zeros : bool
        Whether to flag zero values as issues
    discharge_col : str
        Name of discharge column
    
    Returns
    -------
    tuple
        (cleaned_df, qc_report dict)
    """
    df = df.copy()
    qc_report = {
        'original_records': len(df),
        'issues': [],
        'gaps_detected': [],
        'gaps_filled': [],
        'negative_values': 0,
        'zero_values': 0,
        'nan_values_initial': 0,
        'nan_values_final': 0,
    }
    
    # Count initial NaNs
    qc_report['nan_values_initial'] = df[discharge_col].isna().sum()
    
    # Detect gaps
    gaps = detect_gaps(df, discharge_col=discharge_col)
    qc_report['gaps_detected'] = gaps
    
    # Report long gaps as issues
    long_gaps = [g for g in gaps if g.duration_days > max_gap_days]
    for gap in long_gaps:
        qc_report['issues'].append(
            f"Long gap ({gap.duration_days:.0f} days): {gap.start} to {gap.end}"
        )
    
    # Handle negative values
    neg_mask = df[discharge_col] < 0
    qc_report['negative_values'] = neg_mask.sum()
    if neg_mask.any():
        if flag_negative:
            qc_report['issues'].append(
                f"Found {neg_mask.sum()} negative discharge values (set to NaN)"
            )
            df.loc[neg_mask, discharge_col] = np.nan
        else:
            warnings.warn(f"Found {neg_mask.sum()} negative values")
    
    # Count zeros
    zero_mask = df[discharge_col] == 0
    qc_report['zero_values'] = zero_mask.sum()
    if flag_zeros and zero_mask.any():
        qc_report['issues'].append(
            f"Found {zero_mask.sum()} zero discharge values"
        )
    
    # Fill short gaps
    df, filled = fill_gaps(df, max_gap_days=max_gap_days, discharge_col=discharge_col)
    qc_report['gaps_filled'] = filled
    
    # Count final NaNs
    qc_report['nan_values_final'] = df[discharge_col].isna().sum()
    qc_report['final_records'] = len(df)
    qc_report['valid_records'] = (~df[discharge_col].isna()).sum()
    
    return df, qc_report


def add_qc_flags(
    df: pd.DataFrame,
    discharge_col: str = "discharge",
    quality_col: Optional[str] = "quality"
) -> pd.DataFrame:
    """
    Add QC flag column to DataFrame.
    
    Flags:
    - 0: Good data
    - 1: Interpolated
    - 2: Zero flow (may be valid in intermittent streams)
    - 3: Provisional data
    - 9: Missing/invalid
    
    Parameters
    ----------
    df : pandas.DataFrame
        Streamflow data
    discharge_col : str
        Name of discharge column
    quality_col : str, optional
        Name of existing quality code column
    
    Returns
    -------
    pandas.DataFrame
        DataFrame with 'qc_flag' column added
    """
    df = df.copy()
    df['qc_flag'] = 0  # Default: good
    
    # Flag missing values
    df.loc[df[discharge_col].isna(), 'qc_flag'] = 9
    
    # Flag zeros
    df.loc[df[discharge_col] == 0, 'qc_flag'] = 2
    
    # Flag provisional data (if quality column exists)
    if quality_col and quality_col in df.columns:
        df.loc[df[quality_col] == 'P', 'qc_flag'] = 3
    
    return df
