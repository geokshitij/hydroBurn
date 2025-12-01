"""
Baseflow separation module.

Implements the Lyne-Hollick recursive digital filter for separating
baseflow and quickflow components from streamflow hydrographs.
"""

import numpy as np
import pandas as pd
from typing import Tuple, Union


def lyne_hollick_filter(
    Q: Union[np.ndarray, pd.Series],
    alpha: float = 0.925,
    passes: int = 3,
    reflect: bool = True
) -> np.ndarray:
    """
    Lyne-Hollick recursive digital filter for baseflow separation.
    
    This filter separates the quickflow (direct runoff) component from
    the total streamflow, with baseflow computed as the residual.
    
    Parameters
    ----------
    Q : array or Series
        Total streamflow time series
    alpha : float
        Filter parameter (0.9-0.98). Higher values = more smoothing,
        less baseflow. Recommended: 0.925 (Nathan & McMahon, 1990)
    passes : int
        Number of filter passes. Typically 3 (forward, backward, forward)
        for better separation. Must be odd for consistent results.
    reflect : bool
        Whether to reflect the series at boundaries to reduce edge effects
    
    Returns
    -------
    numpy.ndarray
        Baseflow component (same length as Q)
    
    Notes
    -----
    The filter equation is:
        Qf[i] = alpha * Qf[i-1] + (1 + alpha) / 2 * (Q[i] - Q[i-1])
    
    Where Qf is quickflow. Baseflow is computed as:
        Qb = Q - Qf
    
    References
    ----------
    Lyne, V., & Hollick, M. (1979). Stochastic time-variable rainfall-runoff 
    modelling. Institute of Engineers Australia National Conference.
    
    Nathan, R. J., & McMahon, T. A. (1990). Evaluation of automated techniques 
    for base flow and recession analyses. Water Resources Research, 26(7), 1465-1473.
    """
    # Convert to numpy array
    if isinstance(Q, pd.Series):
        Q = Q.values.copy()
    else:
        Q = np.asarray(Q).copy()
    
    # Handle NaN values
    nan_mask = np.isnan(Q)
    if nan_mask.any():
        # Interpolate NaN values temporarily
        Q_interp = pd.Series(Q).interpolate().values
        Q_interp = np.nan_to_num(Q_interp, nan=0)
    else:
        Q_interp = Q.copy()
    
    n = len(Q_interp)
    
    # Reflect boundaries if requested
    if reflect and n > 10:
        pad = min(30, n // 4)
        Q_padded = np.concatenate([
            Q_interp[pad:0:-1],  # Reflect start
            Q_interp,
            Q_interp[-2:-pad-2:-1]  # Reflect end
        ])
    else:
        Q_padded = Q_interp
        pad = 0
    
    n_padded = len(Q_padded)
    Qf = np.zeros(n_padded)  # Quickflow
    
    # Filter coefficient
    c = (1 + alpha) / 2
    
    for p in range(passes):
        Qf_prev = Qf.copy()
        
        if p % 2 == 0:
            # Forward pass
            for i in range(1, n_padded):
                Qf[i] = alpha * Qf[i-1] + c * (Q_padded[i] - Q_padded[i-1])
        else:
            # Backward pass
            for i in range(n_padded - 2, -1, -1):
                Qf[i] = alpha * Qf[i+1] + c * (Q_padded[i] - Q_padded[i+1])
        
        # Constrain quickflow
        Qf = np.maximum(0, Qf)  # Non-negative
        Qf = np.minimum(Qf, Q_padded)  # Cannot exceed total flow
    
    # Remove padding
    if pad > 0:
        Qf = Qf[pad:-pad]
    
    # Compute baseflow
    Qb = Q_interp - Qf
    
    # Ensure baseflow is non-negative and doesn't exceed total
    Qb = np.maximum(0, Qb)
    Qb = np.minimum(Qb, Q_interp)
    
    # Restore NaN values
    Qb[nan_mask] = np.nan
    
    return Qb


def separate_baseflow(
    df: pd.DataFrame,
    discharge_col: str = "discharge",
    alpha: float = 0.925,
    passes: int = 3
) -> pd.DataFrame:
    """
    Add baseflow and quickflow columns to DataFrame.
    
    Parameters
    ----------
    df : pandas.DataFrame
        Streamflow data with discharge column
    discharge_col : str
        Name of discharge column
    alpha : float
        Lyne-Hollick filter parameter
    passes : int
        Number of filter passes
    
    Returns
    -------
    pandas.DataFrame
        DataFrame with 'baseflow' and 'quickflow' columns added
    """
    df = df.copy()
    
    Q = df[discharge_col].values
    Qb = lyne_hollick_filter(Q, alpha=alpha, passes=passes)
    
    df['baseflow'] = Qb
    df['quickflow'] = df[discharge_col] - df['baseflow']
    
    # Handle edge cases
    df['quickflow'] = df['quickflow'].clip(lower=0)
    
    return df


def compute_baseflow_index(
    df: pd.DataFrame,
    discharge_col: str = "discharge",
    baseflow_col: str = "baseflow",
    window: int = None
) -> Union[float, pd.Series]:
    """
    Compute baseflow index (BFI).
    
    BFI is the ratio of baseflow to total flow, indicating the proportion
    of streamflow derived from groundwater or stored water.
    
    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame with discharge and baseflow columns
    discharge_col : str
        Name of total discharge column
    baseflow_col : str
        Name of baseflow column
    window : int, optional
        Rolling window size (days) for time-varying BFI.
        If None, returns single BFI value for entire series.
    
    Returns
    -------
    float or pandas.Series
        BFI value(s) in range [0, 1]
    
    Notes
    -----
    BFI = sum(Qb) / sum(Q)
    
    Higher BFI indicates more groundwater contribution.
    Typical values:
    - Arid ephemeral streams: 0.1-0.3
    - Humid perennial streams: 0.5-0.8
    - Spring-fed streams: 0.7-0.95
    """
    if window is None:
        # Overall BFI
        total_Q = df[discharge_col].sum()
        total_Qb = df[baseflow_col].sum()
        if total_Q > 0:
            return total_Qb / total_Q
        return np.nan
    else:
        # Rolling BFI
        rolling_Q = df[discharge_col].rolling(window, min_periods=window//2).sum()
        rolling_Qb = df[baseflow_col].rolling(window, min_periods=window//2).sum()
        bfi = rolling_Qb / rolling_Q
        bfi = bfi.clip(0, 1)
        return bfi


def compute_quickflow_fraction(
    df: pd.DataFrame,
    discharge_col: str = "discharge",
    quickflow_col: str = "quickflow",
    window: int = None
) -> Union[float, pd.Series]:
    """
    Compute quickflow fraction (1 - BFI).
    
    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame with discharge and quickflow columns
    discharge_col : str
        Name of total discharge column
    quickflow_col : str
        Name of quickflow column
    window : int, optional
        Rolling window size for time-varying fraction
    
    Returns
    -------
    float or pandas.Series
        Quickflow fraction in range [0, 1]
    """
    if window is None:
        total_Q = df[discharge_col].sum()
        total_Qf = df[quickflow_col].sum()
        if total_Q > 0:
            return total_Qf / total_Q
        return np.nan
    else:
        rolling_Q = df[discharge_col].rolling(window, min_periods=window//2).sum()
        rolling_Qf = df[quickflow_col].rolling(window, min_periods=window//2).sum()
        qf_frac = rolling_Qf / rolling_Q
        qf_frac = qf_frac.clip(0, 1)
        return qf_frac


def compute_annual_bfi(
    df: pd.DataFrame,
    discharge_col: str = "discharge",
    baseflow_col: str = "baseflow"
) -> pd.DataFrame:
    """
    Compute annual baseflow index.
    
    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame with datetime index and flow columns
    discharge_col : str
        Name of total discharge column
    baseflow_col : str
        Name of baseflow column
    
    Returns
    -------
    pandas.DataFrame
        Annual BFI with columns: year, bfi, total_flow, baseflow
    """
    annual = df.groupby(df.index.year).agg({
        discharge_col: 'sum',
        baseflow_col: 'sum'
    }).reset_index()
    
    annual.columns = ['year', 'total_flow', 'baseflow']
    annual['bfi'] = annual['baseflow'] / annual['total_flow']
    annual['bfi'] = annual['bfi'].clip(0, 1)
    
    return annual
