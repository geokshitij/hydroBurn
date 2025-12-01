"""
Flow duration curve module.

Provides functions for computing and comparing flow duration curves.
"""

import numpy as np
import pandas as pd
from typing import Tuple, Dict, List, Optional


def compute_fdc(
    Q: pd.Series,
    n_points: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute flow duration curve.
    
    The FDC shows the percentage of time that flow equals or exceeds
    a given value.
    
    Parameters
    ----------
    Q : pandas.Series
        Discharge time series
    n_points : int, optional
        Number of points to return. If None, returns all points.
    
    Returns
    -------
    tuple
        (exceedance_probability, discharge)
        - exceedance: percentage of time flow is equaled or exceeded (0-100%)
        - discharge: sorted discharge values (descending)
    
    Notes
    -----
    Uses Weibull plotting position:
        P = m / (n + 1) * 100
    where m is rank and n is sample size.
    """
    Q_clean = Q.dropna()
    
    if len(Q_clean) == 0:
        return np.array([]), np.array([])
    
    # Sort descending (highest flow first)
    Q_sorted = np.sort(Q_clean.values)[::-1]
    n = len(Q_sorted)
    
    # Weibull plotting position
    ranks = np.arange(1, n + 1)
    exceedance = ranks / (n + 1) * 100
    
    # Subsample if requested
    if n_points is not None and n_points < n:
        indices = np.linspace(0, n - 1, n_points, dtype=int)
        exceedance = exceedance[indices]
        Q_sorted = Q_sorted[indices]
    
    return exceedance, Q_sorted


def get_fdc_percentiles(
    Q: pd.Series,
    percentiles: List[float] = [1, 5, 10, 25, 50, 75, 90, 95, 99]
) -> Dict[float, float]:
    """
    Get flow values at specific exceedance percentiles.
    
    Parameters
    ----------
    Q : pandas.Series
        Discharge time series
    percentiles : list of float
        Exceedance percentiles to compute (0-100)
    
    Returns
    -------
    dict
        {percentile: discharge_value}
    
    Notes
    -----
    Q10 means flow exceeded 10% of the time (high flow)
    Q90 means flow exceeded 90% of the time (low flow)
    """
    Q_clean = Q.dropna()
    
    if len(Q_clean) == 0:
        return {p: np.nan for p in percentiles}
    
    # For exceedance probability, we need to invert the percentile
    # P(Q >= q) = p means q is the (100-p)th percentile
    result = {}
    for p in percentiles:
        q = np.percentile(Q_clean.values, 100 - p)
        result[p] = q
    
    return result


def compare_fdc(
    Q_pre: pd.Series,
    Q_post: pd.Series,
    percentiles: List[float] = [1, 5, 10, 25, 50, 75, 90, 95, 99]
) -> pd.DataFrame:
    """
    Compare flow duration curves between two periods.
    
    Parameters
    ----------
    Q_pre : pandas.Series
        Pre-fire discharge
    Q_post : pandas.Series
        Post-fire discharge
    percentiles : list of float
        Exceedance percentiles to compare
    
    Returns
    -------
    pandas.DataFrame
        Comparison with columns:
        - exceedance_pct: exceedance probability
        - pre_discharge: pre-fire discharge at this exceedance
        - post_discharge: post-fire discharge
        - change_pct: percent change
        - ratio: post/pre ratio
    """
    pre_pcts = get_fdc_percentiles(Q_pre, percentiles)
    post_pcts = get_fdc_percentiles(Q_post, percentiles)
    
    results = []
    for p in percentiles:
        q_pre = pre_pcts[p]
        q_post = post_pcts[p]
        
        if q_pre > 0:
            change_pct = (q_post - q_pre) / q_pre * 100
            ratio = q_post / q_pre
        else:
            change_pct = np.nan
            ratio = np.nan
        
        results.append({
            'exceedance_pct': p,
            'pre_discharge': q_pre,
            'post_discharge': q_post,
            'change_pct': change_pct,
            'ratio': ratio
        })
    
    return pd.DataFrame(results)


def compute_fdc_signature(Q: pd.Series) -> Dict[str, float]:
    """
    Compute FDC signature metrics.
    
    These metrics characterize the shape of the FDC and are useful
    for comparing hydrologic regimes.
    
    Parameters
    ----------
    Q : pandas.Series
        Discharge time series
    
    Returns
    -------
    dict
        FDC signature metrics:
        - q_mean: mean discharge
        - q_median: median discharge
        - q10: Q10 (high flow)
        - q50: Q50 (median flow)
        - q90: Q90 (low flow)
        - q10_q50_ratio: Q10/Q50 (high flow variability)
        - q50_q90_ratio: Q50/Q90 (low flow variability)
        - fdc_slope: slope of FDC between Q33 and Q66
        - flow_variability: coefficient of variation
        - baseflow_stability: Q90/Q50
    """
    Q_clean = Q.dropna()
    
    if len(Q_clean) == 0:
        return {}
    
    pcts = get_fdc_percentiles(Q, [10, 33, 50, 66, 90])
    
    q10 = pcts[10]
    q33 = pcts[33]
    q50 = pcts[50]
    q66 = pcts[66]
    q90 = pcts[90]
    
    return {
        'q_mean': Q_clean.mean(),
        'q_median': Q_clean.median(),
        'q10': q10,
        'q50': q50,
        'q90': q90,
        'q10_q50_ratio': q10 / q50 if q50 > 0 else np.nan,
        'q50_q90_ratio': q50 / q90 if q90 > 0 else np.nan,
        'fdc_slope': (np.log10(q33) - np.log10(q66)) / (66 - 33)
                     if q33 > 0 and q66 > 0 else np.nan,
        'flow_variability': Q_clean.std() / Q_clean.mean() if Q_clean.mean() > 0 else np.nan,
        'baseflow_stability': q90 / q50 if q50 > 0 else np.nan,
    }


def compute_monthly_fdc(
    df: pd.DataFrame,
    discharge_col: str = "discharge",
    percentiles: List[float] = [10, 50, 90]
) -> pd.DataFrame:
    """
    Compute monthly flow statistics.
    
    Parameters
    ----------
    df : pandas.DataFrame
        Streamflow data with datetime index
    discharge_col : str
        Name of discharge column
    percentiles : list of float
        Exceedance percentiles to compute
    
    Returns
    -------
    pandas.DataFrame
        Monthly statistics
    """
    monthly = df.groupby(df.index.month)[discharge_col].agg([
        'mean', 'median', 'std', 'min', 'max',
        lambda x: np.percentile(x.dropna(), 100 - 10),  # Q10
        lambda x: np.percentile(x.dropna(), 100 - 90),  # Q90
    ])
    monthly.columns = ['mean', 'median', 'std', 'min', 'max', 'q10', 'q90']
    monthly.index.name = 'month'
    
    return monthly.reset_index()
