"""
Flood frequency analysis module.

Provides functions for fitting flood frequency distributions,
computing return period quantiles, and bootstrap confidence intervals.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass


@dataclass
class FloodQuantileResult:
    """Results for a single return period."""
    return_period: float
    estimate: float
    ci_lower: float
    ci_upper: float
    std: float


def extract_annual_maxima(
    df: pd.DataFrame,
    discharge_col: str = "discharge",
    use_water_year: bool = True
) -> pd.DataFrame:
    """
    Extract annual maximum discharge series.
    
    Parameters
    ----------
    df : pandas.DataFrame
        Streamflow data with datetime index
    discharge_col : str
        Name of discharge column
    use_water_year : bool
        If True, use water year (Oct-Sep). If False, use calendar year.
    
    Returns
    -------
    pandas.DataFrame
        Annual maxima with columns: year, max_discharge, max_date
    """
    df = df.copy()
    
    if use_water_year:
        # Water year: Oct 1 to Sep 30
        # WY 2024 = Oct 2023 to Sep 2024
        df['water_year'] = df.index.year
        df.loc[df.index.month >= 10, 'water_year'] += 1
        group_col = 'water_year'
    else:
        group_col = df.index.year
        df['calendar_year'] = df.index.year
        group_col = 'calendar_year'
    
    # Find max for each year
    idx_max = df.groupby(group_col)[discharge_col].idxmax()
    
    results = []
    for year, idx in idx_max.items():
        if pd.isna(idx):
            continue
        results.append({
            'year': year,
            'max_discharge': df.loc[idx, discharge_col],
            'max_date': idx
        })
    
    return pd.DataFrame(results)


def fit_gumbel(annual_maxima: np.ndarray) -> Tuple[float, float]:
    """
    Fit Gumbel (Type I Extreme Value) distribution using L-moments.
    
    The Gumbel distribution is commonly used for flood frequency analysis.
    
    Parameters
    ----------
    annual_maxima : array
        Annual maximum discharge series
    
    Returns
    -------
    tuple
        (location, scale) parameters
        - location (μ): mode of distribution
        - scale (σ): dispersion parameter
    
    Notes
    -----
    L-moment estimators:
        σ = L2 / ln(2)
        μ = L1 - 0.5772 * σ
    
    where 0.5772... is the Euler-Mascheroni constant.
    
    References
    ----------
    Hosking, J. R. M. (1990). L-moments: analysis and estimation of 
    distributions using linear combinations of order statistics.
    """
    x = np.sort(annual_maxima)
    n = len(x)
    
    if n < 2:
        raise ValueError("Need at least 2 observations to fit Gumbel")
    
    # L-moments
    # L1 = mean
    L1 = np.mean(x)
    
    # L2 = 2*b1 - b0 where b_r = (1/n) * sum of (choose(i-1,r) / choose(n-1,r)) * x_i
    b0 = L1
    
    # For b1: sum of (i-1)/(n-1) * x[i] for i = 1 to n
    weights = np.arange(n) / (n - 1)
    b1 = np.mean(weights * x)
    
    L2 = 2 * b1 - b0
    
    # Gumbel parameters
    EULER = 0.5772156649015329  # Euler-Mascheroni constant
    
    scale = L2 / np.log(2)
    location = L1 - EULER * scale
    
    return location, scale


def gumbel_quantile(T: float, location: float, scale: float) -> float:
    """
    Compute T-year flood quantile from Gumbel distribution.
    
    Parameters
    ----------
    T : float
        Return period in years
    location : float
        Gumbel location parameter
    scale : float
        Gumbel scale parameter
    
    Returns
    -------
    float
        Discharge with return period T
    
    Notes
    -----
    The quantile function is:
        Q_T = μ - σ * ln(-ln(1 - 1/T))
    
    where 1/T is the annual exceedance probability.
    """
    if T <= 1:
        raise ValueError("Return period must be > 1")
    
    p = 1 - 1/T  # Non-exceedance probability
    y = -np.log(-np.log(p))  # Gumbel reduced variate
    
    return location + scale * y


def gumbel_cdf(x: float, location: float, scale: float) -> float:
    """
    Compute Gumbel CDF (non-exceedance probability).
    
    Parameters
    ----------
    x : float
        Discharge value
    location : float
        Gumbel location parameter
    scale : float
        Gumbel scale parameter
    
    Returns
    -------
    float
        Non-exceedance probability P(X <= x)
    """
    z = (x - location) / scale
    return np.exp(-np.exp(-z))


def compute_return_period(x: float, location: float, scale: float) -> float:
    """
    Compute return period for a given discharge.
    
    Parameters
    ----------
    x : float
        Discharge value
    location : float
        Gumbel location parameter
    scale : float
        Gumbel scale parameter
    
    Returns
    -------
    float
        Return period in years
    """
    p = gumbel_cdf(x, location, scale)
    if p >= 1:
        return np.inf
    return 1 / (1 - p)


def bootstrap_flood_quantiles(
    annual_maxima: np.ndarray,
    return_periods: List[float] = [2, 5, 10, 25, 50, 100],
    n_bootstrap: int = 1000,
    confidence_level: float = 0.95,
    random_state: Optional[int] = None
) -> Dict[float, FloodQuantileResult]:
    """
    Bootstrap confidence intervals for flood quantiles.
    
    Parameters
    ----------
    annual_maxima : array
        Annual maximum discharge series
    return_periods : list of float
        Return periods to compute
    n_bootstrap : int
        Number of bootstrap samples
    confidence_level : float
        Confidence level (e.g., 0.95 for 95% CI)
    random_state : int, optional
        Random seed for reproducibility
    
    Returns
    -------
    dict
        {return_period: FloodQuantileResult}
    """
    if random_state is not None:
        np.random.seed(random_state)
    
    n = len(annual_maxima)
    alpha = (1 - confidence_level) / 2
    
    # Storage for bootstrap samples
    bootstrap_quantiles = {T: [] for T in return_periods}
    
    # Point estimates
    try:
        loc, scale = fit_gumbel(annual_maxima)
        point_estimates = {T: gumbel_quantile(T, loc, scale) for T in return_periods}
    except Exception:
        return {T: FloodQuantileResult(T, np.nan, np.nan, np.nan, np.nan) 
                for T in return_periods}
    
    # Bootstrap
    for _ in range(n_bootstrap):
        # Resample with replacement
        sample = np.random.choice(annual_maxima, size=n, replace=True)
        
        try:
            loc_boot, scale_boot = fit_gumbel(sample)
            for T in return_periods:
                Q_T = gumbel_quantile(T, loc_boot, scale_boot)
                bootstrap_quantiles[T].append(Q_T)
        except Exception:
            continue
    
    # Compute statistics
    results = {}
    for T in return_periods:
        samples = np.array(bootstrap_quantiles[T])
        if len(samples) > 0:
            results[T] = FloodQuantileResult(
                return_period=T,
                estimate=point_estimates[T],
                ci_lower=np.percentile(samples, alpha * 100),
                ci_upper=np.percentile(samples, (1 - alpha) * 100),
                std=np.std(samples)
            )
        else:
            results[T] = FloodQuantileResult(T, np.nan, np.nan, np.nan, np.nan)
    
    return results


def compare_flood_quantiles(
    pre_maxima: np.ndarray,
    post_maxima: np.ndarray,
    return_periods: List[float] = [2, 5, 10, 25, 50, 100],
    n_bootstrap: int = 1000,
    confidence_level: float = 0.95
) -> pd.DataFrame:
    """
    Compare flood quantiles between pre and post periods.
    
    Parameters
    ----------
    pre_maxima : array
        Pre-fire annual maxima
    post_maxima : array
        Post-fire annual maxima
    return_periods : list of float
        Return periods to compare
    n_bootstrap : int
        Bootstrap iterations
    confidence_level : float
        Confidence level for intervals
    
    Returns
    -------
    pandas.DataFrame
        Comparison table with ratios and confidence intervals
    """
    pre_results = bootstrap_flood_quantiles(
        pre_maxima, return_periods, n_bootstrap, confidence_level
    )
    post_results = bootstrap_flood_quantiles(
        post_maxima, return_periods, n_bootstrap, confidence_level
    )
    
    # Also bootstrap the ratio
    n_pre = len(pre_maxima)
    n_post = len(post_maxima)
    ratio_samples = {T: [] for T in return_periods}
    
    for _ in range(n_bootstrap):
        try:
            pre_sample = np.random.choice(pre_maxima, size=n_pre, replace=True)
            post_sample = np.random.choice(post_maxima, size=n_post, replace=True)
            
            loc_pre, scale_pre = fit_gumbel(pre_sample)
            loc_post, scale_post = fit_gumbel(post_sample)
            
            for T in return_periods:
                q_pre = gumbel_quantile(T, loc_pre, scale_pre)
                q_post = gumbel_quantile(T, loc_post, scale_post)
                if q_pre > 0:
                    ratio_samples[T].append(q_post / q_pre)
        except Exception:
            continue
    
    # Compile results
    alpha = (1 - confidence_level) / 2
    rows = []
    for T in return_periods:
        pre = pre_results[T]
        post = post_results[T]
        
        ratio = post.estimate / pre.estimate if pre.estimate > 0 else np.nan
        
        rs = np.array(ratio_samples[T])
        if len(rs) > 0:
            ratio_ci_lower = np.percentile(rs, alpha * 100)
            ratio_ci_upper = np.percentile(rs, (1 - alpha) * 100)
        else:
            ratio_ci_lower = ratio_ci_upper = np.nan
        
        rows.append({
            'return_period': T,
            'pre_estimate': pre.estimate,
            'pre_ci_lower': pre.ci_lower,
            'pre_ci_upper': pre.ci_upper,
            'post_estimate': post.estimate,
            'post_ci_lower': post.ci_lower,
            'post_ci_upper': post.ci_upper,
            'ratio': ratio,
            'ratio_ci_lower': ratio_ci_lower,
            'ratio_ci_upper': ratio_ci_upper,
            'change_pct': (ratio - 1) * 100 if not np.isnan(ratio) else np.nan
        })
    
    return pd.DataFrame(rows)


def gumbel_plotting_positions(annual_maxima: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute Gumbel plotting positions for probability plot.
    
    Parameters
    ----------
    annual_maxima : array
        Annual maximum series
    
    Returns
    -------
    tuple
        (reduced_variate, sorted_maxima)
        - reduced_variate: Gumbel reduced variate y = -ln(-ln(F))
        - sorted_maxima: discharge values (ascending)
    """
    x = np.sort(annual_maxima)
    n = len(x)
    
    # Gringorten plotting position
    ranks = np.arange(1, n + 1)
    F = (ranks - 0.44) / (n + 0.12)
    
    # Gumbel reduced variate
    y = -np.log(-np.log(F))
    
    return y, x
