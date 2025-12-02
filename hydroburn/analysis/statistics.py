"""
Statistical analysis module.

Provides functions for statistical comparison of pre-fire and post-fire
streamflow characteristics.
"""

import numpy as np
import pandas as pd
from scipy import stats
from typing import Dict, Tuple, List, Optional


def pre_post_statistical_tests(
    pre: np.ndarray,
    post: np.ndarray,
    alpha: float = 0.05
) -> Dict:
    """
    Comprehensive statistical comparison of pre/post distributions.
    
    Performs multiple statistical tests to detect changes in distribution.
    
    Parameters
    ----------
    pre : array
        Pre-fire values
    post : array
        Post-fire values
    alpha : float
        Significance level
    
    Returns
    -------
    dict
        Test results including statistics, p-values, and interpretation
    """
    pre = np.asarray(pre)
    post = np.asarray(post)
    
    # Remove NaN values
    pre = pre[~np.isnan(pre)]
    post = post[~np.isnan(post)]
    
    results = {}
    
    # Check sample sizes
    if len(pre) < 3 or len(post) < 3:
        return {'error': 'Insufficient sample size (need at least 3 in each group)'}
    
    # Mann-Whitney U test (non-parametric, tests for shift)
    try:
        stat, p = stats.mannwhitneyu(pre, post, alternative='two-sided')
        results['mann_whitney'] = {
            'statistic': stat,
            'pvalue': p,
            'significant': p < alpha,
            'interpretation': 'Distributions differ' if p < alpha else 'No significant difference'
        }
    except Exception as e:
        results['mann_whitney'] = {'error': str(e)}
    
    # Kolmogorov-Smirnov test (tests distribution shape)
    try:
        stat, p = stats.ks_2samp(pre, post)
        results['ks_test'] = {
            'statistic': stat,
            'pvalue': p,
            'significant': p < alpha,
            'interpretation': 'Distribution shapes differ' if p < alpha else 'Similar distribution shapes'
        }
    except Exception as e:
        results['ks_test'] = {'error': str(e)}
    
    # Welch's t-test (robust to unequal variances)
    try:
        stat, p = stats.ttest_ind(pre, post, equal_var=False)
        results['welch_t'] = {
            'statistic': stat,
            'pvalue': p,
            'significant': p < alpha,
            'interpretation': 'Means differ' if p < alpha else 'No significant difference in means'
        }
    except Exception as e:
        results['welch_t'] = {'error': str(e)}
    
    # Levene's test (tests variance equality)
    try:
        stat, p = stats.levene(pre, post)
        results['levene'] = {
            'statistic': stat,
            'pvalue': p,
            'significant': p < alpha,
            'interpretation': 'Variances differ' if p < alpha else 'Similar variances'
        }
    except Exception as e:
        results['levene'] = {'error': str(e)}
    
    # Brunner-Munzel test (non-parametric, robust)
    try:
        stat, p = stats.brunnermunzel(pre, post)
        results['brunner_munzel'] = {
            'statistic': stat,
            'pvalue': p,
            'significant': p < alpha,
            'interpretation': 'Distributions differ' if p < alpha else 'No significant difference'
        }
    except Exception as e:
        results['brunner_munzel'] = {'error': str(e)}
        
    # Effect size (Cliff's Delta)
    try:
        # Calculate ranks
        all_values = np.concatenate([pre, post])
        rank_vals = stats.rankdata(np.concatenate([pre, post]))
        
        # Separate ranks
        n = len(pre)
        rank_pre = rank_vals[:n]
        rank_post = rank_vals[n:]
        
        # Calculate Cliff's Delta
        cliffs_delta = (np.sum(rank_pre) / (n * len(post)) - (n + 1) / 2) - \
                       (np.sum(rank_post) / (n * len(pre)) - (n + 1) / 2)
        
        results['effect'] = {
            'cliffs_delta': cliffs_delta,
            'percent_change_median': ((np.median(post) - np.median(pre)) / np.median(pre) * 100) 
                                  if np.median(pre) > 0 else np.nan,
            'percent_change_mean': ((np.mean(post) - np.mean(pre)) / np.mean(pre) * 100)
                                if np.mean(pre) > 0 else np.nan
        }
    except Exception as e:
        results['effect'] = {'error': str(e)}
    
    return results


def perform_before_after_analysis(
    df: pd.DataFrame,
    fire_years: List[int],
    window_years: int,
    discharge_col: str = "discharge"
) -> pd.DataFrame:
    """
    Performs a before-and-after analysis for each fire event.

    For each fire, it calculates summary statistics for a defined window
    before and after the event.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with a DatetimeIndex and discharge data.
    fire_years : List[int]
        A list of years in which significant fires occurred.
    window_years : int
        The number of years to include in the 'before' and 'after' windows.
    discharge_col : str
        The name of the discharge column.

    Returns
    -------
    pd.DataFrame
        A DataFrame containing the summary statistics for each fire's
        before and after period.
    """
    results = []
    window_delta = pd.Timedelta(days=window_years * 365.25)

    for year in fire_years:
        fire_date = pd.to_datetime(f'{year}-01-01')
        if df.index.tz is not None and fire_date.tz is None:
            fire_date = fire_date.tz_localize(df.index.tz)

        # Define before and after periods
        before_start = fire_date - window_delta
        before_end = fire_date - pd.Timedelta(days=1)
        after_start = fire_date
        after_end = fire_date + window_delta

        # Slice the dataframe
        df_before = df.loc[before_start:before_end]
        df_after = df.loc[after_start:after_end]

        if df_before.empty or df_after.empty:
            continue

        # Calculate stats
        stats_before = df_before[discharge_col].describe()
        stats_after = df_after[discharge_col].describe()

        # Combine and store
        period_results = pd.concat([stats_before, stats_after], keys=['before', 'after'])
        period_results['fire_year'] = year
        results.append(period_results.unstack(level=0))

    if not results:
        return pd.DataFrame()

    summary_df = pd.concat(results)
    summary_df = summary_df.reset_index().rename(columns={'index': 'statistic'})
    summary_df = summary_df.set_index(['fire_year', 'statistic'])

    return summary_df


def pettitt_test(x: np.ndarray) -> Tuple[int, float, float]:
    """
    Pettitt's test for detecting a single change point.
    
    Non-parametric test for detecting a shift in the central tendency
    of a time series.
    
    Parameters
    ----------
    x : array
        Time series values
    
    Returns
    -------
    tuple
        (change_point_index, test_statistic_K, p_value)
    
    References
    ----------
    Pettitt, A. N. (1979). A non-parametric approach to the change-point 
    problem. Applied Statistics, 28(2), 126-135.
    """
    x = np.asarray(x)
    x = x[~np.isnan(x)]
    n = len(x)
    
    if n < 4:
        return (0, 0, 1.0)
    
    # Compute U statistic for each potential change point
    U = np.zeros(n)
    
    for t in range(n):
        for i in range(t + 1):
            for j in range(t + 1, n):
                U[t] += np.sign(x[j] - x[i])
    
    # Maximum absolute value of U
    K = np.max(np.abs(U))
    change_point = np.argmax(np.abs(U))
    
    # Approximate p-value
    p_value = 2 * np.exp(-6 * K**2 / (n**3 + n**2))
    p_value = min(1.0, p_value)
    
    return change_point, K, p_value


def mann_kendall_test(x: np.ndarray) -> Dict:
    """
    Mann-Kendall test for monotonic trend.
    
    Non-parametric test for detecting trends in time series.
    
    Parameters
    ----------
    x : array
        Time series values
    
    Returns
    -------
    dict
        Test results including:
        - S: Mann-Kendall statistic
        - Z: normalized test statistic
        - p_value: two-tailed p-value
        - trend: 'increasing', 'decreasing', or 'no trend'
        - tau: Kendall's tau (correlation coefficient)
    """
    x = np.asarray(x)
    x = x[~np.isnan(x)]
    n = len(x)
    
    if n < 4:
        return {'error': 'Need at least 4 observations'}
    
    # Compute S statistic
    S = 0
    for i in range(n - 1):
        for j in range(i + 1, n):
            S += np.sign(x[j] - x[i])
    
    # Compute variance of S
    # Account for ties
    unique, counts = np.unique(x, return_counts=True)
    tie_correction = 0
    for t in counts:
        if t > 1:
            tie_correction += t * (t - 1) * (2 * t + 5)
    
    var_S = (n * (n - 1) * (2 * n + 5) - tie_correction) / 18
    
    # Compute Z statistic
    if S > 0:
        Z = (S - 1) / np.sqrt(var_S)
    elif S < 0:
        Z = (S + 1) / np.sqrt(var_S)
    else:
        Z = 0
    
    # Two-tailed p-value
    p_value = 2 * (1 - stats.norm.cdf(abs(Z)))
    
    # Kendall's tau
    n_pairs = n * (n - 1) / 2
    tau = S / n_pairs
    
    # Trend direction
    if p_value < 0.05:
        trend = 'increasing' if S > 0 else 'decreasing'
    else:
        trend = 'no trend'
    
    return {
        'S': S,
        'var_S': var_S,
        'Z': Z,
        'p_value': p_value,
        'tau': tau,
        'trend': trend,
        'n': n
    }


def sens_slope(x: np.ndarray) -> Tuple[float, float, float]:
    """
    Sen's slope estimator for trend magnitude.
    
    Robust, non-parametric estimate of the slope of a trend.
    
    Parameters
    ----------
    x : array
        Time series values
    
    Returns
    -------
    tuple
        (slope, intercept, slope_ci_95)
        - slope: median of all pairwise slopes
        - intercept: median intercept
        - slope_ci_95: 95% confidence interval width for slope
    """
    x = np.asarray(x)
    x = x[~np.isnan(x)]
    n = len(x)
    
    if n < 2:
        return (np.nan, np.nan, np.nan)
    
    # Compute all pairwise slopes
    slopes = []
    for i in range(n - 1):
        for j in range(i + 1, n):
            if j != i:
                slopes.append((x[j] - x[i]) / (j - i))
    
    slopes = np.array(slopes)
    slope = np.median(slopes)
    
    # Intercept
    times = np.arange(n)
    intercept = np.median(x - slope * times)
    
    # Confidence interval for slope
    slope_ci = np.percentile(slopes, [2.5, 97.5])
    
    return slope, intercept, slope_ci[1] - slope_ci[0]


def compute_summary_statistics(
    pre_series: Optional[pd.Series],
    post_series: Optional[pd.Series]
) -> Dict:
    """
    Compute summary statistics for pre- and post-fire series.
    
    Parameters
    ----------
    pre_series : pd.Series, optional
        Pre-fire discharge series.
    post_series : pd.Series, optional
        Post-fire discharge series.
    
    Returns
    -------
    dict
        A dictionary containing summary statistics for each period.
    """
    
    def _get_stats(series: pd.Series) -> Dict:
        """Helper to calculate stats for a single series."""
        if series is None or series.empty:
            return {
                'n_days': 0, 'n_valid': 0, 'mean': np.nan, 'median': np.nan,
                'std': np.nan, 'cv': np.nan, 'min': np.nan, 'max': np.nan,
                'q10': np.nan, 'q50': np.nan, 'q90': np.nan
            }
        
        mean_val = series.mean()
        return {
            'n_days': len(series),
            'n_valid': series.notna().sum(),
            'mean': mean_val,
            'median': series.median(),
            'std': series.std(),
            'cv': series.std() / mean_val if mean_val > 0 else np.nan,
            'min': series.min(),
            'max': series.max(),
            'q10': series.quantile(0.9),
            'q50': series.quantile(0.5),
            'q90': series.quantile(0.1),
        }

    results = {
        "pre_fire": _get_stats(pre_series),
        "post_fire": _get_stats(post_series)
    }
    
    if pre_series is not None and post_series is not None:
        full_series = pd.concat([pre_series, post_series])
        results["overall"] = _get_stats(full_series)
    elif pre_series is not None:
        results["overall"] = _get_stats(pre_series)
    elif post_series is not None:
        results["overall"] = _get_stats(post_series)
    else:
        results["overall"] = _get_stats(pd.Series(dtype=float))

    return results


def test_seasonality_shift(
    pre_df: pd.DataFrame,
    post_df: pd.DataFrame,
    discharge_col: str = "discharge"
) -> pd.DataFrame:
    """
    Test for shifts in monthly flow patterns.
    
    Parameters
    ----------
    pre_df : pandas.DataFrame
        Pre-fire streamflow
    post_df : pandas.DataFrame
        Post-fire streamflow
    discharge_col : str
        Name of discharge column
    
    Returns
    -------
    pandas.DataFrame
        Monthly comparison with test results
    """
    results = []
    
    for month in range(1, 13):
        pre_monthly = pre_df[pre_df.index.month == month][discharge_col].dropna()
        post_monthly = post_df[post_df.index.month == month][discharge_col].dropna()
        
        if len(pre_monthly) < 3 or len(post_monthly) < 3:
            continue
        
        # Mann-Whitney test
        try:
            stat, p = stats.mannwhitneyu(pre_monthly, post_monthly, alternative='two-sided')
            significant = p < 0.05
        except:
            stat, p, significant = np.nan, np.nan, False
        
        results.append({
            'month': month,
            'pre_mean': pre_monthly.mean(),
            'post_mean': post_monthly.mean(),
            'pre_median': pre_monthly.median(),
            'post_median': post_monthly.median(),
            'change_pct': ((post_monthly.mean() - pre_monthly.mean()) / pre_monthly.mean() * 100)
                         if pre_monthly.mean() > 0 else np.nan,
            'mw_statistic': stat,
            'pvalue': p,
            'significant': significant
        })
    
    return pd.DataFrame(results)


def compare_event_distributions(
    pre_events: List['StormEvent'],
    post_events: List['StormEvent'],
    alpha: float = 0.05
) -> pd.DataFrame:
    """
    Compares the statistical distributions of storm event properties.

    Parameters
    ----------
    pre_events : list
        List of pre-fire StormEvent objects.
    post_events : list
        List of post-fire StormEvent objects.
    alpha : float
        Significance level.

    Returns
    -------
    pandas.DataFrame
        A DataFrame summarizing the statistical comparison for each metric.
    """
    from .events import StormEvent

    if not pre_events or not post_events:
        return pd.DataFrame()

    pre_df = pd.DataFrame([e.to_dict() for e in pre_events])
    post_df = pd.DataFrame([e.to_dict() for e in post_events])

    metrics = {
        'peak_discharge': 'Peak Discharge (m³/s)',
        'total_volume_mm': 'Event Volume (mm)',
        'duration_hours': 'Event Duration (hours)',
        'time_to_peak_hours': 'Time to Peak (hours)',
        'rising_limb_slope': 'Rising Limb Slope',
    }
    
    all_results = []

    for metric, name in metrics.items():
        if metric not in pre_df.columns or metric not in post_df.columns:
            continue
            
        pre_values = pre_df[metric].dropna()
        post_values = post_df[metric].dropna()

        if len(pre_values) < 3 or len(post_values) < 3:
            continue

        stats_results = pre_post_statistical_tests(pre_values, post_values, alpha=alpha)
        
        # Extract key results for summary table
        mann_whitney = stats_results.get('mann_whitney', {})
        effect = stats_results.get('effect', {})
        pre_stats = stats_results.get('pre_stats', {})
        post_stats = stats_results.get('post_stats', {})

        all_results.append({
            'metric': name,
            'pre_mean': pre_stats.get('mean'),
            'post_mean': post_stats.get('mean'),
            'change_mean_pct': effect.get('percent_change_mean'),
            'pvalue': mann_whitney.get('pvalue'),
            'significant': mann_whitney.get('significant'),
        })

    return pd.DataFrame(all_results)
