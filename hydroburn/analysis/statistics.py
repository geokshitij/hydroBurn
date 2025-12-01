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
    
    # Descriptive statistics
    results['pre_stats'] = {
        'n': len(pre),
        'mean': np.mean(pre),
        'median': np.median(pre),
        'std': np.std(pre, ddof=1),
        'min': np.min(pre),
        'max': np.max(pre),
        'q25': np.percentile(pre, 25),
        'q75': np.percentile(pre, 75),
    }
    
    results['post_stats'] = {
        'n': len(post),
        'mean': np.mean(post),
        'median': np.median(post),
        'std': np.std(post, ddof=1),
        'min': np.min(post),
        'max': np.max(post),
        'q25': np.percentile(post, 25),
        'q75': np.percentile(post, 75),
    }
    
    # Effect sizes
    pre_mean = np.mean(pre)
    post_mean = np.mean(post)
    pre_median = np.median(pre)
    post_median = np.median(post)
    
    results['effect'] = {
        'mean_diff': post_mean - pre_mean,
        'median_diff': post_median - pre_median,
        'mean_ratio': post_mean / pre_mean if pre_mean != 0 else np.nan,
        'median_ratio': post_median / pre_median if pre_median != 0 else np.nan,
        'pct_change_mean': ((post_mean - pre_mean) / pre_mean * 100) if pre_mean != 0 else np.nan,
        'pct_change_median': ((post_median - pre_median) / pre_median * 100) if pre_median != 0 else np.nan,
    }
    
    # Cohen's d effect size
    pooled_std = np.sqrt(((len(pre)-1)*np.var(pre, ddof=1) + (len(post)-1)*np.var(post, ddof=1)) / 
                         (len(pre) + len(post) - 2))
    if pooled_std > 0:
        cohens_d = (post_mean - pre_mean) / pooled_std
        results['effect']['cohens_d'] = cohens_d
        
        # Interpret effect size
        d_abs = abs(cohens_d)
        if d_abs < 0.2:
            effect_interp = 'negligible'
        elif d_abs < 0.5:
            effect_interp = 'small'
        elif d_abs < 0.8:
            effect_interp = 'medium'
        else:
            effect_interp = 'large'
        results['effect']['effect_size_interpretation'] = effect_interp
    
    return results


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
    df: pd.DataFrame,
    discharge_col: str = "discharge",
    fire_date: str = None,
    baseflow_col: str = "baseflow"
) -> pd.DataFrame:
    """
    Compute comprehensive summary statistics.
    
    Parameters
    ----------
    df : pandas.DataFrame
        Streamflow data with datetime index
    discharge_col : str
        Name of discharge column
    fire_date : str
        Fire date for pre/post split
    baseflow_col : str
        Name of baseflow column (if available)
    
    Returns
    -------
    pandas.DataFrame
        Summary statistics table
    """
    from ..io.load_streamflow import split_pre_post
    
    stats_list = []
    
    # Overall statistics
    Q = df[discharge_col]
    stats_list.append({
        'period': 'Overall',
        'n_days': len(Q),
        'n_valid': Q.notna().sum(),
        'mean': Q.mean(),
        'median': Q.median(),
        'std': Q.std(),
        'cv': Q.std() / Q.mean() if Q.mean() > 0 else np.nan,
        'min': Q.min(),
        'max': Q.max(),
        'q10': Q.quantile(0.9),  # Q10 = exceeded 10% of time
        'q50': Q.quantile(0.5),
        'q90': Q.quantile(0.1),  # Q90 = exceeded 90% of time
    })
    
    # Pre/post if fire_date provided
    if fire_date:
        pre_df, post_df = split_pre_post(df, fire_date)
        
        for period, sub_df in [('Pre-fire', pre_df), ('Post-fire', post_df)]:
            Q = sub_df[discharge_col]
            row = {
                'period': period,
                'n_days': len(Q),
                'n_valid': Q.notna().sum(),
                'mean': Q.mean(),
                'median': Q.median(),
                'std': Q.std(),
                'cv': Q.std() / Q.mean() if Q.mean() > 0 else np.nan,
                'min': Q.min(),
                'max': Q.max(),
                'q10': Q.quantile(0.9),
                'q50': Q.quantile(0.5),
                'q90': Q.quantile(0.1),
            }
            
            # Add BFI if available
            if baseflow_col in sub_df.columns:
                Qb = sub_df[baseflow_col]
                row['bfi'] = Qb.sum() / Q.sum() if Q.sum() > 0 else np.nan
            
            stats_list.append(row)
    
    return pd.DataFrame(stats_list)


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
            'change_mean_pct': effect.get('pct_change_mean'),
            'pvalue': mann_whitney.get('pvalue'),
            'significant': mann_whitney.get('significant'),
        })

    return pd.DataFrame(all_results)
