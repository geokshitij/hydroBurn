"""
Storm event extraction module.

Provides functions for detecting and characterizing storm events
from streamflow hydrographs.
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import List, Optional, Tuple
from scipy.signal import find_peaks


@dataclass
class StormEvent:
    """Container for storm event characteristics."""
    event_id: int
    start_time: pd.Timestamp
    peak_time: pd.Timestamp
    end_time: pd.Timestamp
    peak_discharge: float  # m³/s
    peak_discharge_mm: Optional[float]  # mm/day (if area provided)
    total_volume_mm: Optional[float]  # Total runoff volume
    quickflow_volume_mm: Optional[float]  # Volume above baseflow
    baseflow_volume_mm: Optional[float]  # Baseflow during event
    time_to_peak_hours: float
    duration_hours: float
    rising_limb_slope: float  # m³/s per hour
    recession_slope: float  # m³/s per hour
    peak_to_volume_ratio: Optional[float]  # Peak / Volume
    
    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            'event_id': self.event_id,
            'start_time': self.start_time,
            'peak_time': self.peak_time,
            'end_time': self.end_time,
            'peak_discharge': self.peak_discharge,
            'peak_discharge_mm': self.peak_discharge_mm,
            'total_volume_mm': self.total_volume_mm,
            'quickflow_volume_mm': self.quickflow_volume_mm,
            'baseflow_volume_mm': self.baseflow_volume_mm,
            'time_to_peak_hours': self.time_to_peak_hours,
            'duration_hours': self.duration_hours,
            'rising_limb_slope': self.rising_limb_slope,
            'recession_slope': self.recession_slope,
            'peak_to_volume_ratio': self.peak_to_volume_ratio,
        }


def find_event_peaks(
    Q: pd.Series,
    threshold_percentile: float = 90,
    min_peak: float = 0.1,
    min_distance_days: float = 2
) -> List[pd.Timestamp]:
    """
    Find peaks in streamflow time series.
    
    Parameters
    ----------
    Q : pandas.Series
        Discharge time series with datetime index
    threshold_percentile : float
        Minimum percentile for peak detection
    min_peak : float
        Minimum absolute peak value
    min_distance_days : float
        Minimum days between independent peaks
    
    Returns
    -------
    list of Timestamps
        Peak times
    """
    Q_clean = Q.dropna()
    if len(Q_clean) == 0:
        return []
    
    # Compute threshold
    threshold = max(
        np.percentile(Q_clean.values, threshold_percentile),
        min_peak
    )
    
    # Estimate samples per day
    if len(Q_clean) > 1:
        dt = (Q_clean.index[-1] - Q_clean.index[0]).total_seconds() / len(Q_clean)
        samples_per_day = 86400 / dt
    else:
        samples_per_day = 1
    
    min_distance = int(min_distance_days * samples_per_day)
    min_distance = max(1, min_distance)
    
    # Find peaks using scipy
    peaks, properties = find_peaks(
        Q_clean.values,
        height=threshold,
        distance=min_distance,
        prominence=threshold * 0.1  # Minimum prominence
    )
    
    peak_times = [Q_clean.index[i] for i in peaks]
    return peak_times


def find_event_bounds(
    Q: pd.Series,
    peak_time: pd.Timestamp,
    Qb: Optional[pd.Series] = None,
    max_duration_days: float = 30
) -> Tuple[pd.Timestamp, pd.Timestamp]:
    """
    Find event start and end times given a peak.
    
    Uses baseflow as threshold if available, otherwise uses
    a fraction of the peak discharge.
    
    Parameters
    ----------
    Q : pandas.Series
        Total discharge
    peak_time : Timestamp
        Time of peak
    Qb : pandas.Series, optional
        Baseflow (if available)
    max_duration_days : float
        Maximum event duration to search
    
    Returns
    -------
    tuple
        (start_time, end_time)
    """
    peak_idx = Q.index.get_loc(peak_time)
    peak_Q = Q.iloc[peak_idx]
    
    # Threshold for event start/end
    if Qb is not None:
        threshold = Qb.iloc[peak_idx] * 1.1  # Just above baseflow
    else:
        threshold = peak_Q * 0.1  # 10% of peak
    
    # Search bounds
    dt = pd.Timedelta(days=max_duration_days)
    search_start = max(0, peak_idx - int(max_duration_days * 2))
    search_end = min(len(Q), peak_idx + int(max_duration_days * 2))
    
    # Find event start (search backward from peak)
    start_idx = peak_idx
    for i in range(peak_idx - 1, search_start - 1, -1):
        if Q.iloc[i] <= threshold or Q.iloc[i] >= Q.iloc[i + 1]:
            start_idx = i + 1
            break
        start_idx = i
    
    # Find event end (search forward from peak)
    end_idx = peak_idx
    for i in range(peak_idx + 1, search_end):
        if Q.iloc[i] <= threshold or Q.iloc[i] >= Q.iloc[i - 1]:
            end_idx = i
            break
        end_idx = i
    
    return Q.index[start_idx], Q.index[end_idx]


def compute_event_metrics(
    Q: pd.Series,
    start_time: pd.Timestamp,
    peak_time: pd.Timestamp,
    end_time: pd.Timestamp,
    Qb: Optional[pd.Series] = None,
    area_km2: Optional[float] = None,
    event_id: int = 0
) -> StormEvent:
    """
    Compute metrics for a single storm event.
    
    Parameters
    ----------
    Q : pandas.Series
        Total discharge (m³/s)
    start_time, peak_time, end_time : Timestamps
        Event timing
    Qb : pandas.Series, optional
        Baseflow component
    area_km2 : float, optional
        Catchment area for depth calculations
    event_id : int
        Event identifier
    
    Returns
    -------
    StormEvent
        Event characteristics
    """
    # Slice event
    event_Q = Q[start_time:end_time]
    peak_Q = Q[peak_time]
    
    # Timing
    time_to_peak = (peak_time - start_time).total_seconds() / 3600  # hours
    duration = (end_time - start_time).total_seconds() / 3600  # hours
    
    # Rising limb slope (m³/s per hour)
    if time_to_peak > 0:
        rising_slope = (peak_Q - Q[start_time]) / time_to_peak
    else:
        rising_slope = 0
    
    # Recession slope
    recession_time = (end_time - peak_time).total_seconds() / 3600
    if recession_time > 0:
        recession_slope = (peak_Q - Q[end_time]) / recession_time
    else:
        recession_slope = 0
    
    # Volume calculations
    total_volume_mm = None
    quickflow_volume_mm = None
    baseflow_volume_mm = None
    peak_mm = None
    peak_to_volume = None
    
    if area_km2 is not None:
        # Convert to mm using trapezoidal integration
        # Assuming daily data, dt = 86400 seconds
        dt_seconds = 86400
        
        # Total volume
        total_volume_m3 = np.trapezoid(event_Q.values, dx=dt_seconds)
        total_volume_mm = (total_volume_m3 / (area_km2 * 1e6)) * 1000
        
        # Peak in mm/day
        peak_mm = (peak_Q * dt_seconds / (area_km2 * 1e6)) * 1000
        
        if Qb is not None:
            event_Qb = Qb[start_time:end_time]
            event_Qf = event_Q - event_Qb
            event_Qf = event_Qf.clip(lower=0)
            
            qf_volume_m3 = np.trapezoid(event_Qf.values, dx=dt_seconds)
            quickflow_volume_mm = (qf_volume_m3 / (area_km2 * 1e6)) * 1000
            
            bf_volume_m3 = np.trapezoid(event_Qb.values, dx=dt_seconds)
            baseflow_volume_mm = (bf_volume_m3 / (area_km2 * 1e6)) * 1000
        
        if total_volume_mm > 0:
            peak_to_volume = peak_mm / total_volume_mm
    
    return StormEvent(
        event_id=event_id,
        start_time=start_time,
        peak_time=peak_time,
        end_time=end_time,
        peak_discharge=peak_Q,
        peak_discharge_mm=peak_mm,
        total_volume_mm=total_volume_mm,
        quickflow_volume_mm=quickflow_volume_mm,
        baseflow_volume_mm=baseflow_volume_mm,
        time_to_peak_hours=time_to_peak,
        duration_hours=duration,
        rising_limb_slope=rising_slope,
        recession_slope=recession_slope,
        peak_to_volume_ratio=peak_to_volume,
    )


def extract_events(
    df: pd.DataFrame,
    discharge_col: str = "discharge",
    baseflow_col: Optional[str] = "baseflow",
    area_km2: Optional[float] = None,
    threshold_percentile: float = 90,
    min_peak: float = 0.1,
    min_separation_days: float = 2,
    max_duration_days: float = 30
) -> List[StormEvent]:
    """
    Extract all storm events from streamflow time series.
    
    Parameters
    ----------
    df : pandas.DataFrame
        Streamflow data with datetime index
    discharge_col : str
        Name of discharge column
    baseflow_col : str, optional
        Name of baseflow column (if available)
    area_km2 : float, optional
        Catchment area for depth calculations
    threshold_percentile : float
        Percentile threshold for peak detection
    min_peak : float
        Minimum peak discharge (m³/s)
    min_separation_days : float
        Minimum days between independent events
    max_duration_days : float
        Maximum event duration
    
    Returns
    -------
    list of StormEvent
        Extracted events sorted by peak time
    """
    Q = df[discharge_col]
    Qb = df[baseflow_col] if baseflow_col and baseflow_col in df.columns else None
    
    # Find peaks
    peak_times = find_event_peaks(
        Q, 
        threshold_percentile=threshold_percentile,
        min_peak=min_peak,
        min_distance_days=min_separation_days
    )
    
    events = []
    used_times = set()
    
    for i, peak_time in enumerate(peak_times):
        # Find event bounds
        start_time, end_time = find_event_bounds(
            Q, peak_time, Qb, max_duration_days
        )
        
        # Skip overlapping events
        event_times = set(Q[start_time:end_time].index)
        if event_times & used_times:
            continue
        used_times.update(event_times)
        
        # Compute metrics
        event = compute_event_metrics(
            Q, start_time, peak_time, end_time,
            Qb=Qb, area_km2=area_km2, event_id=i+1
        )
        events.append(event)
    
    # Sort by peak time
    events.sort(key=lambda e: e.peak_time)
    
    # Renumber
    for i, event in enumerate(events):
        event.event_id = i + 1
    
    return events


def events_to_dataframe(events: List[StormEvent]) -> pd.DataFrame:
    """
    Convert list of events to DataFrame.
    
    Parameters
    ----------
    events : list of StormEvent
        Extracted events
    
    Returns
    -------
    pandas.DataFrame
        Events as DataFrame
    """
    if not events:
        return pd.DataFrame()
    
    return pd.DataFrame([e.to_dict() for e in events])


def compare_event_distributions(
    pre_events: List[StormEvent],
    post_events: List[StormEvent],
    metrics: List[str] = None
) -> pd.DataFrame:
    """
    Compare event metric distributions between pre and post periods.
    
    Parameters
    ----------
    pre_events : list of StormEvent
        Pre-fire events
    post_events : list of StormEvent
        Post-fire events
    metrics : list of str, optional
        Metrics to compare. Default: peaks, volumes, timing
    
    Returns
    -------
    pandas.DataFrame
        Comparison statistics
    """
    if metrics is None:
        metrics = [
            'peak_discharge',
            'quickflow_volume_mm',
            'time_to_peak_hours',
            'rising_limb_slope',
            'peak_to_volume_ratio'
        ]
    
    pre_df = events_to_dataframe(pre_events)
    post_df = events_to_dataframe(post_events)
    
    results = []
    for metric in metrics:
        if metric not in pre_df.columns or metric not in post_df.columns:
            continue
        
        pre_vals = pre_df[metric].dropna()
        post_vals = post_df[metric].dropna()
        
        if len(pre_vals) == 0 or len(post_vals) == 0:
            continue
        
        results.append({
            'metric': metric,
            'pre_n': len(pre_vals),
            'pre_mean': pre_vals.mean(),
            'pre_median': pre_vals.median(),
            'pre_std': pre_vals.std(),
            'post_n': len(post_vals),
            'post_mean': post_vals.mean(),
            'post_median': post_vals.median(),
            'post_std': post_vals.std(),
            'change_mean_pct': ((post_vals.mean() - pre_vals.mean()) / pre_vals.mean() * 100)
                               if pre_vals.mean() != 0 else np.nan,
            'change_median_pct': ((post_vals.median() - pre_vals.median()) / pre_vals.median() * 100)
                                 if pre_vals.median() != 0 else np.nan,
        })
    
    return pd.DataFrame(results)
