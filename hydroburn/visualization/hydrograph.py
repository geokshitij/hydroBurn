"""
Hydrograph plotting module.
"""

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
from typing import Optional
from .style import COLORS

def plot_hydrograph(
    df: pd.DataFrame,
    discharge_col: str = "discharge",
    precip_col: Optional[str] = "precipitation",
    fire_dates: Optional[list] = None,
    post_fire_window_months: Optional[int] = None,
    title: str = "Streamflow Hydrograph",
    ylabel: str = "Discharge (m³/s)",
    output_path: Optional[str] = None
):
    """
    Plot full hydrograph with fire dates and optional precipitation.
    """
    fig, ax = plt.subplots(figsize=(8, 4))
    
    # Plot precipitation if available
    if precip_col and precip_col in df.columns:
        ax_precip = ax.twinx()
        # Invert y-axis for precipitation
        ax_precip.bar(df.index, df[precip_col], color='navy', alpha=0.7, width=2.0, label='Precipitation')
        ax_precip.set_ylabel('Precipitation (mm)')
        ax_precip.set_ylim(top=df[precip_col].max() * 3, bottom=0)  # Scale so bars take up top 1/3
        ax_precip.invert_yaxis()
        # Combine legends later if needed, or just let them be separate
    
    # Plot discharge
    ax.plot(df.index, df[discharge_col], color=COLORS['black'], 
            linewidth=0.8, label='Discharge')
    
    # Add fire date lines
    if fire_dates:
        for i, fire_date in enumerate(fire_dates):
            fire_dt = pd.to_datetime(fire_date)
            if df.index.tz is not None and fire_dt.tz is None:
                fire_dt = fire_dt.tz_localize(df.index.tz)
            
            label = 'Fire Event' if i == 0 else None
            ax.axvline(x=fire_dt, color=COLORS['fire_event'], 
                       linestyle='--', linewidth=1.5, label=label)
            
            # Shade post-fire period for this specific fire if window is defined
            if post_fire_window_months:
                end_dt = fire_dt + pd.DateOffset(months=post_fire_window_months)
                label_shade = 'Post-fire Period' if i == 0 else None
                ax.axvspan(fire_dt, end_dt, color=COLORS['post_fire'], 
                           alpha=0.1, label=label_shade)
        
        # If no window defined, fall back to shading from first fire to end (legacy behavior)
        if not post_fire_window_months:
            first_fire_dt = pd.to_datetime(min(fire_dates))
            if df.index.tz is not None and first_fire_dt.tz is None:
                first_fire_dt = first_fire_dt.tz_localize(df.index.tz)
            ax.axvspan(first_fire_dt, df.index.max(), color=COLORS['post_fire'], 
                       alpha=0.1, label='Post-fire Period')

    ax.legend(loc='upper right')
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_xlabel('Date')
    
    # Format x-axis
    ax.xaxis.set_major_locator(mdates.YearLocator(5))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, bbox_inches='tight')
        plt.close()
    else:
        return fig, ax

def plot_hydrograph_detail(
    df: pd.DataFrame,
    fire_dates: list,
    window_days: int = 365,
    discharge_col: str = "discharge",
    baseflow_col: Optional[str] = "baseflow",
    title: str = "Hydrograph Detail (Around First Fire)",
    ylabel: str = "Discharge (m³/s)",
    output_path: Optional[str] = None
):
    """
    Plot detailed hydrograph around the first fire date.
    """
    if not fire_dates:
        return # Or raise an error

    first_fire_dt = pd.to_datetime(min(fire_dates))
    if df.index.tz is not None and first_fire_dt.tz is None:
        first_fire_dt = first_fire_dt.tz_localize(df.index.tz)
    
    start_date = first_fire_dt - pd.Timedelta(days=window_days)
    end_date = first_fire_dt + pd.Timedelta(days=window_days)
    
    subset = df[(df.index >= start_date) & (df.index <= end_date)]
    
    fig, ax = plt.subplots(figsize=(8, 4))
    
    # Plot total discharge
    ax.plot(subset.index, subset[discharge_col], color=COLORS['black'], 
            linewidth=1.5, label='Total Discharge', alpha=0.8)
    
    # Plot baseflow if available
    if baseflow_col and baseflow_col in subset.columns:
        ax.plot(subset.index, subset[baseflow_col], color=COLORS['baseflow'], 
                linewidth=1.5, linestyle='-', label='Baseflow')
        
        # Fill quickflow
        ax.fill_between(subset.index, subset[baseflow_col], subset[discharge_col],
                        color=COLORS['quickflow'], alpha=0.3, label='Quickflow')
    
    # Fire dates
    for fire_date in fire_dates:
        fire_dt = pd.to_datetime(fire_date)
        if df.index.tz is not None and fire_dt.tz is None:
            fire_dt = fire_dt.tz_localize(df.index.tz)
        
        if start_date <= fire_dt <= end_date:
            ax.axvline(x=fire_dt, color=COLORS['fire_event'], 
                       linestyle='--', linewidth=2, label='Fire Event' if fire_date == min(fire_dates) else None)
    
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.legend(loc='upper left')
    
    # Format x-axis
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, bbox_inches='tight')
        plt.close()
    else:
        return fig, ax
