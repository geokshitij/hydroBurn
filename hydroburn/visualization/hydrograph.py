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
    fire_date: Optional[str] = None,
    title: str = "Streamflow Hydrograph",
    ylabel: str = "Discharge (m³/s)",
    output_path: Optional[str] = None
):
    """
    Plot full hydrograph with fire date.
    """
    fig, ax = plt.subplots(figsize=(8, 4))
    
    # Plot discharge
    ax.plot(df.index, df[discharge_col], color=COLORS['black'], 
            linewidth=0.8, label='Discharge')
    
    # Add fire date line
    if fire_date:
        fire_dt = pd.to_datetime(fire_date)
        if df.index.tz is not None and fire_dt.tz is None:
            fire_dt = fire_dt.tz_localize(df.index.tz)
            
        ax.axvline(x=fire_dt, color=COLORS['fire_event'], 
                   linestyle='--', linewidth=2, label='Fire Date')
        
        # Shade post-fire period
        ax.axvspan(fire_dt, df.index.max(), color=COLORS['post_fire'], 
                   alpha=0.1, label='Post-fire Period')
    
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_xlabel('Date')
    ax.legend(loc='upper right')
    
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
    fire_date: str,
    window_days: int = 365,
    discharge_col: str = "discharge",
    baseflow_col: Optional[str] = "baseflow",
    title: str = "Hydrograph Detail (Pre/Post Fire)",
    ylabel: str = "Discharge (m³/s)",
    output_path: Optional[str] = None
):
    """
    Plot detailed hydrograph around fire date.
    """
    fire_dt = pd.to_datetime(fire_date)
    if df.index.tz is not None and fire_dt.tz is None:
        fire_dt = fire_dt.tz_localize(df.index.tz)
    
    start_date = fire_dt - pd.Timedelta(days=window_days)
    end_date = fire_dt + pd.Timedelta(days=window_days)
    
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
    
    # Fire date
    ax.axvline(x=fire_dt, color=COLORS['fire_event'], 
               linestyle='--', linewidth=2, label='Fire Date')
    
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
