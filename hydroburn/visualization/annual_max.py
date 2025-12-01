"""
Annual maxima plotting module.
"""

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from typing import Optional
from .style import COLORS

def plot_annual_maxima(
    annual_max: pd.DataFrame,
    fire_year: int,
    title: str = "Annual Maximum Discharge",
    ylabel: str = "Peak Discharge (m³/s)",
    output_path: Optional[str] = None
):
    """
    Plot annual maximum series with trend.
    """
    fig, ax = plt.subplots(figsize=(6, 4))
    
    # Split pre/post
    pre = annual_max[annual_max['year'] < fire_year]
    post = annual_max[annual_max['year'] >= fire_year]
    
    # Plot bars or stems
    ax.bar(pre['year'], pre['max_discharge'], color=COLORS['pre_fire'], 
           alpha=0.7, label='Pre-fire Maxima', width=0.8)
    ax.bar(post['year'], post['max_discharge'], color=COLORS['post_fire'], 
           alpha=0.7, label='Post-fire Maxima', width=0.8)
    
    # Add trend line (Pre-fire)
    if len(pre) > 1:
        z = np.polyfit(pre['year'], pre['max_discharge'], 1)
        p = np.poly1d(z)
        ax.plot(pre['year'], p(pre['year']), color=COLORS['pre_fire'], 
                linestyle='--', alpha=0.8, label='Pre-fire Trend')
    
    # Fire year marker
    ax.axvline(x=fire_year - 0.5, color=COLORS['fire_event'], 
               linestyle='--', linewidth=2, label='Fire Year')
    
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_xlabel("Water Year")
    ax.legend()
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, bbox_inches='tight')
        plt.close()
    else:
        return fig, ax
