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
    fire_years: Optional[list] = None,
    title: str = "Annual Maximum Discharge",
    ylabel: str = "Peak Discharge (m³/s)",
    output_path: Optional[str] = None
):
    """
    Plot annual maximum series with trend.
    """
    fig, ax = plt.subplots(figsize=(6, 4))
    
    first_fire_year = min(fire_years) if fire_years else None

    # Split pre/post
    if 'period' in annual_max.columns:
        pre = annual_max[annual_max['period'] == 'Pre-fire']
        post = annual_max[annual_max['period'] == 'Post-fire']
    elif first_fire_year:
        pre = annual_max[annual_max['year'] < first_fire_year]
        post = annual_max[annual_max['year'] >= first_fire_year]
    else:
        pre = annual_max
        post = pd.DataFrame()

    # Plot bars
    # Use slight offset if we have overlapping years to make them visible
    width = 0.4
    if not post.empty and not pre.empty:
        # Check for overlap
        common_years = set(pre['year']).intersection(set(post['year']))
        if common_years:
            ax.bar(pre['year'] - width/2, pre['max_discharge'], color=COLORS['pre_fire'], 
                   alpha=0.8, label='Pre-fire Maxima', width=width)
            ax.bar(post['year'] + width/2, post['max_discharge'], color=COLORS['post_fire'], 
                   alpha=0.8, label='Post-fire Maxima', width=width)
        else:
            ax.bar(pre['year'], pre['max_discharge'], color=COLORS['pre_fire'], 
                   alpha=0.7, label='Pre-fire Maxima', width=0.8)
            ax.bar(post['year'], post['max_discharge'], color=COLORS['post_fire'], 
                   alpha=0.7, label='Post-fire Maxima', width=0.8)
    else:
        ax.bar(pre['year'], pre['max_discharge'], color=COLORS['pre_fire'], 
               alpha=0.7, label='Pre-fire Maxima', width=0.8)
        if not post.empty:
            ax.bar(post['year'], post['max_discharge'], color=COLORS['post_fire'], 
                   alpha=0.7, label='Post-fire Maxima', width=0.8)
    
    # Add trend line (Pre-fire)
    if len(pre) > 1:
        z = np.polyfit(pre['year'], pre['max_discharge'], 1)
        p = np.poly1d(z)
        ax.plot(pre['year'], p(pre['year']), color=COLORS['pre_fire'], 
                linestyle='--', alpha=0.8, label='Pre-fire Trend')
    
    # Fire year markers
    if fire_years:
        for i, fire_year in enumerate(fire_years):
            label = 'Fire Year' if i == 0 else None
            ax.axvline(x=fire_year - 0.5, color=COLORS['fire_event'], 
                       linestyle='--', linewidth=1.5, label=label)
    
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
