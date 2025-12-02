"""
Baseflow visualization module.
"""

import matplotlib.pyplot as plt
import pandas as pd
from typing import Optional
from .style import COLORS

def plot_baseflow_index(
    annual_bfi: pd.DataFrame,
    fire_years: Optional[list] = None,
    title: str = "Annual Baseflow Index (BFI)",
    output_path: Optional[str] = None
):
    """
    Plot annual Baseflow Index time series.
    """
    fig, ax = plt.subplots(figsize=(6, 4))
    
    # Plot BFI line
    ax.plot(annual_bfi['year'], annual_bfi['bfi'], color=COLORS['baseflow'], 
            marker='o', linewidth=1.5, label='Annual BFI')
    
    if fire_years:
        # Fire year markers
        for i, fire_year in enumerate(fire_years):
            label = 'Fire Year' if i == 0 else None
            ax.axvline(x=fire_year - 0.5, color=COLORS['fire_event'], 
                       linestyle='--', linewidth=1.5, label=label)
        
        # Highlight post-fire
        first_fire_year = min(fire_years)
        post_mask = annual_bfi['year'] >= first_fire_year
        if post_mask.any():
            ax.scatter(annual_bfi.loc[post_mask, 'year'], 
                       annual_bfi.loc[post_mask, 'bfi'], 
                       color=COLORS['post_fire'], s=50, zorder=5, label='Post-fire')
    
    ax.set_title(title)
    ax.set_ylabel("Baseflow Index (Baseflow / Total Flow)")
    ax.set_xlabel("Year")
    ax.set_ylim(0, 1)
    ax.legend()
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, bbox_inches='tight')
        plt.close()
    else:
        return fig, ax
