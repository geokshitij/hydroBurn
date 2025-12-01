"""
Flood frequency plotting module.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from typing import Optional, Dict
from .style import COLORS
from ..analysis.flood_freq import gumbel_plotting_positions

def plot_flood_frequency(
    pre_maxima: np.ndarray,
    post_maxima: np.ndarray,
    bootstrap_results: pd.DataFrame,
    title: str = "Flood Frequency Analysis (Gumbel)",
    ylabel: str = "Discharge (m³/s)",
    output_path: Optional[str] = None
):
    """
    Plot flood frequency curves with confidence intervals.
    """
    fig, ax = plt.subplots(figsize=(6, 4))
    
    # Plot observed data points
    y_pre, x_pre = gumbel_plotting_positions(pre_maxima)
    y_post, x_post = gumbel_plotting_positions(post_maxima)
    
    # Convert reduced variate to return period for x-axis
    # T = 1 / (1 - exp(-exp(-y)))
    # But we usually plot linear reduced variate and label with T
    
    ax.scatter(y_pre, x_pre, color=COLORS['pre_fire'], marker='o', 
               s=30, alpha=0.6, label='Pre-fire Observed')
    ax.scatter(y_post, x_post, color=COLORS['post_fire'], marker='^', 
               s=50, alpha=0.8, label='Post-fire Observed')
    
    # Plot fitted curves from bootstrap results
    # We need to convert return periods to reduced variate y
    T_vals = bootstrap_results['return_period'].values
    y_vals = -np.log(-np.log(1 - 1/T_vals))
    
    # Pre-fire curve and CI
    ax.plot(y_vals, bootstrap_results['pre_estimate'], color=COLORS['pre_fire'], 
            linewidth=2, label='Pre-fire Fit')
    ax.fill_between(y_vals, 
                    bootstrap_results['pre_ci_lower'], 
                    bootstrap_results['pre_ci_upper'],
                    color=COLORS['pre_fire'], alpha=0.1)
    
    # Post-fire curve and CI
    ax.plot(y_vals, bootstrap_results['post_estimate'], color=COLORS['post_fire'], 
            linewidth=2, label='Post-fire Fit')
    ax.fill_between(y_vals, 
                    bootstrap_results['post_ci_lower'], 
                    bootstrap_results['post_ci_upper'],
                    color=COLORS['post_fire'], alpha=0.1)
    
    # Configure x-axis (Gumbel scale)
    # Standard ticks for return periods: 2, 5, 10, 25, 50, 100
    ticks_T = np.array([1.01, 2, 5, 10, 25, 50, 100, 200])
    ticks_y = -np.log(-np.log(1 - 1/ticks_T))
    
    ax.set_xticks(ticks_y)
    ax.set_xticklabels([f'{t:g}' for t in ticks_T])
    
    ax.set_xlabel("Return Period (years)")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    ax.grid(True, which='both', alpha=0.3)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, bbox_inches='tight')
        plt.close()
    else:
        return fig, ax
