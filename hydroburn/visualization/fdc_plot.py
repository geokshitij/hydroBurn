"""
Flow duration curve plotting module.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from typing import Optional
from .style import COLORS
from ..analysis.fdc import compute_fdc

def plot_fdc_comparison(
    pre_series: pd.Series,
    post_series: Optional[pd.Series],
    title: str = "Flow Duration Curve Comparison",
    ylabel: str = "Discharge (m³/s)",
    output_path: Optional[str] = None
):
    """
    Plot pre-fire vs post-fire flow duration curves.
    """
    fig, ax = plt.subplots(figsize=(5, 4))

    # Compute and plot Pre-fire FDC
    if pre_series is not None and not pre_series.empty:
        pre_exc, pre_q = compute_fdc(pre_series)
        ax.semilogy(pre_exc, pre_q, color=COLORS['pre_fire'], 
                    linewidth=2, label='Pre-fire')
        q10_pre = np.percentile(pre_series.dropna(), 90)
        ax.scatter([10], [q10_pre], color=COLORS['pre_fire'], zorder=5)
        q50_pre = np.median(pre_series.dropna())
        ax.scatter([50], [q50_pre], color=COLORS['pre_fire'], zorder=5)

    # Compute and plot Post-fire FDC
    if post_series is not None and not post_series.empty:
        post_exc, post_q = compute_fdc(post_series)
        ax.semilogy(post_exc, post_q, color=COLORS['post_fire'], 
                    linewidth=2, label='Post-fire')
        q10_post = np.percentile(post_series.dropna(), 90)
        ax.scatter([10], [q10_post], color=COLORS['post_fire'], zorder=5)
        q50_post = np.median(post_series.dropna())
        ax.scatter([50], [q50_post], color=COLORS['post_fire'], zorder=5)
    
    # Add grid
    ax.grid(True, which="both", ls="-", alpha=0.2)
    
    # Labels
    ax.set_xlabel("Exceedance Probability (%)")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    
    # Set x limits
    ax.set_xlim(0.1, 99.9)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, bbox_inches='tight')
        plt.close()
    else:
        return fig, ax
