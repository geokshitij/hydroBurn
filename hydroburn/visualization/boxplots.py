"""
Boxplot visualization module.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from typing import Optional, List, Dict
from .style import COLORS

def plot_event_boxplots(
    pre_events: pd.DataFrame,
    post_events: pd.DataFrame,
    metrics: List[str] = ['peak_discharge', 'total_volume_mm', 'time_to_peak_hours'],
    metric_labels: Optional[Dict[str, str]] = None,
    output_path: Optional[str] = None
):
    """
    Plot boxplots comparing event metrics.
    """
    if metric_labels is None:
        metric_labels = {
            'peak_discharge': 'Peak Discharge (m³/s)',
            'total_volume_mm': 'Event Volume (mm)',
            'time_to_peak_hours': 'Time to Peak (hours)',
            'rising_limb_slope': 'Rising Limb Slope',
            'peak_to_volume_ratio': 'Peak/Volume Ratio'
        }
    
    # Combine data
    pre_df = pre_events.copy()
    pre_df['Period'] = 'Pre-fire'
    
    post_df = post_events.copy()
    post_df['Period'] = 'Post-fire'
    
    combined = pd.concat([pre_df, post_df], ignore_index=True)
    
    n_metrics = len(metrics)
    fig, axes = plt.subplots(1, n_metrics, figsize=(3 * n_metrics, 4))
    
    if n_metrics == 1:
        axes = [axes]
    
    for i, metric in enumerate(metrics):
        if metric not in combined.columns:
            continue
            
        ax = axes[i]
        sns.boxplot(x='Period', y=metric, data=combined, ax=ax,
                    palette={'Pre-fire': COLORS['pre_fire'], 'Post-fire': COLORS['post_fire']},
                    width=0.5, showfliers=False)
        
        # Add strip plot for individual points
        sns.stripplot(x='Period', y=metric, data=combined, ax=ax,
                      color='black', alpha=0.3, size=3, jitter=True)
        
        ax.set_ylabel(metric_labels.get(metric, metric))
        ax.set_xlabel("")
        ax.set_title(metric_labels.get(metric, metric).split('(')[0])
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, bbox_inches='tight')
        plt.close()
    else:
        return fig, axes

def plot_monthly_boxplots(
    df: pd.DataFrame,
    fire_date: str,
    discharge_col: str = "discharge",
    title: str = "Monthly Flow Distribution",
    ylabel: str = "Discharge (m³/s)",
    output_path: Optional[str] = None
):
    """
    Plot monthly flow distributions pre vs post fire.
    """
    from ..io.load_streamflow import split_pre_post
    
    pre, post = split_pre_post(df, fire_date)
    
    pre = pre.copy()
    pre['Period'] = 'Pre-fire'
    pre['Month'] = pre.index.month
    
    post = post.copy()
    post['Period'] = 'Post-fire'
    post['Month'] = post.index.month
    
    combined = pd.concat([pre, post])
    
    fig, ax = plt.subplots(figsize=(8, 4))
    
    sns.boxplot(x='Month', y=discharge_col, hue='Period', data=combined,
                palette={'Pre-fire': COLORS['pre_fire'], 'Post-fire': COLORS['post_fire']},
                ax=ax, showfliers=False)
    
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_xlabel("Month")
    ax.set_xticklabels(['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                        'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, bbox_inches='tight')
        plt.close()
    else:
        return fig, ax
