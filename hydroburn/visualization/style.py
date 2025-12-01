"""
Visualization style configuration.
"""

import matplotlib.pyplot as plt
import seaborn as sns

# Color palette
COLORS = {
    'pre_fire': '#2166AC',    # Blue
    'post_fire': '#B2182B',   # Red
    'fire_event': '#D95F02',  # Orange
    'baseflow': '#7570B3',    # Purple
    'quickflow': '#1B9E77',   # Teal
    'confidence': '#CCCCCC',  # Gray
    'black': '#000000',
    'grid': '#E0E0E0',
}

def set_hydroburn_style():
    """Set the plotting style for HydroBurn figures."""
    sns.set_style("ticks")
    sns.set_context("paper", font_scale=1.2)
    
    plt.rcParams.update({
        'figure.figsize': (6, 4),
        'figure.dpi': 150,
        'savefig.dpi': 300,
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
        'axes.labelsize': 12,
        'axes.titlesize': 14,
        'legend.fontsize': 10,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'axes.spines.top': False,
        'axes.spines.right': False,
        'axes.grid': True,
        'grid.alpha': 0.3,
        'grid.color': COLORS['grid'],
        'lines.linewidth': 1.5,
    })
