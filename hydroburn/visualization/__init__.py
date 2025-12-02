"""Visualization modules for HydroBurn."""

from .style import set_hydroburn_style, COLORS
from .hydrograph import plot_hydrograph, plot_hydrograph_detail
from .fdc_plot import plot_fdc_comparison
from .annual_max import plot_annual_maxima
from .flood_freq_plot import plot_flood_frequency
from .boxplots import plot_event_boxplots, plot_monthly_boxplots, plot_before_after_summary
from .baseflow_plot import plot_baseflow_index
from .map_catchment import map_catchment

__all__ = [
    "set_hydroburn_style",
    "COLORS",
    "plot_hydrograph",
    "plot_hydrograph_detail",
    "plot_fdc_comparison",
    "plot_annual_maxima",
    "plot_flood_frequency",
    "plot_event_boxplots",
    "plot_monthly_boxplots",
    "plot_baseflow_index",
    "map_catchment",
]
