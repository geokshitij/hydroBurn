"""
Catchment map visualization module.
"""

import matplotlib.pyplot as plt
import geopandas as gpd
from typing import Optional
from .style import COLORS

def plot_catchment_map(
    catchment: gpd.GeoDataFrame,
    title: str = "Catchment Boundary",
    output_path: Optional[str] = None
):
    """
    Plot catchment boundary map.
    """
    fig, ax = plt.subplots(figsize=(5, 5))
    
    # Plot catchment
    catchment.plot(ax=ax, color='white', edgecolor='black', linewidth=2)
    
    # Add centroid/gauge location (approximate)
    centroid = catchment.geometry.centroid.iloc[0]
    ax.scatter([centroid.x], [centroid.y], color=COLORS['post_fire'], 
               marker='^', s=100, label='Gauge / Outlet', zorder=5)
    
    ax.set_title(title)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    
    # Add scale bar (approximate)
    # This is tricky without cartopy, so we'll skip for now or use simple annotation
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, bbox_inches='tight')
        plt.close()
    else:
        return fig, ax
