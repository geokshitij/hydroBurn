"""
Catchment map visualization module.
"""

import matplotlib.pyplot as plt
import geopandas as gpd
from typing import Optional
from .style import COLORS

def map_catchment(
    catchment: gpd.GeoDataFrame,
    fire_history_file: Optional[str] = None,
    title: str = "Catchment Boundary and Fire History",
    output_path: Optional[str] = None
):
    """
    Plot catchment boundary map with fire history.
    """
    fig, ax = plt.subplots(figsize=(6, 6))
    
    # Plot catchment
    catchment.plot(ax=ax, color='white', edgecolor='black', linewidth=1.5, label='Catchment Boundary', zorder=2)
    
    # Plot fire history if available
    if fire_history_file:
        fire_gdf = gpd.read_file(fire_history_file)
        if fire_gdf.crs != catchment.crs:
            fire_gdf = fire_gdf.to_crs(catchment.crs)
        
        # Clip to catchment
        fires_in_catchment = gpd.clip(fire_gdf, catchment)
        
        if not fires_in_catchment.empty:
            fires_in_catchment.plot(ax=ax, marker='.', color=COLORS['fire_event'], 
                                    markersize=5, alpha=0.6, label='Fire Detections', zorder=3)

    ax.set_title(title)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.legend()
    
    # Add scale bar (approximate)
    # This is tricky without cartopy, so we'll skip for now or use simple annotation
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, bbox_inches='tight')
        plt.close()
    else:
        return fig, ax
