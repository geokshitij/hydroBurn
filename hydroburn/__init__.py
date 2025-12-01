"""
HydroBurn: Post-Wildfire Runoff Change Detection Framework

A scientifically defensible, reproducible analysis framework for assessing
wildfire impacts on runoff and flood risk using streamflow and catchment data.

Case Study: 2024 South Fork / Salt Fires, Ruidoso, NM
"""

__version__ = "1.0.0"
__author__ = "HydroBurn Analysis Agent"

from .config import HydroBurnConfig

__all__ = ["HydroBurnConfig", "__version__"]
