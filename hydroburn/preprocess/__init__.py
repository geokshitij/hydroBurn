"""Preprocessing modules for HydroBurn."""

from .qc import quality_control, detect_gaps, fill_gaps
from .units import cfs_to_m3s, m3s_to_cfs, discharge_to_depth
from .baseflow import (
    lyne_hollick_filter, 
    compute_baseflow_index,
    separate_baseflow,
    compute_annual_bfi
)

__all__ = [
    "quality_control",
    "detect_gaps", 
    "fill_gaps",
    "cfs_to_m3s",
    "m3s_to_cfs",
    "discharge_to_depth",
    "lyne_hollick_filter",
    "compute_baseflow_index",
    "separate_baseflow",
    "compute_annual_bfi",
]
