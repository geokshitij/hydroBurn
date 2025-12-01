"""Analysis modules for HydroBurn."""

from .events import extract_events, StormEvent
from .fdc import compute_fdc, compare_fdc
from .flood_freq import (
    fit_gumbel,
    gumbel_quantile,
    bootstrap_flood_quantiles,
    extract_annual_maxima,
    compare_flood_quantiles,
)
from .statistics import (
    pre_post_statistical_tests,
    pettitt_test,
    mann_kendall_test,
    compute_summary_statistics,
    compare_event_distributions,
)

__all__ = [
    "extract_events",
    "StormEvent",
    "compute_fdc",
    "compare_fdc",
    "fit_gumbel",
    "gumbel_quantile",
    "bootstrap_flood_quantiles",
    "extract_annual_maxima",
    "pre_post_statistical_tests",
    "pettitt_test",
    "mann_kendall_test",
    "compute_summary_statistics",
    "compare_event_distributions",
    "compare_flood_quantiles",
]
