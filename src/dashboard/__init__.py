"""
Sprint 5 Dashboard Package — Physics-Informed Pipeline Defence
==============================================================

Exports the data layer builder and all callback / layout helpers used by
the Dash application.

Author: Babak Pirzadi (STRATEGOS Thesis — Sprint 5)
"""

import sys as _sys, os as _os
_src_root = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
_project_root = _os.path.dirname(_src_root)
for _p in [_src_root, _project_root]:
    if _p not in _sys.path:
        _sys.path.insert(0, _p)

from src.dashboard.data_layer import DashboardData, build_dashboard_data, SEAM_FAD_PROFILES
from src.dashboard.callbacks import (
    make_network_figure,
    make_segment_fad_figure,
    make_adversarial_impact_figure,
    make_scenario_comparison_figure,
    make_budget_slider_figure,
    make_coverage_heatmap_figure,
    segment_intel_panel,
)

__all__ = [
    "DashboardData",
    "build_dashboard_data",
    "SEAM_FAD_PROFILES",
    "make_network_figure",
    "make_segment_fad_figure",
    "make_adversarial_impact_figure",
    "make_scenario_comparison_figure",
    "make_budget_slider_figure",
    "make_coverage_heatmap_figure",
    "segment_intel_panel",
]
