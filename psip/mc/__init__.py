"""
psip.mc — Monte Carlo failure probability engine.

Re-exports the complete public API from the underlying mc_failure_probability module.
"""

from src.zone_c.physics.mc_failure_probability import (
    DistributionParams,
    MCResult,
    PipelineSegmentConfig,
    default_distributions_api5l_x65,
    monte_carlo_Pf,
    plot_mc_on_fad,
)

__all__ = [
    "DistributionParams",
    "PipelineSegmentConfig",
    "MCResult",
    "monte_carlo_Pf",
    "default_distributions_api5l_x65",
    "plot_mc_on_fad",
]
