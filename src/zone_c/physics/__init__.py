"""
Zone C Physics Engine
======================

BS 7910:2019 + IIW fatigue assessment framework for pipeline weld joints.

Modules:
    fad_engine:             Option 1 FAD curve and fracture assessment
    fatigue_engine:         IIW S-N curve fatigue life calculation
    mc_failure_probability: Monte Carlo P_f estimation
"""

from .fad_engine import (
    MaterialProperties,
    FlawGeometry,
    PipeGeometry,
    WeldJoint,
    FADAssessmentResult,
    fad_option1,
    compute_Lr_max,
    compute_mu,
    compute_N_hardening,
    assess_flaw,
    hoop_stress_barlow,
    plot_fad,
)

from .fatigue_engine import (
    FatigueParameters,
    FAT_CLASS_TABLE,
    FAT_CLASS_DEFAULTS,
    get_fat_class,
    fatigue_life,
    fatigue_damage,
    cumulative_fatigue_damage,
    remaining_life_years,
    plot_sn_curve,
)

from .mc_failure_probability import (
    DistributionParams,
    PipelineSegmentConfig,
    MCResult,
    monte_carlo_Pf,
    default_distributions_api5l_x65,
    plot_mc_on_fad,
)

__all__ = [
    "MaterialProperties", "FlawGeometry", "PipeGeometry", "WeldJoint",
    "FADAssessmentResult", "fad_option1", "compute_Lr_max", "compute_mu",
    "compute_N_hardening", "assess_flaw", "hoop_stress_barlow", "plot_fad",
    "FatigueParameters", "FAT_CLASS_TABLE", "FAT_CLASS_DEFAULTS",
    "get_fat_class", "fatigue_life", "fatigue_damage",
    "cumulative_fatigue_damage", "remaining_life_years", "plot_sn_curve",
    "DistributionParams", "PipelineSegmentConfig", "MCResult",
    "monte_carlo_Pf", "default_distributions_api5l_x65", "plot_mc_on_fad",
]
