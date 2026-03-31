"""
psip.fad — BS 7910:2019 Level 2 Failure Assessment Diagram engine.

Re-exports the complete public API from the underlying fad_engine module.
"""
from src.zone_c.physics.fad_engine import (
    MaterialProperties,
    FlawGeometry,
    PipeGeometry,
    WeldJoint,
    FADAssessmentResult,
    assess_flaw,
    fad_option1,
    compute_Lr_max,
    compute_mu,
    compute_N_hardening,
    hoop_stress_barlow,
    stress_intensity_surface_flaw,
    reference_stress_axial_surface,
    plot_fad,
)

__all__ = [
    "MaterialProperties",
    "FlawGeometry",
    "PipeGeometry",
    "WeldJoint",
    "FADAssessmentResult",
    "assess_flaw",
    "fad_option1",
    "compute_Lr_max",
    "compute_mu",
    "compute_N_hardening",
    "hoop_stress_barlow",
    "stress_intensity_surface_flaw",
    "reference_stress_axial_surface",
    "plot_fad",
]
