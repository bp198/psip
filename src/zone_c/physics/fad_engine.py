"""
BS 7910:2019 Level 2 Failure Assessment Diagram (FAD) Engine
=============================================================

Implements the Option 1 FAD curve from BS 7910:2019, Section 7.3.3.
This module computes fracture assessment points (Kr, Lr) and determines
whether a flaw in a welded joint is acceptable, tolerable, or unacceptable.

Equations Reference:
    - FAD curve: Eq. 7.26, 7.27, 7.28 (BS 7910:2019, p.7/35)
    - Lr_max:    Eq. 7.25 (BS 7910:2019, p.7/35)
    - Kr:        Eq. 7.38-7.39 (BS 7910:2019, p.7/36)
    - Lr:        Eq. 7.40 (BS 7910:2019, p.7/36)

Author: Babak Pirzadi (STRATEGOS Thesis)
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Optional


# ---------------------------------------------------------------------------
# Data Classes for Material & Flaw Definitions
# ---------------------------------------------------------------------------

@dataclass
class MaterialProperties:
    """Material properties required for FAD assessment.

    Attributes:
        sigma_y:  Yield strength (MPa)
        sigma_u:  Ultimate tensile strength (MPa)
        E:        Young's modulus (MPa), default 207,000 for carbon steel
        K_mat:    Fracture toughness (MPa*sqrt(m))
        poisson:  Poisson's ratio, default 0.3
    """
    sigma_y: float
    sigma_u: float
    E: float = 207_000.0
    K_mat: float = 100.0
    poisson: float = 0.3

    def __post_init__(self):
        if self.sigma_y <= 0 or self.sigma_u <= 0:
            raise ValueError("Yield and ultimate strengths must be positive.")
        if self.sigma_u < self.sigma_y:
            raise ValueError("Ultimate strength must be >= yield strength.")
        if self.E <= 0:
            raise ValueError("Young's modulus must be positive.")
        if self.K_mat <= 0:
            raise ValueError("Fracture toughness must be positive.")


@dataclass
class FlawGeometry:
    """Flaw geometry for a surface or embedded crack.

    Attributes:
        a:     Flaw depth (mm) — half-height for embedded, full depth for surface
        two_c: Flaw length (mm) — full surface length of the flaw
        flaw_type: 'surface' or 'embedded'
    """
    a: float
    two_c: float
    flaw_type: str = "surface"

    def __post_init__(self):
        if self.a <= 0 or self.two_c <= 0:
            raise ValueError("Flaw dimensions must be positive.")
        if self.flaw_type not in ("surface", "embedded"):
            raise ValueError("flaw_type must be 'surface' or 'embedded'.")

    @property
    def c(self) -> float:
        """Half-length of the flaw."""
        return self.two_c / 2.0

    @property
    def aspect_ratio(self) -> float:
        """Flaw aspect ratio a/c."""
        return self.a / self.c


@dataclass
class PipeGeometry:
    """Pipeline cross-section geometry.

    Attributes:
        outer_diameter: Pipe outer diameter (mm)
        wall_thickness: Pipe wall thickness (mm)
    """
    outer_diameter: float
    wall_thickness: float

    def __post_init__(self):
        if self.wall_thickness >= self.outer_diameter / 2:
            raise ValueError("Wall thickness must be less than outer radius.")

    @property
    def inner_diameter(self) -> float:
        return self.outer_diameter - 2 * self.wall_thickness

    @property
    def mean_radius(self) -> float:
        return (self.outer_diameter - self.wall_thickness) / 2.0


@dataclass
class WeldJoint:
    """Weld joint metadata for a pipeline segment.

    Attributes:
        weld_type:   'butt', 'fillet', or 'socket'
        fat_class:   IIW FAT classification (e.g., 71, 80, 90)
        scf:         Stress concentration factor at weld toe
        misalignment: Axial misalignment (mm), default 0
        as_welded:   True if as-welded (no PWHT), False if post-weld heat treated
    """
    weld_type: str = "butt"
    fat_class: int = 71
    scf: float = 1.0
    misalignment: float = 0.0
    as_welded: bool = True

    def __post_init__(self):
        valid_types = ("butt", "fillet", "socket")
        if self.weld_type not in valid_types:
            raise ValueError(f"weld_type must be one of {valid_types}.")
        if self.fat_class <= 0:
            raise ValueError("FAT class must be positive.")


# ---------------------------------------------------------------------------
# FAD Curve Functions — BS 7910:2019, Section 7.3.3 (Option 1)
# ---------------------------------------------------------------------------

def compute_mu(E: float, sigma_y: float) -> float:
    """Compute the mu parameter for the Option 1 FAD curve.

    BS 7910:2019, Section 7.3.3:
        mu = min(0.001 * E / sigma_y, 0.6)

    Args:
        E:       Young's modulus (MPa)
        sigma_y: Yield strength (MPa)

    Returns:
        mu parameter (dimensionless)
    """
    return min(0.001 * E / sigma_y, 0.6)


def compute_N_hardening(sigma_y: float, sigma_u: float) -> float:
    """Compute the strain hardening exponent N for the FAD curve.

    BS 7910:2019, Section 7.3.3:
        N = 0.3 * (1 - sigma_y / sigma_u)

    Args:
        sigma_y: Yield strength (MPa)
        sigma_u: Ultimate tensile strength (MPa)

    Returns:
        N parameter (dimensionless)
    """
    return 0.3 * (1.0 - sigma_y / sigma_u)


def compute_Lr_max(sigma_y: float, sigma_u: float) -> float:
    """Compute the cut-off load ratio Lr_max.

    BS 7910:2019, Eq. 7.25:
        Lr_max = (sigma_y + sigma_u) / (2 * sigma_y)

    Args:
        sigma_y: Yield strength (MPa)
        sigma_u: Ultimate tensile strength (MPa)

    Returns:
        Lr_max (dimensionless)
    """
    return (sigma_y + sigma_u) / (2.0 * sigma_y)


def fad_option1(Lr: float | np.ndarray, mat: MaterialProperties) -> float | np.ndarray:
    """Evaluate the Option 1 FAD curve f(Lr).

    BS 7910:2019, Eq. 7.26-7.28:
        For Lr <= 1:
            f(Lr) = (1 + 0.5*Lr^2)^(-0.5) * [0.3 + 0.7*exp(-mu*Lr^6)]

        For 1 < Lr < Lr_max:
            f(Lr) = f(1) * Lr^((N-1)/(2N))

        For Lr >= Lr_max:
            f(Lr) = 0

    Args:
        Lr:  Load ratio (scalar or array)
        mat: MaterialProperties instance

    Returns:
        f(Lr) — the FAD curve ordinate (scalar or array)
    """
    Lr = np.asarray(Lr, dtype=float)
    scalar_input = Lr.ndim == 0
    Lr = np.atleast_1d(Lr)

    mu = compute_mu(mat.E, mat.sigma_y)
    N = compute_N_hardening(mat.sigma_y, mat.sigma_u)
    Lr_max = compute_Lr_max(mat.sigma_y, mat.sigma_u)

    result = np.zeros_like(Lr)

    # Region 1: Lr <= 1 (Eq. 7.26)
    mask1 = Lr <= 1.0
    Lr1 = Lr[mask1]
    result[mask1] = (1.0 + 0.5 * Lr1**2) ** (-0.5) * (
        0.3 + 0.7 * np.exp(-mu * Lr1**6)
    )

    # f(Lr=1) for transition region
    f_at_1 = (1.0 + 0.5) ** (-0.5) * (0.3 + 0.7 * np.exp(-mu))

    # Region 2: 1 < Lr < Lr_max (Eq. 7.27)
    mask2 = (Lr > 1.0) & (Lr < Lr_max)
    Lr2 = Lr[mask2]
    if N > 0:
        exponent = (N - 1.0) / (2.0 * N)
        result[mask2] = f_at_1 * Lr2**exponent
    else:
        # Edge case: perfectly plastic (N->0)
        result[mask2] = 0.0

    # Region 3: Lr >= Lr_max (Eq. 7.28)
    mask3 = Lr >= Lr_max
    result[mask3] = 0.0

    if scalar_input:
        return float(result[0])
    return result


# ---------------------------------------------------------------------------
# Stress Intensity Factor (K_I) — Simplified for Pipeline Surface Flaws
# ---------------------------------------------------------------------------

def stress_intensity_surface_flaw(
    sigma_m: float,
    sigma_b: float,
    a: float,
    c: float,
    B: float,
    W: float = np.inf,
    M_k_m: float = 1.0,
    M_k_b: float = 1.0,
) -> float:
    """Compute stress intensity factor K_I for a semi-elliptical surface flaw.

    Simplified from BS 7910:2019, Annex M (Figure M.3):
        K_I = sqrt(pi * a) * [sigma_m * Y_m * M_k_m + sigma_b * Y_b * M_k_b]

    where Y_m and Y_b are geometry correction factors computed using the
    Newman-Raju solution (simplified for pipeline applications).

    Args:
        sigma_m:  Membrane (hoop) stress (MPa)
        sigma_b:  Bending stress (MPa)
        a:        Flaw depth (mm)
        c:        Flaw half-length (mm)
        B:        Wall thickness (mm)
        W:        Plate/pipe width (mm), default inf for wide plate
        M_k_m:    Weld toe magnification factor for membrane stress
        M_k_b:    Weld toe magnification factor for bending stress

    Returns:
        K_I in MPa*sqrt(mm)
    """
    # Flaw shape parameter (complete elliptic integral approximation)
    aspect = a / c
    if aspect <= 1.0:
        phi = 1.0 + 1.464 * aspect**1.65
    else:
        phi = 1.0 + 1.464 * (1.0 / aspect)**1.65

    Q = phi  # shape factor (sometimes called Phi^2 in some references)

    # Finite thickness correction (simplified Newman-Raju)
    f_w = 1.0  # for a/B < 0.8 in wide plates
    a_over_B = min(a / B, 0.95)  # Cap at 0.95 to avoid singularity
    if a_over_B > 0.1:
        # Simplified correction: sec(pi*a/(2*B))^0.5 for membrane
        cos_val = np.cos(np.pi * a_over_B / 2.0)
        cos_val = max(cos_val, 0.01)  # Prevent division by zero
        f_w = (1.0 / cos_val) ** 0.5

    # Geometry correction factors (simplified for deepest point of surface flaw)
    M1 = 1.13 - 0.09 * aspect
    M2 = -0.54 + 0.89 / (0.2 + aspect)
    M3 = 0.5 - 1.0 / (0.65 + aspect) + 14.0 * (1.0 - aspect)**24

    g = 1.0  # at deepest point (phi_angle = pi/2)

    Y_m = (M1 + M2 * a_over_B**2 + M3 * a_over_B**4) * g * f_w / np.sqrt(Q)
    Y_b = Y_m * (1.0 - 0.25 * a_over_B**2)  # Simplified bending correction

    K_I = np.sqrt(np.pi * a) * (sigma_m * Y_m * M_k_m + sigma_b * Y_b * M_k_b)
    return K_I


# ---------------------------------------------------------------------------
# Reference Stress for Cylindrical Shell with Axial Surface Flaw
# ---------------------------------------------------------------------------

def reference_stress_axial_surface(
    sigma_m: float,
    sigma_b: float,
    a: float,
    two_c: float,
    B: float,
    R_mean: float,
) -> float:
    """Compute reference stress for an axial surface flaw in a cylinder.

    Based on BS 7910:2019, Annex P — simplified local collapse solution
    for a surface flaw in a pressurized cylinder.

    Args:
        sigma_m: Membrane (hoop) stress (MPa)
        sigma_b: Bending stress (MPa)
        a:       Flaw depth (mm)
        two_c:   Flaw surface length (mm)
        B:       Wall thickness (mm)
        R_mean:  Mean radius (mm)

    Returns:
        sigma_ref (MPa)
    """
    alpha = a / B
    # Bulging factor for cylinder (Folias factor approximation)
    M_s = np.sqrt(1.0 + 1.255 * (two_c / 2.0)**2 / (R_mean * B)
                  - 0.0135 * (two_c / 2.0)**4 / (R_mean * B)**2)

    # Net section stress approach (local collapse)
    alpha_prime = alpha / M_s
    sigma_ref = (sigma_m + sigma_b) / (1.0 - alpha_prime)

    return sigma_ref


# ---------------------------------------------------------------------------
# Hoop Stress from Internal Pressure (Barlow's Equation)
# ---------------------------------------------------------------------------

def hoop_stress_barlow(pressure: float, outer_diameter: float, wall_thickness: float) -> float:
    """Compute hoop stress using Barlow's equation.

    sigma_h = P * D / (2 * t)

    Args:
        pressure:       Internal pressure (MPa)
        outer_diameter: Pipe outer diameter (mm)
        wall_thickness: Pipe wall thickness (mm)

    Returns:
        Hoop stress (MPa)
    """
    return pressure * outer_diameter / (2.0 * wall_thickness)


# ---------------------------------------------------------------------------
# FAD Assessment Point & Result
# ---------------------------------------------------------------------------

@dataclass
class FADAssessmentResult:
    """Result of a single FAD assessment point.

    Attributes:
        Kr:       Fracture ratio K_I / K_mat
        Lr:       Load ratio sigma_ref / sigma_y
        f_Lr:     FAD curve value at Lr
        Lr_max:   Cut-off load ratio
        is_acceptable: True if point lies inside the FAD envelope (Kr <= f(Lr))
        reserve_factor: Ratio of distance to FAD curve vs distance to point
    """
    Kr: float
    Lr: float
    f_Lr: float
    Lr_max: float
    is_acceptable: bool
    reserve_factor: float


def assess_flaw(
    mat: MaterialProperties,
    flaw: FlawGeometry,
    pipe: PipeGeometry,
    weld: WeldJoint,
    pressure: float,
    sigma_b: float = 0.0,
    sigma_residual: float = 0.0,
) -> FADAssessmentResult:
    """Perform a BS 7910 Option 1 FAD assessment for a flaw in a pipeline weld.

    This is the central function that combines material properties, flaw geometry,
    pipe geometry, and loading to determine whether a flaw is acceptable.

    Args:
        mat:             MaterialProperties instance
        flaw:            FlawGeometry instance
        pipe:            PipeGeometry instance
        weld:            WeldJoint instance
        pressure:        Internal operating pressure (MPa)
        sigma_b:         Applied bending stress (MPa)
        sigma_residual:  Residual stress at weld (MPa), default 0

    Returns:
        FADAssessmentResult with all computed values
    """
    # Step 1: Compute applied stresses
    sigma_m = hoop_stress_barlow(pressure, pipe.outer_diameter, pipe.wall_thickness)

    # Step 2: Apply weld SCF to local stress
    sigma_m_local = sigma_m * weld.scf

    # Step 3: Compute stress intensity factor K_I
    K_I = stress_intensity_surface_flaw(
        sigma_m=sigma_m_local + sigma_residual,
        sigma_b=sigma_b,
        a=flaw.a,
        c=flaw.c,
        B=pipe.wall_thickness,
    )

    # Convert K_I from MPa*sqrt(mm) to MPa*sqrt(m) for consistency with K_mat
    K_I_m = K_I / np.sqrt(1000.0)

    # Step 4: Compute Kr
    Kr = K_I_m / mat.K_mat

    # Step 5: Compute reference stress and Lr
    sigma_ref = reference_stress_axial_surface(
        sigma_m=sigma_m_local,
        sigma_b=sigma_b,
        a=flaw.a,
        two_c=flaw.two_c,
        B=pipe.wall_thickness,
        R_mean=pipe.mean_radius,
    )
    Lr = sigma_ref / mat.sigma_y

    # Step 6: Evaluate FAD curve at Lr
    f_Lr = fad_option1(Lr, mat)
    Lr_max = compute_Lr_max(mat.sigma_y, mat.sigma_u)

    # Step 7: Determine acceptability
    is_acceptable = (Kr <= f_Lr) and (Lr < Lr_max)

    # Step 8: Compute reserve factor (distance-based)
    # Reserve factor = OB/OA where O is origin, A is assessment point, B is FAD curve
    distance_to_point = np.sqrt(Kr**2 + Lr**2)
    if distance_to_point > 0:
        # Scale factor to reach the FAD curve along the OA line
        # Solve: Kr_scaled <= f(Lr_scaled) along the radial line
        theta = np.arctan2(Kr, Lr) if Lr > 0 else np.pi / 2
        # Approximate reserve factor
        if f_Lr > 0:
            reserve_factor = f_Lr / Kr if Kr > 0 else float("inf")
        else:
            reserve_factor = 0.0
    else:
        reserve_factor = float("inf")

    return FADAssessmentResult(
        Kr=Kr,
        Lr=Lr,
        f_Lr=f_Lr,
        Lr_max=Lr_max,
        is_acceptable=is_acceptable,
        reserve_factor=reserve_factor,
    )


# ---------------------------------------------------------------------------
# Plotting Utility
# ---------------------------------------------------------------------------

def plot_fad(
    mat: MaterialProperties,
    assessment_points: Optional[list[FADAssessmentResult]] = None,
    labels: Optional[list[str]] = None,
    title: str = "BS 7910 Option 1 Failure Assessment Diagram",
    save_path: Optional[str] = None,
):
    """Plot the FAD curve with optional assessment points.

    Args:
        mat:               MaterialProperties for the FAD curve
        assessment_points: List of FADAssessmentResult to plot
        labels:            Labels for each assessment point
        title:             Plot title
        save_path:         If provided, save figure to this path
    """
    import matplotlib.pyplot as plt

    Lr_max = compute_Lr_max(mat.sigma_y, mat.sigma_u)
    Lr_array = np.linspace(0, Lr_max * 1.1, 500)
    f_array = fad_option1(Lr_array, mat)

    fig, ax = plt.subplots(figsize=(10, 8))

    # FAD envelope
    ax.plot(Lr_array, f_array, "b-", linewidth=2.5, label="Option 1 FAD Curve")
    ax.fill_between(Lr_array, f_array, alpha=0.08, color="blue")

    # Lr_max cut-off line
    ax.axvline(x=Lr_max, color="red", linestyle="--", linewidth=1.5,
               label=f"$L_{{r,max}}$ = {Lr_max:.3f}")

    # Assessment points
    if assessment_points:
        for i, pt in enumerate(assessment_points):
            color = "green" if pt.is_acceptable else "red"
            marker = "o" if pt.is_acceptable else "X"
            label = labels[i] if labels and i < len(labels) else f"Point {i+1}"
            status = "ACCEPTABLE" if pt.is_acceptable else "UNACCEPTABLE"
            ax.plot(pt.Lr, pt.Kr, marker, color=color, markersize=12,
                    markeredgecolor="black", markeredgewidth=1.0,
                    label=f"{label} ({status}, RF={pt.reserve_factor:.2f})")

    ax.set_xlabel("$L_r$ (Load Ratio)", fontsize=14)
    ax.set_ylabel("$K_r$ (Fracture Ratio)", fontsize=14)
    ax.set_title(title, fontsize=16, fontweight="bold")
    ax.set_xlim(0, Lr_max * 1.15)
    ax.set_ylim(0, 1.2)
    ax.legend(loc="upper right", fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_aspect("auto")

    # Annotate zones
    ax.text(0.3, 0.3, "ACCEPTABLE\nZONE", fontsize=14, color="green",
            alpha=0.5, ha="center", fontweight="bold")

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    return fig, ax
