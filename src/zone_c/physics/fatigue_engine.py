"""
IIW Fatigue Assessment Engine
==============================

Implements fatigue life calculations based on:
    - IIW Recommendations for Fatigue Design of Welded Joints and Components
      (Hobbacher, 2nd Edition, Springer)
    - BS 7910:2019 Section 8

S-N Curve Equation (IIW Eq. 3.1):
    N = C / (Delta_sigma)^m

Where:
    - N = fatigue life in cycles
    - C = FAT^m * 2e6 (derived from FAT class at 2 million cycles)
    - m = 3.0 for normal stress (default)
    - m = 5.0 for shear stress
    - Knee point at N = 1e7 (normal) or 1e8 (shear)
    - Beyond knee point: m = 22 (variable amplitude) or infinite life (constant amplitude)

Author: Babak Pirzadi (STRATEGOS Thesis)
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from typing import Optional


# ---------------------------------------------------------------------------
# FAT Class Registry — IIW Table 3.1 & 3.2 (Steel Values)
# ---------------------------------------------------------------------------

# Key: (weld_type, condition) -> FAT class (MPa at 2e6 cycles)
FAT_CLASS_TABLE: dict[tuple[str, str], int] = {
    # === BUTT WELDS (Table 3.1) ===
    ("butt", "ground_flush"):                   112,
    ("butt", "as_welded_both_sides"):            90,
    ("butt", "as_welded_one_side_backing"):       80,
    ("butt", "as_welded_one_side_no_backing"):    71,
    ("butt", "partial_penetration"):              63,
    ("butt", "field_girth_weld"):                 71,  # pipeline girth welds

    # === FILLET WELDS (Table 3.1) ===
    ("fillet", "transverse_non_load_carrying"):  100,
    ("fillet", "cruciform_load_carrying"):         90,
    ("fillet", "lap_joint"):                       80,
    ("fillet", "longitudinal_attachment"):          71,
    ("fillet", "cover_plate"):                     63,

    # === SOCKET WELDS ===
    ("socket", "standard"):                        56,

    # === SHEAR (Table 3.2) ===
    ("butt", "shear_full_penetration"):           100,
    ("fillet", "shear"):                            80,
}

# Default FAT classes when condition is not specified
FAT_CLASS_DEFAULTS: dict[str, int] = {
    "butt":   71,    # Conservative: as-welded, single-side, no backing
    "fillet": 80,    # Conservative: lap joint level
    "socket": 56,
}


@dataclass
class FatigueParameters:
    """Parameters for S-N curve fatigue assessment.

    Attributes:
        fat_class:           FAT classification (MPa at 2e6 cycles)
        m_slope:             S-N curve slope (default 3.0 for normal stress)
        m_slope_endurance:   Slope beyond knee point (default 22 for variable amplitude)
        knee_point_cycles:   Knee point in cycles (default 1e7)
        stress_type:         'normal' or 'shear'
        variable_amplitude:  If True, use m=22 beyond knee; if False, infinite life
    """
    fat_class: int = 71
    m_slope: float = 3.0
    m_slope_endurance: float = 22.0
    knee_point_cycles: float = 1e7
    stress_type: str = "normal"
    variable_amplitude: bool = True

    def __post_init__(self):
        if self.stress_type == "shear":
            self.m_slope = 5.0
            self.knee_point_cycles = 1e8

    @property
    def C(self) -> float:
        """S-N curve constant C = FAT^m * N_ref (at N_ref = 2e6 cycles)."""
        return (self.fat_class ** self.m_slope) * 2e6

    @property
    def C_endurance(self) -> float:
        """S-N curve constant beyond the knee point."""
        delta_sigma_knee = self.fat_class * (2e6 / self.knee_point_cycles) ** (1.0 / self.m_slope)
        return (delta_sigma_knee ** self.m_slope_endurance) * self.knee_point_cycles


def get_fat_class(weld_type: str, condition: Optional[str] = None) -> int:
    """Look up FAT class from the IIW table.

    Args:
        weld_type:  'butt', 'fillet', or 'socket'
        condition:  Specific weld condition string (see FAT_CLASS_TABLE)

    Returns:
        FAT class value (MPa)
    """
    if condition:
        key = (weld_type, condition)
        if key in FAT_CLASS_TABLE:
            return FAT_CLASS_TABLE[key]

    return FAT_CLASS_DEFAULTS.get(weld_type, 71)


# ---------------------------------------------------------------------------
# S-N Curve Fatigue Life Calculation
# ---------------------------------------------------------------------------

def fatigue_life(
    delta_sigma: float | np.ndarray,
    params: FatigueParameters,
) -> float | np.ndarray:
    """Compute fatigue life N for a given stress range.

    IIW S-N Curve (Eq. 3.1):
        N = C / (Delta_sigma)^m   for N <= knee point
        N = C_e / (Delta_sigma)^m_e   for N > knee point (variable amplitude)
        N = inf   for N > knee point (constant amplitude)

    Args:
        delta_sigma: Applied stress range (MPa), scalar or array
        params:      FatigueParameters instance

    Returns:
        Fatigue life in cycles (scalar or array)
    """
    delta_sigma = np.asarray(delta_sigma, dtype=float)
    scalar_input = delta_sigma.ndim == 0
    delta_sigma = np.atleast_1d(delta_sigma)

    # Stress range at knee point
    delta_sigma_knee = (params.C / params.knee_point_cycles) ** (1.0 / params.m_slope)

    N = np.zeros_like(delta_sigma)

    # Above knee point stress: use main S-N curve
    mask_high = delta_sigma >= delta_sigma_knee
    N[mask_high] = params.C / (delta_sigma[mask_high] ** params.m_slope)

    # Below knee point stress: endurance region
    mask_low = delta_sigma < delta_sigma_knee
    if params.variable_amplitude:
        N[mask_low] = params.C_endurance / (delta_sigma[mask_low] ** params.m_slope_endurance)
    else:
        N[mask_low] = np.inf  # Constant amplitude: infinite life below knee

    # Zero or negative stress range
    mask_zero = delta_sigma <= 0
    N[mask_zero] = np.inf

    if scalar_input:
        return float(N[0])
    return N


def fatigue_damage(
    delta_sigma: float,
    n_cycles: float,
    params: FatigueParameters,
) -> float:
    """Compute Miner's rule cumulative damage for a single stress block.

    D = n / N

    Args:
        delta_sigma: Stress range (MPa)
        n_cycles:    Number of applied cycles at this stress range
        params:      FatigueParameters instance

    Returns:
        Damage fraction D (failure at D >= 1.0)
    """
    N = fatigue_life(delta_sigma, params)
    if np.isinf(N):
        return 0.0
    return n_cycles / N


def cumulative_fatigue_damage(
    stress_spectrum: list[tuple[float, float]],
    params: FatigueParameters,
) -> float:
    """Compute total Miner's rule cumulative damage for a stress spectrum.

    D_total = sum(n_i / N_i)

    Args:
        stress_spectrum: List of (delta_sigma_i, n_cycles_i) tuples
        params:          FatigueParameters instance

    Returns:
        Total cumulative damage D (failure at D >= 1.0)
    """
    D_total = 0.0
    for delta_sigma, n_cycles in stress_spectrum:
        D_total += fatigue_damage(delta_sigma, n_cycles, params)
    return D_total


# ---------------------------------------------------------------------------
# Remaining Fatigue Life Estimation
# ---------------------------------------------------------------------------

def remaining_life_years(
    delta_sigma: float,
    cycles_per_year: float,
    params: FatigueParameters,
    accumulated_damage: float = 0.0,
) -> float:
    """Estimate remaining fatigue life in years.

    Args:
        delta_sigma:        Constant-amplitude stress range (MPa)
        cycles_per_year:    Number of stress cycles per year (e.g., from pressure cycling)
        params:             FatigueParameters instance
        accumulated_damage: Pre-existing damage fraction from prior service (0 to 1)

    Returns:
        Remaining life in years (inf if below endurance limit)
    """
    N_total = fatigue_life(delta_sigma, params)
    if np.isinf(N_total):
        return float("inf")

    remaining_cycles = N_total * (1.0 - accumulated_damage)
    if remaining_cycles <= 0:
        return 0.0

    return remaining_cycles / cycles_per_year


# ---------------------------------------------------------------------------
# Plotting Utility
# ---------------------------------------------------------------------------

def plot_sn_curve(
    params: FatigueParameters,
    delta_sigma_range: Optional[tuple[float, float]] = None,
    save_path: Optional[str] = None,
    title: Optional[str] = None,
):
    """Plot the S-N curve for given fatigue parameters.

    Args:
        params:            FatigueParameters instance
        delta_sigma_range: (min, max) stress range in MPa
        save_path:         If provided, save figure to this path
        title:             Plot title
    """
    import matplotlib.pyplot as plt

    if delta_sigma_range is None:
        delta_sigma_range = (10.0, 500.0)

    delta_sigma = np.logspace(
        np.log10(delta_sigma_range[0]),
        np.log10(delta_sigma_range[1]),
        500,
    )
    N = fatigue_life(delta_sigma, params)

    fig, ax = plt.subplots(figsize=(10, 7))

    # S-N curve
    ax.loglog(N, delta_sigma, "b-", linewidth=2.5,
              label=f"FAT {params.fat_class} (m={params.m_slope})")

    # Mark FAT class point (2e6 cycles)
    ax.plot(2e6, params.fat_class, "ro", markersize=10, markeredgecolor="black",
            label=f"FAT {params.fat_class} MPa @ 2×10⁶ cycles")

    # Mark knee point
    delta_sigma_knee = (params.C / params.knee_point_cycles) ** (1.0 / params.m_slope)
    ax.plot(params.knee_point_cycles, delta_sigma_knee, "g^", markersize=10,
            markeredgecolor="black",
            label=f"Knee point: {delta_sigma_knee:.1f} MPa @ {params.knee_point_cycles:.0e}")

    ax.set_xlabel("Fatigue Life N (cycles)", fontsize=14)
    ax.set_ylabel("Stress Range Δσ (MPa)", fontsize=14)
    ax.set_title(title or f"S-N Curve — FAT {params.fat_class}", fontsize=16, fontweight="bold")
    ax.legend(loc="upper right", fontsize=10)
    ax.grid(True, which="both", alpha=0.3)
    ax.set_xlim(1e4, 1e10)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    return fig, ax
