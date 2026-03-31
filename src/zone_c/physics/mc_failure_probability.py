"""
Monte Carlo Failure Probability Engine
========================================

Computes physics-based probability of failure (P_f) for pipeline weld joints
using Monte Carlo simulation over the BS 7910 FAD framework.

This module is the critical "bridge" that transforms:
    Material properties + Flaw distributions + Operating conditions
    → P_f (probability of failure per weld joint)
    → V(e) (vulnerability payoff for game-theoretic model)

The P_f values computed here become the physics-informed payoffs in the
Bayesian Stackelberg game engine (Layer 3).

Input distributions (calibrated from PHMSA data and IIW standards):
    - Defect size:           Lognormal (a, two_c)
    - Material toughness:    Weibull (K_mat)
    - Yield strength:        Normal (sigma_y)
    - Operating pressure:    Normal (P)
    - SCF:                   Uniform (weld geometry variability)

Author: Babak Pirzadi (STRATEGOS Thesis)
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Optional

from .fad_engine import (
    MaterialProperties,
    FlawGeometry,
    PipeGeometry,
    WeldJoint,
    assess_flaw,
    FADAssessmentResult,
)


# ---------------------------------------------------------------------------
# Distribution Definitions
# ---------------------------------------------------------------------------

@dataclass
class DistributionParams:
    """Parameters defining a probability distribution for Monte Carlo sampling.

    Attributes:
        dist_type:  Distribution type ('normal', 'lognormal', 'weibull', 'uniform')
        param1:     First parameter (mean/shape/lower depending on dist_type)
        param2:     Second parameter (std/scale/upper depending on dist_type)
        lower_bound: Physical lower bound (e.g., 0 for non-negative quantities)
        upper_bound: Physical upper bound (optional)
    """
    dist_type: str
    param1: float
    param2: float
    lower_bound: Optional[float] = None
    upper_bound: Optional[float] = None

    def sample(self, n: int, rng: np.random.Generator) -> np.ndarray:
        """Draw n samples from this distribution.

        Args:
            n:   Number of samples
            rng: NumPy random generator

        Returns:
            Array of n samples
        """
        if self.dist_type == "normal":
            samples = rng.normal(self.param1, self.param2, n)
        elif self.dist_type == "lognormal":
            # param1 = mean of underlying normal, param2 = std of underlying normal
            samples = rng.lognormal(self.param1, self.param2, n)
        elif self.dist_type == "weibull":
            # param1 = shape (k), param2 = scale (lambda)
            samples = self.param2 * rng.weibull(self.param1, n)
        elif self.dist_type == "uniform":
            samples = rng.uniform(self.param1, self.param2, n)
        else:
            raise ValueError(f"Unknown distribution type: {self.dist_type}")

        # Apply physical bounds
        if self.lower_bound is not None:
            samples = np.maximum(samples, self.lower_bound)
        if self.upper_bound is not None:
            samples = np.minimum(samples, self.upper_bound)

        return samples


# ---------------------------------------------------------------------------
# Pipeline Segment Configuration
# ---------------------------------------------------------------------------

@dataclass
class PipelineSegmentConfig:
    """Configuration for a single pipeline segment's probabilistic properties.

    This represents one edge in the network graph, fully parameterized for
    Monte Carlo simulation.

    Attributes:
        segment_id:     Unique identifier for this pipeline segment
        pipe:           PipeGeometry (deterministic)
        weld:           WeldJoint (deterministic baseline)
        dist_defect_a:      Distribution for flaw depth (mm)
        dist_defect_2c:     Distribution for flaw surface length (mm)
        dist_K_mat:         Distribution for fracture toughness (MPa*sqrt(m))
        dist_sigma_y:       Distribution for yield strength (MPa)
        dist_sigma_u:       Distribution for ultimate strength (MPa)
        dist_pressure:      Distribution for operating pressure (MPa)
        dist_scf:           Distribution for SCF variability
        sigma_residual:     Residual stress at weld (MPa), default = sigma_y (conservative)
    """
    segment_id: str
    pipe: PipeGeometry
    weld: WeldJoint
    dist_defect_a: DistributionParams
    dist_defect_2c: DistributionParams
    dist_K_mat: DistributionParams
    dist_sigma_y: DistributionParams
    dist_sigma_u: DistributionParams
    dist_pressure: DistributionParams
    dist_scf: DistributionParams
    sigma_residual: Optional[float] = None  # None means = sigma_y (conservative)


# ---------------------------------------------------------------------------
# Default Distribution Presets for Pipeline Steels
# ---------------------------------------------------------------------------

def default_distributions_api5l_x65() -> dict:
    """Return default distribution parameters for API 5L X65 pipeline steel.

    Values based on published literature and PHMSA statistical analysis.
    """
    return {
        "dist_defect_a": DistributionParams(
            dist_type="lognormal", param1=0.7, param2=0.8,
            lower_bound=0.5, upper_bound=15.0,
        ),
        "dist_defect_2c": DistributionParams(
            dist_type="lognormal", param1=2.0, param2=0.7,
            lower_bound=2.0, upper_bound=200.0,
        ),
        "dist_K_mat": DistributionParams(
            dist_type="weibull", param1=4.0, param2=120.0,
            lower_bound=30.0,
        ),
        "dist_sigma_y": DistributionParams(
            dist_type="normal", param1=480.0, param2=25.0,
            lower_bound=400.0,
        ),
        "dist_sigma_u": DistributionParams(
            dist_type="normal", param1=565.0, param2=20.0,
            lower_bound=520.0,
        ),
        "dist_pressure": DistributionParams(
            dist_type="normal", param1=7.0, param2=0.5,
            lower_bound=0.1,
        ),
        "dist_scf": DistributionParams(
            dist_type="uniform", param1=1.0, param2=2.5,
        ),
    }


# ---------------------------------------------------------------------------
# Monte Carlo Simulation Engine
# ---------------------------------------------------------------------------

@dataclass
class MCResult:
    """Result of a Monte Carlo failure probability simulation.

    Attributes:
        segment_id:     Pipeline segment identifier
        n_simulations:  Number of MC iterations run
        n_failures:     Number of iterations where the flaw was unacceptable
        P_f:            Estimated probability of failure
        P_f_lower:      Lower bound of 95% CI on P_f
        P_f_upper:      Upper bound of 95% CI on P_f
        mean_Kr:        Mean fracture ratio across all simulations
        mean_Lr:        Mean load ratio across all simulations
        mean_reserve:   Mean reserve factor for acceptable cases
        all_Kr:         Array of all Kr values (for post-processing)
        all_Lr:         Array of all Lr values (for post-processing)
    """
    segment_id: str
    n_simulations: int
    n_failures: int
    P_f: float
    P_f_lower: float
    P_f_upper: float
    mean_Kr: float
    mean_Lr: float
    mean_reserve: float
    all_Kr: np.ndarray = field(default_factory=lambda: np.array([]))
    all_Lr: np.ndarray = field(default_factory=lambda: np.array([]))


def monte_carlo_Pf(
    config: PipelineSegmentConfig,
    n_simulations: int = 10_000,
    seed: Optional[int] = None,
    confidence: float = 0.95,
) -> MCResult:
    """Run Monte Carlo simulation to estimate failure probability P_f.

    For each iteration:
        1. Sample material properties, flaw size, pressure, and SCF
        2. Run BS 7910 FAD assessment
        3. Record whether the flaw is unacceptable (outside FAD envelope)
        4. P_f = n_failures / n_simulations with Wilson CI

    Args:
        config:        PipelineSegmentConfig with distributions
        n_simulations: Number of MC iterations (default 10,000)
        seed:          Random seed for reproducibility
        confidence:    Confidence level for CI (default 0.95)

    Returns:
        MCResult with P_f and statistics
    """
    rng = np.random.default_rng(seed)

    # Sample all random variables
    a_samples = config.dist_defect_a.sample(n_simulations, rng)
    two_c_samples = config.dist_defect_2c.sample(n_simulations, rng)
    K_mat_samples = config.dist_K_mat.sample(n_simulations, rng)
    sigma_y_samples = config.dist_sigma_y.sample(n_simulations, rng)
    sigma_u_samples = config.dist_sigma_u.sample(n_simulations, rng)
    pressure_samples = config.dist_pressure.sample(n_simulations, rng)
    scf_samples = config.dist_scf.sample(n_simulations, rng)

    # Ensure sigma_u >= sigma_y
    sigma_u_samples = np.maximum(sigma_u_samples, sigma_y_samples + 10.0)

    # Run FAD assessments
    n_failures = 0
    Kr_values = np.zeros(n_simulations)
    Lr_values = np.zeros(n_simulations)
    reserve_factors = np.zeros(n_simulations)

    for i in range(n_simulations):
        mat = MaterialProperties(
            sigma_y=sigma_y_samples[i],
            sigma_u=sigma_u_samples[i],
            K_mat=K_mat_samples[i],
        )
        flaw = FlawGeometry(a=a_samples[i], two_c=two_c_samples[i])

        # Update weld SCF for this sample
        weld = WeldJoint(
            weld_type=config.weld.weld_type,
            fat_class=config.weld.fat_class,
            scf=scf_samples[i],
            as_welded=config.weld.as_welded,
        )

        sigma_res = config.sigma_residual if config.sigma_residual is not None else sigma_y_samples[i]

        try:
            result = assess_flaw(
                mat=mat,
                flaw=flaw,
                pipe=config.pipe,
                weld=weld,
                pressure=pressure_samples[i],
                sigma_residual=sigma_res,
            )
            Kr_values[i] = result.Kr
            Lr_values[i] = result.Lr
            reserve_factors[i] = result.reserve_factor

            if not result.is_acceptable:
                n_failures += 1
        except (ValueError, ZeroDivisionError):
            # Treat numerical errors as failures (conservative)
            n_failures += 1
            Kr_values[i] = np.nan
            Lr_values[i] = np.nan

    # Compute P_f with Wilson score confidence interval
    P_f = n_failures / n_simulations
    z = _z_score(confidence)
    P_f_lower, P_f_upper = _wilson_ci(n_failures, n_simulations, z)

    # Statistics
    valid_mask = ~np.isnan(Kr_values)
    acceptable_mask = valid_mask & (reserve_factors > 0)

    return MCResult(
        segment_id=config.segment_id,
        n_simulations=n_simulations,
        n_failures=n_failures,
        P_f=P_f,
        P_f_lower=P_f_lower,
        P_f_upper=P_f_upper,
        mean_Kr=float(np.nanmean(Kr_values)),
        mean_Lr=float(np.nanmean(Lr_values)),
        mean_reserve=float(np.nanmean(reserve_factors[acceptable_mask]))
            if acceptable_mask.any() else 0.0,
        all_Kr=Kr_values,
        all_Lr=Lr_values,
    )


def _z_score(confidence: float) -> float:
    """Get z-score for a given confidence level."""
    from scipy.stats import norm
    return norm.ppf((1 + confidence) / 2)


def _wilson_ci(successes: int, n: int, z: float) -> tuple[float, float]:
    """Wilson score confidence interval for a proportion."""
    if n == 0:
        return (0.0, 1.0)
    p_hat = successes / n
    denominator = 1 + z**2 / n
    center = (p_hat + z**2 / (2 * n)) / denominator
    margin = z * np.sqrt(p_hat * (1 - p_hat) / n + z**2 / (4 * n**2)) / denominator
    lower = max(0.0, center - margin)
    upper = min(1.0, center + margin)
    return (lower, upper)


# ---------------------------------------------------------------------------
# Plotting Utility
# ---------------------------------------------------------------------------

def plot_mc_on_fad(
    mc_result: MCResult,
    mat: MaterialProperties,
    save_path: Optional[str] = None,
    title: Optional[str] = None,
):
    """Plot Monte Carlo samples on the FAD diagram.

    Args:
        mc_result: MCResult from monte_carlo_Pf
        mat:       MaterialProperties for drawing the FAD curve
        save_path: If provided, save figure to this path
        title:     Plot title
    """
    from .fad_engine import plot_fad, fad_option1, compute_Lr_max

    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(12, 9))

    # Draw FAD curve
    Lr_max = compute_Lr_max(mat.sigma_y, mat.sigma_u)
    Lr_array = np.linspace(0, Lr_max * 1.1, 500)
    f_array = fad_option1(Lr_array, mat)
    ax.plot(Lr_array, f_array, "b-", linewidth=2.5, label="FAD Curve (Option 1)")
    ax.fill_between(Lr_array, f_array, alpha=0.05, color="blue")
    ax.axvline(x=Lr_max, color="red", linestyle="--", linewidth=1.5, alpha=0.7)

    # Plot MC samples — filter valid and finite
    valid = (~np.isnan(mc_result.all_Kr) & ~np.isnan(mc_result.all_Lr)
             & np.isfinite(mc_result.all_Kr) & np.isfinite(mc_result.all_Lr))
    Kr = mc_result.all_Kr[valid]
    Lr = mc_result.all_Lr[valid]

    # Cap extreme outliers for visualization (keep data, just cap display)
    Kr_cap = np.clip(Kr, 0, 5.0)
    Lr_cap = np.clip(Lr, 0, Lr_max * 2.0)

    # Determine which points are inside/outside the FAD
    f_at_Lr = fad_option1(Lr_cap, mat)
    inside = (Kr_cap <= f_at_Lr) & (Lr_cap < Lr_max)
    outside = ~inside

    ax.scatter(Lr_cap[inside], Kr_cap[inside], s=3, alpha=0.15, color="green", label="Acceptable")
    ax.scatter(Lr_cap[outside], Kr_cap[outside], s=3, alpha=0.3, color="red", label="Unacceptable")

    ax.set_xlabel("$L_r$ (Load Ratio)", fontsize=14)
    ax.set_ylabel("$K_r$ (Fracture Ratio)", fontsize=14)
    ax.set_title(
        title or f"MC Simulation — Segment {mc_result.segment_id}\n"
                 f"P_f = {mc_result.P_f:.4f} [{mc_result.P_f_lower:.4f}, {mc_result.P_f_upper:.4f}] "
                 f"(n={mc_result.n_simulations:,})",
        fontsize=14, fontweight="bold",
    )
    ax.legend(fontsize=11)
    ax.set_xlim(0, Lr_max * 1.3)
    y_max = max(2.0, np.percentile(Kr_cap, 99) * 1.2) if len(Kr_cap) > 0 else 2.0
    ax.set_ylim(0, min(y_max, 5.0))
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    return fig, ax
