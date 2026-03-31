"""
psip.api.models — Pydantic request and response models for all API endpoints.

Every field has a description so the auto-generated OpenAPI docs at /docs
are fully self-explanatory for operators, engineers, and regulators.
"""
from __future__ import annotations

from typing import Dict, List, Optional
from pydantic import BaseModel, Field


# ─────────────────────────────────────────────────────────────────────────────
# SHARED / UTILITY
# ─────────────────────────────────────────────────────────────────────────────

class HealthResponse(BaseModel):
    status: str = Field(..., description="Service status: 'ok'")
    version: str = Field(..., description="psip package version")
    engines: List[str] = Field(..., description="Available computation engines")


# ─────────────────────────────────────────────────────────────────────────────
# FAD ENDPOINT  —  POST /api/fad/assess
# ─────────────────────────────────────────────────────────────────────────────

class FADRequest(BaseModel):
    """Input for a BS 7910:2019 Level 2 Failure Assessment Diagram evaluation."""

    # Material
    sigma_y: float = Field(..., gt=0, description="Yield strength (MPa). Typical X65: 448 MPa.")
    sigma_u: float = Field(..., gt=0, description="Ultimate tensile strength (MPa). Typical X65: 531 MPa.")
    K_mat: float   = Field(..., gt=0, description="Fracture toughness (MPa·√m). Typical range: 80–200.")
    E: float       = Field(207_000.0, gt=0, description="Young's modulus (MPa). Default: 207,000 (carbon steel).")

    # Pipe geometry
    outer_diameter: float  = Field(..., gt=0, description="Pipe outer diameter (mm). E.g., 914 mm (36 inch).")
    wall_thickness: float  = Field(..., gt=0, description="Pipe wall thickness (mm). E.g., 14.3 mm.")
    pressure: float        = Field(..., gt=0, description="Operating internal pressure (MPa). E.g., 7.5 MPa.")

    # Flaw geometry
    flaw_depth: float  = Field(..., gt=0, description="Flaw depth *a* (mm) — through-thickness direction.")
    flaw_length: float = Field(..., gt=0, description="Flaw surface length *2c* (mm).")

    # Weld
    weld_type: str  = Field("butt",  description="Weld type: 'butt', 'fillet', or 'socket'.")
    fat_class: int  = Field(71,      description="IIW FAT classification (e.g., 71, 80, 90).")
    scf: float      = Field(1.0, gt=0, description="Stress concentration factor at weld toe. Default: 1.0.")

    model_config = {
        "json_schema_extra": {
            "example": {
                "sigma_y": 448.0, "sigma_u": 531.0, "K_mat": 120.0,
                "outer_diameter": 914.0, "wall_thickness": 14.3,
                "pressure": 7.5, "flaw_depth": 3.0, "flaw_length": 20.0,
                "weld_type": "butt", "fat_class": 71, "scf": 1.5,
            }
        }
    }


class FADResponse(BaseModel):
    """Result of the BS 7910:2019 Level 2 FAD assessment."""

    is_acceptable: bool  = Field(..., description="True if flaw lies inside the FAD envelope (safe).")
    Kr: float            = Field(..., description="Fracture ratio K_I / K_mat. Must be < f(Lr) for safety.")
    Lr: float            = Field(..., description="Load ratio σ_ref / σ_y. Must be < Lr_max.")
    f_Lr: float          = Field(..., description="FAD curve value f(Lr) — the allowable Kr at this Lr.")
    Lr_max: float        = Field(..., description="Cut-off load ratio (plastic collapse limit).")
    reserve_factor: float = Field(..., description="Safety margin: ratio of distance to FAD curve vs distance to assessment point. > 1.0 = safe.")
    assessment_point: Dict[str, float] = Field(..., description="The (Lr, Kr) point plotted on the FAD.")


# ─────────────────────────────────────────────────────────────────────────────
# MONTE CARLO ENDPOINT  —  POST /api/mc/simulate
# ─────────────────────────────────────────────────────────────────────────────

class MCRequest(BaseModel):
    """Input for a Monte Carlo failure probability simulation."""

    n_simulations: int  = Field(10_000, ge=100, le=100_000,
                                description="Number of Monte Carlo trials. Range: 100–100,000.")
    outer_diameter: float = Field(914.0,  gt=0, description="Pipe outer diameter (mm).")
    wall_thickness: float = Field(14.3,   gt=0, description="Pipe wall thickness (mm).")
    scf: float            = Field(1.5,    gt=0, description="Stress concentration factor at weld toe.")
    segment_id: str       = Field("default", description="Human-readable segment identifier for logging.")
    random_seed: Optional[int] = Field(42, description="Random seed for reproducibility. None = random.")

    model_config = {
        "json_schema_extra": {
            "example": {
                "n_simulations": 10000,
                "outer_diameter": 914.0,
                "wall_thickness": 14.3,
                "scf": 1.5,
                "segment_id": "TAP-SEG-001",
                "random_seed": 42,
            }
        }
    }


class MCResponse(BaseModel):
    """Result of the Monte Carlo failure probability simulation."""

    segment_id: str    = Field(..., description="Segment identifier echoed from the request.")
    n_simulations: int = Field(..., description="Number of trials executed.")
    n_failures: int    = Field(..., description="Number of trials where the flaw was assessed as unacceptable.")
    P_f: float         = Field(..., description="Point estimate of failure probability P_f = n_failures / n_simulations.")
    P_f_lower: float   = Field(..., description="Lower bound of 95% Wilson confidence interval on P_f.")
    P_f_upper: float   = Field(..., description="Upper bound of 95% Wilson confidence interval on P_f.")
    mean_Kr: float     = Field(..., description="Mean fracture ratio Kr across all simulated flaws.")
    mean_Lr: float     = Field(..., description="Mean load ratio Lr across all simulated flaws.")
    mean_reserve: float = Field(..., description="Mean reserve factor for simulations inside the FAD envelope.")
    risk_level: str    = Field(..., description="Qualitative risk band: 'LOW' (<0.1), 'MEDIUM' (0.1–0.5), 'HIGH' (>0.5).")


# ─────────────────────────────────────────────────────────────────────────────
# GAME ENDPOINT  —  POST /api/game/solve
# ─────────────────────────────────────────────────────────────────────────────

class AttackerPrior(BaseModel):
    """Prior probability for one attacker type."""
    attacker_type: str = Field(..., description="Attacker type: 'strategic', 'opportunistic', or 'state_actor'.")
    prior: float       = Field(..., ge=0, le=1, description="Prior probability. All priors must sum to 1.0.")


class GameRequest(BaseModel):
    """
    Input for the Bayesian Stackelberg Security Equilibrium solver.

    Uses the synthetic Gulf Coast network (20 nodes, 22 segments) by default.
    Pass segment_pf_overrides to inject your own P_f values.
    """

    budget: float = Field(..., gt=0, le=1.0,
                          description="Defender coverage budget B ∈ (0, 1]. Fraction of the network that can be actively defended.")
    n_nodes: int  = Field(20, ge=4, le=50,
                          description="Number of nodes in the synthetic test network. Default: 20 (Gulf Coast model).")
    n_segments: int = Field(22, ge=3, le=100,
                            description="Number of segments in the synthetic test network. Default: 22.")
    random_seed: int = Field(42, description="Seed for synthetic network generation.")
    attacker_priors: Optional[List[AttackerPrior]] = Field(
        None,
        description="Prior probabilities over attacker types. Must sum to 1.0. "
                    "Default: STRATEGIC=0.50, OPPORTUNISTIC=0.30, STATE_ACTOR=0.20."
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "budget": 0.40,
                "n_nodes": 20,
                "n_segments": 22,
                "random_seed": 42,
                "attacker_priors": [
                    {"attacker_type": "strategic",     "prior": 0.50},
                    {"attacker_type": "opportunistic", "prior": 0.30},
                    {"attacker_type": "state_actor",   "prior": 0.20},
                ]
            }
        }
    }


class GameResponse(BaseModel):
    """Result of the Bayesian Stackelberg Security Equilibrium computation."""

    equilibrium_type: str              = Field(..., description="'bayesian_sse' or 'sse_single'.")
    budget_used: float                 = Field(..., description="Total coverage budget consumed: Σ c_i.")
    defender_utility: float            = Field(..., description="Expected defender payoff at the SSE.")
    attacker_utility: float            = Field(..., description="Expected attacker payoff at the SSE.")
    coverage_effectiveness: float      = Field(..., description="Expected loss reduction vs. zero coverage (%).")
    coverage_by_segment: Dict[str, float] = Field(..., description="Coverage allocation per segment: {segment_id: c_i*}.")
    attacker_strategy: Dict[str, float]   = Field(..., description="Attacker mixed strategy at equilibrium: {segment_id: prob}.")
    top_3_defended: List[str]          = Field(..., description="The three segments receiving the highest coverage allocation.")
    n_segments: int                    = Field(..., description="Total number of segments in the network.")


# ─────────────────────────────────────────────────────────────────────────────
# ADVERSARIAL ENDPOINT  —  POST /api/adversarial/attack
# ─────────────────────────────────────────────────────────────────────────────

class AdversarialRequest(BaseModel):
    """
    Input for an adversarial attack on the WeldDefectMLP NDE classifier.

    The API trains a fresh WeldDefectMLP on synthetic NDE data, then
    attacks it. This replicates the thesis Sprint 4 evaluation pipeline.
    """

    method: str   = Field("pgd", description="Attack method: 'fgsm', 'bim', or 'pgd'.")
    epsilon: float = Field(0.30, gt=0, le=1.0,
                           description="L∞ perturbation budget. Default: 0.30 (thesis calibration).")
    n_steps: int   = Field(40, ge=1, le=200,
                           description="Iteration count for BIM/PGD. Ignored for FGSM.")
    n_samples: int = Field(200, ge=10, le=2000,
                           description="Number of NDE test samples to attack.")
    random_seed: int = Field(42, description="Seed for synthetic data generation and training.")
    physics_scaled: bool = Field(
        True,
        description="If True, applies physics-informed ε scaling: ε_eff = ε × SCF/1.5. "
                    "Ensures adversarial perturbations remain physically plausible."
    )
    scf: float = Field(1.5, gt=0, description="Weld SCF used for physics-scaled ε. Only used if physics_scaled=True.")

    model_config = {
        "json_schema_extra": {
            "example": {
                "method": "pgd", "epsilon": 0.30, "n_steps": 40,
                "n_samples": 200, "random_seed": 42,
                "physics_scaled": True, "scf": 1.5,
            }
        }
    }


class AdversarialResponse(BaseModel):
    """Result of an adversarial attack evaluation."""

    method: str                  = Field(..., description="Attack method used: 'fgsm', 'bim', or 'pgd'.")
    epsilon_requested: float     = Field(..., description="L∞ budget as provided by the caller.")
    epsilon_effective: float     = Field(..., description="Effective ε after physics scaling (= ε_req if not scaled).")
    n_samples: int               = Field(..., description="Number of NDE samples attacked.")
    clean_accuracy: float        = Field(..., description="Model accuracy on unperturbed inputs (%).")
    adversarial_accuracy: float  = Field(..., description="Model accuracy on adversarial inputs (%). Lower = stronger attack.")
    attack_success_rate: float   = Field(..., description="Fraction of correctly-classified clean samples flipped by the attack (%).")
    mean_l_inf: float            = Field(..., description="Mean L∞ norm of adversarial perturbations.")
    mean_l2: float               = Field(..., description="Mean L2 norm of adversarial perturbations.")
    class_breakdown: Dict[str, float] = Field(..., description="Attack success rate per defect class: {class_name: asr%}.")


# ─────────────────────────────────────────────────────────────────────────────
# NETWORK INFO ENDPOINT  —  GET /api/network/summary
# ─────────────────────────────────────────────────────────────────────────────

class NetworkSummaryResponse(BaseModel):
    """Summary statistics for the default Gulf Coast synthetic network."""

    name: str              = Field(..., description="Network identifier.")
    n_nodes: int           = Field(..., description="Number of pipeline nodes.")
    n_segments: int        = Field(..., description="Number of pipeline segments.")
    total_length_km: float = Field(..., description="Total network length (km).")
    pf_min: float          = Field(..., description="Minimum P_f across all segments.")
    pf_max: float          = Field(..., description="Maximum P_f across all segments.")
    pf_mean: float         = Field(..., description="Mean P_f across all segments.")
    segments: List[Dict]   = Field(..., description="Per-segment data: id, length_km, P_f, material.")
