"""
psip — Pipeline Security & Integrity Platform
==============================================

Physics-Informed Game-Theoretic Defense of Pipeline Infrastructure.

This package exposes a clean, stable public API over four computation engines:

    psip.run_fad()          — BS 7910:2019 Level 2 FAD assessment
    psip.run_mc()           — Monte Carlo failure probability (P_f)
    psip.run_game()         — Bayesian Stackelberg Security Equilibrium
    psip.run_adversarial()  — FGSM / BIM / PGD adversarial attacks on NDE classifier

Sub-packages (direct engine access):
    psip.fad        — FAD engine classes and functions
    psip.mc         — Monte Carlo engine
    psip.game       — Stackelberg game engine
    psip.nde        — WeldDefectMLP classifier
    psip.adversarial — Adversarial attack implementations
    psip.network    — Pipeline graph data model
    psip.fatigue    — IIW S-N fatigue life engine

Example usage
-------------
>>> import psip
>>>
>>> # 1. Assess a single weld flaw
>>> result = psip.run_fad(
...     sigma_y=448.0, sigma_u=531.0, K_mat=120.0,
...     outer_diameter=914.0, wall_thickness=14.3,
...     pressure=7.5, flaw_depth=3.0, flaw_length=20.0,
... )
>>> print(f"Status: {result.status}, Kr={result.Kr:.3f}, Lr={result.Lr:.3f}")
>>>
>>> # 2. Monte Carlo P_f for a segment
>>> pf_result = psip.run_mc(n_simulations=10_000)
>>> print(f"P_f = {pf_result.pf:.4f} (95% CI: {pf_result.ci_lower:.4f}–{pf_result.ci_upper:.4f})")
>>>
>>> # 3. Solve Bayesian Stackelberg game for a network
>>> game_solution = psip.run_game(network, budget=0.40)
>>> print(f"Risk reduction: {game_solution.risk_reduction_pct:.1f}%")
>>>
>>> # 4. Run adversarial attack on a trained classifier
>>> attack_result = psip.run_adversarial(model, X_test, y_test, method="pgd")
>>> print(f"Attack success rate: {attack_result.attack_success_rate:.1%}")

Author:  Babak Pirzadi (STRATEGOS Thesis)
Version: 0.1.0
"""

from __future__ import annotations

__version__ = "0.1.0"
__author__ = "Babak Pirzadi"
__all__ = ["run_fad", "run_mc", "run_game", "run_adversarial"]

# ---------------------------------------------------------------------------
# Sub-package imports (lazy-style: only pulled in when psip itself is used)
# ---------------------------------------------------------------------------
from psip import fad, mc, game, nde, adversarial, network, fatigue


# ---------------------------------------------------------------------------
# High-level convenience functions
# ---------------------------------------------------------------------------

def run_fad(
    sigma_y: float,
    sigma_u: float,
    K_mat: float,
    outer_diameter: float,
    wall_thickness: float,
    pressure: float,
    flaw_depth: float,
    flaw_length: float,
    E: float = 207_000.0,
    scf: float = 1.0,
    weld_type: str = "butt",
    fat_class: int = 71,
):
    """
    Run a BS 7910:2019 Level 2 FAD assessment for a single weld flaw.

    Parameters
    ----------
    sigma_y : float
        Yield strength (MPa).
    sigma_u : float
        Ultimate tensile strength (MPa).
    K_mat : float
        Fracture toughness (MPa·√m).
    outer_diameter : float
        Pipe outer diameter (mm).
    wall_thickness : float
        Pipe wall thickness (mm).
    pressure : float
        Operating pressure (MPa).
    flaw_depth : float
        Flaw depth *a* (mm) — semi-axis in through-thickness direction.
    flaw_length : float
        Flaw surface length *2c* (mm).
    E : float, optional
        Young's modulus (MPa), default 207,000 (carbon steel).
    scf : float, optional
        Stress concentration factor at weld toe, default 1.0.
    weld_type : str, optional
        Weld type: 'butt', 'fillet', or 'socket'. Default 'butt'.
    fat_class : int, optional
        IIW FAT classification (e.g., 71, 80, 90). Default 71.

    Returns
    -------
    FADAssessmentResult
        Contains: is_acceptable (bool), Kr, Lr, f_Lr (FAD curve value),
        Lr_max (cut-off), and reserve_factor (safety margin).
    """
    mat = fad.MaterialProperties(sigma_y=sigma_y, sigma_u=sigma_u, E=E, K_mat=K_mat)
    pipe = fad.PipeGeometry(outer_diameter=outer_diameter, wall_thickness=wall_thickness)
    flaw_geom = fad.FlawGeometry(a=flaw_depth, two_c=flaw_length)
    weld = fad.WeldJoint(weld_type=weld_type, fat_class=fat_class, scf=scf)
    return fad.assess_flaw(mat=mat, pipe=pipe, flaw=flaw_geom, weld=weld, pressure=pressure)


def run_mc(
    n_simulations: int = 10_000,
    segment_config=None,
    random_seed: int | None = 42,
    outer_diameter: float = 914.0,
    wall_thickness: float = 14.3,
    scf: float = 1.5,
):
    """
    Run a Monte Carlo failure probability simulation for a pipeline segment.

    Parameters
    ----------
    n_simulations : int, optional
        Number of Monte Carlo trials. Default 10,000.
    segment_config : PipelineSegmentConfig, optional
        Full segment configuration. If None, builds a default API 5L X65
        configuration using the supplied geometry parameters.
    random_seed : int or None, optional
        NumPy random seed for reproducibility. Default 42.
    outer_diameter : float, optional
        Pipe outer diameter (mm). Used when segment_config is None. Default 914.0.
    wall_thickness : float, optional
        Pipe wall thickness (mm). Used when segment_config is None. Default 14.3.
    scf : float, optional
        Nominal stress concentration factor. Used when segment_config is None. Default 1.5.

    Returns
    -------
    MCResult
        Contains: P_f (point estimate), P_f_lower/P_f_upper (95% Wilson CI),
        n_failures, n_simulations, mean_Kr, mean_Lr, mean_reserve.
    """
    if segment_config is None:
        defaults = mc.default_distributions_api5l_x65()
        pipe = fad.PipeGeometry(outer_diameter=outer_diameter, wall_thickness=wall_thickness)
        weld = fad.WeldJoint(weld_type="butt", fat_class=71, scf=scf)
        segment_config = mc.PipelineSegmentConfig(
            segment_id="psip-default",
            pipe=pipe,
            weld=weld,
            **defaults,
        )
    return mc.monte_carlo_Pf(
        config=segment_config,
        n_simulations=n_simulations,
        seed=random_seed,
    )


def run_game(network_graph, budget: float, attacker_priors: dict | None = None):
    """
    Solve the Bayesian Stackelberg Security Equilibrium for a pipeline network.

    Parameters
    ----------
    network_graph : networkx.DiGraph
        Directed pipeline network graph. Each edge must carry 'pf' (failure
        probability) and 'value' (segment consequence value) attributes.
        Use psip.network.PipelineNetwork to construct a compliant graph.
    budget : float
        Total defender coverage budget B ∈ (0, 1]. Represents the fraction
        of the network that can be actively monitored/defended simultaneously.
    attacker_priors : dict, optional
        Prior probability over attacker types. Keys must be AttackerType enum
        members. Default: {STRATEGIC: 0.50, OPPORTUNISTIC: 0.30,
        STATE_ACTOR: 0.20} as calibrated in the thesis.

    Returns
    -------
    StackelbergSolution
        Contains: coverage (dict segment→c_i*), sse_defender_utility,
        risk_reduction_pct, attacker_best_response, and per-type SSEResults.
    """
    from src.zone_c.game.stackelberg_game import AttackerType, AttackerProfile

    if attacker_priors is None:
        attacker_priors = {
            AttackerType.STRATEGIC: 0.50,
            AttackerType.OPPORTUNISTIC: 0.30,
            AttackerType.STATE_ACTOR: 0.20,
        }

    targets = game.build_target_nodes_from_network(network_graph)
    profiles = [
        AttackerProfile(attacker_type=atype, prior_prob=prob)
        for atype, prob in attacker_priors.items()
    ]
    config = game.GameConfig(targets=targets, attacker_profiles=profiles, budget_fraction=budget)
    return game.solve_bayesian_stackelberg(config)


def run_adversarial(
    model,
    X: "np.ndarray",
    y: "np.ndarray",
    method: str = "pgd",
    epsilon: float = 0.30,
    n_steps: int = 40,
    step_size: float = 0.01,
):
    """
    Run an adversarial attack on the WeldDefectMLP NDE classifier.

    Parameters
    ----------
    model : WeldDefectMLP
        A trained WeldDefectMLP instance.
    X : np.ndarray, shape (N, 32)
        NDE feature vectors (normalised).
    y : np.ndarray, shape (N,) int
        True class labels (0=Clean, 1=Porosity, 2=Crack, 3=Lack-of-fusion).
    method : str, optional
        Attack method: 'fgsm', 'bim', or 'pgd'. Default 'pgd'.
    epsilon : float, optional
        L∞ perturbation budget. Default 0.30 (as per thesis calibration).
    n_steps : int, optional
        Number of iteration steps (BIM/PGD only). Default 40.
    step_size : float, optional
        Per-step perturbation size α (BIM/PGD only). Default 0.01.

    Returns
    -------
    AttackResult
        Contains: X_adv, attack_success_rate, per_sample_success,
        original_predictions, adversarial_predictions.
    """
    import numpy as np

    config = adversarial.AttackConfig(
        epsilon=epsilon,
        n_steps=n_steps,
        step_size=step_size,
    )

    method = method.lower()
    if method == "fgsm":
        return adversarial.fgsm_attack(model=model, X=X, y=y, config=config)
    elif method == "bim":
        return adversarial.bim_attack(model=model, X=X, y=y, config=config)
    elif method == "pgd":
        return adversarial.pgd_attack(model=model, X=X, y=y, config=config)
    else:
        raise ValueError(f"Unknown attack method '{method}'. Choose from: 'fgsm', 'bim', 'pgd'.")
