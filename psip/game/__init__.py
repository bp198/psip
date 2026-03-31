"""
psip.game — Bayesian Stackelberg Security Game engine.

Re-exports the complete public API from the underlying stackelberg_game module.
"""
from src.zone_c.game.stackelberg_game import (
    AttackerType,
    AttackerProfile,
    TargetNode,
    GameConfig,
    SSEResult,
    StackelbergSolution,
    solve_bayesian_stackelberg,
    solve_strong_stackelberg_equilibrium,
    build_target_nodes_from_network,
    compute_attacker_utilities,
    compute_segment_value,
    compute_betweenness_weights,
    budget_sensitivity_analysis,
)

__all__ = [
    "AttackerType",
    "AttackerProfile",
    "TargetNode",
    "GameConfig",
    "SSEResult",
    "StackelbergSolution",
    "solve_bayesian_stackelberg",
    "solve_strong_stackelberg_equilibrium",
    "build_target_nodes_from_network",
    "compute_attacker_utilities",
    "compute_segment_value",
    "compute_betweenness_weights",
    "budget_sensitivity_analysis",
]
