"""Zone C – Game Engine package (Sprint 3)."""

from .stackelberg_game import (
    AttackerType,
    AttackerProfile,
    TargetNode,
    GameConfig,
    SSEResult,
    StackelbergSolution,
    DEFAULT_ATTACKER_PROFILES,
    build_target_nodes_from_network,
    compute_attacker_utilities,
    compute_betweenness_weights,
    solve_strong_stackelberg_equilibrium,
    solve_bayesian_stackelberg,
    budget_sensitivity_analysis,
)

__all__ = [
    "AttackerType",
    "AttackerProfile",
    "TargetNode",
    "GameConfig",
    "SSEResult",
    "StackelbergSolution",
    "DEFAULT_ATTACKER_PROFILES",
    "build_target_nodes_from_network",
    "compute_attacker_utilities",
    "compute_betweenness_weights",
    "solve_strong_stackelberg_equilibrium",
    "solve_bayesian_stackelberg",
    "budget_sensitivity_analysis",
]
