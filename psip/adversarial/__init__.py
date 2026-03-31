"""
psip.adversarial — Gradient-based adversarial attack implementations.

Re-exports the complete public API from the underlying adversarial_attacks module.
"""
from src.zone_a.adversarial_attacks import (
    AttackConfig,
    AttackResult,
    fgsm_attack,
    bim_attack,
    pgd_attack,
    epsilon_sweep,
)

__all__ = [
    "AttackConfig",
    "AttackResult",
    "fgsm_attack",
    "bim_attack",
    "pgd_attack",
    "epsilon_sweep",
]
