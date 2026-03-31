"""
psip.fatigue — IIW S-N fatigue life engine (Miner's rule).

Re-exports the complete public API from the underlying fatigue_engine module.
"""

from src.zone_c.physics.fatigue_engine import (
    FatigueParameters,
    cumulative_fatigue_damage,
    fatigue_damage,
    fatigue_life,
    get_fat_class,
    plot_sn_curve,
    remaining_life_years,
)

__all__ = [
    "FatigueParameters",
    "get_fat_class",
    "fatigue_life",
    "fatigue_damage",
    "cumulative_fatigue_damage",
    "remaining_life_years",
    "plot_sn_curve",
]
