"""
psip.fatigue — IIW S-N fatigue life engine (Miner's rule).

Re-exports the complete public API from the underlying fatigue_engine module.
"""
from src.zone_c.physics.fatigue_engine import (
    FatigueParameters,
    get_fat_class,
    fatigue_life,
    fatigue_damage,
    cumulative_fatigue_damage,
    remaining_life_years,
    plot_sn_curve,
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
