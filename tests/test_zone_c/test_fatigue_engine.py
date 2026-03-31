"""
Test Suite for IIW Fatigue Engine
==================================

Validates fatigue life calculations against IIW S-N curve properties.
"""

import numpy as np
import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from src.zone_c.physics.fatigue_engine import (
    FatigueParameters,
    fatigue_life,
    fatigue_damage,
    cumulative_fatigue_damage,
    remaining_life_years,
    get_fat_class,
    FAT_CLASS_TABLE,
)


class TestSNCurve:

    def test_fat_class_at_2e6(self):
        """At N=2e6 cycles, stress range must equal FAT class value."""
        params = FatigueParameters(fat_class=71, m_slope=3.0)
        # N = C / delta_sigma^m => delta_sigma = (C/N)^(1/m)
        delta_sigma_at_2e6 = (params.C / 2e6) ** (1.0 / params.m_slope)
        assert abs(delta_sigma_at_2e6 - 71.0) < 0.01

    def test_fat90_at_2e6(self):
        """FAT 90 at 2e6 cycles."""
        params = FatigueParameters(fat_class=90, m_slope=3.0)
        delta_sigma_at_2e6 = (params.C / 2e6) ** (1.0 / params.m_slope)
        assert abs(delta_sigma_at_2e6 - 90.0) < 0.01

    def test_higher_stress_shorter_life(self):
        """Higher stress range must give shorter fatigue life."""
        params = FatigueParameters(fat_class=71)
        N_low = fatigue_life(50.0, params)
        N_high = fatigue_life(100.0, params)
        assert N_high < N_low

    def test_m3_cube_law(self):
        """For m=3: doubling stress should reduce life by factor of 8."""
        params = FatigueParameters(fat_class=71, m_slope=3.0)
        N_1 = fatigue_life(100.0, params)
        N_2 = fatigue_life(200.0, params)
        ratio = N_1 / N_2
        assert abs(ratio - 8.0) < 0.1  # 2^3 = 8

    def test_zero_stress_infinite_life(self):
        """Zero stress range should give infinite life."""
        params = FatigueParameters(fat_class=71)
        N = fatigue_life(0.0, params)
        assert np.isinf(N)

    def test_array_input(self):
        """Fatigue life must handle array inputs."""
        params = FatigueParameters(fat_class=71)
        stresses = np.array([50.0, 100.0, 200.0])
        N = fatigue_life(stresses, params)
        assert N.shape == (3,)
        assert N[0] > N[1] > N[2]

    def test_shear_m5(self):
        """Shear stress type should use m=5."""
        params = FatigueParameters(fat_class=80, stress_type="shear")
        assert params.m_slope == 5.0
        assert params.knee_point_cycles == 1e8


class TestFatigueDamage:

    def test_damage_at_full_life(self):
        """Damage should be 1.0 when n_cycles == N."""
        params = FatigueParameters(fat_class=71)
        N = fatigue_life(100.0, params)
        D = fatigue_damage(100.0, N, params)
        assert abs(D - 1.0) < 1e-6

    def test_damage_partial(self):
        """Half the cycles should give 0.5 damage."""
        params = FatigueParameters(fat_class=71)
        N = fatigue_life(100.0, params)
        D = fatigue_damage(100.0, N / 2.0, params)
        assert abs(D - 0.5) < 1e-6

    def test_cumulative_damage(self):
        """Miner's rule cumulative damage."""
        params = FatigueParameters(fat_class=71)
        spectrum = [(100.0, 1e5), (80.0, 2e5), (60.0, 5e5)]
        D = cumulative_fatigue_damage(spectrum, params)
        assert D > 0
        assert isinstance(D, float)


class TestRemainingLife:

    def test_remaining_life_positive(self):
        """Remaining life should be positive for moderate loading."""
        params = FatigueParameters(fat_class=71)
        years = remaining_life_years(
            delta_sigma=50.0, cycles_per_year=365.0, params=params,
        )
        assert years > 0
        assert not np.isinf(years)

    def test_accumulated_damage_reduces_life(self):
        """Pre-existing damage should reduce remaining life."""
        params = FatigueParameters(fat_class=71)
        life_fresh = remaining_life_years(50.0, 365.0, params, accumulated_damage=0.0)
        life_damaged = remaining_life_years(50.0, 365.0, params, accumulated_damage=0.5)
        assert life_damaged < life_fresh
        assert abs(life_damaged - life_fresh / 2.0) < 1.0


class TestFATClassLookup:

    def test_butt_ground_flush(self):
        assert get_fat_class("butt", "ground_flush") == 112

    def test_fillet_default(self):
        assert get_fat_class("fillet") == 80

    def test_socket_default(self):
        assert get_fat_class("socket") == 56

    def test_unknown_returns_default(self):
        """Unknown condition should return default for weld type."""
        assert get_fat_class("butt", "nonexistent_condition") == 71
