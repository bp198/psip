"""
Test Suite for BS 7910 FAD Engine
==================================

Validates the FAD implementation against known analytical values
and BS 7910:2019 boundary conditions.
"""

import numpy as np
import pytest
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from src.zone_c.physics.fad_engine import (
    MaterialProperties,
    FlawGeometry,
    PipeGeometry,
    WeldJoint,
    fad_option1,
    compute_Lr_max,
    compute_mu,
    compute_N_hardening,
    hoop_stress_barlow,
    assess_flaw,
)


# ---------------------------------------------------------------------------
# Material Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def api5l_x65():
    """API 5L X65 pipeline steel — typical properties."""
    return MaterialProperties(
        sigma_y=450.0,    # MPa (65 ksi min)
        sigma_u=535.0,    # MPa (77 ksi min)
        E=207_000.0,      # MPa
        K_mat=100.0,      # MPa*sqrt(m) — moderate toughness
    )


@pytest.fixture
def api5l_x52():
    """API 5L X52 pipeline steel — older vintage pipe."""
    return MaterialProperties(
        sigma_y=358.0,    # MPa (52 ksi min)
        sigma_u=455.0,    # MPa (66 ksi min)
        E=207_000.0,
        K_mat=80.0,
    )


@pytest.fixture
def standard_pipe():
    """20-inch (508mm OD) pipeline, 12.7mm wall — typical transmission."""
    return PipeGeometry(outer_diameter=508.0, wall_thickness=12.7)


@pytest.fixture
def standard_flaw():
    """Moderate surface flaw: 3mm deep, 20mm long."""
    return FlawGeometry(a=3.0, two_c=20.0, flaw_type="surface")


# ---------------------------------------------------------------------------
# Test: FAD Curve Fundamental Properties
# ---------------------------------------------------------------------------

class TestFADCurve:
    """Validate FAD curve against BS 7910 mathematical requirements."""

    def test_fad_at_zero_is_one(self, api5l_x65):
        """f(Lr=0) must equal 1.0 (BS 7910 Eq 7.26 with Lr=0)."""
        f_0 = fad_option1(0.0, api5l_x65)
        assert abs(f_0 - 1.0) < 1e-10

    def test_fad_monotonically_decreasing(self, api5l_x65):
        """FAD curve must be monotonically decreasing for 0 <= Lr <= Lr_max."""
        Lr_max = compute_Lr_max(api5l_x65.sigma_y, api5l_x65.sigma_u)
        Lr = np.linspace(0, Lr_max - 0.001, 200)
        f = fad_option1(Lr, api5l_x65)
        diffs = np.diff(f)
        assert np.all(diffs <= 1e-10), "FAD curve must be monotonically decreasing."

    def test_fad_at_Lr_max_is_zero(self, api5l_x65):
        """f(Lr_max) must equal 0 (BS 7910 Eq 7.28)."""
        Lr_max = compute_Lr_max(api5l_x65.sigma_y, api5l_x65.sigma_u)
        f_max = fad_option1(Lr_max, api5l_x65)
        assert f_max == 0.0

    def test_fad_beyond_Lr_max_is_zero(self, api5l_x65):
        """f(Lr > Lr_max) must be 0."""
        Lr_max = compute_Lr_max(api5l_x65.sigma_y, api5l_x65.sigma_u)
        f_beyond = fad_option1(Lr_max + 0.5, api5l_x65)
        assert f_beyond == 0.0

    def test_fad_continuity_at_Lr_1(self, api5l_x65):
        """FAD curve must be continuous at Lr=1 (transition between Eq 7.26 and 7.27)."""
        f_below = fad_option1(0.9999, api5l_x65)
        f_at_1 = fad_option1(1.0, api5l_x65)
        f_above = fad_option1(1.0001, api5l_x65)
        # Check continuity within tolerance
        assert abs(f_at_1 - f_below) < 0.01
        assert abs(f_at_1 - f_above) < 0.01

    def test_fad_array_input(self, api5l_x65):
        """FAD must handle array inputs correctly."""
        Lr = np.array([0.0, 0.5, 1.0, 1.2])
        f = fad_option1(Lr, api5l_x65)
        assert f.shape == (4,)
        assert f[0] > f[1] > f[2] > f[3]  # Decreasing

    def test_fad_values_between_0_and_1(self, api5l_x65):
        """All FAD values must be in [0, 1]."""
        Lr = np.linspace(0, 2.0, 500)
        f = fad_option1(Lr, api5l_x65)
        assert np.all(f >= 0.0)
        assert np.all(f <= 1.0 + 1e-10)


# ---------------------------------------------------------------------------
# Test: Material Parameter Calculations
# ---------------------------------------------------------------------------

class TestMaterialParameters:

    def test_mu_for_steel(self, api5l_x65):
        """mu should be min(0.001*E/sigma_y, 0.6)."""
        mu = compute_mu(api5l_x65.E, api5l_x65.sigma_y)
        expected = min(0.001 * 207_000 / 450.0, 0.6)
        assert abs(mu - expected) < 1e-10
        assert mu == pytest.approx(0.46, abs=0.01)

    def test_N_hardening(self, api5l_x65):
        """N = 0.3 * (1 - sigma_y/sigma_u)."""
        N = compute_N_hardening(api5l_x65.sigma_y, api5l_x65.sigma_u)
        expected = 0.3 * (1.0 - 450.0 / 535.0)
        assert abs(N - expected) < 1e-10
        assert N > 0.0

    def test_Lr_max(self, api5l_x65):
        """Lr_max = (sigma_y + sigma_u) / (2*sigma_y)."""
        Lr_max = compute_Lr_max(api5l_x65.sigma_y, api5l_x65.sigma_u)
        expected = (450.0 + 535.0) / (2.0 * 450.0)
        assert abs(Lr_max - expected) < 1e-10
        assert Lr_max > 1.0  # Must be > 1 for strain-hardening materials

    def test_Lr_max_x52(self, api5l_x52):
        """Check Lr_max for a different steel grade."""
        Lr_max = compute_Lr_max(api5l_x52.sigma_y, api5l_x52.sigma_u)
        expected = (358.0 + 455.0) / (2.0 * 358.0)
        assert abs(Lr_max - expected) < 1e-10


# ---------------------------------------------------------------------------
# Test: Hoop Stress (Barlow's Equation)
# ---------------------------------------------------------------------------

class TestHoopStress:

    def test_barlow_basic(self):
        """Barlow: sigma_h = P*D/(2*t)."""
        sigma_h = hoop_stress_barlow(pressure=7.0, outer_diameter=508.0, wall_thickness=12.7)
        expected = 7.0 * 508.0 / (2.0 * 12.7)
        assert abs(sigma_h - expected) < 1e-10
        assert sigma_h > 0

    def test_barlow_high_pressure(self):
        """Higher pressure gives higher hoop stress."""
        sigma_low = hoop_stress_barlow(5.0, 508.0, 12.7)
        sigma_high = hoop_stress_barlow(10.0, 508.0, 12.7)
        assert sigma_high > sigma_low


# ---------------------------------------------------------------------------
# Test: Full FAD Assessment
# ---------------------------------------------------------------------------

class TestAssessment:

    def test_small_flaw_acceptable(self, api5l_x65, standard_pipe):
        """A small flaw under moderate pressure should be acceptable."""
        small_flaw = FlawGeometry(a=1.0, two_c=10.0)
        weld = WeldJoint(weld_type="butt", fat_class=71, scf=1.2)
        result = assess_flaw(
            mat=api5l_x65, flaw=small_flaw, pipe=standard_pipe,
            weld=weld, pressure=5.0,
        )
        assert result.is_acceptable
        assert result.Kr > 0
        assert result.Lr > 0
        assert result.reserve_factor > 1.0

    def test_large_flaw_unacceptable(self, api5l_x65, standard_pipe):
        """A very large flaw should be unacceptable."""
        large_flaw = FlawGeometry(a=10.0, two_c=100.0)
        weld = WeldJoint(weld_type="butt", fat_class=71, scf=2.5)
        result = assess_flaw(
            mat=api5l_x65, flaw=large_flaw, pipe=standard_pipe,
            weld=weld, pressure=10.0, sigma_residual=450.0,
        )
        # With high SCF, high pressure, large flaw, and full residual stress
        # this should fail
        assert not result.is_acceptable or result.reserve_factor < 1.0

    def test_higher_pressure_reduces_safety(self, api5l_x65, standard_pipe, standard_flaw):
        """Higher pressure should reduce the reserve factor."""
        weld = WeldJoint(weld_type="butt", fat_class=71, scf=1.5)
        result_low = assess_flaw(
            mat=api5l_x65, flaw=standard_flaw, pipe=standard_pipe,
            weld=weld, pressure=3.0,
        )
        result_high = assess_flaw(
            mat=api5l_x65, flaw=standard_flaw, pipe=standard_pipe,
            weld=weld, pressure=10.0,
        )
        assert result_high.Kr >= result_low.Kr
        assert result_high.Lr >= result_low.Lr

    def test_higher_scf_increases_risk(self, api5l_x65, standard_pipe, standard_flaw):
        """Higher SCF should increase Kr and Lr."""
        weld_low = WeldJoint(weld_type="butt", fat_class=71, scf=1.0)
        weld_high = WeldJoint(weld_type="butt", fat_class=71, scf=3.0)
        result_low = assess_flaw(
            mat=api5l_x65, flaw=standard_flaw, pipe=standard_pipe,
            weld=weld_low, pressure=7.0,
        )
        result_high = assess_flaw(
            mat=api5l_x65, flaw=standard_flaw, pipe=standard_pipe,
            weld=weld_high, pressure=7.0,
        )
        assert result_high.Kr > result_low.Kr


# ---------------------------------------------------------------------------
# Test: Input Validation
# ---------------------------------------------------------------------------

class TestInputValidation:

    def test_invalid_material_raises(self):
        """Negative yield strength should raise ValueError."""
        with pytest.raises(ValueError):
            MaterialProperties(sigma_y=-100, sigma_u=500, K_mat=100)

    def test_sigma_u_less_than_sigma_y_raises(self):
        with pytest.raises(ValueError):
            MaterialProperties(sigma_y=500, sigma_u=400, K_mat=100)

    def test_invalid_flaw_raises(self):
        with pytest.raises(ValueError):
            FlawGeometry(a=-1.0, two_c=10.0)

    def test_invalid_pipe_raises(self):
        with pytest.raises(ValueError):
            PipeGeometry(outer_diameter=100, wall_thickness=60)  # wall > radius
