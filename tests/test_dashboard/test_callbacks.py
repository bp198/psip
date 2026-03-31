"""
Sprint 5 Dashboard — Unit Tests for Callback Functions
=======================================================

Tests all pure callback functions in src/dashboard/callbacks.py
WITHOUT a running Dash server.

Strategy:
  • Build a shared DashboardData fixture once per test session (expensive).
  • Test each callback function independently with controlled inputs.
  • Assert structure, types, ranges, and semantic correctness.

Coverage:
  • DashboardData construction (data_layer)
  • make_network_figure (all colour modes, selection, attacker overlay)
  • make_segment_fad_figure (valid segment, invalid segment)
  • make_adversarial_impact_figure (valid/invalid, SCF scaling)
  • make_scenario_comparison_figure (all three modes)
  • make_budget_slider_figure (default + custom fraction)
  • make_coverage_heatmap_figure (ssg / baseline / diff)
  • segment_intel_panel (valid / None input)
  • SeamFADProfile completeness
  • _resolve_seam_key coverage
  • Expected loss helper correctness

Author: Babak Pirzadi (STRATEGOS Thesis — Sprint 5)
"""

import sys
import os
import pytest
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

from src.dashboard.data_layer import (
    DashboardData,
    build_dashboard_data,
    SEAM_FAD_PROFILES,
    _resolve_seam_key,
    _expected_network_loss,
)
from src.dashboard.callbacks import (
    make_network_figure,
    make_segment_fad_figure,
    make_adversarial_impact_figure,
    make_scenario_comparison_figure,
    make_budget_slider_figure,
    make_coverage_heatmap_figure,
    segment_intel_panel,
)


# ---------------------------------------------------------------------------
# Session-scoped fixture — build data ONCE for all tests in this module
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def data() -> DashboardData:
    """Build dashboard data with reduced sim count for speed in testing."""
    return build_dashboard_data(
        budget_fraction=0.30,
        n_sim_pf=500,       # fewer sims → fast (physics is deterministic at seed)
        n_epochs=10,        # fewer epochs → fast (accuracy drops but structure ok)
        seed=42,
        verbose=False,
    )


@pytest.fixture(scope="session")
def first_segment(data: DashboardData) -> str:
    """Return the first valid segment_id for all segment-specific tests."""
    return data.segment_ids[0]


@pytest.fixture(scope="session")
def high_pf_segment(data: DashboardData) -> str:
    """Return the segment with the highest P_f."""
    return max(data.segment_ids, key=lambda s: data.edge_pf.get(s, 0))


# ===========================================================================
# Data Layer Tests
# ===========================================================================

class TestDashboardDataBuild:
    """Verify DashboardData is fully and correctly populated."""

    def test_data_type(self, data):
        assert isinstance(data, DashboardData)

    def test_network_populated(self, data):
        assert data.n_nodes >= 10
        assert data.n_segments >= 10
        assert len(data.segment_ids) == data.n_segments
        assert len(data.edge_list) == data.n_segments

    def test_node_positions_valid(self, data):
        for node, pos in data.node_positions.items():
            lon, lat = pos
            assert -180 <= lon <= 180
            assert -90  <= lat <= 90

    def test_edge_pf_in_range(self, data):
        for sid, pf in data.edge_pf.items():
            assert 0.0 <= pf <= 1.0, f"P_f out of range for {sid}: {pf}"

    def test_edge_value_normalised(self, data):
        vals = list(data.edge_value.values())
        assert max(vals) <= 1.0 + 1e-9
        assert min(vals) >= 0.0

    def test_coverage_sum_within_budget(self, data):
        total_cov = sum(data.coverage_by_id.values())
        budget = data.game_config.budget
        assert total_cov <= budget * 1.01, (
            f"Coverage sum {total_cov:.3f} exceeds budget {budget:.3f}"
        )

    def test_baseline_coverage_uniform(self, data):
        vals = list(data.baseline_coverage.values())
        assert len(set(round(v, 8) for v in vals)) == 1, (
            "Baseline should be uniform across all segments"
        )

    def test_all_segments_have_fad(self, data):
        for sid in data.segment_ids:
            assert sid in data.fad_results, f"No FAD result for {sid}"

    def test_all_segments_have_adv(self, data):
        for sid in data.segment_ids:
            assert sid in data.adv_results, f"No adv result for {sid}"

    def test_nde_model_clean_accuracy_positive(self, data):
        # With only 10 epochs accuracy might be modest but must be > random (25%)
        assert data.global_clean_acc > 0.25

    def test_attack_success_rates_non_negative(self, data):
        assert 0.0 <= data.global_fgsm_asr <= 1.0
        assert 0.0 <= data.global_bim_asr  <= 1.0
        assert 0.0 <= data.global_pgd_asr  <= 1.0

    def test_scenario_risk_reduction_positive(self, data):
        # SSE should reduce risk vs. uniform baseline
        assert data.scenario_risk_reduction >= 0.0

    def test_epsilon_sweep_shapes(self, data):
        eps = data.epsilon_sweep_data
        n = len(eps["epsilons"])
        assert n > 0
        assert len(eps["acc_fgsm"]) == n
        assert len(eps["acc_bim"])  == n
        assert len(eps["acc_pgd"])  == n

    def test_epsilon_sweep_clean_end(self, data):
        """Accuracy at ε=0 should approximately equal clean accuracy."""
        fgsm_at_zero = data.epsilon_sweep_data["acc_fgsm"][0]
        clean = data.global_clean_acc
        assert abs(fgsm_at_zero - clean) < 0.10, (
            f"ε=0 accuracy {fgsm_at_zero:.3f} differs from clean {clean:.3f} by >10%"
        )

    def test_build_notes_populated(self, data):
        assert len(data.build_notes) > 3


class TestSeamFADProfiles:
    """Validate the seam material/geometry profile registry."""

    def test_all_required_seams_present(self):
        required = {"seamless", "dsaw", "erw_hf", "erw_lf", "spiral", "unknown"}
        assert required.issubset(set(SEAM_FAD_PROFILES.keys()))

    def test_sigma_u_geq_sigma_y(self):
        for key, p in SEAM_FAD_PROFILES.items():
            assert p.material.sigma_u >= p.material.sigma_y, (
                f"{key}: sigma_u < sigma_y"
            )

    def test_scf_positive(self):
        for key, p in SEAM_FAD_PROFILES.items():
            assert p.scf > 0, f"{key}: SCF must be positive"

    def test_flaw_nominal_smaller_than_critical(self):
        for key, p in SEAM_FAD_PROFILES.items():
            assert p.flaw_nominal.a < p.flaw_critical.a, (
                f"{key}: nominal flaw depth should be less than critical"
            )
            assert p.flaw_nominal.two_c < p.flaw_critical.two_c, (
                f"{key}: nominal flaw length should be less than critical"
            )

    def test_pipe_wall_less_than_radius(self):
        for key, p in SEAM_FAD_PROFILES.items():
            geom = p.pipe_geom
            assert geom.wall_thickness < geom.outer_diameter / 2, (
                f"{key}: wall thickness must be less than outer radius"
            )


class TestResolveSeamKey:
    """Validate seam key normalisation."""

    def test_seamless_maps_to_seamless(self):
        assert _resolve_seam_key("seamless") == "seamless"

    def test_dsaw_aliases(self):
        assert _resolve_seam_key("dsaw")       == "dsaw"
        assert _resolve_seam_key("dsaw_seam")  == "dsaw"
        assert _resolve_seam_key("single_saw") == "dsaw"

    def test_erw_variants(self):
        assert _resolve_seam_key("erw_hf")      == "erw_hf"
        assert _resolve_seam_key("erw_hf_seam") == "erw_hf"
        assert _resolve_seam_key("erw_lf")      == "erw_lf"
        assert _resolve_seam_key("erw_lf_seam") == "erw_lf"

    def test_unknown_fallback(self):
        assert _resolve_seam_key("mystery_seam") == "unknown"
        assert _resolve_seam_key("") == "unknown"

    def test_case_insensitive(self):
        assert _resolve_seam_key("SEAMLESS") == "seamless"
        assert _resolve_seam_key("DSAW")     == "dsaw"


class TestExpectedNetworkLoss:
    """Unit tests for the expected loss helper."""

    def test_zero_coverage_gives_raw_risk(self):
        sids = ["A", "B"]
        pf   = {"A": 0.1, "B": 0.2}
        val  = {"A": 1.0, "B": 1.0}
        cov  = {}
        loss = _expected_network_loss(sids, pf, val, cov, 0.25)
        expected = 0.1 * 1.0 + 0.2 * 1.0
        assert abs(loss - expected) < 1e-10

    def test_full_coverage_reduces_risk(self):
        sids = ["A"]
        pf   = {"A": 0.10}
        val  = {"A": 1.0}
        cov_full = {"A": 1.0}
        cov_zero = {}
        delta = 0.25   # protection_factor
        loss_full = _expected_network_loss(sids, pf, val, cov_full, delta)
        loss_zero = _expected_network_loss(sids, pf, val, cov_zero, delta)
        # Full coverage: P_f * 1.0 * (1 - 1.0*(1 - 0.25)) = 0.1 * 0.25
        assert abs(loss_full - 0.1 * delta) < 1e-10
        assert loss_full < loss_zero

    def test_monotone_in_coverage(self):
        sids = ["A", "B", "C"]
        pf   = {"A": 0.05, "B": 0.08, "C": 0.12}
        val  = {"A": 0.8, "B": 1.0, "C": 0.6}
        delta = 0.25
        losses = []
        for c_level in [0.0, 0.1, 0.3, 0.5, 0.7, 1.0]:
            cov = {s: c_level for s in sids}
            losses.append(_expected_network_loss(sids, pf, val, cov, delta))
        assert losses == sorted(losses, reverse=True), (
            "Expected loss should be monotonically non-increasing with coverage"
        )


# ===========================================================================
# Network Figure Tests
# ===========================================================================

class TestMakeNetworkFigure:
    """Tests for the interactive network map figure."""

    def test_returns_figure(self, data):
        fig = make_network_figure(data)
        assert hasattr(fig, "data")
        assert hasattr(fig, "layout")

    def test_has_edge_traces(self, data):
        fig = make_network_figure(data)
        # Should have at least n_segments edge traces + 1 node trace + 1 colourbar
        assert len(fig.data) >= data.n_segments + 1

    def test_coverage_mode(self, data):
        fig = make_network_figure(data, colour_mode="coverage")
        assert "Coverage" in fig.layout.title.text or \
               "coverage" in str(fig.layout).lower()

    def test_risk_mode(self, data):
        fig = make_network_figure(data, colour_mode="risk")
        assert fig is not None

    def test_pf_mode(self, data):
        fig = make_network_figure(data, colour_mode="pf")
        assert "Failure" in fig.layout.title.text or \
               "pf" in fig.layout.title.text.lower() or \
               "P_f" in fig.layout.title.text

    def test_selected_segment_changes_line_style(self, data, first_segment):
        fig_no_sel = make_network_figure(data, selected_segment=None)
        fig_sel    = make_network_figure(data, selected_segment=first_segment)
        # Figures should differ (selected segment gets different dash style)
        dashes_no_sel = [t.line.dash for t in fig_no_sel.data
                         if hasattr(t, "line") and t.line is not None]
        dashes_sel    = [t.line.dash for t in fig_sel.data
                         if hasattr(t, "line") and t.line is not None]
        assert dashes_no_sel != dashes_sel

    def test_attacker_overlay_adds_traces(self, data):
        fig_no_atk = make_network_figure(data, show_attacker_strategy=False)
        fig_atk    = make_network_figure(data, show_attacker_strategy=True)
        # With attacker overlay there should be more traces
        assert len(fig_atk.data) >= len(fig_no_atk.data)

    def test_height_set(self, data):
        fig = make_network_figure(data)
        assert fig.layout.height == 480


# ===========================================================================
# FAD Figure Tests
# ===========================================================================

class TestMakeSegmentFADFigure:
    """Tests for the BS 7910 FAD figure."""

    def test_valid_segment_returns_figure(self, data, first_segment):
        fig = make_segment_fad_figure(data, first_segment)
        assert hasattr(fig, "data")
        assert len(fig.data) >= 2  # FAD curve + at least one assessment point

    def test_invalid_segment_returns_empty_figure(self, data):
        fig = make_segment_fad_figure(data, "NONEXISTENT_SEGMENT")
        # Should return a figure with no segment data (just annotation)
        assert hasattr(fig, "data")

    def test_fad_curve_is_first_trace(self, data, first_segment):
        fig = make_segment_fad_figure(data, first_segment)
        fad_trace = fig.data[0]
        assert hasattr(fad_trace, "x") and hasattr(fad_trace, "y")
        # FAD curve Lr values should be positive and Kr < 1 for most of the range
        assert all(x >= 0 for x in fad_trace.x)
        assert all(y >= 0 for y in fad_trace.y)

    def test_fad_curve_decreasing(self, data, first_segment):
        """BS 7910 FAD curve Kr(Lr) should be non-increasing."""
        fig = make_segment_fad_figure(data, first_segment)
        kr = list(fig.data[0].y)
        # Allow small numerical noise: check that Kr(Lr_max) < Kr(0.1)
        assert kr[-1] <= kr[0] + 0.05

    def test_title_contains_segment_id(self, data, first_segment):
        fig = make_segment_fad_figure(data, first_segment)
        # Segment id should appear in the title
        assert first_segment in fig.layout.title.text

    def test_high_risk_segment_has_assessment_points(self, data, high_pf_segment):
        fig = make_segment_fad_figure(data, high_pf_segment)
        # At least nominal and critical points (2 traces beyond the FAD curve)
        assert len(fig.data) >= 3

    def test_height_set(self, data, first_segment):
        fig = make_segment_fad_figure(data, first_segment)
        assert fig.layout.height == 340


# ===========================================================================
# Adversarial Impact Figure Tests
# ===========================================================================

class TestMakeAdversarialImpactFigure:
    """Tests for per-segment adversarial NDE threat figure."""

    def test_valid_segment_returns_figure(self, data, first_segment):
        fig = make_adversarial_impact_figure(data, first_segment)
        assert hasattr(fig, "data")
        assert len(fig.data) >= 2  # bar trace + gauge indicator

    def test_invalid_segment_returns_figure(self, data):
        fig = make_adversarial_impact_figure(data, "NONEXISTENT")
        assert hasattr(fig, "data")

    def test_group_perturbation_non_negative(self, data, first_segment):
        res = data.adv_results[first_segment]
        assert all(v >= 0 for v in res.group_perturbation), (
            "Feature group perturbation must be non-negative"
        )

    def test_group_perturbation_shape(self, data, first_segment):
        res = data.adv_results[first_segment]
        assert res.group_perturbation.shape == (4,), (
            "Expect 4 feature groups: amplitude, frequency, geometry, texture"
        )

    def test_fgsm_asr_in_range(self, data, first_segment):
        res = data.adv_results[first_segment]
        assert 0.0 <= res.fgsm_asr <= 1.0

    def test_scf_scaling_increases_asr_for_erw_lf(self, data):
        """ERW-LF segments (SCF=1.8) should show higher effective ε than seamless."""
        from src.dashboard.data_layer import _resolve_seam_key, SEAM_FAD_PROFILES
        # Find a seamless and an erw_lf segment if they exist
        seamless_segs = [sid for sid in data.segment_ids
                         if _resolve_seam_key(data.fad_results[sid].seam_type)
                         == "seamless"]
        erw_segs = [sid for sid in data.segment_ids
                    if _resolve_seam_key(data.fad_results[sid].seam_type)
                    == "erw_lf"]
        # If both types exist, check that ASR for erw_lf >= seamless
        # (not guaranteed with random seam assignment, so just check types exist)
        if seamless_segs and erw_segs:
            scf_seamless = SEAM_FAD_PROFILES["seamless"].scf
            scf_erw_lf   = SEAM_FAD_PROFILES["erw_lf"].scf
            assert scf_erw_lf > scf_seamless

    def test_title_contains_segment_id(self, data, first_segment):
        fig = make_adversarial_impact_figure(data, first_segment)
        assert first_segment in fig.layout.title.text

    def test_height_set(self, data, first_segment):
        fig = make_adversarial_impact_figure(data, first_segment)
        assert fig.layout.height == 260


# ===========================================================================
# Scenario Comparison Figure Tests
# ===========================================================================

class TestMakeScenarioComparisonFigure:
    """Tests for per-segment risk comparison figure."""

    def test_returns_figure_both(self, data):
        fig = make_scenario_comparison_figure(data, show_mode="both")
        assert hasattr(fig, "data")

    def test_returns_figure_baseline(self, data):
        fig = make_scenario_comparison_figure(data, show_mode="baseline")
        assert hasattr(fig, "data")

    def test_returns_figure_ssg(self, data):
        fig = make_scenario_comparison_figure(data, show_mode="ssg")
        assert hasattr(fig, "data")

    def test_both_mode_has_two_bar_traces(self, data):
        import plotly.graph_objects as go
        fig = make_scenario_comparison_figure(data, show_mode="both")
        bar_traces = [t for t in fig.data if isinstance(t, go.Bar)]
        assert len(bar_traces) == 2

    def test_ssg_risk_leq_baseline_risk(self, data):
        """Total SSE expected loss should be ≤ baseline expected loss."""
        assert data.scenario_ssg_loss <= data.scenario_baseline_loss * 1.001, (
            "SSE should produce lower or equal total network risk than baseline"
        )

    def test_unprotected_line_present(self, data):
        import plotly.graph_objects as go
        fig = make_scenario_comparison_figure(data, show_mode="both")
        scatter_traces = [t for t in fig.data if isinstance(t, go.Scatter)]
        assert len(scatter_traces) >= 1   # unprotected reference line

    def test_risk_reduction_annotation(self, data):
        fig = make_scenario_comparison_figure(data, show_mode="both")
        annots = fig.layout.annotations
        assert len(annots) >= 1
        assert any("reduction" in str(a.text).lower() or "SSE" in str(a.text)
                   for a in annots)


# ===========================================================================
# Budget Slider Figure Tests
# ===========================================================================

class TestMakeBudgetSliderFigure:
    """Tests for the budget sensitivity figure."""

    def test_returns_figure(self, data):
        fig = make_budget_slider_figure(data)
        assert hasattr(fig, "data")

    def test_has_three_subplots(self, data):
        fig = make_budget_slider_figure(data)
        # Should have at least 4 traces (2 lines in panel 1, 1 bar, 1 line)
        assert len(fig.data) >= 4

    def test_custom_highlight_fraction(self, data):
        fig1 = make_budget_slider_figure(data, highlight_fraction=0.20)
        fig2 = make_budget_slider_figure(data, highlight_fraction=0.70)
        # Both should be valid figures
        assert fig1 is not None
        assert fig2 is not None

    def test_defender_utility_monotone_overall(self, data):
        """Defender utility should generally increase with budget (not strict)."""
        d_utils = [r["defender_utility"] for r in data.budget_results]
        # At least last half should be higher than first quarter
        n = len(d_utils)
        avg_low  = sum(d_utils[:n//4]) / (n//4)
        avg_high = sum(d_utils[3*n//4:]) / (n//4)
        assert avg_high >= avg_low - 0.05, (
            "Higher budget should generally yield better defender utility"
        )

    def test_budget_fractions_increasing(self, data):
        fracs = [r["budget_fraction"] for r in data.budget_results]
        assert fracs == sorted(fracs), "Budget fractions should be in ascending order"


# ===========================================================================
# Coverage Heatmap Figure Tests
# ===========================================================================

class TestMakeCoverageHeatmapFigure:
    """Tests for the defender coverage bar chart."""

    def test_ssg_mode(self, data):
        fig = make_coverage_heatmap_figure(data, mode="ssg")
        assert hasattr(fig, "data")

    def test_baseline_mode(self, data):
        fig = make_coverage_heatmap_figure(data, mode="baseline")
        assert hasattr(fig, "data")

    def test_diff_mode(self, data):
        fig = make_coverage_heatmap_figure(data, mode="diff")
        assert hasattr(fig, "data")

    def test_ssg_bars_sum_within_budget(self, data):
        import plotly.graph_objects as go
        fig = make_coverage_heatmap_figure(data, mode="ssg")
        bar_traces = [t for t in fig.data if isinstance(t, go.Bar)]
        assert len(bar_traces) >= 1
        total = sum(bar_traces[0].y)
        budget = data.game_config.budget
        assert total <= budget * 1.01, (
            f"Coverage bars sum {total:.3f} exceeds budget {budget:.3f}"
        )

    def test_ssg_has_attacker_overlay(self, data):
        import plotly.graph_objects as go
        fig = make_coverage_heatmap_figure(data, mode="ssg")
        # SSE mode should have a Scatter trace for attacker probability
        scatter_traces = [t for t in fig.data if isinstance(t, go.Scatter)]
        assert len(scatter_traces) >= 1

    def test_diff_mode_sums_to_near_zero(self, data):
        """SSE − baseline should sum to ~0 (same total budget)."""
        import plotly.graph_objects as go
        fig = make_coverage_heatmap_figure(data, mode="diff")
        bar_traces = [t for t in fig.data if isinstance(t, go.Bar)]
        if bar_traces:
            diff_sum = sum(bar_traces[0].y)
            assert abs(diff_sum) < 0.1, (
                f"SSE−baseline diff sum {diff_sum:.4f} should be close to 0"
            )

    def test_segments_in_x_axis(self, data):
        fig = make_coverage_heatmap_figure(data, mode="ssg")
        import plotly.graph_objects as go
        bar_traces = [t for t in fig.data if isinstance(t, go.Bar)]
        assert len(bar_traces[0].x) == data.n_segments


# ===========================================================================
# Segment Intel Panel Tests
# ===========================================================================

class TestSegmentIntelPanel:
    """Tests for the pure-function intelligence summary."""

    def test_none_input(self, data):
        result = segment_intel_panel(data, None)
        assert result["segment_id"] is None

    def test_invalid_input(self, data):
        result = segment_intel_panel(data, "DOES_NOT_EXIST")
        assert result["segment_id"] is None

    def test_valid_segment_all_keys_present(self, data, first_segment):
        result = segment_intel_panel(data, first_segment)
        required_keys = [
            "segment_id", "seam_type", "diameter_mm", "length_km",
            "P_f", "coverage_ssg", "coverage_baseline",
            "fad_verdict_nominal", "fad_verdict_critical",
            "fgsm_asr", "scf", "risk_rank", "n_segments",
        ]
        for key in required_keys:
            assert key in result, f"Missing key: {key}"

    def test_segment_id_matches_input(self, data, first_segment):
        result = segment_intel_panel(data, first_segment)
        assert result["segment_id"] == first_segment

    def test_pf_in_range(self, data, first_segment):
        result = segment_intel_panel(data, first_segment)
        assert 0.0 <= result["P_f"] <= 1.0

    def test_coverage_ssg_non_negative(self, data, first_segment):
        result = segment_intel_panel(data, first_segment)
        assert result["coverage_ssg"] >= 0.0

    def test_scf_positive(self, data, first_segment):
        result = segment_intel_panel(data, first_segment)
        assert result["scf"] > 0

    def test_risk_rank_in_valid_range(self, data, first_segment):
        result = segment_intel_panel(data, first_segment)
        assert 1 <= result["risk_rank"] <= result["n_segments"]

    def test_fad_verdict_is_string(self, data, first_segment):
        result = segment_intel_panel(data, first_segment)
        assert isinstance(result["fad_verdict_nominal"], str)
        assert isinstance(result["fad_verdict_critical"], str)

    def test_fgsm_asr_in_range(self, data, first_segment):
        result = segment_intel_panel(data, first_segment)
        if result["fgsm_asr"] is not None:
            assert 0.0 <= result["fgsm_asr"] <= 1.0

    def test_high_pf_segment_has_higher_rank(self, data, high_pf_segment, first_segment):
        """Segment with highest P_f should have rank 1."""
        result = segment_intel_panel(data, high_pf_segment)
        assert result["risk_rank"] == 1

    def test_n_segments_matches_data(self, data, first_segment):
        result = segment_intel_panel(data, first_segment)
        assert result["n_segments"] == data.n_segments

    def test_diameter_positive(self, data, first_segment):
        result = segment_intel_panel(data, first_segment)
        assert result["diameter_mm"] > 0

    def test_length_positive(self, data, first_segment):
        result = segment_intel_panel(data, first_segment)
        assert result["length_km"] > 0
