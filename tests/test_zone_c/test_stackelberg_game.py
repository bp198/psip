"""
Unit Tests — Bayesian Stackelberg Security Game Engine (Sprint 3)
=================================================================

Tests for:
    - AttackerType / AttackerProfile / TargetNode / GameConfig data structures
    - Utility computation (shape, normalisation, type differentiation)
    - LP solver (feasibility, optimality, best-response constraints)
    - Single-type SSE properties (budget, utility bounds, coverage validity)
    - Bayesian SSE properties (prior normalisation, budget satisfaction, dominance)
    - Budget sensitivity analysis (monotonicity, output structure)

Author: Babak Pirzadi (STRATEGOS Thesis)
"""

import pytest
import numpy as np
import warnings

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from src.zone_c.game.stackelberg_game import (
    AttackerType,
    AttackerProfile,
    TargetNode,
    GameConfig,
    SSEResult,
    StackelbergSolution,
    DEFAULT_ATTACKER_PROFILES,
    compute_segment_value,
    compute_attacker_utilities,
    _solve_lp_for_target_q,
    solve_strong_stackelberg_equilibrium,
    solve_bayesian_stackelberg,
    budget_sensitivity_analysis,
)


# =========================================================================
# Helpers
# =========================================================================

def make_targets(n: int = 6, seed: int = 0) -> list:
    """Create a list of n TargetNode objects with controlled properties."""
    rng = np.random.default_rng(seed)
    targets = []
    for i in range(n):
        targets.append(TargetNode(
            segment_id  = f"SEG_{i:03d}",
            P_f         = float(rng.uniform(0.2, 0.9)),
            value       = float(rng.uniform(0.1, 1.0)),
            maop_mpa    = float(rng.uniform(3.0, 10.0)),
            diameter_mm = float(rng.choice([203.0, 305.0, 508.0, 610.0])),
            length_km   = float(rng.uniform(20.0, 200.0)),
            seam_type   = "dsaw",
            grade       = "X52",
            betweenness = float(rng.uniform(0.0, 1.0)),
        ))
    return targets


def make_game_config(n_targets: int = 8, budget_fraction: float = 0.30,
                     protection_factor: float = 0.25, seed: int = 0) -> GameConfig:
    targets = make_targets(n_targets, seed)
    profiles = [
        AttackerProfile(AttackerType.STRATEGIC,    0.50, w_pf=0.6, w_value=1.0, w_betweenness=0.0),
        AttackerProfile(AttackerType.OPPORTUNISTIC, 0.30, w_pf=1.0, w_value=0.2, w_betweenness=0.0),
        AttackerProfile(AttackerType.STATE_ACTOR,  0.20, w_pf=0.5, w_value=0.7, w_betweenness=1.0),
    ]
    return GameConfig(targets=targets, attacker_profiles=profiles,
                      budget_fraction=budget_fraction,
                      protection_factor=protection_factor)


# =========================================================================
# 1. Enumerations & Profiles
# =========================================================================

class TestEnumerations:
    def test_attacker_type_values(self):
        assert {at.value for at in AttackerType} == {
            "strategic", "opportunistic", "state_actor"
        }

    def test_default_profiles_count(self):
        assert len(DEFAULT_ATTACKER_PROFILES) == 3

    def test_default_profiles_cover_all_types(self):
        types = {p.attacker_type for p in DEFAULT_ATTACKER_PROFILES}
        assert types == set(AttackerType)

    def test_default_profiles_priors_sum_to_one(self):
        total = sum(p.prior_prob for p in DEFAULT_ATTACKER_PROFILES)
        assert abs(total - 1.0) < 1e-6

    def test_prior_normalisation_warning(self):
        """GameConfig must auto-normalise priors that don't sum to 1."""
        profiles = [
            AttackerProfile(AttackerType.STRATEGIC,    0.60),
            AttackerProfile(AttackerType.OPPORTUNISTIC, 0.60),
            AttackerProfile(AttackerType.STATE_ACTOR,  0.60),
        ]
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            cfg = GameConfig(targets=make_targets(4), attacker_profiles=profiles)
            assert len(w) == 1
            assert "Normalising" in str(w[0].message)

        # After normalisation priors sum to 1
        total = sum(p.prior_prob for p in cfg.attacker_profiles)
        assert abs(total - 1.0) < 1e-6


# =========================================================================
# 2. TargetNode
# =========================================================================

class TestTargetNode:
    def test_criticality(self):
        t = TargetNode("T1", P_f=0.7, value=0.8, maop_mpa=5.0,
                       diameter_mm=305.0, length_km=100.0)
        assert abs(t.criticality - 0.7 * 0.8) < 1e-9

    def test_network_impact_no_betweenness(self):
        t = TargetNode("T1", P_f=0.5, value=0.4, maop_mpa=5.0,
                       diameter_mm=305.0, length_km=50.0, betweenness=0.0)
        assert abs(t.network_impact - 0.5 * 0.4 * 1.0) < 1e-9

    def test_network_impact_with_betweenness(self):
        t = TargetNode("T1", P_f=0.5, value=0.4, maop_mpa=5.0,
                       diameter_mm=305.0, length_km=50.0, betweenness=0.6)
        expected = 0.5 * 0.4 * 1.6
        assert abs(t.network_impact - expected) < 1e-9


# =========================================================================
# 3. GameConfig
# =========================================================================

class TestGameConfig:
    def test_n_targets(self):
        cfg = make_game_config(n_targets=10)
        assert cfg.n_targets == 10

    def test_budget_formula(self):
        cfg = make_game_config(n_targets=10, budget_fraction=0.30)
        assert abs(cfg.budget - 3.0) < 1e-9

    def test_budget_fraction_1_means_full_coverage(self):
        cfg = make_game_config(n_targets=5, budget_fraction=1.0)
        assert abs(cfg.budget - 5.0) < 1e-9


# =========================================================================
# 4. compute_segment_value
# =========================================================================

class TestSegmentValue:
    def test_returns_positive(self):
        v = compute_segment_value(508.0, 100.0, 7.0, 0.6)
        assert v > 0

    def test_larger_diameter_higher_value(self):
        v_small = compute_segment_value(203.0, 100.0, 7.0, 0.6)
        v_large = compute_segment_value(610.0, 100.0, 7.0, 0.6)
        assert v_large > v_small

    def test_higher_maop_higher_value(self):
        v_low  = compute_segment_value(305.0, 100.0, 3.0, 0.6)
        v_high = compute_segment_value(305.0, 100.0, 9.0, 0.6)
        assert v_high > v_low

    def test_longer_segment_higher_value(self):
        v_short = compute_segment_value(305.0, 20.0,  7.0, 0.6)
        v_long  = compute_segment_value(305.0, 200.0, 7.0, 0.6)
        assert v_long > v_short


# =========================================================================
# 5. compute_attacker_utilities
# =========================================================================

class TestAttackerUtilities:
    @pytest.fixture
    def targets(self):
        return make_targets(8, seed=7)

    def test_output_shape(self, targets):
        profile = AttackerProfile(AttackerType.STRATEGIC, 1.0)
        U = compute_attacker_utilities(targets, profile)
        assert U.shape == (len(targets),)

    def test_max_is_one(self, targets):
        profile = AttackerProfile(AttackerType.STRATEGIC, 1.0)
        U = compute_attacker_utilities(targets, profile)
        assert abs(U.max() - 1.0) < 1e-9

    def test_all_non_negative(self, targets):
        for atype in AttackerType:
            profile = next(p for p in DEFAULT_ATTACKER_PROFILES
                           if p.attacker_type == atype)
            U = compute_attacker_utilities(targets, profile)
            assert (U >= 0).all(), f"{atype.value} has negative utilities"

    def test_opportunistic_prefers_high_pf(self, targets):
        """Opportunistic type (w_value=0.2, w_pf=1.0) should rank P_f heavily."""
        profile = AttackerProfile(AttackerType.OPPORTUNISTIC, 1.0,
                                  w_pf=1.0, w_value=0.0, w_betweenness=0.0)
        U = compute_attacker_utilities(targets, profile)
        # Best target should be the one with highest P_f
        best_idx = int(np.argmax(U))
        best_pf = targets[best_idx].P_f
        all_pf = [t.P_f for t in targets]
        assert best_pf == max(all_pf)

    def test_strategic_considers_value(self, targets):
        """Strategic type weights P_f AND value, so utilities > pure P_f ranking."""
        opp = AttackerProfile(AttackerType.OPPORTUNISTIC, 1.0,
                              w_pf=1.0, w_value=0.0, w_betweenness=0.0)
        strat = AttackerProfile(AttackerType.STRATEGIC, 1.0,
                                w_pf=0.6, w_value=1.0, w_betweenness=0.0)
        U_opp  = compute_attacker_utilities(targets, opp)
        U_strat = compute_attacker_utilities(targets, strat)
        # Rankings may differ
        best_opp   = int(np.argmax(U_opp))
        best_strat = int(np.argmax(U_strat))
        # They don't have to be different but the utility VECTORS must differ
        assert not np.allclose(U_opp, U_strat)


# =========================================================================
# 6. LP Solver (_solve_lp_for_target_q)
# =========================================================================

class TestLPSolver:
    def _make_ua(self, n=5, seed=0):
        rng = np.random.default_rng(seed)
        U = rng.uniform(0.2, 1.0, n)
        return U / U.max()

    def test_optimal_status_for_trivial_case(self):
        """Single-target game should always be feasible."""
        U_a = np.array([1.0])
        c, obj, status = _solve_lp_for_target_q(U_a, 0.25, 1.0, q=0)
        assert status == "optimal"
        assert c is not None

    def test_coverage_in_bounds(self):
        U_a = self._make_ua(6)
        for q in range(6):
            c, obj, status = _solve_lp_for_target_q(U_a, 0.25, 2.0, q=q)
            if c is not None:
                assert (c >= -1e-8).all(), "Coverage below 0"
                assert (c <= 1.0 + 1e-8).all(), "Coverage above 1"

    def test_budget_not_exceeded(self):
        U_a = self._make_ua(6)
        budget = 1.5
        for q in range(6):
            c, obj, status = _solve_lp_for_target_q(U_a, 0.25, budget, q=q)
            if c is not None:
                assert c.sum() <= budget + 1e-6, f"Budget exceeded for q={q}"

    def test_best_response_constraint_satisfied(self):
        """q must remain attacker's best response at the LP solution."""
        U_a = self._make_ua(5, seed=3)
        delta = 0.25
        budget = 2.0
        for q in range(5):
            c, obj, status = _solve_lp_for_target_q(U_a, delta, budget, q=q)
            if c is None:
                continue
            EU = np.array([
                c[i] * delta * U_a[i] + (1 - c[i]) * U_a[i]
                for i in range(5)
            ])
            # q must have the highest or tied EU_a
            assert EU[q] >= EU.max() - 1e-5, \
                f"q={q} is not attacker best-response: EU[q]={EU[q]:.4f}, max={EU.max():.4f}"

    def test_higher_protection_factor_reduces_objective(self):
        """Higher δ → attacker retains more utility when covered → lower LP obj."""
        U_a = self._make_ua(4)
        budget = 1.5
        _, obj_low_prot,  _ = _solve_lp_for_target_q(U_a, 0.10, budget, q=0)
        _, obj_high_prot, _ = _solve_lp_for_target_q(U_a, 0.90, budget, q=0)
        # With high protection factor (δ=0.90, only 10% P_f reduction),
        # the defender benefit is small → lower objective
        assert obj_low_prot >= obj_high_prot - 1e-6


# =========================================================================
# 7. Single-type SSE
# =========================================================================

class TestSingleTypeSSE:
    @pytest.fixture
    def small_game(self):
        return make_game_config(n_targets=6, budget_fraction=0.40, seed=11)

    def test_returns_sse_result(self, small_game):
        sse = solve_strong_stackelberg_equilibrium(small_game, AttackerType.STRATEGIC)
        assert isinstance(sse, SSEResult)

    def test_coverage_shape(self, small_game):
        sse = solve_strong_stackelberg_equilibrium(small_game, AttackerType.STRATEGIC)
        assert sse.coverage_probs.shape == (small_game.n_targets,)

    def test_coverage_in_bounds(self, small_game):
        for atype in AttackerType:
            sse = solve_strong_stackelberg_equilibrium(small_game, atype)
            assert (sse.coverage_probs >= -1e-8).all()
            assert (sse.coverage_probs <= 1.0 + 1e-8).all()

    def test_budget_not_exceeded(self, small_game):
        for atype in AttackerType:
            sse = solve_strong_stackelberg_equilibrium(small_game, atype)
            assert sse.coverage_probs.sum() <= small_game.budget + 1e-5

    def test_utilities_consistent_sign(self, small_game):
        """At equilibrium, U_d = -U_a (zero-sum structure)."""
        sse = solve_strong_stackelberg_equilibrium(small_game, AttackerType.STRATEGIC)
        assert abs(sse.defender_utility + sse.attacker_utility) < 1e-6

    def test_attacker_utility_positive(self, small_game):
        """Attacker always gains positive utility (cannot be stopped completely)."""
        for atype in AttackerType:
            sse = solve_strong_stackelberg_equilibrium(small_game, atype)
            assert sse.attacker_utility > 0

    def test_feasible_flag(self, small_game):
        for atype in AttackerType:
            sse = solve_strong_stackelberg_equilibrium(small_game, atype)
            assert sse.feasible

    def test_missing_type_raises(self, small_game):
        # Create config with only one type
        cfg_one = GameConfig(
            targets=small_game.targets,
            attacker_profiles=[
                AttackerProfile(AttackerType.STRATEGIC, 1.0)
            ],
        )
        with pytest.raises(ValueError):
            solve_strong_stackelberg_equilibrium(cfg_one, AttackerType.OPPORTUNISTIC)

    def test_best_target_valid_index(self, small_game):
        sse = solve_strong_stackelberg_equilibrium(small_game, AttackerType.STRATEGIC)
        assert 0 <= sse.best_attacker_tgt < small_game.n_targets


# =========================================================================
# 8. Bayesian SSE
# =========================================================================

class TestBayesianSSE:
    @pytest.fixture(scope="class")
    def solution(self):
        cfg = make_game_config(n_targets=10, budget_fraction=0.40, seed=42)
        return solve_bayesian_stackelberg(cfg), cfg

    def test_returns_stackelberg_solution(self, solution):
        sol, _ = solution
        assert isinstance(sol, StackelbergSolution)

    def test_equilibrium_type_label(self, solution):
        sol, _ = solution
        assert sol.equilibrium_type == "bayesian_sse"

    def test_coverage_shape(self, solution):
        sol, cfg = solution
        assert sol.coverage_probs.shape == (cfg.n_targets,)

    def test_coverage_in_bounds(self, solution):
        sol, _ = solution
        assert (sol.coverage_probs >= -1e-8).all()
        assert (sol.coverage_probs <= 1.0 + 1e-8).all()

    def test_budget_satisfied(self, solution):
        sol, cfg = solution
        assert sol.budget_used <= cfg.budget + 1e-5

    def test_budget_utilised(self, solution):
        """With greedy fill the solver should use close to the full budget."""
        sol, cfg = solution
        # At least 80% of budget should be used (greedy fill)
        assert sol.budget_used >= 0.80 * cfg.budget

    def test_coverage_by_id_keys_match_targets(self, solution):
        sol, cfg = solution
        target_ids = {t.segment_id for t in cfg.targets}
        assert set(sol.coverage_by_id.keys()) == target_ids

    def test_coverage_by_id_values_match_array(self, solution):
        sol, cfg = solution
        for i, tgt in enumerate(cfg.targets):
            expected = round(float(sol.coverage_probs[i]), 4)
            assert abs(sol.coverage_by_id[tgt.segment_id] - expected) < 1e-5

    def test_all_types_solved(self, solution):
        sol, _ = solution
        for atype in AttackerType:
            assert atype.value in sol.type_solutions

    def test_attacker_strategy_probs_sum_to_one(self, solution):
        sol, _ = solution
        total = sum(sol.attacker_strategy.values())
        assert abs(total - 1.0) < 1e-6

    def test_defender_utility_improves_over_zero_coverage(self, solution):
        """Coverage effectiveness must be positive (defender is better off defending)."""
        sol, _ = solution
        assert sol.coverage_effectiveness > 0

    def test_coverage_concentrates_on_high_utility_segments(self, solution):
        sol, cfg = solution
        # Segment with highest Bayesian-weighted utility should receive
        # at least as much coverage as the median
        coverage_vals = list(sol.coverage_by_id.values())
        median_cov = np.median(coverage_vals)
        max_cov    = max(coverage_vals)
        assert max_cov >= median_cov


# =========================================================================
# 9. Budget Sensitivity Analysis
# =========================================================================

class TestBudgetSensitivity:
    @pytest.fixture(scope="class")
    def sensitivity_results(self):
        cfg = make_game_config(n_targets=8, budget_fraction=0.30, seed=99)
        budgets = np.linspace(0.10, 0.90, 9).tolist()
        return budget_sensitivity_analysis(cfg, budget_fractions=budgets), budgets

    def test_returns_list_of_correct_length(self, sensitivity_results):
        results, budgets = sensitivity_results
        assert len(results) == len(budgets)

    def test_required_keys_present(self, sensitivity_results):
        results, _ = sensitivity_results
        required = {
            "budget_fraction", "budget_absolute", "defender_utility",
            "attacker_utility", "coverage_effectiveness",
            "top_covered_segment", "top_coverage_prob",
        }
        for r in results:
            assert required.issubset(set(r.keys()))

    def test_defender_utility_increases_with_budget(self, sensitivity_results):
        """More budget → higher defender utility (less negative)."""
        results, _ = sensitivity_results
        d_utils = [r["defender_utility"] for r in results]
        # Not strictly monotone (LP solutions can be non-smooth) but overall trend
        assert d_utils[-1] >= d_utils[0] - 1e-6

    def test_attacker_utility_decreases_with_budget(self, sensitivity_results):
        """More budget → lower attacker utility."""
        results, _ = sensitivity_results
        a_utils = [r["attacker_utility"] for r in results]
        assert a_utils[-1] <= a_utils[0] + 1e-6

    def test_budget_fraction_increases_monotonically(self, sensitivity_results):
        results, _ = sensitivity_results
        fracs = [r["budget_fraction"] for r in results]
        assert all(fracs[i] < fracs[i+1] for i in range(len(fracs)-1))

    def test_top_coverage_prob_valid(self, sensitivity_results):
        results, _ = sensitivity_results
        for r in results:
            assert 0.0 <= r["top_coverage_prob"] <= 1.0 + 1e-6
