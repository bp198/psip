"""
Bayesian Stackelberg Security Game Engine  (Sprint 3)
======================================================

Implements a two-player Defender–Attacker Stackelberg Security Game (SSG)
where physics-informed P_f values from Sprint 1 directly drive the payoff
structure over the pipeline network graph from Sprint 2.

Game Structure
--------------
  Leader  (Defender): commits first to a mixed coverage strategy
                       c = (c_1, …, c_N),  Σ c_i ≤ B,  0 ≤ c_i ≤ 1
  Follower (Attacker): observes defender strategy, best-responds by
                       attacking the segment with highest expected utility

Payoffs
-------
  Attacker utility attacking segment i:
      uncovered:  U_a(i) = P_f(i) · v(i)          (full damage)
      covered:    U_a_c(i) = δ · P_f(i) · v(i)    (residual damage, δ = protection factor)

  Defender utility when attacker hits i:
      uncovered:  U_d(i)   = −U_a(i)               (loss equals attacker gain)
      covered:    U_d_c(i) = −U_a_c(i)

Bayesian Extension
------------------
  The defender faces K attacker types θ_k with prior beliefs p_k = P(θ_k).
  Each type has a distinct utility function over targets.

  Types implemented:
      STRATEGIC    – maximises P_f · segment_value  (rational, well-resourced)
      OPPORTUNISTIC – maximises P_f only            (unsophisticated, cost-driven)
      STATE_ACTOR  – maximises P_f · value · betweenness  (targets network chokepoints)

Solution Concept
----------------
  Strong Stackelberg Equilibrium (SSE) computed via LP enumeration:
    For each candidate attacker target q ∈ {1…N}:
        Solve LP_q: maximise defender utility subject to
                    (a) budget constraint,
                    (b) q remains attacker best-response (best-response constraints).
    The SSE is the feasible LP_q with the highest defender objective.

  Bayesian SSG:
    Solve SSE separately for each type k → coverage vectors c^k.
    Bayesian optimal coverage: c* = argmax_c  Σ_k p_k · U_d(c, BR_k(c)).
    Approximated by the prior-weighted convex combination c* ≈ Σ_k p_k · c^k,
    with a correction LP that re-normalises to the budget constraint.

    The approximation is tight when attacker types have non-overlapping
    high-value targets; exact when K = 1.

References
----------
  Paruchuri et al. (2008) "Playing games for security: An efficient exact
      algorithm for solving Bayesian Stackelberg games." AAMAS.
  Tambe (2011) "Security and Game Theory." Cambridge University Press.
  Conitzer & Sandholm (2006) "Computing the optimal strategy to commit to."
      ACM EC.

Author: Babak Pirzadi (STRATEGOS Thesis)
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple

import networkx as nx
import numpy as np
from scipy.optimize import linprog


# =========================================================================
# Enumerations
# =========================================================================

class AttackerType(Enum):
    """Attacker behavioural type for Bayesian game."""
    STRATEGIC    = "strategic"      # Rational, maximises expected damage value
    OPPORTUNISTIC = "opportunistic" # Targets easiest-to-compromise segments
    STATE_ACTOR  = "state_actor"    # Targets structural network chokepoints


# =========================================================================
# Data Classes
# =========================================================================

@dataclass
class TargetNode:
    """
    Physics-informed representation of a pipeline segment as a game target.

    All monetary/value quantities are normalised to [0, 1] after construction
    so that utilities are on a common scale.

    Attributes
    ----------
    segment_id   : Unique identifier (from PipeSegment.segment_id)
    P_f          : Monte Carlo failure probability ∈ (0, 1)
    value        : Normalised infrastructure value ∈ [0, 1]
    maop_mpa     : Maximum Allowable Operating Pressure (MPa)
    diameter_mm  : Outer diameter (mm)
    length_km    : Segment length (km)
    seam_type    : PHMSA seam type string
    grade        : API 5L grade string
    betweenness  : Normalised betweenness centrality ∈ [0, 1]
    """
    segment_id:  str
    P_f:         float
    value:       float
    maop_mpa:    float
    diameter_mm: float
    length_km:   float
    seam_type:   str   = "unknown"
    grade:       str   = "X52"
    betweenness: float = 0.0

    @property
    def criticality(self) -> float:
        """Product P_f · value — key attacker metric for STRATEGIC type."""
        return self.P_f * self.value

    @property
    def network_impact(self) -> float:
        """Product P_f · value · betweenness — key metric for STATE_ACTOR."""
        return self.P_f * self.value * (1.0 + self.betweenness)


@dataclass
class AttackerProfile:
    """
    Characterises one attacker type in the Bayesian game.

    Attributes
    ----------
    attacker_type : AttackerType enum value
    prior_prob    : Defender's prior belief P(θ) — must sum to 1 across profiles
    w_pf          : Weight on failure probability in utility function
    w_value       : Weight on segment infrastructure value
    w_betweenness : Weight on betweenness centrality (network chokepoint importance)
    """
    attacker_type:  AttackerType
    prior_prob:     float
    w_pf:           float = 1.0
    w_value:        float = 1.0
    w_betweenness:  float = 0.0


@dataclass
class GameConfig:
    """
    Full specification for a Stackelberg Security Game instance.

    Attributes
    ----------
    targets           : Ordered list of TargetNode objects (pipeline segments)
    attacker_profiles : List of AttackerProfile (one per type)
    budget_fraction   : Fraction of total targets that can be covered simultaneously
                        (B = budget_fraction × N, so B=0.3 means 30% coverage)
    protection_factor : Residual risk fraction when a segment is covered
                        (δ=0.25 means coverage reduces effective P_f by 75%)
    name              : Human-readable game instance label
    """
    targets:           List[TargetNode]
    attacker_profiles: List[AttackerProfile]
    budget_fraction:   float = 0.30
    protection_factor: float = 0.25
    name:              str   = "pipeline_ssg"

    def __post_init__(self):
        priors = [p.prior_prob for p in self.attacker_profiles]
        if abs(sum(priors) - 1.0) > 1e-6:
            warnings.warn(
                f"Attacker type priors sum to {sum(priors):.4f} ≠ 1.0. "
                "Normalising automatically.",
                stacklevel=2,
            )
            total = sum(priors)
            for p in self.attacker_profiles:
                p.prior_prob /= total

    @property
    def n_targets(self) -> int:
        return len(self.targets)

    @property
    def budget(self) -> float:
        """Maximum total coverage: B = budget_fraction × N."""
        return self.budget_fraction * self.n_targets


@dataclass
class SSEResult:
    """
    Result of a single-type Strong Stackelberg Equilibrium.

    Attributes
    ----------
    coverage_probs    : Array shape (N,) — c_i for each target
    best_attacker_tgt : Index of the attacker's best-response target
    defender_utility  : Defender's expected utility at equilibrium
    attacker_utility  : Attacker's expected utility at equilibrium
    lp_status         : scipy linprog status string
    feasible          : Whether a valid SSE was found
    """
    coverage_probs:    np.ndarray
    best_attacker_tgt: int
    defender_utility:  float
    attacker_utility:  float
    lp_status:         str
    feasible:          bool


@dataclass
class StackelbergSolution:
    """
    Full solution to the (possibly Bayesian) Stackelberg Security Game.

    Attributes
    ----------
    coverage_probs       : Array shape (N,) — final coverage allocation
    coverage_by_id       : {segment_id: c_i} — human-readable coverage map
    attacker_strategy    : {segment_id: attack_prob} — attacker mixed strategy
    defender_utility     : Expected defender payoff at equilibrium
    attacker_utility     : Expected attacker payoff at equilibrium
    budget_used          : Σ c_i (should be ≤ GameConfig.budget)
    equilibrium_type     : "sse_single" | "bayesian_sse"
    type_solutions       : Per-type SSEResult for Bayesian games
    payoff_attacker      : Shape (N,) — attacker uncovered utilities U_a(i)
    payoff_defender      : Shape (N,) — defender utilities at equilibrium
    coverage_effectiveness: Expected loss reduction vs. zero coverage
    """
    coverage_probs:         np.ndarray
    coverage_by_id:         Dict[str, float]
    attacker_strategy:      Dict[str, float]
    defender_utility:       float
    attacker_utility:       float
    budget_used:            float
    equilibrium_type:       str
    type_solutions:         Dict[str, SSEResult] = field(default_factory=dict)
    payoff_attacker:        np.ndarray = field(default_factory=lambda: np.array([]))
    payoff_defender:        np.ndarray = field(default_factory=lambda: np.array([]))
    coverage_effectiveness: float = 0.0


# =========================================================================
# Utility Functions
# =========================================================================

def compute_segment_value(
    diameter_mm: float,
    length_km:   float,
    maop_mpa:    float,
    P_f:         float,
) -> float:
    """
    Compute a raw infrastructure value score for a pipeline segment.

    The proxy captures:
      - Physical energy flux capacity: π/4 · D² · MAOP  (pressure × area)
      - Exposure length: length_km
      - Vulnerability weighting: P_f amplifies the expected impact

    Raw score = (diameter_m)² · maop_mpa · sqrt(length_km)
    Values are normalised after computing all targets.

    Returns
    -------
    float : Raw (un-normalised) value score > 0
    """
    diameter_m = diameter_mm / 1000.0
    return (diameter_m ** 2) * maop_mpa * np.sqrt(max(length_km, 1.0))


def compute_betweenness_weights(graph: nx.DiGraph) -> Dict[str, float]:
    """
    Compute normalised betweenness centrality for all edges in the graph.

    Betweenness centrality of edge (u,v) is the fraction of all-pairs shortest
    paths that pass through (u,v). High betweenness = network chokepoint.

    Returns
    -------
    dict : {segment_id: normalised_betweenness ∈ [0, 1]}
    """
    bc = nx.edge_betweenness_centrality(graph, normalized=True)
    result = {}
    for (u, v), centrality in bc.items():
        seg_id = graph[u][v].get("segment_id", f"{u}_{v}")
        result[seg_id] = centrality
    # Normalise to [0, 1]
    if result:
        max_bc = max(result.values()) or 1.0
        result = {k: v / max_bc for k, v in result.items()}
    return result


def build_target_nodes_from_network(
    network,  # PipelineNetwork instance
) -> List[TargetNode]:
    """
    Convert Sprint 2 PipelineNetwork edges into TargetNode objects.

    Computes:
      - Raw infrastructure value from diameter, MAOP, length
      - Normalised value (min-max over all segments)
      - Normalised edge betweenness centrality

    Parameters
    ----------
    network : PipelineNetwork
        Fully constructed network with P_f attached (after attach_pf_values())

    Returns
    -------
    List[TargetNode] : One TargetNode per network edge, sorted by segment_id
    """
    graph = network.graph

    # Compute betweenness centrality
    betweenness = compute_betweenness_weights(graph)

    # Collect raw data
    raw_targets = []
    for u, v, data in graph.edges(data=True):
        seg_id  = data["segment_id"]
        P_f     = data.get("P_f", 0.5)
        diam    = data["diameter_mm"]
        wall    = data["wall_mm"]
        maop    = data.get("maop_mpa", 5.0)
        length  = data["length_km"]
        seam    = data.get("seam_type", "unknown")
        grade   = data.get("grade", "X52")

        raw_value = compute_segment_value(diam, length, maop, P_f)
        bc = betweenness.get(seg_id, 0.0)

        raw_targets.append({
            "segment_id":  seg_id,
            "P_f":         P_f,
            "raw_value":   raw_value,
            "maop_mpa":    maop,
            "diameter_mm": diam,
            "length_km":   length,
            "seam_type":   seam,
            "grade":       grade,
            "betweenness": bc,
        })

    # Normalise value to [0, 1]
    raw_vals = np.array([t["raw_value"] for t in raw_targets])
    v_min, v_max = raw_vals.min(), raw_vals.max()
    v_range = v_max - v_min if v_max > v_min else 1.0

    targets = []
    for t in sorted(raw_targets, key=lambda x: x["segment_id"]):
        norm_value = (t["raw_value"] - v_min) / v_range
        # Clip to small positive value so every segment has nonzero value
        norm_value = max(norm_value, 0.01)
        targets.append(TargetNode(
            segment_id  = t["segment_id"],
            P_f         = t["P_f"],
            value       = norm_value,
            maop_mpa    = t["maop_mpa"],
            diameter_mm = t["diameter_mm"],
            length_km   = t["length_km"],
            seam_type   = t["seam_type"],
            grade       = t["grade"],
            betweenness = t["betweenness"],
        ))

    return targets


def compute_attacker_utilities(
    targets:  List[TargetNode],
    profile:  AttackerProfile,
) -> np.ndarray:
    """
    Compute attacker utility vector U_a(i) for each target under a given type.

    U_a(i) = w_pf · P_f(i)  +  w_value · value(i)  +  w_betweenness · betweenness(i)

    The result is normalised so that max(U_a) = 1.

    Parameters
    ----------
    targets : List[TargetNode]
    profile : AttackerProfile

    Returns
    -------
    np.ndarray shape (N,) : Normalised attacker utilities
    """
    N = len(targets)
    U = np.zeros(N)
    for i, tgt in enumerate(targets):
        U[i] = (
            profile.w_pf          * tgt.P_f
            + profile.w_value     * tgt.value
            + profile.w_betweenness * tgt.betweenness
        )
    # Normalise to [0, 1] so different types are comparable
    u_max = U.max()
    if u_max > 0:
        U /= u_max
    return U


# =========================================================================
# LP Solver — Strong Stackelberg Equilibrium (single type)
# =========================================================================

def _solve_lp_for_target_q(
    U_a:    np.ndarray,   # Attacker uncovered utilities (N,)
    delta:  float,        # Protection factor (residual risk fraction)
    budget: float,        # Maximum Σ c_i
    q:      int,          # Candidate attacker target
) -> Tuple[Optional[np.ndarray], float, str]:
    """
    Solve LP_q: find coverage probabilities c that maximise defender utility
    assuming the attacker will target segment q.

    LP formulation (N variables: c_0 … c_{N-1})
    ─────────────────────────────────────────────
    Minimise:   −(1−δ)·U_a[q]·c_q           [maximise coverage of target q]

    Subject to:
      (1) Budget:   Σ_i c_i ≤ B
      (2) Best-response ∀ j ≠ q:
              EU_a(q|c) ≥ EU_a(j|c)
            ⟺ (1−δ)·[U_a[q]·c_q − U_a[j]·c_j] ≥ U_a[j] − U_a[q]
            ⟺ −(1−δ)·U_a[q]·c_q + (1−δ)·U_a[j]·c_j ≤ U_a[q] − U_a[j]   (as ≤)
      (3) Bounds: 0 ≤ c_i ≤ 1

    Returns
    -------
    (c, obj_val, status) where c is the coverage array if feasible, else None
    """
    N = len(U_a)
    eps = (1.0 - delta)  # benefit factor

    # Objective: minimise −eps·U_a[q]·c_q
    c_obj = np.zeros(N)
    c_obj[q] = -eps * U_a[q]

    # Constraint (1): budget
    A_budget = np.ones((1, N))
    b_budget = np.array([budget])

    # Constraints (2): best-response (one per j ≠ q)
    # Enforce EU_a(q|c) ≥ EU_a(j|c)  for all j ≠ q
    #   ↔  eps·(c_q·U_a[q] − c_j·U_a[j]) ≤ U_a[q] − U_a[j]   (as ≤ for linprog)
    A_br_rows = []
    b_br_rows = []
    for j in range(N):
        if j == q:
            continue
        row = np.zeros(N)
        row[q] = +eps * U_a[q]   # positive coefficient on c_q
        row[j] = -eps * U_a[j]   # negative coefficient on c_j
        A_br_rows.append(row)
        b_br_rows.append(U_a[q] - U_a[j])

    if A_br_rows:
        A_ub = np.vstack([A_budget, np.array(A_br_rows)])
        b_ub = np.concatenate([b_budget, np.array(b_br_rows)])
    else:
        # N == 1 edge case
        A_ub = A_budget
        b_ub = b_budget

    bounds = [(0.0, 1.0)] * N

    result = linprog(
        c_obj,
        A_ub=A_ub, b_ub=b_ub,
        bounds=bounds,
        method="highs",
    )

    if result.status == 0:  # optimal
        return result.x, -result.fun, "optimal"
    elif result.status == 2:
        return None, -np.inf, "infeasible"
    else:
        return None, -np.inf, f"lp_status_{result.status}"


def solve_strong_stackelberg_equilibrium(
    game_config:    GameConfig,
    attacker_type:  AttackerType,
) -> SSEResult:
    """
    Compute the Strong Stackelberg Equilibrium for a single attacker type.

    Algorithm:
      Enumerate all N candidate attacker targets q.
      For each q, solve LP_q (see _solve_lp_for_target_q).
      The SSE is the feasible LP with the highest defender objective value.

    Complexity: O(N²) per LP × N LPs = O(N³) for N targets.

    Parameters
    ----------
    game_config   : GameConfig with targets, budget, protection factor
    attacker_type : The attacker type to solve for

    Returns
    -------
    SSEResult
    """
    targets  = game_config.targets
    N        = game_config.n_targets
    delta    = game_config.protection_factor
    budget   = game_config.budget

    # Find the matching profile
    profile = next(
        (p for p in game_config.attacker_profiles if p.attacker_type == attacker_type),
        None
    )
    if profile is None:
        raise ValueError(f"AttackerType {attacker_type} not found in game_config.attacker_profiles")

    U_a = compute_attacker_utilities(targets, profile)

    best_coverage   = None
    best_obj        = -np.inf
    best_q          = 0
    best_status     = "no_feasible_lp"

    for q in range(N):
        c, obj_val, status = _solve_lp_for_target_q(U_a, delta, budget, q)
        if c is not None and obj_val > best_obj:
            best_coverage = c.copy()
            best_obj      = obj_val
            best_q        = q
            best_status   = status

    if best_coverage is None:
        # Fallback: uniform coverage up to budget
        best_coverage = np.full(N, min(1.0, budget / N))
        best_q        = int(np.argmax(U_a))
        best_status   = "fallback_uniform"

    # Compute attacker's expected utility at equilibrium
    c_q       = best_coverage[best_q]
    U_a_eq    = c_q * delta * U_a[best_q] + (1 - c_q) * U_a[best_q]
    U_d_eq    = -(U_a_eq)

    return SSEResult(
        coverage_probs    = best_coverage,
        best_attacker_tgt = best_q,
        defender_utility  = U_d_eq,
        attacker_utility  = U_a_eq,
        lp_status         = best_status,
        feasible          = best_status in ("optimal", "fallback_uniform"),
    )


# =========================================================================
# Bayesian Stackelberg Equilibrium
# =========================================================================

def _bayesian_coverage_lp(
    type_coverages:  List[np.ndarray],            # c^k from each SSE_k
    priors:          List[float],                 # p_k for each type
    budget:          float,                       # Maximum Σ c_i
    ua_profiles:     Optional[List[np.ndarray]] = None,  # U_a^k for greedy spread
) -> np.ndarray:
    """
    Compute the Bayesian optimal coverage vector that fully utilises the budget.

    Algorithm:
      Step 1 – Weighted combination:
          c = Σ_k p_k · c^k  (prior-weighted SSE coverages)
          Scale down if Σ c > B.

      Step 2 – Greedy budget fill:
          Any budget remaining after Step 1 is allocated greedily to segments
          with the highest Bayesian-weighted attacker utility, prioritising
          currently uncovered (c_i < 1) segments.  This ensures the full budget
          is utilised and produces meaningful variation in the sensitivity analysis.

    The greedy fill embeds the "layered defence" principle: once the primary
    threat target is fully covered the defender should protect the next most
    attractive target, and so on, until the budget is exhausted.

    Parameters
    ----------
    type_coverages : Per-type SSE coverage vectors (shape (N,) each)
    priors         : Prior probability weights for each type
    budget         : Maximum Σ c_i allowed
    ua_profiles    : Per-type normalised attacker utility arrays for greedy ordering

    Returns
    -------
    np.ndarray shape (N,) : Coverage probabilities summing to ≤ budget
    """
    K = len(type_coverages)
    N = len(type_coverages[0])

    # Step 1: weighted combination
    c_combined = np.zeros(N)
    for k in range(K):
        c_combined += priors[k] * type_coverages[k]

    total = c_combined.sum()
    if total > budget:
        c_combined *= budget / total
    c_combined = np.clip(c_combined, 0.0, 1.0)

    # Step 2: greedy fill of remaining budget
    budget_remaining = budget - c_combined.sum()
    if budget_remaining > 1e-4 and ua_profiles is not None:
        # Bayesian-weighted attacker utility across all types
        U_w = np.zeros(N)
        for k in range(K):
            U_w += priors[k] * ua_profiles[k]

        # Priority: highest weighted utility on least-covered segments
        priority = sorted(
            range(N),
            key=lambda i: -U_w[i] * (1.0 - c_combined[i]),
        )
        for i in priority:
            if c_combined[i] < 1.0 - 1e-6 and budget_remaining > 1e-6:
                add = min(1.0 - c_combined[i], budget_remaining)
                c_combined[i] += add
                budget_remaining -= add

    return np.clip(c_combined, 0.0, 1.0)


def solve_bayesian_stackelberg(game_config: GameConfig) -> StackelbergSolution:
    """
    Compute the Bayesian Strong Stackelberg Equilibrium over all attacker types.

    For each type k:
        1. Solve SSE_k using LP enumeration.
        2. Record coverage c^k and defender utility U_d^k.

    Bayesian coverage:
        c* = project(Σ_k p_k · c^k) onto budget set.

    Bayesian defender utility:
        U_d* = Σ_k p_k · U_d(c*, BR_k(c*))

    Parameters
    ----------
    game_config : GameConfig

    Returns
    -------
    StackelbergSolution
    """
    targets  = game_config.targets
    N        = game_config.n_targets
    delta    = game_config.protection_factor
    budget   = game_config.budget
    profiles = game_config.attacker_profiles

    type_solutions:  Dict[str, SSEResult] = {}
    type_coverages:  List[np.ndarray]     = []
    priors:          List[float]          = [p.prior_prob for p in profiles]

    # ── Step 1: solve SSE for each attacker type ──────────────────────────
    ua_profiles_list: List[np.ndarray] = []
    for profile in profiles:
        sse = solve_strong_stackelberg_equilibrium(game_config, profile.attacker_type)
        type_solutions[profile.attacker_type.value] = sse
        type_coverages.append(sse.coverage_probs)
        ua_profiles_list.append(compute_attacker_utilities(targets, profile))

    # ── Step 2: Bayesian coverage combination + greedy fill ───────────────
    c_star = _bayesian_coverage_lp(
        type_coverages, priors, budget, ua_profiles=ua_profiles_list
    )

    # ── Step 3: evaluate Bayesian defender & attacker utilities ──────────
    U_d_bayesian = 0.0
    U_a_bayesian = 0.0
    attacker_strategy: Dict[str, float] = {}

    for profile in profiles:
        U_a = compute_attacker_utilities(targets, profile)
        # Attacker best-responds to c_star
        EU_a = np.array([
            c_star[i] * delta * U_a[i] + (1 - c_star[i]) * U_a[i]
            for i in range(N)
        ])
        q_star = int(np.argmax(EU_a))
        seg_id = targets[q_star].segment_id

        U_a_type = EU_a[q_star]
        U_d_type = -U_a_type

        U_d_bayesian += profile.prior_prob * U_d_type
        U_a_bayesian += profile.prior_prob * U_a_type

        # Accumulate attacker strategy (weighted)
        attacker_strategy[seg_id] = (
            attacker_strategy.get(seg_id, 0.0) + profile.prior_prob
        )

    # ── Step 4: compute coverage effectiveness vs. zero-coverage baseline ─
    baseline_utility = 0.0
    for profile in profiles:
        U_a = compute_attacker_utilities(targets, profile)
        q_baseline = int(np.argmax(U_a))
        baseline_utility += profile.prior_prob * (-U_a[q_baseline])

    coverage_effectiveness = (U_d_bayesian - baseline_utility) / max(abs(baseline_utility), 1e-9)

    # ── Step 5: build attacker utility payoff array (weighted) ───────────
    U_a_weighted = np.zeros(N)
    for profile in profiles:
        U_a_weighted += profile.prior_prob * compute_attacker_utilities(targets, profile)

    U_d_payoff = np.array([
        c_star[i] * (-delta * U_a_weighted[i]) + (1 - c_star[i]) * (-U_a_weighted[i])
        for i in range(N)
    ])

    coverage_by_id = {
        targets[i].segment_id: round(float(c_star[i]), 4)
        for i in range(N)
    }

    return StackelbergSolution(
        coverage_probs         = c_star,
        coverage_by_id         = coverage_by_id,
        attacker_strategy      = attacker_strategy,
        defender_utility       = U_d_bayesian,
        attacker_utility       = U_a_bayesian,
        budget_used            = float(c_star.sum()),
        equilibrium_type       = "bayesian_sse",
        type_solutions         = type_solutions,
        payoff_attacker        = U_a_weighted,
        payoff_defender        = U_d_payoff,
        coverage_effectiveness = coverage_effectiveness,
    )


# =========================================================================
# Sensitivity Analysis — Budget Sweep
# =========================================================================

def budget_sensitivity_analysis(
    game_config:       GameConfig,
    budget_fractions:  Optional[List[float]] = None,
) -> List[Dict]:
    """
    Solve the Bayesian SSG across a range of defender budget fractions.

    For each budget level B ∈ budget_fractions:
        1. Create a modified GameConfig with budget_fraction = B.
        2. Solve Bayesian SSE.
        3. Record (B, defender_utility, attacker_utility, budget_used,
                   coverage_effectiveness, top_covered_segment).

    Parameters
    ----------
    game_config      : Base GameConfig (budget_fraction will be overridden)
    budget_fractions : List of fractions ∈ (0, 1].  Defaults to 11 points.

    Returns
    -------
    List[dict] : One dict per budget level.
    """
    if budget_fractions is None:
        budget_fractions = np.linspace(0.05, 0.95, 19).tolist()

    results = []
    for bf in budget_fractions:
        cfg_copy = GameConfig(
            targets           = game_config.targets,
            attacker_profiles = game_config.attacker_profiles,
            budget_fraction   = bf,
            protection_factor = game_config.protection_factor,
            name              = game_config.name,
        )
        sol = solve_bayesian_stackelberg(cfg_copy)

        top_seg = max(sol.coverage_by_id, key=sol.coverage_by_id.get)
        results.append({
            "budget_fraction":       bf,
            "budget_absolute":       sol.budget_used,
            "defender_utility":      sol.defender_utility,
            "attacker_utility":      sol.attacker_utility,
            "coverage_effectiveness": sol.coverage_effectiveness,
            "top_covered_segment":   top_seg,
            "top_coverage_prob":     sol.coverage_by_id[top_seg],
        })

    return results


# =========================================================================
# Convenience: Default Bayesian Game Configuration
# =========================================================================

DEFAULT_ATTACKER_PROFILES = [
    AttackerProfile(
        attacker_type  = AttackerType.STRATEGIC,
        prior_prob     = 0.50,
        w_pf           = 0.6,
        w_value        = 1.0,
        w_betweenness  = 0.0,
    ),
    AttackerProfile(
        attacker_type  = AttackerType.OPPORTUNISTIC,
        prior_prob     = 0.30,
        w_pf           = 1.0,
        w_value        = 0.2,
        w_betweenness  = 0.0,
    ),
    AttackerProfile(
        attacker_type  = AttackerType.STATE_ACTOR,
        prior_prob     = 0.20,
        w_pf           = 0.5,
        w_value        = 0.7,
        w_betweenness  = 1.0,
    ),
]
