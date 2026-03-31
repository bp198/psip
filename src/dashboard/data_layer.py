"""
Dashboard Data Layer — Sprint 5
================================

Pre-computes ALL model outputs at application startup so that dashboard
callbacks can respond interactively without re-running expensive models.

Pipeline:
  1. Reconstruct Sprint 2 network (seed=42, identical to all sprints)
  2. Solve Bayesian Stackelberg SSE + budget sensitivity (Sprint 3)
  3. Build per-segment FAD assessment profiles (Sprint 1)
  4. Train WeldDefectMLP + run adversarial attacks (Sprint 4)
  5. Package everything into DashboardData for the callback layer

Author: Babak Pirzadi (STRATEGOS Thesis — Sprint 5)
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional

# Ensure src/ is importable regardless of working directory
import sys as _sys
import os as _os
_src_root = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
_project_root = _os.path.dirname(_src_root)
for _p in [_src_root, _project_root]:
    if _p not in _sys.path:
        _sys.path.insert(0, _p)

# Zone C — Network & Game
from src.zone_c.network.pipeline_graph import PipelineNetwork
from src.zone_c.physics.fad_engine import (
    MaterialProperties, FlawGeometry, PipeGeometry, WeldJoint,
    fad_option1, compute_Lr_max, assess_flaw,
)
from src.zone_c.game.stackelberg_game import (
    AttackerType,
    GameConfig,
    DEFAULT_ATTACKER_PROFILES,
    build_target_nodes_from_network,
    solve_bayesian_stackelberg,
    budget_sensitivity_analysis,
    StackelbergSolution,
)

# Zone A — NDE & Adversarial
from src.zone_a import (
    DefectClass, CLASS_NAMES, N_FEATURES, N_CLASSES,
    generate_nde_dataset, normalise_features,
    WeldDefectMLP, TrainerConfig, train_model,
    AttackConfig,
    fgsm_attack, bim_attack, pgd_attack, epsilon_sweep,
)


# ---------------------------------------------------------------------------
# FAD material profiles per seam type
# ---------------------------------------------------------------------------

@dataclass
class SeamFADProfile:
    """Material + geometry + flaw profile for a specific seam type."""
    seam_type: str
    material: MaterialProperties
    pipe_geom: PipeGeometry
    flaw_nominal: FlawGeometry       # Nominal (acceptable) flaw
    flaw_critical: FlawGeometry      # Near-critical flaw used for dashboard display
    scf: float = 1.0
    description: str = ""


# Seam-type-specific material/geometry data based on PHMSA fleet + BS 7910 guidance
SEAM_FAD_PROFILES: Dict[str, SeamFADProfile] = {
    "seamless": SeamFADProfile(
        seam_type="seamless",
        material=MaterialProperties(sigma_y=414, sigma_u=517, E=207_000, K_mat=130.0),
        pipe_geom=PipeGeometry(outer_diameter=610.0, wall_thickness=9.5),
        flaw_nominal=FlawGeometry(a=1.5, two_c=10.0, flaw_type="surface"),
        flaw_critical=FlawGeometry(a=4.2, two_c=22.0, flaw_type="surface"),
        scf=1.0,
        description="API 5L X60 Seamless — typical compressor station piping",
    ),
    "dsaw": SeamFADProfile(
        seam_type="dsaw",
        material=MaterialProperties(sigma_y=448, sigma_u=552, E=207_000, K_mat=110.0),
        pipe_geom=PipeGeometry(outer_diameter=762.0, wall_thickness=11.1),
        flaw_nominal=FlawGeometry(a=2.0, two_c=15.0, flaw_type="surface"),
        flaw_critical=FlawGeometry(a=5.5, two_c=30.0, flaw_type="surface"),
        scf=1.25,
        description="API 5L X65 DSAW — long-distance transmission trunk line",
    ),
    "erw_hf": SeamFADProfile(
        seam_type="erw_hf",
        material=MaterialProperties(sigma_y=359, sigma_u=455, E=207_000, K_mat=85.0),
        pipe_geom=PipeGeometry(outer_diameter=508.0, wall_thickness=7.9),
        flaw_nominal=FlawGeometry(a=1.8, two_c=12.0, flaw_type="surface"),
        flaw_critical=FlawGeometry(a=4.0, two_c=20.0, flaw_type="surface"),
        scf=1.5,
        description="API 5L X52 HF-ERW — distribution-grade seam (higher SCF)",
    ),
    "erw_lf": SeamFADProfile(
        seam_type="erw_lf",
        material=MaterialProperties(sigma_y=317, sigma_u=414, E=207_000, K_mat=65.0),
        pipe_geom=PipeGeometry(outer_diameter=406.0, wall_thickness=6.4),
        flaw_nominal=FlawGeometry(a=1.5, two_c=10.0, flaw_type="surface"),
        flaw_critical=FlawGeometry(a=3.0, two_c=16.0, flaw_type="surface"),
        scf=1.8,
        description="API 5L X46 LF-ERW — vintage (pre-1970) low-frequency seam",
    ),
    "spiral": SeamFADProfile(
        seam_type="spiral",
        material=MaterialProperties(sigma_y=359, sigma_u=455, E=207_000, K_mat=90.0),
        pipe_geom=PipeGeometry(outer_diameter=610.0, wall_thickness=8.7),
        flaw_nominal=FlawGeometry(a=2.0, two_c=14.0, flaw_type="surface"),
        flaw_critical=FlawGeometry(a=4.8, two_c=26.0, flaw_type="surface"),
        scf=1.4,
        description="API 5L X52 Spiral-SAW — storage or lateral spur",
    ),
    "unknown": SeamFADProfile(
        seam_type="unknown",
        material=MaterialProperties(sigma_y=345, sigma_u=448, E=207_000, K_mat=80.0),
        pipe_geom=PipeGeometry(outer_diameter=508.0, wall_thickness=8.7),
        flaw_nominal=FlawGeometry(a=2.0, two_c=14.0, flaw_type="surface"),
        flaw_critical=FlawGeometry(a=4.5, two_c=24.0, flaw_type="surface"),
        scf=1.6,
        description="Unknown seam — conservative material properties assumed",
    ),
}

# Normalise seam key lookup (handles aliases from pipeline_graph.py)
_SEAM_KEY_MAP = {
    "seamless":      "seamless",
    "dsaw":          "dsaw",
    "dsaw_seam":     "dsaw",
    "erw_hf":        "erw_hf",
    "erw_hf_seam":   "erw_hf",
    "erw_lf":        "erw_lf",
    "erw_lf_seam":   "erw_lf",
    "erw_unknown":   "erw_hf",
    "flash_welded":  "erw_lf",
    "lap_welded":    "erw_lf",
    "spiral":        "spiral",
    "single_saw":    "dsaw",
    "furnace_butt":  "erw_lf",
    "unknown":       "unknown",
    "girth_weld_field": "unknown",
}


def _resolve_seam_key(raw: str) -> str:
    """Map any seam string to a key in SEAM_FAD_PROFILES."""
    return _SEAM_KEY_MAP.get(raw.lower(), "unknown")


# ---------------------------------------------------------------------------
# Pre-computed per-segment FAD result
# ---------------------------------------------------------------------------

@dataclass
class SegmentFADResult:
    """FAD assessment result for one pipeline segment at a chosen flaw size."""
    segment_id: str
    seam_type: str
    fad_key: str
    # FAD curve arrays for plotting
    Lr_curve: np.ndarray
    Kr_curve: np.ndarray
    # Assessment points
    Lr_nominal: float
    Kr_nominal: float
    verdict_nominal: str
    Lr_critical: float
    Kr_critical: float
    verdict_critical: str
    # Scalar risk proxy
    P_f: float


# ---------------------------------------------------------------------------
# Pre-computed per-segment adversarial result
# ---------------------------------------------------------------------------

@dataclass
class SegmentAdvResult:
    """Adversarial impact on NDE for one pipeline segment."""
    segment_id: str
    seam_type: str
    # FGSM attack result at ε=0.30 using flaw features mapped to this seam
    fgsm_asr: float      # Attack success rate (probability defect is missed)
    fgsm_adv_acc: float  # Model accuracy under FGSM
    # Per-class adversarial confusion array (4×4)
    confusion_clean: np.ndarray   # shape (4, 4)
    confusion_fgsm: np.ndarray    # shape (4, 4)
    # Feature importances (mean absolute perturbation per feature group)
    group_perturbation: np.ndarray  # shape (4,) — amplitude, freq, geom, texture


# ---------------------------------------------------------------------------
# Main Dashboard Data Container
# ---------------------------------------------------------------------------

@dataclass
class DashboardData:
    """All pre-computed data needed by the Sprint 5 dashboard.

    Built once at startup by build_dashboard_data() and passed to all
    callback functions.
    """
    # ── Network ──────────────────────────────────────────────────────────────
    network: PipelineNetwork
    edge_list: List[Tuple[str, str]]          # (u, v) tuples in graph order
    segment_ids: List[str]
    node_positions: Dict[str, Tuple[float, float]]  # node → (lon, lat)
    node_types: Dict[str, str]
    edge_pf: Dict[str, float]                # segment_id → P_f
    edge_value: Dict[str, float]             # segment_id → normalised value

    # ── Game (Sprint 3) ───────────────────────────────────────────────────────
    game_config: GameConfig
    ssg_solution: StackelbergSolution
    budget_results: List[Dict]               # from budget_sensitivity_analysis
    coverage_by_id: Dict[str, float]         # segment_id → c_i
    attacker_strategy: Dict[str, float]      # segment_id → attack prob
    # Baseline: uniform coverage = budget / N
    baseline_coverage: Dict[str, float]

    # ── FAD (Sprint 1 physics) ────────────────────────────────────────────────
    fad_results: Dict[str, SegmentFADResult]  # segment_id → FAD result

    # ── Adversarial NDE (Sprint 4) ────────────────────────────────────────────
    nde_model: WeldDefectMLP
    adv_results: Dict[str, SegmentAdvResult]  # segment_id → adversarial result
    global_clean_acc: float
    global_fgsm_asr: float
    global_bim_asr: float
    global_pgd_asr: float
    epsilon_sweep_data: Dict[str, np.ndarray]  # keys: epsilons, acc_fgsm, acc_bim, acc_pgd

    # ── Scenario comparison ───────────────────────────────────────────────────
    # Expected loss under baseline (uniform) vs physics-informed (SSE) coverage
    scenario_baseline_loss: float
    scenario_ssg_loss: float
    scenario_risk_reduction: float           # (baseline - ssg) / baseline

    # ── Metadata ─────────────────────────────────────────────────────────────
    n_segments: int
    n_nodes: int
    budget_fraction: float
    build_notes: List[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Helper: compute expected network loss under a coverage allocation
# ---------------------------------------------------------------------------

def _expected_network_loss(
    segment_ids: List[str],
    edge_pf: Dict[str, float],
    edge_value: Dict[str, float],
    coverage: Dict[str, float],
    protection_factor: float,
) -> float:
    """Sum over segments of: P_f * value * (1 - c_i*(1-delta))."""
    total = 0.0
    for sid in segment_ids:
        pf = edge_pf.get(sid, 0.0)
        v  = edge_value.get(sid, 1.0)
        ci = coverage.get(sid, 0.0)
        effective_pf = pf * (1.0 - ci * (1.0 - protection_factor))
        total += effective_pf * v
    return total


# ---------------------------------------------------------------------------
# Helper: compute per-segment FAD result
# ---------------------------------------------------------------------------

def _compute_segment_fad(segment_id: str, seam_raw: str, P_f: float) -> SegmentFADResult:
    """Compute BS 7910 FAD assessment for one segment."""
    fad_key = _resolve_seam_key(seam_raw)
    profile = SEAM_FAD_PROFILES[fad_key]

    mat  = profile.material
    pipe = profile.pipe_geom

    # FAD curve: 200 points from 0 to Lr_max
    Lr_max = compute_Lr_max(mat.sigma_y, mat.sigma_u)
    Lr_arr = np.linspace(0.01, Lr_max * 0.99, 200)
    Kr_arr = np.array([float(fad_option1(lr, mat)) for lr in Lr_arr])

    # WeldJoint with seam-specific SCF
    weld = WeldJoint(
        weld_type="butt",
        fat_class=71,
        scf=profile.scf,
        as_welded=True,
    )

    # Operating pressure: 72% SMYS using Barlow's formula
    # sigma_h = P * D / (2*t)  →  P = sigma_h * 2t / D
    sigma_maop = 0.72 * mat.sigma_y         # hoop stress at MAOP (MPa)
    pressure   = sigma_maop * 2 * pipe.wall_thickness / pipe.outer_diameter  # MPa

    # Nominal assessment point
    res_nom = assess_flaw(
        mat=mat,
        flaw=profile.flaw_nominal,
        pipe=pipe,
        weld=weld,
        pressure=pressure,
    )

    # Critical assessment point
    res_crit = assess_flaw(
        mat=mat,
        flaw=profile.flaw_critical,
        pipe=pipe,
        weld=weld,
        pressure=pressure,
    )

    verdict_nominal  = "ACCEPTABLE"   if res_nom.is_acceptable  else "UNACCEPTABLE"
    verdict_critical = "ACCEPTABLE"   if res_crit.is_acceptable else "UNACCEPTABLE"

    return SegmentFADResult(
        segment_id=segment_id,
        seam_type=seam_raw,
        fad_key=fad_key,
        Lr_curve=Lr_arr,
        Kr_curve=Kr_arr,
        Lr_nominal=res_nom.Lr,
        Kr_nominal=res_nom.Kr,
        verdict_nominal=verdict_nominal,
        Lr_critical=res_crit.Lr,
        Kr_critical=res_crit.Kr,
        verdict_critical=verdict_critical,
        P_f=P_f,
    )


# ---------------------------------------------------------------------------
# Helper: compute per-segment adversarial impact
# ---------------------------------------------------------------------------

def _compute_segment_adv(
    segment_id: str,
    seam_raw: str,
    model: WeldDefectMLP,
    X_te: np.ndarray,
    y_te: np.ndarray,
    epsilon: float = 0.30,
) -> SegmentAdvResult:
    """Run FGSM on a seam-type-stratified subset of the test set."""
    from sklearn.metrics import confusion_matrix  # noqa: E402

    # Map seam to a DefectClass distribution weight
    # Segments with higher SCF are more likely to carry CRACK-type defects
    fad_key = _resolve_seam_key(seam_raw)
    scf = SEAM_FAD_PROFILES[fad_key].scf

    # Select a subset weighted by seam vulnerability (use all test samples;
    # seam-specific weighting is expressed in the perturbation budget scaling)
    eps_scaled = min(epsilon * scf / 1.5, 1.0)   # scale ε by relative SCF

    cfg = AttackConfig(epsilon=eps_scaled, n_steps=1,
                       clip_min=-5.0, clip_max=5.0)
    result = fgsm_attack(model, X_te, y_te, cfg)

    # Per-class confusion matrices
    cm_clean = confusion_matrix(y_te, result.y_clean_pred,
                                labels=list(range(N_CLASSES)))
    cm_fgsm  = confusion_matrix(y_te, result.y_adv_pred,
                                labels=list(range(N_CLASSES)))

    # Feature group mean absolute perturbation
    delta = result.perturbation           # (N_test, 32)
    group_pert = np.array([
        np.abs(delta[:, i*8:(i+1)*8]).mean()
        for i in range(4)
    ])

    return SegmentAdvResult(
        segment_id=segment_id,
        seam_type=seam_raw,
        fgsm_asr=result.attack_success_rate,
        fgsm_adv_acc=result.adv_acc,
        confusion_clean=cm_clean,
        confusion_fgsm=cm_fgsm,
        group_perturbation=group_pert,
    )


# ---------------------------------------------------------------------------
# Main builder
# ---------------------------------------------------------------------------

def build_dashboard_data(
    budget_fraction: float = 0.30,
    n_sim_pf: int = 5_000,
    n_epochs: int = 80,
    seed: int = 42,
    verbose: bool = True,
) -> DashboardData:
    """Build all dashboard data from scratch.

    This function is called ONCE at application startup.  It reproduces
    every sprint result with deterministic seeds, then assembles the
    DashboardData object consumed by all callback functions.

    Args:
        budget_fraction: Defender budget as fraction of total segments (0–1).
        n_sim_pf:        Monte-Carlo simulations per segment for P_f.
        n_epochs:        Training epochs for WeldDefectMLP.
        seed:            Master random seed.
        verbose:         Print progress messages.

    Returns:
        Fully populated DashboardData instance.
    """
    notes: List[str] = []

    def log(msg: str) -> None:
        if verbose:
            print(f"  [Dashboard] {msg}")
        notes.append(msg)

    # ── 1. Network ──────────────────────────────────────────────────────────
    log("Building Gulf Coast network (Sprint 2, seed=42)...")
    net = PipelineNetwork("Gulf_Coast_Transmission_Dashboard")
    net.generate_synthetic(n_nodes=20, n_segments=30, seed=seed)
    net.attach_pf_values(n_simulations=n_sim_pf, seed=seed)

    G = net.graph
    edge_list    = list(G.edges())
    segment_ids  = [G[u][v].get("segment_id", f"{u}-{v}") for u, v in edge_list]
    node_positions = {n: (G.nodes[n].get("lon", 0.0), G.nodes[n].get("lat", 0.0))
                      for n in G.nodes()}
    node_types   = {n: G.nodes[n].get("type", "junction") for n in G.nodes()}
    edge_pf      = {G[u][v].get("segment_id", f"{u}-{v}"): G[u][v].get("P_f", 0.0)
                    for u, v in edge_list}
    edge_value   = {}
    for u, v in edge_list:
        sid = G[u][v].get("segment_id", f"{u}-{v}")
        # Normalised value: diameter-weighted length / sum
        d   = G[u][v].get("diameter_mm", 508.0)
        L   = G[u][v].get("length_km", 1.0)
        edge_value[sid] = d * L
    # Normalise to [0, 1]
    max_val = max(edge_value.values()) or 1.0
    edge_value = {k: v / max_val for k, v in edge_value.items()}
    n_segs = len(segment_ids)
    n_nodes = len(G.nodes())
    log(f"Network: {n_nodes} nodes, {n_segs} segments, "
        f"P_f range [{min(edge_pf.values()):.4f}, {max(edge_pf.values()):.4f}]")

    # ── 2. Game (Sprint 3) ──────────────────────────────────────────────────
    log("Solving Bayesian Stackelberg SSE (Sprint 3)...")
    targets = build_target_nodes_from_network(net)
    game_config = GameConfig(
        targets=targets,
        attacker_profiles=DEFAULT_ATTACKER_PROFILES,
        budget_fraction=budget_fraction,
        protection_factor=0.25,
        name="Dashboard_SSG",
    )
    ssg_solution = solve_bayesian_stackelberg(game_config)
    log(f"SSE: defender_utility={ssg_solution.defender_utility:.4f}, "
        f"coverage_effectiveness={ssg_solution.coverage_effectiveness*100:+.1f}%")

    budget_results = budget_sensitivity_analysis(
        game_config,
        budget_fractions=np.linspace(0.05, 0.90, 18).tolist(),
    )

    # Baseline: uniform coverage
    uniform_c = game_config.budget / n_segs
    baseline_coverage = {sid: uniform_c for sid in segment_ids}

    # ── 3. FAD per segment ──────────────────────────────────────────────────
    log("Computing BS 7910 FAD assessments per segment...")
    fad_results: Dict[str, SegmentFADResult] = {}
    for u, v in edge_list:
        sid  = G[u][v].get("segment_id", f"{u}-{v}")
        seam = G[u][v].get("seam_type", "unknown")
        pf   = edge_pf.get(sid, 0.0)
        fad_results[sid] = _compute_segment_fad(sid, seam, pf)
    log(f"FAD assessments: {len(fad_results)} segments computed")

    # ── 4. NDE Model + Adversarial Attacks (Sprint 4) ──────────────────────
    log("Generating NDE dataset + training WeldDefectMLP...")
    dataset = generate_nde_dataset(n_samples_per_class=500, seed=seed,
                                   noise_level=0.18)
    train_ds, val_ds, test_ds = dataset.split(train_fraction=0.70,
                                              val_fraction=0.15, seed=seed)
    X_tr, X_te, _, _ = normalise_features(train_ds.X, test_ds.X)
    _, X_val, _, _   = normalise_features(train_ds.X, val_ds.X)
    y_tr, y_val, y_te = train_ds.y, val_ds.y, test_ds.y

    nde_model = WeldDefectMLP(n_input=N_FEATURES, n_hidden1=128, n_hidden2=64,
                               n_classes=N_CLASSES, seed=seed)
    tr_cfg = TrainerConfig(n_epochs=n_epochs, batch_size=64, lr_max=0.05,
                           lr_min=1e-4, momentum=0.9, l2_lambda=1e-4,
                           dropout_p=0.30, seed=seed, verbose=verbose,
                           print_every=20)
    nde_model = train_model(nde_model, X_tr, y_tr, X_val, y_val, tr_cfg)
    global_clean_acc = nde_model.accuracy(X_te, y_te)
    log(f"NDE model clean accuracy: {global_clean_acc*100:.1f}%")

    # Global attacks at nominal ε=0.30
    eps_nom = 0.30
    cfg_fgsm = AttackConfig(epsilon=eps_nom, n_steps=1,
                             clip_min=-5.0, clip_max=5.0)
    cfg_bim  = AttackConfig(epsilon=eps_nom, n_steps=20,
                             clip_min=-5.0, clip_max=5.0)
    cfg_pgd  = AttackConfig(epsilon=eps_nom, n_steps=20, random_start=True,
                             clip_min=-5.0, clip_max=5.0)
    res_fgsm = fgsm_attack(nde_model, X_te, y_te, cfg_fgsm)
    res_bim  = bim_attack(nde_model, X_te, y_te, cfg_bim)
    res_pgd  = pgd_attack(nde_model, X_te, y_te, cfg_pgd, seed=seed)
    global_fgsm_asr = res_fgsm.attack_success_rate
    global_bim_asr  = res_bim.attack_success_rate
    global_pgd_asr  = res_pgd.attack_success_rate
    log(f"Attacks (ε=0.30): FGSM ASR={global_fgsm_asr*100:.1f}%  "
        f"BIM={global_bim_asr*100:.1f}%  PGD={global_pgd_asr*100:.1f}%")

    # Epsilon sweep
    eps_arr = np.linspace(0.0, 1.0, 16)
    _, acc_fgsm = epsilon_sweep(nde_model, X_te, y_te, eps_arr, "fgsm", seed=seed)
    _, acc_bim  = epsilon_sweep(nde_model, X_te, y_te, eps_arr, "bim",
                                 n_steps=15, seed=seed)
    _, acc_pgd  = epsilon_sweep(nde_model, X_te, y_te, eps_arr, "pgd",
                                 n_steps=20, seed=seed)
    eps_sweep_data = {
        "epsilons": eps_arr,
        "acc_fgsm": acc_fgsm,
        "acc_bim":  acc_bim,
        "acc_pgd":  acc_pgd,
        "clean_acc": np.full_like(eps_arr, global_clean_acc),
    }

    # Per-segment adversarial impact
    log("Computing per-segment adversarial impacts...")
    adv_results: Dict[str, SegmentAdvResult] = {}
    for u, v in edge_list:
        sid  = G[u][v].get("segment_id", f"{u}-{v}")
        seam = G[u][v].get("seam_type", "unknown")
        adv_results[sid] = _compute_segment_adv(
            sid, seam, nde_model, X_te, y_te, epsilon=eps_nom
        )
    log(f"Per-segment adversarial results: {len(adv_results)} segments")

    # ── 5. Scenario Comparison ──────────────────────────────────────────────
    log("Computing scenario comparison (baseline vs. SSE)...")
    pf_ssg = _expected_network_loss(
        segment_ids, edge_pf, edge_value,
        ssg_solution.coverage_by_id, game_config.protection_factor,
    )
    pf_base = _expected_network_loss(
        segment_ids, edge_pf, edge_value,
        baseline_coverage, game_config.protection_factor,
    )
    pf_none = _expected_network_loss(
        segment_ids, edge_pf, edge_value,
        {}, game_config.protection_factor,
    )
    risk_reduction = (pf_base - pf_ssg) / pf_base if pf_base > 0 else 0.0
    log(f"Network risk — Unprotected: {pf_none:.4f}  "
        f"Baseline: {pf_base:.4f}  SSE: {pf_ssg:.4f}  "
        f"Reduction: {risk_reduction*100:.1f}%")

    log("DashboardData build complete.")

    return DashboardData(
        network=net,
        edge_list=edge_list,
        segment_ids=segment_ids,
        node_positions=node_positions,
        node_types=node_types,
        edge_pf=edge_pf,
        edge_value=edge_value,
        game_config=game_config,
        ssg_solution=ssg_solution,
        budget_results=budget_results,
        coverage_by_id=ssg_solution.coverage_by_id,
        attacker_strategy=ssg_solution.attacker_strategy,
        baseline_coverage=baseline_coverage,
        fad_results=fad_results,
        nde_model=nde_model,
        adv_results=adv_results,
        global_clean_acc=global_clean_acc,
        global_fgsm_asr=global_fgsm_asr,
        global_bim_asr=global_bim_asr,
        global_pgd_asr=global_pgd_asr,
        epsilon_sweep_data=eps_sweep_data,
        scenario_baseline_loss=pf_base,
        scenario_ssg_loss=pf_ssg,
        scenario_risk_reduction=risk_reduction,
        n_segments=n_segs,
        n_nodes=n_nodes,
        budget_fraction=budget_fraction,
        build_notes=notes,
    )
