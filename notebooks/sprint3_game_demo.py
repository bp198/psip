"""
Sprint 3 Demo — Bayesian Stackelberg Security Game
====================================================

Loads the Sprint 2 Gulf Coast synthetic network, builds the game
configuration with three Bayesian attacker types, solves for the
Bayesian Strong Stackelberg Equilibrium, and generates:

    fig13_stackelberg_coverage_map.png  – network with coverage probabilities
    fig14_budget_sensitivity.png        – expected loss vs. defender budget

Author: Babak Pirzadi (STRATEGOS Thesis)
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.cm as cm
import networkx as nx

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.zone_c.network.pipeline_graph import PipelineNetwork
from src.zone_c.game.stackelberg_game import (
    AttackerType,
    AttackerProfile,
    GameConfig,
    DEFAULT_ATTACKER_PROFILES,
    build_target_nodes_from_network,
    solve_strong_stackelberg_equilibrium,
    solve_bayesian_stackelberg,
    budget_sensitivity_analysis,
)

FIGURES_DIR = os.path.join(os.path.dirname(__file__), "..", "docs", "figures")
os.makedirs(FIGURES_DIR, exist_ok=True)

ETC_FIGS = "/sessions/magical-peaceful-curie/mnt/etc/figures"
os.makedirs(ETC_FIGS, exist_ok=True)


# =========================================================================
# 1. Reconstruct Sprint 2 Network
# =========================================================================
print("=" * 70)
print("SPRINT 3: Bayesian Stackelberg Security Game")
print("=" * 70)

print("\n  Reconstructing Sprint 2 Gulf Coast network (seed=42)...")
net = PipelineNetwork("Gulf_Coast_Transmission_SSG")
net.generate_synthetic(n_nodes=20, n_segments=30, seed=42)
net.attach_pf_values(n_simulations=5_000, seed=42)

s = net.summary()
print(f"  Network: {s['n_nodes']} nodes | {s['n_edges']} edges | "
      f"{s['total_length_km']} km")
print(f"  P_f range: [{s['P_f_range'][0]:.4f}, {s['P_f_range'][1]:.4f}] "
      f"| mean: {s['P_f_mean']:.4f}")


# =========================================================================
# 2. Build Game Configuration
# =========================================================================
print("\n  Building game configuration...")

targets = build_target_nodes_from_network(net)
N = len(targets)
print(f"  Targets: {N} pipeline segments")
print(f"  Value range: [{min(t.value for t in targets):.3f}, "
      f"{max(t.value for t in targets):.3f}]")
print(f"  Betweenness range: [{min(t.betweenness for t in targets):.3f}, "
      f"{max(t.betweenness for t in targets):.3f}]")

game_config = GameConfig(
    targets           = targets,
    attacker_profiles = DEFAULT_ATTACKER_PROFILES,
    budget_fraction   = 0.30,     # defend 30% of segments simultaneously
    protection_factor = 0.25,     # coverage reduces P_f to 25% of original
    name              = "Gulf_Coast_30pct_budget",
)

print(f"\n  Budget: {game_config.budget_fraction*100:.0f}% of {N} segments "
      f"= {game_config.budget:.1f} coverage units")
print(f"  Protection factor δ = {game_config.protection_factor} "
      f"(75% P_f reduction when covered)")
print("\n  Attacker type priors:")
for p in DEFAULT_ATTACKER_PROFILES:
    print(f"    {p.attacker_type.value:<15} prior={p.prior_prob:.2f}  "
          f"w_pf={p.w_pf}  w_val={p.w_value}  w_bc={p.w_betweenness}")


# =========================================================================
# 3. Solve SSE for Each Individual Attacker Type
# =========================================================================
print("\n" + "=" * 70)
print("  Single-Type SSE Results")
print("=" * 70)

for atype in AttackerType:
    sse = solve_strong_stackelberg_equilibrium(game_config, atype)
    top_i = sse.best_attacker_tgt
    top_tgt = targets[top_i]
    cov_used = sse.coverage_probs.sum()
    print(f"  {atype.value:<15}  "
          f"attacker_tgt={top_tgt.segment_id}  "
          f"P_f={top_tgt.P_f:.3f}  "
          f"U_d={sse.defender_utility:.4f}  "
          f"U_a={sse.attacker_utility:.4f}  "
          f"Σc={cov_used:.2f}  status={sse.lp_status}")


# =========================================================================
# 4. Solve Bayesian SSE
# =========================================================================
print("\n" + "=" * 70)
print("  Bayesian Stackelberg Equilibrium (all types)")
print("=" * 70)

solution = solve_bayesian_stackelberg(game_config)

print(f"\n  Equilibrium type:      {solution.equilibrium_type}")
print(f"  Defender utility:      {solution.defender_utility:.6f}")
print(f"  Attacker utility:      {solution.attacker_utility:.6f}")
print(f"  Budget used:           {solution.budget_used:.3f} / {game_config.budget:.3f}")
print(f"  Coverage effectiveness:{solution.coverage_effectiveness*100:+.1f}%  "
      f"(vs. zero coverage)")

print("\n  Top-10 coverage allocations:")
sorted_cov = sorted(solution.coverage_by_id.items(), key=lambda x: -x[1])
for seg_id, ci in sorted_cov[:10]:
    tgt = next(t for t in targets if t.segment_id == seg_id)
    print(f"    {seg_id:<28}  c={ci:.4f}  P_f={tgt.P_f:.3f}  "
          f"v={tgt.value:.3f}  bc={tgt.betweenness:.3f}  seam={tgt.seam_type}")

print("\n  Attacker strategy (Bayesian-weighted):")
sorted_atk = sorted(solution.attacker_strategy.items(), key=lambda x: -x[1])
for seg_id, prob in sorted_atk:
    print(f"    target {seg_id:<28}  prob={prob:.3f}")


# =========================================================================
# 5. Budget Sensitivity Analysis
# =========================================================================
print("\n  Running budget sensitivity analysis (19 budget levels)...")
budget_results = budget_sensitivity_analysis(
    game_config,
    budget_fractions=np.linspace(0.05, 0.90, 18).tolist(),
)
print("  Done.")

# Compute zero-budget baseline
baseline = budget_results[0]
print(f"\n  Baseline (5% budget) defender utility: {baseline['defender_utility']:.4f}")
print(f"  Full (90% budget)  defender utility: {budget_results[-1]['defender_utility']:.4f}")


# =========================================================================
# 6. Figure 13 — Coverage Probability Map
# =========================================================================
print("\n" + "=" * 70)
print("Generating Figure 13: Stackelberg Coverage Probability Map")
print("=" * 70)

fig, axes = plt.subplots(1, 2, figsize=(16, 7))
fig.suptitle(
    "Bayesian Stackelberg Security Game — Gulf Coast Pipeline Network\n"
    f"Budget = {game_config.budget_fraction*100:.0f}% | δ = {game_config.protection_factor}",
    fontsize=13, fontweight="bold", y=1.01,
)

G = net.graph
pos = {n: (G.nodes[n]["lon"], G.nodes[n]["lat"]) for n in G.nodes()}

node_type_colors = {
    "source":     "#1a5276",
    "compressor": "#784212",
    "junction":   "#616a6b",
    "delivery":   "#1e8449",
    "storage":    "#7d3c98",
    "valve":      "#b7950b",
}

for ax_idx, (ax, title, edge_values, edge_label, cmap_name) in enumerate([
    (axes[0], "(a) Vulnerability Map — P_f",
     [G[u][v].get("P_f", 0) for u, v in G.edges()], "P_f", "Reds"),
    (axes[1], "(b) Coverage Probability Map — c_i",
     [solution.coverage_by_id.get(G[u][v].get("segment_id", ""), 0)
      for u, v in G.edges()], "Coverage probability c_i", "Blues"),
]):
    ax.set_title(title, fontsize=11, fontweight="bold", pad=8)
    ax.set_xlabel("Longitude", fontsize=9)
    ax.set_ylabel("Latitude", fontsize=9)

    cmap = cm.get_cmap(cmap_name)
    vals = np.array(edge_values)
    vmin, vmax = vals.min(), vals.max()
    norm = plt.Normalize(vmin=vmin, vmax=vmax)

    # Draw edges coloured by value
    for (u, v), val in zip(G.edges(), edge_values):
        x0, y0 = pos[u]
        x1, y1 = pos[v]
        color = cmap(norm(val))
        lw = 1.5 + 3.0 * val  # thicker = higher value
        ax.plot([x0, x1], [y0, y1], color=color, linewidth=lw,
                alpha=0.85, solid_capstyle="round", zorder=2)

    # Edge midpoint labels for high-value segments
    if ax_idx == 1:  # coverage map
        for u, v in G.edges():
            seg_id = G[u][v].get("segment_id", "")
            ci = solution.coverage_by_id.get(seg_id, 0)
            if ci > 0.25:
                mx = (pos[u][0] + pos[v][0]) / 2
                my = (pos[u][1] + pos[v][1]) / 2
                ax.annotate(f"{ci:.2f}", (mx, my), fontsize=6.5,
                            ha="center", va="center",
                            bbox=dict(boxstyle="round,pad=0.1",
                                      fc="white", ec="steelblue", alpha=0.8))

    # Draw nodes
    for node, data in G.nodes(data=True):
        ntype = data.get("type", "junction")
        color = node_type_colors.get(ntype, "#95a5a6")
        size  = 120 if ntype in ("source", "compressor") else 60
        ax.scatter(*pos[node], s=size, c=color, zorder=4,
                   edgecolors="white", linewidths=0.8)
        if ntype in ("source", "delivery", "compressor"):
            ax.annotate(
                data.get("name", node).split("_")[0],
                pos[node], fontsize=6.5, ha="center", va="bottom",
                xytext=(0, 5), textcoords="offset points",
            )

    # Colorbar
    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cb = plt.colorbar(sm, ax=ax, shrink=0.6, pad=0.02)
    cb.set_label(edge_label, fontsize=9)
    cb.ax.tick_params(labelsize=8)

    # Node legend
    legend_handles = [
        mpatches.Patch(color=c, label=t.title())
        for t, c in node_type_colors.items()
        if any(d.get("type") == t for _, d in G.nodes(data=True))
    ]
    ax.legend(handles=legend_handles, loc="lower left",
              fontsize=7.5, framealpha=0.9, title="Node type",
              title_fontsize=8)
    ax.grid(True, alpha=0.2, linestyle="--")
    ax.tick_params(labelsize=8)

plt.tight_layout()
for path in [
    os.path.join(FIGURES_DIR, "fig13_stackelberg_coverage_map.png"),
    os.path.join(ETC_FIGS, "fig13_stackelberg_coverage_map.png"),
]:
    plt.savefig(path, dpi=150, bbox_inches="tight")
plt.close()
print(f"  Saved fig13_stackelberg_coverage_map.png")


# =========================================================================
# 7. Figure 14 — Budget Sensitivity
# =========================================================================
print("\nGenerating Figure 14: Budget Sensitivity Analysis")

fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle(
    "Stackelberg Security Game — Defender Budget Sensitivity Analysis\n"
    "Gulf Coast Pipeline Network (22 segments, PHMSA-calibrated)",
    fontsize=12, fontweight="bold", y=1.02,
)

budget_fracs  = [r["budget_fraction"]       for r in budget_results]
d_utils       = [r["defender_utility"]      for r in budget_results]
a_utils       = [r["attacker_utility"]      for r in budget_results]
effectiveness = [r["coverage_effectiveness"] * 100 for r in budget_results]

NAVY = "#1a3a5c"
RED  = "#c0392b"
GREEN = "#1e8449"

# ── Panel (a): Defender & attacker utilities vs. budget ──────────────────
ax = axes[0]
ax.plot(budget_fracs, d_utils, "o-", color=NAVY, linewidth=2.0,
        markersize=5, label="Defender utility")
ax.plot(budget_fracs, a_utils, "s--", color=RED, linewidth=2.0,
        markersize=5, label="Attacker utility")
ax.axvline(game_config.budget_fraction, color="gray", linestyle=":",
           linewidth=1.5, label=f"Nominal budget ({game_config.budget_fraction*100:.0f}%)")
ax.set_xlabel("Defender budget fraction B/N", fontsize=10)
ax.set_ylabel("Expected utility", fontsize=10)
ax.set_title("(a) Utility vs. Budget", fontsize=10, fontweight="bold")
ax.legend(fontsize=8.5)
ax.grid(True, alpha=0.3)
ax.tick_params(labelsize=8)

# ── Panel (b): Coverage effectiveness (% improvement over zero budget) ───
ax = axes[1]
ax.bar(budget_fracs, effectiveness, width=0.04,
       color=[GREEN if e > 0 else RED for e in effectiveness],
       alpha=0.85, edgecolor="white", linewidth=0.5)
ax.axhline(0, color="black", linewidth=0.8)
ax.axvline(game_config.budget_fraction, color="gray", linestyle=":",
           linewidth=1.5)
ax.set_xlabel("Defender budget fraction B/N", fontsize=10)
ax.set_ylabel("Coverage effectiveness (%)", fontsize=10)
ax.set_title("(b) Coverage Effectiveness\nvs. Zero-Budget Baseline", fontsize=10, fontweight="bold")
ax.grid(True, alpha=0.3, axis="y")
ax.tick_params(labelsize=8)

# ── Panel (c): Marginal benefit of additional budget ─────────────────────
ax = axes[2]
marginal = np.gradient(d_utils, budget_fracs)
ax.plot(budget_fracs, marginal, "D-", color="#8e44ad", linewidth=2.0,
        markersize=5, label="∂U_d / ∂B")
ax.axhline(0, color="black", linewidth=0.8)
ax.axvline(game_config.budget_fraction, color="gray", linestyle=":",
           linewidth=1.5, label=f"Nominal ({game_config.budget_fraction*100:.0f}%)")

# Mark diminishing returns zone
dim_returns_idx = np.argmax(np.array(marginal) < 0) if any(m < 0 for m in marginal) else -1
if dim_returns_idx > 0:
    ax.axvspan(budget_fracs[dim_returns_idx], budget_fracs[-1],
               alpha=0.08, color=RED, label="Diminishing returns")

ax.set_xlabel("Defender budget fraction B/N", fontsize=10)
ax.set_ylabel("Marginal defender utility gain", fontsize=10)
ax.set_title("(c) Marginal Value of Additional\nDefense Resources", fontsize=10, fontweight="bold")
ax.legend(fontsize=8.5)
ax.grid(True, alpha=0.3)
ax.tick_params(labelsize=8)

plt.tight_layout()
for path in [
    os.path.join(FIGURES_DIR, "fig14_budget_sensitivity.png"),
    os.path.join(ETC_FIGS, "fig14_budget_sensitivity.png"),
]:
    plt.savefig(path, dpi=150, bbox_inches="tight")
plt.close()
print(f"  Saved fig14_budget_sensitivity.png")


# =========================================================================
# 8. Summary
# =========================================================================
print("\n" + "=" * 70)
print("SPRINT 3 GAME DEMO — COMPLETE")
print("=" * 70)

print(f"""
  Game: {game_config.name}
  Segments: {N} | Budget: {game_config.budget_fraction*100:.0f}% | δ={game_config.protection_factor}

  Bayesian SSE Results:
    Defender utility:      {solution.defender_utility:+.6f}
    Attacker utility:      {solution.attacker_utility:+.6f}
    Budget used:           {solution.budget_used:.3f} / {game_config.budget:.3f}
    Coverage effectiveness:{solution.coverage_effectiveness*100:+.1f}%

  Per-type SSE summary:""")

for atype in AttackerType:
    sse = solution.type_solutions[atype.value]
    print(f"    {atype.value:<15}  "
          f"U_d={sse.defender_utility:+.4f}  "
          f"U_a={sse.attacker_utility:+.4f}  "
          f"tgt={targets[sse.best_attacker_tgt].segment_id}")

print(f"""
  Figures:
    fig13 → Network coverage probability map
    fig14 → Budget sensitivity (utility, effectiveness, marginal return)

  Payoff matrix: READY for Sprint 4 / thesis Chapter 4
""")
