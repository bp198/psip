"""
Sprint 5 — Static Figure Export
=================================

Generates publication-quality PNG figures and self-contained interactive
HTML files from the Sprint 5 dashboard without a running Dash server.

Outputs:
    fig18_dashboard_network_intelligence.png  — Network map + FAD + adversarial
    fig18_dashboard_network_intelligence.html — Interactive Plotly HTML
    fig19_stackelberg_coverage_heatmap.png    — SSE coverage + baseline comparison
    fig19_stackelberg_coverage_heatmap.html
    fig20_scenario_comparison.png             — Risk scenario + robustness curves
    fig20_scenario_comparison.html

Author: Babak Pirzadi (STRATEGOS Thesis — Sprint 5)
"""

import sys
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import plotly.io as pio

from src.dashboard.data_layer import build_dashboard_data
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
# Output directories
# ---------------------------------------------------------------------------

DOCS_FIGS = os.path.join(ROOT, "docs", "figures")
ETC_FIGS  = "/sessions/magical-peaceful-curie/mnt/etc/figures"
os.makedirs(DOCS_FIGS, exist_ok=True)
os.makedirs(ETC_FIGS,  exist_ok=True)

print("=" * 70)
print("SPRINT 5: Static Figure Export")
print("=" * 70)

# ---------------------------------------------------------------------------
# Build dashboard data (5 000 MC sims, 80 NDE epochs — full quality)
# ---------------------------------------------------------------------------

print("\n  Building dashboard data (full quality: 5k MC sims, 80 NDE epochs)...")
DATA = build_dashboard_data(
    budget_fraction=0.30,
    n_sim_pf=5_000,
    n_epochs=80,
    seed=42,
    verbose=True,
)
print(f"\n  Network: {DATA.n_nodes} nodes | {DATA.n_segments} segments")
print(f"  SSE coverage effectiveness: {DATA.ssg_solution.coverage_effectiveness*100:+.1f}%")
print(f"  Scenario risk reduction:    {DATA.scenario_risk_reduction*100:.1f}%")

# Choose the highest-P_f segment for the click-to-inspect demo
demo_segment = max(DATA.segment_ids, key=lambda s: DATA.edge_pf.get(s, 0))
print(f"\n  Demo segment (highest P_f): {demo_segment}  "
      f"P_f={DATA.edge_pf[demo_segment]:.4f}")


# ===========================================================================
# FIGURE 18 — Network Intelligence Dashboard
# ===========================================================================
print("\n" + "=" * 70)
print("Generating Figure 18: Network Intelligence Dashboard")
print("=" * 70)

# ── Plotly interactive HTML version ─────────────────────────────────────────
fig_net   = make_network_figure(DATA, colour_mode="pf",
                                selected_segment=demo_segment)
fig_fad   = make_segment_fad_figure(DATA, demo_segment)
fig_adv   = make_adversarial_impact_figure(DATA, demo_segment)

# Compose into a 3-row HTML figure
fig18_html = make_subplots(
    rows=3, cols=1,
    row_heights=[0.45, 0.30, 0.25],
    subplot_titles=[
        f"Pipeline Network — Failure Probability P_f  "
        f"(selected: {demo_segment})",
        f"BS 7910 Level 2 FAD — {demo_segment}",
        f"Adversarial NDE Threat — {demo_segment}",
    ],
    vertical_spacing=0.08,
)

for trace in fig_net.data:
    fig18_html.add_trace(trace, row=1, col=1)
for trace in fig_fad.data:
    fig18_html.add_trace(trace, row=2, col=1)
for trace in fig_adv.data:
    # Skip Indicator traces (not supported in subplots) — add separately
    try:
        fig18_html.add_trace(trace, row=3, col=1)
    except Exception:
        pass

fig18_html.update_layout(
    title=dict(
        text=(f"Figure 18 — Network Intelligence Dashboard<br>"
              f"<sup>Sprint 5: Click-to-Inspect · BS 7910 FAD · "
              f"Adversarial NDE Threat  |  "
              f"Selected: {demo_segment}  P_f={DATA.edge_pf[demo_segment]:.4f}</sup>"),
        font=dict(size=14, color="#1a3a5c"),
    ),
    height=1000,
    showlegend=False,
    plot_bgcolor="white",
    paper_bgcolor="white",
)

# Copy axis labels from sub-figures
fig18_html.update_xaxes(title_text="Longitude", row=1, col=1,
                         showgrid=True, gridcolor="#f5f5f5")
fig18_html.update_yaxes(title_text="Latitude",  row=1, col=1,
                         showgrid=True, gridcolor="#f5f5f5")
fig18_html.update_xaxes(title_text="L_r (Load Ratio)",     row=2, col=1,
                         showgrid=True, gridcolor="#f5f5f5")
fig18_html.update_yaxes(title_text="K_r (Toughness Ratio)",row=2, col=1,
                         showgrid=True, gridcolor="#f5f5f5")
fig18_html.update_xaxes(title_text="Feature Group",         row=3, col=1)
fig18_html.update_yaxes(title_text="Mean |Δ feature|",      row=3, col=1,
                         showgrid=True, gridcolor="#f5f5f5")

for path in [os.path.join(DOCS_FIGS, "fig18_dashboard_network_intelligence.html"),
             os.path.join(ETC_FIGS,  "fig18_dashboard_network_intelligence.html")]:
    pio.write_html(fig18_html, file=path, full_html=True,
                   include_plotlyjs="cdn", auto_open=False)
print(f"  Saved fig18 HTML")

# ── Matplotlib static PNG version ───────────────────────────────────────────
fig18_png, axes = plt.subplots(2, 2, figsize=(18, 14))
fig18_png.suptitle(
    f"Figure 18 — Network Intelligence Dashboard\n"
    f"Sprint 5: Network Map · FAD Assessment · Adversarial Threat\n"
    f"Selected segment: {demo_segment}  (P_f = {DATA.edge_pf[demo_segment]:.4f})",
    fontsize=13, fontweight="bold", y=1.01,
)

G    = DATA.network.graph
pos  = DATA.node_positions

# ── Panel (a): P_f network map ───────────────────────────────────────────────
import matplotlib.cm as cm

ax = axes[0, 0]
ax.set_title("(a) Failure Probability Map P_f", fontsize=11, fontweight="bold")
pf_vals = np.array([DATA.edge_pf.get(G[u][v].get("segment_id", ""), 0)
                    for u, v in DATA.edge_list])
vmin, vmax = pf_vals.min(), pf_vals.max()
cmap = cm.get_cmap("Reds")
norm = plt.Normalize(vmin=vmin, vmax=vmax)
for (u, v), pf in zip(DATA.edge_list, pf_vals):
    x0, y0 = pos[u]; x1, y1 = pos[v]
    col = cmap(norm(pf)); lw = 1.5 + 4.0 * (pf - vmin) / (vmax - vmin + 1e-9)
    sid = G[u][v].get("segment_id", "")
    dash = (0, (3, 2)) if sid == demo_segment else "solid"
    ax.plot([x0, x1], [y0, y1], color=col, linewidth=lw, linestyle=dash,
            alpha=0.85, solid_capstyle="round", zorder=2)
NODE_C = {"source": "#1a5276", "compressor": "#784212",
          "junction": "#7f8c8d", "delivery": "#1e8449",
          "storage": "#7d3c98", "valve": "#b7950b"}
for node, ntype in DATA.node_types.items():
    c = NODE_C.get(ntype, "#7f8c8d"); sz = 90 if ntype in ("source", "compressor") else 45
    ax.scatter(*pos[node], s=sz, c=c, zorder=4, edgecolors="white", linewidths=0.8)
sm = cm.ScalarMappable(cmap=cmap, norm=norm); sm.set_array([])
plt.colorbar(sm, ax=ax, shrink=0.6, pad=0.02, label="P_f")
ax.set_xlabel("Longitude", fontsize=9); ax.set_ylabel("Latitude", fontsize=9)
ax.grid(True, alpha=0.2, linestyle="--"); ax.tick_params(labelsize=8)

# ── Panel (b): SSE coverage map ──────────────────────────────────────────────
ax = axes[0, 1]
ax.set_title("(b) SSE Coverage Probability c_i", fontsize=11, fontweight="bold")
cov_vals = np.array([DATA.coverage_by_id.get(G[u][v].get("segment_id",""), 0)
                     for u, v in DATA.edge_list])
cmap_b = cm.get_cmap("Blues")
normb = plt.Normalize(vmin=cov_vals.min(), vmax=cov_vals.max())
for (u, v), ci in zip(DATA.edge_list, cov_vals):
    x0, y0 = pos[u]; x1, y1 = pos[v]
    col = cmap_b(normb(ci)); lw = 1.0 + 4.0 * normb(ci)
    ax.plot([x0, x1], [y0, y1], color=col, linewidth=lw, alpha=0.85,
            solid_capstyle="round", zorder=2)
    if ci > 0.20:
        mx, my = (x0+x1)/2, (y0+y1)/2
        ax.annotate(f"{ci:.2f}", (mx, my), fontsize=6.5, ha="center", va="center",
                    bbox=dict(boxstyle="round,pad=0.1", fc="white",
                              ec="steelblue", alpha=0.8))
for node, ntype in DATA.node_types.items():
    c = NODE_C.get(ntype, "#7f8c8d"); sz = 90 if ntype in ("source", "compressor") else 45
    ax.scatter(*pos[node], s=sz, c=c, zorder=4, edgecolors="white", linewidths=0.8)
smb = cm.ScalarMappable(cmap=cmap_b, norm=normb); smb.set_array([])
plt.colorbar(smb, ax=ax, shrink=0.6, pad=0.02, label="c_i")
ax.set_xlabel("Longitude", fontsize=9); ax.set_ylabel("Latitude", fontsize=9)
ax.grid(True, alpha=0.2, linestyle="--"); ax.tick_params(labelsize=8)

# ── Panel (c): BS 7910 FAD for demo segment ───────────────────────────────────
ax = axes[1, 0]
fad_res = DATA.fad_results[demo_segment]
ax.set_title(f"(c) BS 7910 FAD — {demo_segment}\nSeam: {fad_res.seam_type}  P_f={fad_res.P_f:.4f}",
             fontsize=10, fontweight="bold")
ax.plot(fad_res.Lr_curve, fad_res.Kr_curve, color="#1a3a5c", linewidth=2.5,
        label="BS 7910 Level 2 FAD")
ax.fill_between(fad_res.Lr_curve, fad_res.Kr_curve, 0, alpha=0.05, color="#1a3a5c")
Lr_max = float(fad_res.Lr_curve[-1])
ax.axvline(Lr_max, color="#1a3a5c", linestyle="--", linewidth=1.2,
           label=f"Lr_max = {Lr_max:.2f}")
nom_col  = "#1e8449" if fad_res.verdict_nominal  == "ACCEPTABLE" else "#e67e22"
crit_col = "#c0392b" if fad_res.verdict_critical != "ACCEPTABLE" else "#1e8449"
ax.scatter(fad_res.Lr_nominal, fad_res.Kr_nominal, s=100, c=nom_col,
           marker="o", zorder=5, label=f"Nominal ({fad_res.verdict_nominal})")
ax.scatter(fad_res.Lr_critical, fad_res.Kr_critical, s=120, c=crit_col,
           marker="D", zorder=5, label=f"Critical ({fad_res.verdict_critical})")
ax.annotate("SAFE\nREGION", (0.15, 0.28), fontsize=10,
            color=(30/255, 142/255, 73/255, 0.5), ha="center")
ax.set_xlim(0, Lr_max * 1.05); ax.set_ylim(0, 1.3)
ax.set_xlabel("Lr (Load Ratio)", fontsize=9); ax.set_ylabel("Kr (Toughness Ratio)", fontsize=9)
ax.legend(fontsize=8, loc="upper right"); ax.grid(True, alpha=0.25)
ax.tick_params(labelsize=8)

# ── Panel (d): Adversarial impact ────────────────────────────────────────────
ax = axes[1, 1]
adv_res = DATA.adv_results[demo_segment]
groups  = ["Amplitude\n(0–7)", "Frequency\n(8–15)",
           "Geometry\n(16–23)", "Texture\n(24–31)"]
gcolors = ["#2980b9", "#27ae60", "#8e44ad", "#e67e22"]
bars = ax.bar(groups, adv_res.group_perturbation, color=gcolors, alpha=0.85,
              edgecolor="white", linewidth=0.5)
ax.set_title(f"(d) Adversarial NDE Threat — {demo_segment}\n"
             f"FGSM ASR = {adv_res.fgsm_asr*100:.1f}%  "
             f"(ε scaled by SCF)",
             fontsize=10, fontweight="bold")
ax.set_ylabel("Mean |Δ feature|", fontsize=9)
ax.grid(True, axis="y", alpha=0.3); ax.tick_params(labelsize=8)
for bar, pert in zip(bars, adv_res.group_perturbation):
    ax.annotate(f"{pert:.3f}", xy=(bar.get_x() + bar.get_width()/2, pert),
                ha="center", va="bottom", fontsize=9, fontweight="600")

plt.tight_layout()
for path in [os.path.join(DOCS_FIGS, "fig18_dashboard_network_intelligence.png"),
             os.path.join(ETC_FIGS,  "fig18_dashboard_network_intelligence.png")]:
    plt.savefig(path, dpi=150, bbox_inches="tight")
plt.close()
print(f"  Saved fig18 PNG")


# ===========================================================================
# FIGURE 19 — Stackelberg Coverage Heatmap
# ===========================================================================
print("\nGenerating Figure 19: Stackelberg Coverage Heatmap")

fig19_html = make_subplots(
    rows=2, cols=2,
    subplot_titles=[
        "SSE Coverage Allocation c_i",
        "Uniform Baseline Coverage",
        "SSE − Baseline (Gain)",
        "Budget Sensitivity Analysis",
    ],
    vertical_spacing=0.14,
    horizontal_spacing=0.12,
)

for trace in make_coverage_heatmap_figure(DATA, mode="ssg").data:
    fig19_html.add_trace(trace, row=1, col=1)
for trace in make_coverage_heatmap_figure(DATA, mode="baseline").data:
    fig19_html.add_trace(trace, row=1, col=2)
for trace in make_coverage_heatmap_figure(DATA, mode="diff").data:
    fig19_html.add_trace(trace, row=2, col=1)
for trace in make_budget_slider_figure(DATA).data:
    fig19_html.add_trace(trace, row=2, col=2)

fig19_html.update_layout(
    title=dict(
        text=(f"Figure 19 — Stackelberg Defender Coverage & Budget Sensitivity<br>"
              f"<sup>Sprint 5: Bayesian SSE vs. Uniform Baseline | "
              f"Budget = {DATA.budget_fraction:.0%} | "
              f"δ = {DATA.game_config.protection_factor} | "
              f"Risk reduction = {DATA.scenario_risk_reduction*100:.1f}%</sup>"),
        font=dict(size=14, color="#1a3a5c"),
    ),
    height=820,
    showlegend=False,
    plot_bgcolor="white",
    paper_bgcolor="white",
)

for path in [os.path.join(DOCS_FIGS, "fig19_stackelberg_coverage_heatmap.html"),
             os.path.join(ETC_FIGS,  "fig19_stackelberg_coverage_heatmap.html")]:
    pio.write_html(fig19_html, file=path, full_html=True,
                   include_plotlyjs="cdn", auto_open=False)
print(f"  Saved fig19 HTML")

# ── PNG version ───────────────────────────────────────────────────────────────
fig19_png, axes = plt.subplots(2, 2, figsize=(18, 11))
fig19_png.suptitle(
    f"Figure 19 — Stackelberg Defender Coverage & Budget Sensitivity\n"
    f"Bayesian SSE vs. Uniform Baseline | Budget = {DATA.budget_fraction:.0%} | "
    f"Risk Reduction = {DATA.scenario_risk_reduction*100:.1f}%",
    fontsize=12, fontweight="bold", y=1.01,
)

NAVY, RED, GREEN, AMBER, PURPLE = "#1a3a5c", "#c0392b", "#1e8449", "#d4a017", "#7d3c98"

sids_sorted = sorted(DATA.segment_ids,
                     key=lambda s: -DATA.coverage_by_id.get(s, 0))
labels_s = [s.replace("SEG_", "").replace("_", "–") for s in sids_sorted]
cov_ssg  = [DATA.coverage_by_id.get(s, 0)  for s in sids_sorted]
cov_base = [DATA.baseline_coverage.get(s, 0) for s in sids_sorted]
cov_diff = [c - b for c, b in zip(cov_ssg, cov_base)]

# (a) SSE allocation
ax = axes[0, 0]
pf_s  = np.array([DATA.edge_pf.get(s, 0) for s in sids_sorted])
pf_n  = (pf_s - pf_s.min()) / (pf_s.max() - pf_s.min() + 1e-9)
bar_c = [plt.cm.Blues(0.4 + 0.5 * t) for t in pf_n]
ax.bar(range(len(sids_sorted)), cov_ssg, color=bar_c, edgecolor="white", linewidth=0.3)
ax.axhline(DATA.game_config.budget / DATA.n_segments, color="gray",
           linestyle=":", linewidth=1.5, label="Uniform budget")
ax.set_title("(a) SSE Coverage Allocation c_i\n(colour ∝ P_f)",
             fontsize=10, fontweight="bold")
ax.set_xticks(range(len(labels_s))); ax.set_xticklabels(labels_s, rotation=60,
                                                          ha="right", fontsize=5.5)
ax.set_ylabel("Coverage c_i", fontsize=9); ax.legend(fontsize=8)
ax.grid(True, axis="y", alpha=0.3); ax.tick_params(labelsize=7)

# (b) Baseline
ax = axes[0, 1]
ax.bar(range(len(sids_sorted)), cov_base, color=AMBER, alpha=0.85, edgecolor="white", linewidth=0.3)
ax.set_title("(b) Uniform Baseline Coverage",
             fontsize=10, fontweight="bold")
ax.set_xticks(range(len(labels_s))); ax.set_xticklabels(labels_s, rotation=60,
                                                          ha="right", fontsize=5.5)
ax.set_ylabel("Coverage c_i", fontsize=9)
ax.grid(True, axis="y", alpha=0.3); ax.tick_params(labelsize=7)

# (c) Diff
ax = axes[1, 0]
col_diff = [GREEN if d >= 0 else RED for d in cov_diff]
ax.bar(range(len(sids_sorted)), cov_diff, color=col_diff, alpha=0.85, edgecolor="white", linewidth=0.3)
ax.axhline(0, color="black", linewidth=0.8)
ax.set_title("(c) SSE − Baseline (coverage gain from optimisation)",
             fontsize=10, fontweight="bold")
ax.set_xticks(range(len(labels_s))); ax.set_xticklabels(labels_s, rotation=60,
                                                          ha="right", fontsize=5.5)
ax.set_ylabel("Δ c_i", fontsize=9)
ax.grid(True, axis="y", alpha=0.3); ax.tick_params(labelsize=7)

# (d) Budget sensitivity
ax = axes[1, 1]
bfracs  = [r["budget_fraction"] for r in DATA.budget_results]
d_utils = [r["defender_utility"] for r in DATA.budget_results]
a_utils = [r["attacker_utility"] for r in DATA.budget_results]
ax.plot(bfracs, d_utils, "o-", color=NAVY, linewidth=2, markersize=4,
        label="Defender U_d")
ax.plot(bfracs, a_utils, "s--", color=RED, linewidth=2, markersize=4,
        label="Attacker U_a")
ax.axvline(DATA.budget_fraction, color="gray", linestyle=":", linewidth=1.5,
           label=f"Nominal ({DATA.budget_fraction:.0%})")
ax.set_title("(d) Budget Sensitivity — Utility vs. B/N",
             fontsize=10, fontweight="bold")
ax.set_xlabel("Budget fraction B/N", fontsize=9)
ax.set_ylabel("Expected utility", fontsize=9)
ax.legend(fontsize=8); ax.grid(True, alpha=0.25); ax.tick_params(labelsize=8)

plt.tight_layout()
for path in [os.path.join(DOCS_FIGS, "fig19_stackelberg_coverage_heatmap.png"),
             os.path.join(ETC_FIGS,  "fig19_stackelberg_coverage_heatmap.png")]:
    plt.savefig(path, dpi=150, bbox_inches="tight")
plt.close()
print(f"  Saved fig19 PNG")


# ===========================================================================
# FIGURE 20 — Scenario Comparison + Robustness
# ===========================================================================
print("\nGenerating Figure 20: Scenario Comparison & Adversarial Robustness")

# ── HTML ─────────────────────────────────────────────────────────────────────
fig20_html = make_subplots(
    rows=2, cols=1,
    row_heights=[0.55, 0.45],
    subplot_titles=[
        "Per-Segment Residual Risk — Physics-Informed SSE vs. Uniform Baseline",
        "WeldDefectMLP Adversarial Robustness — Accuracy vs. ε (L∞)",
    ],
    vertical_spacing=0.12,
)

for trace in make_scenario_comparison_figure(DATA, show_mode="both").data:
    fig20_html.add_trace(trace, row=1, col=1)

eps_data = DATA.epsilon_sweep_data
epsilons  = eps_data["epsilons"]
for name, arr, col, dash in [
    ("FGSM",                eps_data["acc_fgsm"], "#d4a017", "solid"),
    ("BIM",                 eps_data["acc_bim"],  "#1a3a5c", "dash"),
    ("PGD (random start)",  eps_data["acc_pgd"],  "#c0392b", "solid"),
]:
    fig20_html.add_trace(go.Scatter(
        x=epsilons, y=arr * 100,
        mode="lines+markers", name=name,
        line=dict(color=col, width=2, dash=dash),
        marker=dict(size=5),
    ), row=2, col=1)

fig20_html.add_hline(y=DATA.global_clean_acc * 100,
                      line_dash="dot", line_color="green", line_width=1.8,
                      annotation_text=f"Clean {DATA.global_clean_acc*100:.1f}%",
                      annotation_position="top right",
                      annotation_font_size=10, row=2, col=1)
fig20_html.add_vline(x=0.30, line_dash="dash", line_color="lightgray",
                      line_width=1.2,
                      annotation_text="ε=0.30",
                      annotation_font_size=9, row=2, col=1)

fig20_html.update_xaxes(title_text="Segment", row=1, col=1, tickangle=-45)
fig20_html.update_yaxes(title_text="Expected Loss P_f·v·(1-c_i·(1-δ))",
                         row=1, col=1)
fig20_html.update_xaxes(title_text="Perturbation budget ε (L∞ norm)",
                         row=2, col=1, range=[0, 1])
fig20_html.update_yaxes(title_text="Test accuracy (%)", row=2, col=1)

fig20_html.update_layout(
    title=dict(
        text=(f"Figure 20 — Scenario Comparison & Adversarial Robustness<br>"
              f"<sup>Sprint 5: Risk reduction {DATA.scenario_risk_reduction*100:.1f}% | "
              f"FGSM ASR {DATA.global_fgsm_asr*100:.1f}% | "
              f"BIM ASR {DATA.global_bim_asr*100:.1f}% | "
              f"PGD ASR {DATA.global_pgd_asr*100:.1f}% (ε=0.30)</sup>"),
        font=dict(size=13, color="#1a3a5c"),
    ),
    height=860,
    plot_bgcolor="white",
    paper_bgcolor="white",
    legend=dict(x=0.75, y=0.25, bgcolor="rgba(255,255,255,0.9)",
                bordercolor="#ccc", borderwidth=1),
)

for path in [os.path.join(DOCS_FIGS, "fig20_scenario_comparison.html"),
             os.path.join(ETC_FIGS,  "fig20_scenario_comparison.html")]:
    pio.write_html(fig20_html, file=path, full_html=True,
                   include_plotlyjs="cdn", auto_open=False)
print(f"  Saved fig20 HTML")

# ── PNG ───────────────────────────────────────────────────────────────────────
fig20_png, axes = plt.subplots(2, 1, figsize=(16, 12))
fig20_png.suptitle(
    f"Figure 20 — Scenario Comparison & Adversarial Robustness\n"
    f"Physics-Informed SSE vs. Uniform Baseline | "
    f"Risk Reduction = {DATA.scenario_risk_reduction*100:.1f}%",
    fontsize=12, fontweight="bold", y=1.01,
)

# ── Panel (a): scenario comparison ──────────────────────────────────────────
ax = axes[0]
delta = 1.0 - DATA.game_config.protection_factor
sids = DATA.segment_ids
risk_base = np.array([DATA.edge_pf.get(s, 0) * DATA.edge_value.get(s, 1)
                      * (1.0 - DATA.baseline_coverage.get(s, 0) * delta)
                      for s in sids])
risk_ssg  = np.array([DATA.edge_pf.get(s, 0) * DATA.edge_value.get(s, 1)
                      * (1.0 - DATA.coverage_by_id.get(s, 0) * delta)
                      for s in sids])
risk_none = np.array([DATA.edge_pf.get(s, 0) * DATA.edge_value.get(s, 1)
                      for s in sids])
x = np.arange(len(sids))
w = 0.38
labels_all = [s.replace("SEG_", "").replace("_", "–") for s in sids]
ax.bar(x - w/2, risk_base, width=w, label="Uniform Baseline",
       color=AMBER, alpha=0.85, edgecolor="white", linewidth=0.3)
ax.bar(x + w/2, risk_ssg,  width=w, label="Physics-Informed SSE",
       color=NAVY, alpha=0.85, edgecolor="white", linewidth=0.3)
ax.plot(x, risk_none, "k:", linewidth=1.5, label="Unprotected P_f·v", alpha=0.7)
ax.set_title("(a) Per-Segment Residual Risk — Baseline vs. Physics-Informed SSE",
             fontsize=11, fontweight="bold")
ax.set_xticks(x); ax.set_xticklabels(labels_all, rotation=60,
                                      ha="right", fontsize=6.5)
ax.set_ylabel("Expected Loss  P_f · v · (1 − c_i·(1−δ))", fontsize=9)
ax.legend(fontsize=9); ax.grid(True, axis="y", alpha=0.3); ax.tick_params(labelsize=7)
rr = DATA.scenario_risk_reduction * 100
ax.text(0.99, 0.97, f"SSE risk reduction vs. baseline:\n{rr:.1f}%",
        transform=ax.transAxes, ha="right", va="top",
        fontsize=10, fontweight="bold", color=NAVY,
        bbox=dict(boxstyle="round", fc="white", ec=NAVY, alpha=0.85))

# ── Panel (b): robustness curves ─────────────────────────────────────────────
ax = axes[1]
eps = eps_data["epsilons"]
ax.plot(eps, eps_data["acc_fgsm"] * 100, "o-", color=AMBER, linewidth=2,
        markersize=5, label="FGSM (1-step)")
ax.plot(eps, eps_data["acc_bim"]  * 100, "s--", color=NAVY, linewidth=2,
        markersize=5, label="BIM (20-step)")
ax.plot(eps, eps_data["acc_pgd"]  * 100, "D-", color=RED, linewidth=2.5,
        markersize=5, label="PGD (20-step, random start)")
ax.axhline(DATA.global_clean_acc * 100, color=GREEN, linestyle=":",
           linewidth=1.8, label=f"Clean acc. {DATA.global_clean_acc*100:.1f}%")
ax.axhline(25.0, color="gray", linestyle=":", linewidth=1,
           label="Random baseline (25%)")
ax.axvline(0.30, color="lightgray", linestyle="--", linewidth=1.2,
           label="ε = 0.30")
ax.set_title("(b) WeldDefectMLP Adversarial Robustness — Accuracy vs. ε",
             fontsize=11, fontweight="bold")
ax.set_xlabel("Perturbation budget ε (L∞ norm)", fontsize=9)
ax.set_ylabel("Test accuracy (%)", fontsize=9)
ax.set_xlim(0, 1.0); ax.set_ylim(0, 110)
ax.legend(fontsize=9, loc="upper right"); ax.grid(True, alpha=0.3)
ax.tick_params(labelsize=8)

plt.tight_layout()
for path in [os.path.join(DOCS_FIGS, "fig20_scenario_comparison.png"),
             os.path.join(ETC_FIGS,  "fig20_scenario_comparison.png")]:
    plt.savefig(path, dpi=150, bbox_inches="tight")
plt.close()
print(f"  Saved fig20 PNG")


# ===========================================================================
# Summary
# ===========================================================================
print("\n" + "=" * 70)
print("SPRINT 5 STATIC EXPORT — COMPLETE")
print("=" * 70)
print(f"""
  Network:  {DATA.n_nodes} nodes | {DATA.n_segments} segments
  SSE coverage effectiveness:  {DATA.ssg_solution.coverage_effectiveness*100:+.1f}%
  Scenario risk reduction:     {DATA.scenario_risk_reduction*100:.1f}%
  NDE clean accuracy:          {DATA.global_clean_acc*100:.1f}%
  FGSM ASR (ε=0.30):          {DATA.global_fgsm_asr*100:.1f}%
  BIM  ASR (ε=0.30):          {DATA.global_bim_asr*100:.1f}%
  PGD  ASR (ε=0.30):          {DATA.global_pgd_asr*100:.1f}%

  Figures:
    fig18 → Network Intelligence Dashboard (map + FAD + adversarial)
    fig19 → Stackelberg Coverage Heatmap (SSE / baseline / diff / sensitivity)
    fig20 → Scenario Comparison + Adversarial Robustness Curves

  Saved to:
    {DOCS_FIGS}
    {ETC_FIGS}

  Dashboard app:
    cd strategos-pipeline-defense
    python notebooks/sprint5_dashboard.py
    → http://localhost:8050
""")
