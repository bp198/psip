"""
Dashboard Callback Functions — Sprint 5
=========================================

All figure-generation functions in this module are PURE FUNCTIONS:
they take DashboardData + user-selected parameters and return Plotly
figure dictionaries.  They have no Dash dependency, so they can be
unit-tested directly without an application context.

The Dash app in sprint5_dashboard.py registers these as callbacks.

Author: Babak Pirzadi (STRATEGOS Thesis — Sprint 5)
"""

from __future__ import annotations

from typing import Dict, List, Optional, Any

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import sys as _sys, os as _os
_src_root = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
_project_root = _os.path.dirname(_src_root)
for _p in [_src_root, _project_root]:
    if _p not in _sys.path:
        _sys.path.insert(0, _p)

from src.dashboard.data_layer import DashboardData, SEAM_FAD_PROFILES, _resolve_seam_key

# ---------------------------------------------------------------------------
# Colour palette (consistent with thesis figures)
# ---------------------------------------------------------------------------

NAVY   = "#1a3a5c"
RED    = "#c0392b"
GREEN  = "#1e8449"
AMBER  = "#d4a017"
PURPLE = "#7d3c98"
TEAL   = "#148f77"
ORANGE = "#ca6f1e"

NODE_COLOURS = {
    "source":     "#1a5276",
    "compressor": "#784212",
    "junction":   "#7f8c8d",
    "delivery":   "#1e8449",
    "storage":    "#7d3c98",
    "valve":      "#b7950b",
}

CLASS_NAMES = ["CLEAN", "POROSITY", "CRACK", "LOF"]


# ---------------------------------------------------------------------------
# Helper: edge colour + width from scalar value
# ---------------------------------------------------------------------------

def _edge_colour(val: float, vmin: float, vmax: float,
                 colorscale: str = "Reds") -> str:
    """Map a scalar in [vmin, vmax] to a hex colour using a Plotly colorscale."""
    import plotly.colors as pc
    if vmax <= vmin:
        t = 0.5
    else:
        t = max(0.0, min(1.0, (val - vmin) / (vmax - vmin)))
    # Sample from colorscale
    colours = pc.sequential.__dict__.get(colorscale, pc.sequential.Reds)
    idx = int(t * (len(colours) - 1))
    return colours[min(idx, len(colours) - 1)]


def _linear_interp_colour(t: float, low: str, high: str) -> str:
    """Linear RGB interpolation between two hex colours."""
    def hex2rgb(h: str):
        h = h.lstrip("#")
        return tuple(int(h[i:i+2], 16) for i in (0, 2, 4))
    r1, g1, b1 = hex2rgb(low)
    r2, g2, b2 = hex2rgb(high)
    r = int(r1 + t * (r2 - r1))
    g = int(g1 + t * (g2 - g1))
    b = int(b1 + t * (b2 - b1))
    return f"#{r:02x}{g:02x}{b:02x}"


# ---------------------------------------------------------------------------
# 1. Network Map Figure
# ---------------------------------------------------------------------------

def make_network_figure(
    data: DashboardData,
    colour_mode: str = "pf",          # "pf" | "coverage" | "risk"
    selected_segment: Optional[str] = None,
    show_attacker_strategy: bool = False,
) -> go.Figure:
    """Build the interactive pipeline network map.

    Args:
        data:                 Pre-computed dashboard data.
        colour_mode:          Edge colouring — failure probability, SSE coverage,
                              or composite risk (P_f × (1 - c_i)).
        selected_segment:     Highlight this segment_id if provided.
        show_attacker_strategy: Overlay attacker probability bubbles on nodes.

    Returns:
        Plotly Figure with one trace per edge + one scatter trace for nodes.
    """
    G = data.network.graph
    fig = go.Figure()

    # ── Determine edge scalar values ─────────────────────────────────────────
    if colour_mode == "coverage":
        edge_vals = {sid: data.coverage_by_id.get(sid, 0.0)
                     for sid in data.segment_ids}
        scale_label = "SSE Coverage c_i"
        colorscale  = "Blues"
        low_col, high_col = "#d6eaf8", "#154360"
    elif colour_mode == "risk":
        edge_vals = {}
        for sid in data.segment_ids:
            pf = data.edge_pf.get(sid, 0.0)
            ci = data.coverage_by_id.get(sid, 0.0)
            edge_vals[sid] = pf * (1.0 - ci * (1.0 - data.game_config.protection_factor))
        scale_label = "Residual Risk P_f·(1-c_i·(1-δ))"
        colorscale  = "YlOrRd"
        low_col, high_col = "#fef9e7", "#922b21"
    else:  # "pf"
        edge_vals = data.edge_pf
        scale_label = "Failure Probability P_f"
        colorscale  = "Reds"
        low_col, high_col = "#fdedec", "#7b241c"

    vals_array = np.array(list(edge_vals.values()))
    vmin, vmax = float(vals_array.min()), float(vals_array.max())
    if vmax <= vmin:
        vmax = vmin + 1e-9

    # ── Draw edges ───────────────────────────────────────────────────────────
    for u, v in data.edge_list:
        sid  = G[u][v].get("segment_id", f"{u}-{v}")
        val  = edge_vals.get(sid, 0.0)
        t    = (val - vmin) / (vmax - vmin)
        col  = _linear_interp_colour(t, low_col, high_col)
        lw   = 1.5 + 3.5 * t
        dash = "dot" if sid == selected_segment else "solid"

        x0, y0 = data.node_positions[u]
        x1, y1 = data.node_positions[v]
        mx, my  = (x0 + x1) / 2, (y0 + y1) / 2

        # Hover text
        seam  = G[u][v].get("seam_type", "unknown")
        diam  = G[u][v].get("diameter_mm", 0)
        length = G[u][v].get("length_km", 0)
        ci    = data.coverage_by_id.get(sid, 0.0)
        hover = (f"<b>{sid}</b><br>"
                 f"P_f = {data.edge_pf.get(sid, 0):.4f}<br>"
                 f"Coverage c_i = {ci:.3f}<br>"
                 f"Seam: {seam}<br>"
                 f"Ø {diam:.0f} mm | {length:.1f} km<br>"
                 f"<i>Click to inspect</i>")

        fig.add_trace(go.Scatter(
            x=[x0, mx, x1], y=[y0, my, y1],
            mode="lines",
            line=dict(color=col, width=lw + (2 if sid == selected_segment else 0),
                      dash=dash),
            hoverinfo="text",
            hovertext=hover,
            customdata=[sid, sid, sid],
            name=sid,
            showlegend=False,
        ))

    # ── Draw nodes ───────────────────────────────────────────────────────────
    node_x, node_y, node_col, node_size, node_text, node_hover = [], [], [], [], [], []
    for node, ntype in data.node_types.items():
        x, y = data.node_positions[node]
        node_x.append(x); node_y.append(y)
        node_col.append(NODE_COLOURS.get(ntype, "#7f8c8d"))
        node_size.append(14 if ntype in ("source", "compressor") else 8)
        label = G.nodes[node].get("name", node).split("_")[0]
        node_text.append(label if ntype in ("source", "delivery", "compressor") else "")
        node_hover.append(f"<b>{label}</b><br>Type: {ntype}")

    fig.add_trace(go.Scatter(
        x=node_x, y=node_y,
        mode="markers+text",
        marker=dict(color=node_col, size=node_size,
                    line=dict(color="white", width=1)),
        text=node_text,
        textposition="top center",
        textfont=dict(size=8),
        hoverinfo="text",
        hovertext=node_hover,
        showlegend=False,
        name="Nodes",
    ))

    # ── Attacker strategy bubbles ────────────────────────────────────────────
    if show_attacker_strategy and data.attacker_strategy:
        for u, v in data.edge_list:
            sid  = G[u][v].get("segment_id", f"{u}-{v}")
            prob = data.attacker_strategy.get(sid, 0.0)
            if prob < 0.01:
                continue
            x0, y0 = data.node_positions[u]
            x1, y1 = data.node_positions[v]
            mx, my = (x0 + x1) / 2, (y0 + y1) / 2
            fig.add_trace(go.Scatter(
                x=[mx], y=[my],
                mode="markers",
                marker=dict(color=RED, size=max(6, prob * 60),
                            opacity=0.55,
                            line=dict(color=RED, width=1)),
                hoverinfo="text",
                hovertext=f"Attack prob: {prob:.3f}",
                showlegend=False,
                name=f"atk_{sid}",
            ))

    # ── Colourbar (fake scatter for legend) ─────────────────────────────────
    cb_vals = np.linspace(vmin, vmax, 100)
    fig.add_trace(go.Scatter(
        x=[None] * 100, y=[None] * 100,
        mode="markers",
        marker=dict(
            color=cb_vals,
            colorscale=colorscale,
            cmin=vmin, cmax=vmax,
            showscale=True,
            colorbar=dict(
                title=dict(text=scale_label, side="right", font=dict(size=10)),
                thickness=14, len=0.6, x=1.01,
                tickfont=dict(size=9),
            ),
        ),
        showlegend=False,
        hoverinfo="none",
    ))

    fig.update_layout(
        title=dict(
            text=(f"Gulf Coast Pipeline Network — "
                  f"{'Failure Probability P_f' if colour_mode=='pf' else 'SSE Coverage' if colour_mode=='coverage' else 'Residual Risk'}"),
            font=dict(size=13, color=NAVY),
        ),
        xaxis=dict(title="Longitude", showgrid=True, gridcolor="#f0f0f0",
                   zeroline=False),
        yaxis=dict(title="Latitude", showgrid=True, gridcolor="#f0f0f0",
                   zeroline=False),
        plot_bgcolor="white",
        paper_bgcolor="white",
        margin=dict(l=40, r=80, t=50, b=40),
        height=480,
        hovermode="closest",
    )
    return fig


# ---------------------------------------------------------------------------
# 2. Segment FAD Figure (click-to-inspect)
# ---------------------------------------------------------------------------

def make_segment_fad_figure(
    data: DashboardData,
    segment_id: str,
) -> go.Figure:
    """Build BS 7910 FAD diagram for a selected segment.

    Draws the FAD curve, the nominal assessment point (green ✓),
    and the critical assessment point (red ✗).

    Args:
        data:        Dashboard data.
        segment_id:  The selected segment_id.

    Returns:
        Plotly Figure.
    """
    if segment_id not in data.fad_results:
        fig = go.Figure()
        fig.add_annotation(text="No segment selected",
                           xref="paper", yref="paper", x=0.5, y=0.5,
                           showarrow=False, font=dict(size=14, color="gray"))
        fig.update_layout(height=320, plot_bgcolor="white")
        return fig

    r = data.fad_results[segment_id]
    profile = SEAM_FAD_PROFILES[r.fad_key]

    fig = go.Figure()

    # ── FAD curve ────────────────────────────────────────────────────────────
    fig.add_trace(go.Scatter(
        x=r.Lr_curve, y=r.Kr_curve,
        mode="lines",
        line=dict(color=NAVY, width=2.5),
        name="BS 7910 Level 2 FAD",
        fill="tonexty",
        fillcolor="rgba(26,58,92,0.06)",
    ))

    # ── Lr_max cut-off vertical ──────────────────────────────────────────────
    Lr_max = float(r.Lr_curve[-1])
    fig.add_shape(type="line",
                  x0=Lr_max, x1=Lr_max, y0=0, y1=1.2,
                  line=dict(color=NAVY, dash="dash", width=1.2))
    fig.add_annotation(x=Lr_max, y=1.15,
                       text=f"L<sub>r,max</sub>={Lr_max:.2f}",
                       showarrow=False, font=dict(size=9, color=NAVY))

    # ── Nominal assessment point ─────────────────────────────────────────────
    nom_col = GREEN if r.verdict_nominal == "ACCEPTABLE" else AMBER
    fig.add_trace(go.Scatter(
        x=[r.Lr_nominal], y=[r.Kr_nominal],
        mode="markers+text",
        marker=dict(color=nom_col, size=12, symbol="circle",
                    line=dict(color="white", width=1.5)),
        text=[f"{'✓' if r.verdict_nominal=='ACCEPTABLE' else '⚠'} Nominal"],
        textposition="top right",
        textfont=dict(size=9),
        name=f"Nominal flaw ({r.verdict_nominal})",
        hovertemplate=(f"Nominal flaw<br>"
                       f"L_r = {r.Lr_nominal:.3f}<br>"
                       f"K_r = {r.Kr_nominal:.3f}<br>"
                       f"Verdict: {r.verdict_nominal}"),
    ))

    # ── Critical assessment point ────────────────────────────────────────────
    crit_col = RED if r.verdict_critical in ("UNACCEPTABLE", "BEYOND_Lr_MAX") else AMBER
    fig.add_trace(go.Scatter(
        x=[r.Lr_critical], y=[r.Kr_critical],
        mode="markers+text",
        marker=dict(color=crit_col, size=13, symbol="diamond",
                    line=dict(color="white", width=1.5)),
        text=[f"{'✗' if crit_col==RED else '⚠'} Critical"],
        textposition="top left",
        textfont=dict(size=9),
        name=f"Critical flaw ({r.verdict_critical})",
        hovertemplate=(f"Critical flaw<br>"
                       f"L_r = {r.Lr_critical:.3f}<br>"
                       f"K_r = {r.Kr_critical:.3f}<br>"
                       f"Verdict: {r.verdict_critical}"),
    ))

    # ── Safe region shading ──────────────────────────────────────────────────
    fig.add_annotation(
        x=0.15, y=0.3,
        text="SAFE<br>REGION",
        showarrow=False,
        font=dict(size=11, color="rgba(26,142,73,0.5)"),
    )

    fig.update_layout(
        title=dict(
            text=(f"BS 7910 FAD — {segment_id}<br>"
                  f"<sup>Seam: {r.seam_type} | SCF={profile.scf:.1f} | "
                  f"P_f={r.P_f:.4f}</sup>"),
            font=dict(size=11, color=NAVY),
        ),
        xaxis=dict(title="L_r (Load Ratio)", range=[0, Lr_max * 1.05],
                   showgrid=True, gridcolor="#f5f5f5"),
        yaxis=dict(title="K_r (Toughness Ratio)", range=[0, 1.3],
                   showgrid=True, gridcolor="#f5f5f5"),
        plot_bgcolor="white", paper_bgcolor="white",
        legend=dict(x=0.01, y=0.99, bgcolor="rgba(255,255,255,0.85)",
                    bordercolor="#cccccc", borderwidth=1, font=dict(size=9)),
        margin=dict(l=50, r=20, t=70, b=40),
        height=340,
    )
    return fig


# ---------------------------------------------------------------------------
# 3. Adversarial Impact Figure (click-to-inspect)
# ---------------------------------------------------------------------------

def make_adversarial_impact_figure(
    data: DashboardData,
    segment_id: str,
) -> go.Figure:
    """Bar chart of feature-group adversarial perturbation for a segment.

    The ε used for this segment is scaled by its seam SCF, giving a
    physics-informed adversarial threat model:
        ε_eff = ε_nominal × (SCF / 1.5)

    Args:
        data:        Dashboard data.
        segment_id:  Selected segment_id.

    Returns:
        Plotly Figure (2 panels: group perturbation + ASR annotation).
    """
    if segment_id not in data.adv_results:
        fig = go.Figure()
        fig.add_annotation(text="No segment selected",
                           xref="paper", yref="paper", x=0.5, y=0.5,
                           showarrow=False, font=dict(size=14, color="gray"))
        fig.update_layout(height=220, plot_bgcolor="white")
        return fig

    res = data.adv_results[segment_id]
    fad_key = _resolve_seam_key(res.seam_type)
    scf = SEAM_FAD_PROFILES[fad_key].scf
    eps_eff = min(0.30 * scf / 1.5, 1.0)

    groups = ["Amplitude\n(0–7)", "Frequency\n(8–15)",
              "Geometry\n(16–23)", "Texture\n(24–31)"]
    group_colours = ["#2980b9", "#27ae60", "#8e44ad", "#e67e22"]
    pert = res.group_perturbation

    fig = make_subplots(
        rows=1, cols=2,
        column_widths=[0.65, 0.35],
        subplot_titles=["Feature Group Perturbation |δ|",
                        "Adversarial Threat Summary"],
        specs=[[{"type": "xy"}, {"type": "indicator"}]],
    )

    # ── Panel a: group perturbation bars ────────────────────────────────────
    fig.add_trace(go.Bar(
        x=groups,
        y=pert.tolist(),
        marker_color=group_colours,
        marker_line_color="white",
        marker_line_width=0.5,
        name="Mean |δ|",
        hovertemplate="%{x}<br>Mean |δ| = %{y:.4f}",
        showlegend=False,
    ), row=1, col=1)

    fig.update_yaxes(title_text="Mean |Δ feature|", row=1, col=1,
                     gridcolor="#f5f5f5")
    fig.update_xaxes(row=1, col=1)

    # ── Panel b: ASR gauge via indicator ────────────────────────────────────
    asr_pct = res.fgsm_asr * 100
    fig.add_trace(go.Indicator(
        mode="gauge+number+delta",
        value=asr_pct,
        number=dict(suffix="%", font=dict(size=22, color=RED)),
        title=dict(text=f"FGSM ASR<br><sup>ε_eff={eps_eff:.2f} | SCF={scf:.1f}</sup>",
                   font=dict(size=11)),
        gauge=dict(
            axis=dict(range=[0, 100], tickwidth=1),
            bar=dict(color=RED, thickness=0.3),
            bgcolor="white",
            bordercolor="#cccccc",
            steps=[
                dict(range=[0, 20],  color="#eafaf1"),
                dict(range=[20, 50], color="#fef9e7"),
                dict(range=[50, 100], color="#fdedec"),
            ],
            threshold=dict(line=dict(color="black", width=2),
                           thickness=0.75, value=data.global_fgsm_asr * 100),
        ),
        delta=dict(reference=data.global_fgsm_asr * 100,
                   relative=False,
                   suffix="% vs. global",
                   font=dict(size=11)),
    ), row=1, col=2)

    fig.update_layout(
        title=dict(
            text=(f"Adversarial NDE Threat — {segment_id}<br>"
                  f"<sup>FGSM attack on WeldDefectMLP | "
                  f"Seam: {res.seam_type}</sup>"),
            font=dict(size=11, color=NAVY),
        ),
        plot_bgcolor="white", paper_bgcolor="white",
        margin=dict(l=40, r=20, t=70, b=40),
        height=260,
    )
    return fig


# ---------------------------------------------------------------------------
# 4. Scenario Comparison Figure
# ---------------------------------------------------------------------------

def make_scenario_comparison_figure(
    data: DashboardData,
    show_mode: str = "both",    # "baseline" | "ssg" | "both"
) -> go.Figure:
    """Grouped bar chart: per-segment residual risk under baseline vs SSE.

    Args:
        data:       Dashboard data.
        show_mode:  Which scenarios to show.

    Returns:
        Plotly Figure.
    """
    sids = data.segment_ids
    delta = 1.0 - data.game_config.protection_factor

    risk_none = [data.edge_pf.get(s, 0) * data.edge_value.get(s, 1)
                 for s in sids]
    risk_base = [data.edge_pf.get(s, 0) * data.edge_value.get(s, 1)
                 * (1.0 - data.baseline_coverage.get(s, 0) * delta)
                 for s in sids]
    risk_ssg  = [data.edge_pf.get(s, 0) * data.edge_value.get(s, 1)
                 * (1.0 - data.coverage_by_id.get(s, 0) * delta)
                 for s in sids]

    # Short labels
    labels = [s.replace("SEG_", "").replace("_", "–") for s in sids]

    fig = go.Figure()

    if show_mode in ("both", "baseline"):
        fig.add_trace(go.Bar(
            x=labels, y=risk_base,
            name="Uniform Baseline",
            marker_color=AMBER,
            marker_line_width=0,
            opacity=0.85,
            hovertemplate="<b>%{x}</b><br>Baseline risk: %{y:.4f}",
        ))

    if show_mode in ("both", "ssg"):
        fig.add_trace(go.Bar(
            x=labels, y=risk_ssg,
            name="Physics-Informed SSE",
            marker_color=NAVY,
            marker_line_width=0,
            opacity=0.85,
            hovertemplate="<b>%{x}</b><br>SSE risk: %{y:.4f}",
        ))

    # ── Unprotected reference line ───────────────────────────────────────────
    max_risk = max(max(risk_none), 1e-9)
    fig.add_trace(go.Scatter(
        x=labels, y=risk_none,
        mode="lines",
        line=dict(color=RED, dash="dot", width=1.5),
        name="Unprotected P_f·v",
        hovertemplate="<b>%{x}</b><br>Unprotected: %{y:.4f}",
    ))

    # ── Annotation: total risk reduction ────────────────────────────────────
    rr = data.scenario_risk_reduction * 100
    fig.add_annotation(
        xref="paper", yref="paper",
        x=0.99, y=0.97,
        text=(f"<b>SSE vs. Baseline<br>"
              f"Risk reduction: {rr:.1f}%</b>"),
        showarrow=False,
        font=dict(size=11, color=NAVY),
        align="right",
        bgcolor="rgba(255,255,255,0.85)",
        bordercolor=NAVY,
        borderwidth=1,
    )

    fig.update_layout(
        title=dict(
            text="Scenario Comparison — Defender Allocation Strategy",
            font=dict(size=13, color=NAVY),
        ),
        barmode="group",
        xaxis=dict(title="Pipeline Segment", tickangle=-45,
                   tickfont=dict(size=8), showgrid=False),
        yaxis=dict(title="Expected Loss  P_f · v · (1 − c_i·(1−δ))",
                   showgrid=True, gridcolor="#f5f5f5"),
        legend=dict(x=0.01, y=0.99, bgcolor="rgba(255,255,255,0.85)",
                    bordercolor="#cccccc", borderwidth=1),
        plot_bgcolor="white", paper_bgcolor="white",
        margin=dict(l=50, r=20, t=50, b=110),
        height=380,
    )
    return fig


# ---------------------------------------------------------------------------
# 5. Budget Slider Figure
# ---------------------------------------------------------------------------

def make_budget_slider_figure(
    data: DashboardData,
    highlight_fraction: Optional[float] = None,
) -> go.Figure:
    """Utility vs budget curve with an interactive vertical marker.

    Args:
        data:               Dashboard data.
        highlight_fraction: Current slider value to mark (0–1).

    Returns:
        Plotly Figure.
    """
    fracs  = [r["budget_fraction"]  for r in data.budget_results]
    d_util = [r["defender_utility"] for r in data.budget_results]
    a_util = [r["attacker_utility"] for r in data.budget_results]
    eff    = [r["coverage_effectiveness"] * 100 for r in data.budget_results]
    marginal = list(np.gradient(d_util, fracs))

    fig = make_subplots(
        rows=1, cols=3,
        subplot_titles=[
            "Utility vs. Budget",
            "Coverage Effectiveness",
            "Marginal Return ∂U_d/∂B",
        ],
        horizontal_spacing=0.1,
    )

    # ── Panel 1: utilities ───────────────────────────────────────────────────
    fig.add_trace(go.Scatter(
        x=fracs, y=d_util,
        mode="lines+markers", name="Defender U_d",
        line=dict(color=NAVY, width=2),
        marker=dict(size=5),
        hovertemplate="B/N=%{x:.2f}<br>U_d=%{y:.4f}",
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=fracs, y=a_util,
        mode="lines+markers", name="Attacker U_a",
        line=dict(color=RED, width=2, dash="dash"),
        marker=dict(size=5),
        hovertemplate="B/N=%{x:.2f}<br>U_a=%{y:.4f}",
    ), row=1, col=1)

    # ── Panel 2: effectiveness bars ──────────────────────────────────────────
    fig.add_trace(go.Bar(
        x=fracs, y=eff,
        name="Effectiveness",
        marker_color=[GREEN if e >= 0 else RED for e in eff],
        marker_line_width=0,
        width=0.04,
        showlegend=False,
        hovertemplate="B/N=%{x:.2f}<br>ΔU_d=%{y:.1f}%",
    ), row=1, col=2)
    fig.add_shape(type="line", x0=0, x1=1, y0=0, y1=0,
                  line=dict(color="black", width=0.8),
                  row=1, col=2)

    # ── Panel 3: marginal return ─────────────────────────────────────────────
    fig.add_trace(go.Scatter(
        x=fracs, y=marginal,
        mode="lines+markers", name="∂U_d/∂B",
        line=dict(color=PURPLE, width=2),
        marker=dict(size=5),
        showlegend=False,
        hovertemplate="B/N=%{x:.2f}<br>∂U_d/∂B=%{y:.4f}",
    ), row=1, col=3)
    fig.add_shape(type="line", x0=0, x1=1, y0=0, y1=0,
                  line=dict(color="black", width=0.8),
                  row=1, col=3)

    # ── Highlight nominal budget ─────────────────────────────────────────────
    nom = highlight_fraction or data.budget_fraction
    for col in range(1, 4):
        fig.add_vline(x=nom, line_dash="dot", line_color="gray",
                      line_width=1.5, row=1, col=col)

    fig.update_xaxes(title_text="Budget Fraction B/N", range=[0, 1])
    fig.update_layout(
        title=dict(text="Stackelberg Game — Budget Sensitivity Analysis",
                   font=dict(size=13, color=NAVY)),
        legend=dict(x=0.01, y=0.99, bgcolor="rgba(255,255,255,0.85)",
                    bordercolor="#cccccc", borderwidth=1, font=dict(size=9)),
        plot_bgcolor="white", paper_bgcolor="white",
        margin=dict(l=50, r=20, t=60, b=50),
        height=320,
    )
    return fig


# ---------------------------------------------------------------------------
# 6. Coverage Heatmap Figure (Stackelberg defender allocation)
# ---------------------------------------------------------------------------

def make_coverage_heatmap_figure(
    data: DashboardData,
    mode: str = "ssg",           # "ssg" | "baseline" | "diff"
) -> go.Figure:
    """Sorted bar chart of defender coverage probabilities c_i per segment.

    Args:
        data:  Dashboard data.
        mode:  "ssg"      = Bayesian SSE allocation
               "baseline" = uniform allocation
               "diff"     = SSE − baseline (gain from optimisation)

    Returns:
        Plotly Figure.
    """
    sids = sorted(data.segment_ids,
                  key=lambda s: -data.coverage_by_id.get(s, 0))

    if mode == "baseline":
        vals   = [data.baseline_coverage.get(s, 0) for s in sids]
        colour = AMBER
        title_extra = "Uniform Baseline"
    elif mode == "diff":
        vals   = [data.coverage_by_id.get(s, 0) - data.baseline_coverage.get(s, 0)
                  for s in sids]
        colour = [GREEN if v >= 0 else RED for v in vals]
        title_extra = "SSE − Baseline (gain from physics-informed allocation)"
    else:
        vals   = [data.coverage_by_id.get(s, 0) for s in sids]
        colour = NAVY
        title_extra = "Bayesian Stackelberg SSE"

    labels = [s.replace("SEG_", "").replace("_", "–") for s in sids]

    # Colour by P_f for SSE mode to show risk-weighted allocation
    if mode == "ssg":
        pf_vals = np.array([data.edge_pf.get(s, 0) for s in sids])
        pf_norm = (pf_vals - pf_vals.min()) / (pf_vals.max() - pf_vals.min() + 1e-9)
        colour  = [_linear_interp_colour(t, "#2980b9", "#154360")
                   for t in pf_norm.tolist()]

    # Attacker strategy overlay
    atk_vals = [data.attacker_strategy.get(s, 0) for s in sids]

    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=labels,
        y=vals,
        marker_color=colour,
        marker_line_width=0,
        name=title_extra,
        hovertemplate=(
            "<b>%{x}</b><br>"
            "Coverage c_i = %{y:.4f}<br>"
        ),
    ))

    if mode == "ssg":
        # Overlay attacker strategy as line
        fig.add_trace(go.Scatter(
            x=labels, y=atk_vals,
            mode="lines+markers",
            name="Attacker probability q_i",
            line=dict(color=RED, dash="dot", width=1.8),
            marker=dict(size=5, color=RED),
            yaxis="y2",
            hovertemplate="<b>%{x}</b><br>Atk. prob = %{y:.3f}",
        ))
        fig.update_layout(
            yaxis2=dict(title="Attacker probability",
                        overlaying="y", side="right",
                        showgrid=False,
                        range=[0, max(atk_vals) * 1.4 + 0.01]),
        )

    fig.add_hline(y=data.game_config.budget / data.n_segments,
                  line_dash="dot", line_color="gray", line_width=1.2,
                  annotation_text=f"Uniform budget = {data.game_config.budget/data.n_segments:.3f}",
                  annotation_position="top right",
                  annotation_font_size=9)

    fig.update_layout(
        title=dict(
            text=f"Defender Coverage Probabilities c_i — {title_extra}",
            font=dict(size=13, color=NAVY),
        ),
        xaxis=dict(title="Segment", tickangle=-45,
                   tickfont=dict(size=7.5), showgrid=False),
        yaxis=dict(title="Coverage probability c_i",
                   range=[min(0, min(vals)) - 0.01, max(vals) * 1.15 + 0.01],
                   showgrid=True, gridcolor="#f5f5f5"),
        legend=dict(x=0.01, y=0.99, bgcolor="rgba(255,255,255,0.85)",
                    bordercolor="#cccccc", borderwidth=1, font=dict(size=9)),
        plot_bgcolor="white", paper_bgcolor="white",
        margin=dict(l=50, r=60, t=50, b=110),
        height=340,
    )
    return fig


# ---------------------------------------------------------------------------
# 7. Segment Intelligence Panel (combined FAD + adversarial summary dict)
# ---------------------------------------------------------------------------

def segment_intel_panel(
    data: DashboardData,
    segment_id: Optional[str],
) -> Dict[str, Any]:
    """Return a summary dict for the selected segment's intelligence panel.

    Used both by the Dash callback (converts to HTML children) and by tests.

    Returns a dict with keys:
        segment_id, seam_type, diameter_mm, length_km, P_f,
        coverage_ssg, coverage_baseline, fad_verdict_nominal,
        fad_verdict_critical, fgsm_asr, scf, risk_rank
    """
    if segment_id is None or segment_id not in data.fad_results:
        return {"segment_id": None}

    G = data.network.graph
    edge_data = None
    for u, v in data.edge_list:
        if G[u][v].get("segment_id") == segment_id:
            edge_data = G[u][v]
            break

    fad  = data.fad_results[segment_id]
    adv  = data.adv_results.get(segment_id)
    fad_key = _resolve_seam_key(fad.seam_type)
    scf = SEAM_FAD_PROFILES[fad_key].scf

    # Risk rank among all segments (1 = highest risk)
    sorted_by_risk = sorted(data.segment_ids,
                            key=lambda s: -data.edge_pf.get(s, 0))
    rank = sorted_by_risk.index(segment_id) + 1 if segment_id in sorted_by_risk else -1

    return {
        "segment_id":          segment_id,
        "seam_type":           fad.seam_type,
        "diameter_mm":         edge_data.get("diameter_mm", 0) if edge_data else 0,
        "length_km":           edge_data.get("length_km", 0) if edge_data else 0,
        "P_f":                 fad.P_f,
        "coverage_ssg":        data.coverage_by_id.get(segment_id, 0),
        "coverage_baseline":   data.baseline_coverage.get(segment_id, 0),
        "fad_verdict_nominal": fad.verdict_nominal,
        "fad_verdict_critical":fad.verdict_critical,
        "fgsm_asr":            adv.fgsm_asr if adv else None,
        "scf":                 scf,
        "risk_rank":           rank,
        "n_segments":          data.n_segments,
    }
