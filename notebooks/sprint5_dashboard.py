"""
Sprint 5 — Interactive Dashboard
=================================

Physics-Informed Game-Theoretic Defence of Pipeline Infrastructure
STRATEGOS MSc Thesis — Babak Pirzadi

Usage:
    cd strategos-pipeline-defense
    python notebooks/sprint5_dashboard.py

Then open  http://localhost:8050  in your browser.

The dashboard integrates all four prior sprints into three tabs:

  Tab 1 — Network Intelligence
      • Interactive pipeline map (P_f / SSE coverage / residual risk)
      • Click-to-inspect: BS 7910 FAD + FGSM adversarial impact per segment

  Tab 2 — Stackelberg Defender
      • Optimal coverage allocation heat-map (SSE vs. uniform baseline)
      • Budget sensitivity analysis with interactive slider

  Tab 3 — Scenario Comparison
      • Per-segment expected loss: uniform baseline vs. physics-informed SSE
      • Adversarial robustness curves (FGSM / BIM / PGD vs. ε)
"""

import sys
import os

# ── Path setup ───────────────────────────────────────────────────────────────
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from dash import Dash, Input, Output, State, html, callback_context
import plotly.graph_objects as go
import numpy as np

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
from src.dashboard.layout import build_layout, _kpi_item

# ---------------------------------------------------------------------------
# Bootstrap: build all data at startup (deterministic, seed=42)
# ---------------------------------------------------------------------------

print("=" * 70)
print("SPRINT 5: Interactive Dashboard — loading all sprint data...")
print("=" * 70)

DATA = build_dashboard_data(
    budget_fraction=0.30,
    n_sim_pf=5_000,
    n_epochs=80,
    seed=42,
    verbose=True,
)

print("\n  Dashboard data ready.")
print(f"  Network: {DATA.n_nodes} nodes | {DATA.n_segments} segments")
print(f"  SSE coverage effectiveness: "
      f"{DATA.ssg_solution.coverage_effectiveness*100:+.1f}%")
print(f"  Scenario risk reduction: {DATA.scenario_risk_reduction*100:.1f}%")
print(f"  Global clean NDE accuracy: {DATA.global_clean_acc*100:.1f}%")
print(f"  FGSM ASR: {DATA.global_fgsm_asr*100:.1f}%  "
      f"BIM: {DATA.global_bim_asr*100:.1f}%  "
      f"PGD: {DATA.global_pgd_asr*100:.1f}%")

# ---------------------------------------------------------------------------
# Dash App
# ---------------------------------------------------------------------------

app = Dash(
    __name__,
    title="STRATEGOS Pipeline Defence Dashboard",
    suppress_callback_exceptions=True,
)

app.layout = build_layout()

# ---------------------------------------------------------------------------
# Callback: KPI strip (fires once on page load)
# ---------------------------------------------------------------------------

@app.callback(
    Output("kpi-strip", "children"),
    Input("main-tabs", "value"),
)
def update_kpi_strip(_tab):
    rr = DATA.scenario_risk_reduction * 100
    eff = DATA.ssg_solution.coverage_effectiveness * 100
    return [
        _kpi_item("Segments",
                  str(DATA.n_segments), "#2c3e50"),
        _kpi_item("Budget",
                  f"{DATA.budget_fraction:.0%}", "#1a5276"),
        _kpi_item("SSE Coverage Eff.",
                  f"{eff:+.1f}", "#1e8449", "%"),
        _kpi_item("Risk Reduction",
                  f"{rr:.1f}", "#c0392b", "%"),
        _kpi_item("NDE Clean Acc.",
                  f"{DATA.global_clean_acc*100:.1f}", "#7d3c98", "%"),
        _kpi_item("FGSM ASR (ε=0.30)",
                  f"{DATA.global_fgsm_asr*100:.1f}", "#e74c3c", "%"),
        _kpi_item("BIM ASR (ε=0.30)",
                  f"{DATA.global_bim_asr*100:.1f}", "#e74c3c", "%"),
        _kpi_item("PGD ASR (ε=0.30)",
                  f"{DATA.global_pgd_asr*100:.1f}", "#c0392b", "%"),
    ]


# ---------------------------------------------------------------------------
# Callback: store clicked segment from network map
# ---------------------------------------------------------------------------

@app.callback(
    Output("selected-segment", "data"),
    Input("network-map", "clickData"),
    State("selected-segment", "data"),
    prevent_initial_call=True,
)
def store_selected_segment(click_data, current):
    """Extract segment_id from clickData and persist in Store."""
    if click_data is None:
        return current
    points = click_data.get("points", [])
    if not points:
        return current
    # customdata was set to segment_id in make_network_figure
    cd = points[0].get("customdata")
    if isinstance(cd, str) and cd in DATA.fad_results:
        return cd
    return current


# ---------------------------------------------------------------------------
# Callback: network map figure
# ---------------------------------------------------------------------------

@app.callback(
    Output("network-map", "figure"),
    Input("colour-mode", "value"),
    Input("network-options", "value"),
    Input("selected-segment", "data"),
)
def update_network_map(colour_mode, options, selected_segment):
    show_atk = "show_attacker" in (options or [])
    return make_network_figure(
        DATA,
        colour_mode=colour_mode,
        selected_segment=selected_segment,
        show_attacker_strategy=show_atk,
    )


# ---------------------------------------------------------------------------
# Callback: segment intelligence panel (text)
# ---------------------------------------------------------------------------

@app.callback(
    Output("segment-intel-panel", "children"),
    Input("selected-segment", "data"),
)
def update_segment_intel(segment_id):
    info = segment_intel_panel(DATA, segment_id)
    if info.get("segment_id") is None:
        return html.Div("👆  Select a segment on the map",
                        style=dict(color="#6c757d", textAlign="center",
                                   paddingTop="40px", fontSize="12px"))

    risk_col = ("#c0392b" if info["P_f"] > 0.05 else
                "#e67e22" if info["P_f"] > 0.02 else "#1e8449")
    fad_col  = ("#c0392b" if info["fad_verdict_nominal"] == "UNACCEPTABLE"
                else "#1e8449")

    def row(label, value, colour="#2c3e50"):
        return html.Div(
            [html.Span(label, style=dict(color="#6c757d", fontSize="11px",
                                          width="140px", display="inline-block")),
             html.Span(str(value), style=dict(fontWeight="600", color=colour,
                                               fontSize="11px"))],
            style=dict(marginBottom="4px"),
        )

    asr_str = (f"{info['fgsm_asr']*100:.1f}%"
               if info["fgsm_asr"] is not None else "n/a")
    asr_col = ("#c0392b" if (info["fgsm_asr"] or 0) > 0.2 else
               "#e67e22" if (info["fgsm_asr"] or 0) > 0.1 else "#1e8449")

    return html.Div(
        [
            html.Div(
                info["segment_id"],
                style=dict(fontWeight="700", color="#1a3a5c",
                           fontSize="13px", marginBottom="8px",
                           borderBottom="1px solid #dee2e6",
                           paddingBottom="5px"),
            ),
            row("Seam type:",       info["seam_type"]),
            row("Diameter:",        f"{info['diameter_mm']:.0f} mm"),
            row("Length:",          f"{info['length_km']:.1f} km"),
            row("Failure Prob P_f:", f"{info['P_f']:.4f}", risk_col),
            row("Risk rank:",       f"{info['risk_rank']} / {info['n_segments']}",
                risk_col),
            html.Hr(style=dict(margin="6px 0", borderColor="#eee")),
            row("SSE coverage c_i:",    f"{info['coverage_ssg']:.4f}", "#1a5276"),
            row("Baseline coverage:",   f"{info['coverage_baseline']:.4f}", "#e67e22"),
            row("SCF (weld):",          f"{info['scf']:.2f}"),
            html.Hr(style=dict(margin="6px 0", borderColor="#eee")),
            row("FAD nominal:",     info["fad_verdict_nominal"],  fad_col),
            row("FAD critical:",    info["fad_verdict_critical"],
                "#c0392b" if info["fad_verdict_critical"] != "ACCEPTABLE"
                else "#1e8449"),
            html.Hr(style=dict(margin="6px 0", borderColor="#eee")),
            row("FGSM ASR (ε scaled):", asr_str, asr_col),
        ],
        style=dict(fontSize="11px"),
    )


# ---------------------------------------------------------------------------
# Callback: FAD figure
# ---------------------------------------------------------------------------

@app.callback(
    Output("fad-figure", "figure"),
    Input("selected-segment", "data"),
)
def update_fad(segment_id):
    return make_segment_fad_figure(DATA, segment_id or "")


# ---------------------------------------------------------------------------
# Callback: adversarial impact figure
# ---------------------------------------------------------------------------

@app.callback(
    Output("adv-figure", "figure"),
    Input("selected-segment", "data"),
)
def update_adv(segment_id):
    return make_adversarial_impact_figure(DATA, segment_id or "")


# ---------------------------------------------------------------------------
# Callback: coverage heatmap
# ---------------------------------------------------------------------------

@app.callback(
    Output("coverage-heatmap", "figure"),
    Input("coverage-mode", "value"),
)
def update_coverage_heatmap(mode):
    return make_coverage_heatmap_figure(DATA, mode=mode)


# ---------------------------------------------------------------------------
# Callback: budget sensitivity
# ---------------------------------------------------------------------------

@app.callback(
    Output("budget-sensitivity", "figure"),
    Input("budget-slider", "value"),
)
def update_budget_sensitivity(slider_val):
    return make_budget_slider_figure(DATA, highlight_fraction=slider_val)


# ---------------------------------------------------------------------------
# Callback: scenario comparison
# ---------------------------------------------------------------------------

@app.callback(
    Output("scenario-comparison", "figure"),
    Input("scenario-mode", "value"),
)
def update_scenario(mode):
    return make_scenario_comparison_figure(DATA, show_mode=mode)


# ---------------------------------------------------------------------------
# Callback: robustness curves (static — lives on scenario tab)
# ---------------------------------------------------------------------------

@app.callback(
    Output("robustness-curves", "figure"),
    Input("main-tabs", "value"),
)
def update_robustness_curves(_tab):
    """Render adversarial robustness curves on the scenario tab."""
    eps   = DATA.epsilon_sweep_data["epsilons"]
    fgsm  = DATA.epsilon_sweep_data["acc_fgsm"] * 100
    bim   = DATA.epsilon_sweep_data["acc_bim"]  * 100
    pgd   = DATA.epsilon_sweep_data["acc_pgd"]  * 100
    clean = DATA.global_clean_acc * 100

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=eps, y=fgsm, mode="lines+markers", name="FGSM",
        line=dict(color="#d4a017", width=2), marker=dict(size=5),
    ))
    fig.add_trace(go.Scatter(
        x=eps, y=bim, mode="lines+markers", name="BIM",
        line=dict(color="#1a3a5c", width=2, dash="dash"), marker=dict(size=5),
    ))
    fig.add_trace(go.Scatter(
        x=eps, y=pgd, mode="lines+markers", name="PGD (random start)",
        line=dict(color="#c0392b", width=2.5), marker=dict(size=5),
    ))
    fig.add_hline(y=clean, line_dash="dot", line_color="green", line_width=1.8,
                  annotation_text=f"Clean acc. {clean:.1f}%",
                  annotation_position="top right",
                  annotation_font_size=10)
    fig.add_hline(y=25.0, line_dash="dot", line_color="gray", line_width=1,
                  annotation_text="Random (25%)", annotation_font_size=9,
                  annotation_position="bottom right")
    fig.add_vline(x=0.30, line_dash="dash", line_color="lightgray",
                  line_width=1.2, annotation_text="ε=0.30",
                  annotation_font_size=9)

    fig.update_layout(
        title=dict(
            text="WeldDefectMLP Adversarial Robustness — Accuracy vs. ε (L∞)",
            font=dict(size=12, color="#1a3a5c"),
        ),
        xaxis=dict(title="Perturbation budget ε", range=[0, 1],
                   showgrid=True, gridcolor="#f5f5f5"),
        yaxis=dict(title="Test accuracy (%)", range=[0, 110],
                   showgrid=True, gridcolor="#f5f5f5"),
        legend=dict(x=0.75, y=0.99, bgcolor="rgba(255,255,255,0.85)",
                    bordercolor="#cccccc", borderwidth=1),
        plot_bgcolor="white", paper_bgcolor="white",
        margin=dict(l=50, r=20, t=50, b=40),
        height=320,
    )
    return fig


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    port = int(os.environ.get("DASH_PORT", 8050))
    print(f"\n{'='*70}")
    print(f"  Dashboard running at  http://localhost:{port}")
    print(f"  Press Ctrl+C to stop.")
    print(f"{'='*70}\n")
    app.run(debug=False, host="0.0.0.0", port=port)
