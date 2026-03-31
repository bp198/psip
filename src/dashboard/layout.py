"""
Dash Application Layout — Sprint 5
=====================================

Builds the complete HTML/Dash component tree for the Sprint 5 dashboard.
All layout is declared here; callbacks are wired in sprint5_dashboard.py.

Author: Babak Pirzadi (STRATEGOS Thesis — Sprint 5)
"""

from dash import html, dcc

# ---------------------------------------------------------------------------
# Colour tokens
# ---------------------------------------------------------------------------

NAVY  = "#1a3a5c"
LIGHT = "#f4f6f8"
WHITE = "#ffffff"
BORDER = "#dee2e6"

# ---------------------------------------------------------------------------
# Re-usable card wrapper
# ---------------------------------------------------------------------------

def _card(children, style=None):
    base = dict(
        background=WHITE,
        border=f"1px solid {BORDER}",
        borderRadius="6px",
        padding="12px 16px",
        marginBottom="12px",
        boxShadow="0 1px 3px rgba(0,0,0,.06)",
    )
    if style:
        base.update(style)
    return html.Div(children, style=base)


def _header_badge(text, colour="#e8f4fd"):
    return html.Span(
        text,
        style=dict(
            background=colour,
            color=NAVY,
            borderRadius="4px",
            padding="2px 8px",
            fontSize="11px",
            fontWeight="600",
            marginLeft="8px",
        ),
    )


# ---------------------------------------------------------------------------
# Top navbar
# ---------------------------------------------------------------------------

def _navbar():
    return html.Div(
        [
            html.Div(
                [
                    html.H1(
                        "STRATEGOS Pipeline Defence Dashboard",
                        style=dict(
                            color=WHITE, margin="0", fontSize="18px",
                            fontWeight="700", letterSpacing="0.5px",
                        ),
                    ),
                    html.Div(
                        [
                            _header_badge("Sprint 5", "#3498db"),
                            _header_badge("Physics-Informed", "#1abc9c"),
                            _header_badge("Game-Theoretic", "#e67e22"),
                            _header_badge("Adversarial NDE", "#e74c3c"),
                        ],
                        style=dict(display="flex", alignItems="center",
                                   marginLeft="24px"),
                    ),
                ],
                style=dict(display="flex", alignItems="center"),
            ),
            html.Div(
                "Gulf Coast Transmission Network · B. Pirzadi · STRATEGOS MSc",
                style=dict(color="rgba(255,255,255,0.65)", fontSize="11px"),
            ),
        ],
        style=dict(
            background=NAVY,
            padding="12px 24px",
            display="flex",
            justifyContent="space-between",
            alignItems="center",
            marginBottom="0",
            borderBottom="3px solid #e74c3c",
        ),
    )


# ---------------------------------------------------------------------------
# KPI strip (top summary row)
# ---------------------------------------------------------------------------

def _kpi_strip():
    kpi_style = dict(
        flex="1",
        textAlign="center",
        padding="10px 6px",
        borderRight=f"1px solid {BORDER}",
    )
    return html.Div(
        id="kpi-strip",
        children=[html.Div("Loading...", style=dict(color="gray"))],
        style=dict(
            display="flex",
            background=WHITE,
            border=f"1px solid {BORDER}",
            borderRadius="6px",
            marginBottom="12px",
            overflow="hidden",
        ),
    )


def _kpi_item(label: str, value: str, colour: str = NAVY, suffix: str = ""):
    return html.Div(
        [
            html.Div(
                [html.Span(value, style=dict(fontSize="22px",
                                             fontWeight="700",
                                             color=colour)),
                 html.Span(suffix, style=dict(fontSize="13px",
                                              color=colour,
                                              marginLeft="2px"))],
            ),
            html.Div(label, style=dict(fontSize="10px", color="#6c757d",
                                       marginTop="2px", textTransform="uppercase",
                                       letterSpacing="0.5px")),
        ],
        style=dict(flex="1", textAlign="center", padding="10px 6px",
                   borderRight=f"1px solid {BORDER}"),
    )


# ---------------------------------------------------------------------------
# Tab 1 — Network Intelligence
# ---------------------------------------------------------------------------

def _tab_network():
    return html.Div(
        [
            # ── Controls row ─────────────────────────────────────────────────
            _card([
                html.Div(
                    [
                        html.Label("Edge Colouring:",
                                   style=dict(fontWeight="600", marginRight="8px",
                                              fontSize="12px")),
                        dcc.RadioItems(
                            id="colour-mode",
                            options=[
                                {"label": " Failure Probability P_f",    "value": "pf"},
                                {"label": " SSE Coverage c_i",           "value": "coverage"},
                                {"label": " Residual Risk P_f·(1-c_i·(1-δ))", "value": "risk"},
                            ],
                            value="pf",
                            inline=True,
                            inputStyle={"marginRight": "4px"},
                            style={"fontSize": "12px"},
                        ),
                    ],
                    style=dict(display="flex", alignItems="center",
                               marginBottom="8px"),
                ),
                html.Div(
                    [
                        html.Label("Options:",
                                   style=dict(fontWeight="600", marginRight="8px",
                                              fontSize="12px")),
                        dcc.Checklist(
                            id="network-options",
                            options=[
                                {"label": " Show attacker strategy bubbles",
                                 "value": "show_attacker"},
                            ],
                            value=[],
                            inline=True,
                            inputStyle={"marginRight": "4px"},
                            style={"fontSize": "12px"},
                        ),
                    ],
                    style=dict(display="flex", alignItems="center"),
                ),
            ], style=dict(padding="10px 16px", marginBottom="8px")),

            # ── Network map + segment intel (2 columns) ──────────────────────
            html.Div(
                [
                    # Left: map
                    html.Div(
                        [_card([
                            dcc.Graph(
                                id="network-map",
                                config={"scrollZoom": True,
                                        "displayModeBar": True,
                                        "modeBarButtonsToRemove": ["select2d", "lasso2d"]},
                                style={"height": "480px"},
                            ),
                            html.Div(
                                "Click any pipeline segment to inspect its "
                                "FAD assessment and adversarial threat profile.",
                                style=dict(fontSize="11px", color="#6c757d",
                                           marginTop="4px", textAlign="center"),
                            ),
                        ])],
                        style=dict(flex="0 0 62%"),
                    ),
                    # Right: segment intel
                    html.Div(
                        [
                            _card([
                                html.H4("Segment Intelligence",
                                        style=dict(margin="0 0 8px 0",
                                                   fontSize="13px",
                                                   color=NAVY,
                                                   borderBottom=f"2px solid {NAVY}",
                                                   paddingBottom="6px")),
                                html.Div(id="segment-intel-panel",
                                         children=[
                                             html.Div(
                                                 "👆  Select a segment on the map",
                                                 style=dict(color="#6c757d",
                                                            textAlign="center",
                                                            paddingTop="40px",
                                                            fontSize="12px"),
                                             )
                                         ]),
                            ], style=dict(marginBottom="8px")),
                            _card([
                                html.H4("BS 7910 FAD Assessment",
                                        style=dict(margin="0 0 6px 0",
                                                   fontSize="12px", color=NAVY)),
                                dcc.Graph(id="fad-figure",
                                          config={"displayModeBar": False},
                                          style={"height": "340px"}),
                            ], style=dict(marginBottom="8px")),
                            _card([
                                html.H4("Adversarial NDE Threat",
                                        style=dict(margin="0 0 6px 0",
                                                   fontSize="12px", color=NAVY)),
                                dcc.Graph(id="adv-figure",
                                          config={"displayModeBar": False},
                                          style={"height": "260px"}),
                            ]),
                        ],
                        style=dict(flex="0 0 38%", paddingLeft="10px"),
                    ),
                ],
                style=dict(display="flex", gap="0px"),
            ),
        ],
    )


# ---------------------------------------------------------------------------
# Tab 2 — Stackelberg Defender
# ---------------------------------------------------------------------------

def _tab_game():
    return html.Div(
        [
            # Budget slider + coverage toggle
            _card([
                html.Div(
                    [
                        html.Label("Coverage display:",
                                   style=dict(fontWeight="600", fontSize="12px",
                                              marginRight="8px")),
                        dcc.RadioItems(
                            id="coverage-mode",
                            options=[
                                {"label": " SSE Allocation",     "value": "ssg"},
                                {"label": " Uniform Baseline",   "value": "baseline"},
                                {"label": " SSE − Baseline gain", "value": "diff"},
                            ],
                            value="ssg",
                            inline=True,
                            inputStyle={"marginRight": "4px"},
                            style={"fontSize": "12px"},
                        ),
                    ],
                    style=dict(display="flex", alignItems="center",
                               marginBottom="8px"),
                ),
                html.Div(
                    [
                        html.Label("Budget fraction (for sensitivity highlighting):",
                                   style=dict(fontWeight="600", fontSize="12px",
                                              marginRight="8px")),
                        dcc.Slider(
                            id="budget-slider",
                            min=0.05, max=0.90, step=0.05, value=0.30,
                            marks={v: f"{v:.0%}" for v in
                                   [0.05, 0.10, 0.20, 0.30, 0.40, 0.50,
                                    0.60, 0.70, 0.80, 0.90]},
                            tooltip={"placement": "bottom",
                                     "always_visible": False},
                        ),
                    ],
                    style=dict(display="flex", alignItems="center",
                               flex="1"),
                ),
            ], style=dict(padding="10px 16px")),

            # Coverage heatmap
            _card([
                html.H4("Optimal Defender Coverage Distribution",
                        style=dict(margin="0 0 6px 0", fontSize="13px", color=NAVY)),
                dcc.Graph(id="coverage-heatmap",
                          config={"displayModeBar": False},
                          style={"height": "340px"}),
            ]),

            # Budget sensitivity
            _card([
                html.H4("Budget Sensitivity Analysis",
                        style=dict(margin="0 0 6px 0", fontSize="13px", color=NAVY)),
                dcc.Graph(id="budget-sensitivity",
                          config={"displayModeBar": False},
                          style={"height": "320px"}),
            ]),
        ],
    )


# ---------------------------------------------------------------------------
# Tab 3 — Scenario Comparison
# ---------------------------------------------------------------------------

def _tab_scenario():
    return html.Div(
        [
            _card([
                html.Div(
                    [
                        html.Label("Show:",
                                   style=dict(fontWeight="600", fontSize="12px",
                                              marginRight="8px")),
                        dcc.RadioItems(
                            id="scenario-mode",
                            options=[
                                {"label": " Both strategies",         "value": "both"},
                                {"label": " Baseline only",           "value": "baseline"},
                                {"label": " Physics-Informed only",   "value": "ssg"},
                            ],
                            value="both",
                            inline=True,
                            inputStyle={"marginRight": "4px"},
                            style={"fontSize": "12px"},
                        ),
                    ],
                    style=dict(display="flex", alignItems="center"),
                ),
            ], style=dict(padding="10px 16px")),

            _card([
                html.H4("Per-Segment Residual Risk Comparison",
                        style=dict(margin="0 0 6px 0", fontSize="13px", color=NAVY)),
                dcc.Graph(id="scenario-comparison",
                          config={"displayModeBar": True},
                          style={"height": "380px"}),
            ]),

            _card([
                html.H4("Adversarial Robustness Curves",
                        style=dict(margin="0 0 6px 0", fontSize="13px", color=NAVY)),
                dcc.Graph(id="robustness-curves",
                          config={"displayModeBar": False},
                          style={"height": "320px"}),
            ]),
        ],
    )


# ---------------------------------------------------------------------------
# Root layout builder
# ---------------------------------------------------------------------------

def build_layout() -> html.Div:
    """Construct the complete Dash application layout.

    Returns:
        The root html.Div to assign to app.layout.
    """
    return html.Div(
        [
            _navbar(),
            html.Div(
                [
                    # KPI strip
                    _kpi_strip(),

                    # Hidden store for selected segment
                    dcc.Store(id="selected-segment", data=None),

                    # Tabs
                    dcc.Tabs(
                        id="main-tabs",
                        value="tab-network",
                        children=[
                            dcc.Tab(
                                label="🗺  Network Intelligence",
                                value="tab-network",
                                children=[_tab_network()],
                                style=dict(fontSize="12px"),
                                selected_style=dict(
                                    fontWeight="700",
                                    borderTop=f"3px solid {NAVY}",
                                    fontSize="12px",
                                ),
                            ),
                            dcc.Tab(
                                label="♟  Stackelberg Defender",
                                value="tab-game",
                                children=[_tab_game()],
                                style=dict(fontSize="12px"),
                                selected_style=dict(
                                    fontWeight="700",
                                    borderTop="3px solid #e67e22",
                                    fontSize="12px",
                                ),
                            ),
                            dcc.Tab(
                                label="⚔  Scenario Comparison",
                                value="tab-scenario",
                                children=[_tab_scenario()],
                                style=dict(fontSize="12px"),
                                selected_style=dict(
                                    fontWeight="700",
                                    borderTop="3px solid #e74c3c",
                                    fontSize="12px",
                                ),
                            ),
                        ],
                        style=dict(marginBottom="0"),
                    ),
                ],
                style=dict(padding="12px 20px", maxWidth="1600px",
                           margin="0 auto"),
            ),
        ],
        style=dict(fontFamily="'Inter', 'Segoe UI', Arial, sans-serif",
                   background=LIGHT, minHeight="100vh"),
    )
