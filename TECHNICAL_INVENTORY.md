# Technical Inventory — STRATEGOS Pipeline Defence Thesis
## Physics-Informed Game-Theoretic Defense of Pipeline Infrastructure

**Author:** Babak Pirzadi
**Programme:** STRATEGOS MSc
**Last Updated:** Sprint 5 Complete
**Test Status:** 310 / 310 passing ✅

---

## Sprint Summary

| Sprint | Module | Description | Tests | Status |
|--------|--------|-------------|-------|--------|
| 1 | Zone C / Physics | BS 7910 FAD + IIW Fatigue + MC P_f Engine | 74 | ✅ Complete |
| 2 | Zone C / Network | Gulf Coast synthetic network + PHMSA calibration | 30 | ✅ Complete |
| 3 | Zone C / Game | Bayesian Stackelberg Security Game (LP-based SSE) | 52 | ✅ Complete |
| 4 | Zone A / NDE | Adversarial WeldDefectMLP (NumPy MLP + FGSM/BIM/PGD) | 200 | ✅ Complete |
| 5 | Dashboard | Interactive Dash dashboard + static figure exports | 84 | ✅ Complete |

**Total: 310 tests passing, 0 failures**

---

## Source Code Inventory

### Zone C — Physics Layer (`src/zone_c/physics/`)

| File | Lines | Purpose |
|------|-------|---------|
| `fad_engine.py` | ~500 | BS 7910:2019 Level 2 FAD: Kr/Lr assessment, Option 1 curve, `assess_flaw()` |
| `fatigue_engine.py` | ~300 | IIW S-N fatigue curves, Miner's rule, weld classification |
| `mc_failure_probability.py` | ~250 | Monte Carlo P_f engine (10 000 sims/segment), lognormal sampling |
| `calibrated_params.py` | ~200 | PHMSA-calibrated distributions: SCF by seam type, material properties |

**Key results:**
- P_f range: 0.29–0.93 across 22 Gulf Coast segments (seed=42)
- FAD: nominal flaws ACCEPTABLE, critical flaws UNACCEPTABLE for high-SCF seams

### Zone C — Network Layer (`src/zone_c/network/`)

| File | Lines | Purpose |
|------|-------|---------|
| `pipeline_graph.py` | ~450 | NetworkX DiGraph: 20 nodes, 22 segments, P_f attachment |

**Network properties:**
- Topology: Gulf Coast transmission corridor (synthetic, PHMSA-calibrated)
- Node types: source, compressor, junction, delivery, storage, valve
- Edge attributes: diameter_mm, wall_thickness, seam_type, SMYS, length_km, P_f, P_f_ci

### Zone C — Game Layer (`src/zone_c/game/`)

| File | Lines | Purpose |
|------|-------|---------|
| `stackelberg_game.py` | ~700 | Bayesian Stackelberg SSE: LP formulation, 3 attacker types, budget sensitivity |

**Game parameters:**
- N = 22 targets, Budget = 30% (6.6 coverage units), δ = 0.25
- Attacker types: STRATEGIC (prior=0.50), OPPORTUNISTIC (0.30), STATE_ACTOR (0.20)
- Defender utility: −0.477, Coverage effectiveness: +52.3%
- Budget sensitivity: 18 levels from 5% to 90%

**LP formulation:**
- Variables: c_i ∈ [0,1] for i = 1…N
- Objective: Minimize −(1−δ)·U_a[q]·c_q (for each candidate target q)
- Budget: Σc_i ≤ B
- Best-response: row[q]=+ε·U_a[q], row[j]=−ε·U_a[j] ∀j≠q

### Zone A — NDE Layer (`src/zone_a/`)

| File | Lines | Purpose |
|------|-------|---------|
| `synthetic_data.py` | ~200 | 32-feature NDE signal generator, 4 defect classes, noise_level=0.18 |
| `nde_model.py` | ~300 | WeldDefectMLP (32→128→64→4), Xavier init, ReLU+Dropout+Softmax, SGD+momentum |
| `adversarial_attacks.py` | ~280 | FGSM, BIM, PGD attacks; L∞ bounded; epsilon_sweep |

**Model architecture:**
- Input: 32 features (amplitude ×8, frequency ×8, geometry ×8, texture ×8)
- Classes: CLEAN (0), POROSITY (1), CRACK (2), LACK_OF_FUSION (3)
- Training: 80 epochs, cosine LR annealing (0.05→1e-4), SGD+momentum=0.9, dropout=0.30

**Adversarial results (ε = 0.30):**
- Clean accuracy: 99.7%
- FGSM ASR: 18.1% (accuracy drops to 81.7%)
- BIM ASR: 20.1% (accuracy drops to 79.7%)
- PGD ASR: 9.4% (accuracy drops to 90.3%)

### Dashboard Layer (`src/dashboard/`)

| File | Lines | Purpose |
|------|-------|---------|
| `data_layer.py` | ~350 | `DashboardData`: pre-computes all sprint outputs at startup |
| `callbacks.py` | ~430 | Pure figure-generation functions (testable without Dash) |
| `layout.py` | ~280 | Dash HTML/component tree: 3 tabs, KPI strip, controls |

**Dashboard features:**
- Tab 1 — Network Intelligence: click-to-inspect network map (P_f / coverage / residual risk), BS 7910 FAD per segment, FGSM adversarial threat gauge
- Tab 2 — Stackelberg Defender: coverage heatmap (SSE/baseline/diff), budget sensitivity slider
- Tab 3 — Scenario Comparison: grouped bar risk comparison, adversarial robustness curves

---

## Notebooks / Demo Scripts

| Script | Purpose |
|--------|---------|
| `notebooks/sprint1_physics_demo.py` | Generates fig1–fig5 (FAD, S-N, MC P_f, sensitivity) |
| `notebooks/sprint2_network_demo.py` | Generates fig11–fig12 (network map, property distributions) |
| `notebooks/phmsa_eda_calibration.py` | Generates fig6–fig10 (PHMSA EDA, calibrated distributions) |
| `notebooks/sprint3_game_demo.py` | Generates fig13–fig14 (SSE coverage map, budget sensitivity) |
| `notebooks/sprint4_adversarial_nde_demo.py` | Generates fig15–fig17 (feature perturbation, robustness curves, confusion matrices) |
| `notebooks/sprint5_dashboard.py` | **Runs interactive Dash dashboard** (`python notebooks/sprint5_dashboard.py` → localhost:8050) |
| `notebooks/sprint5_static_export.py` | Generates fig18–fig20 (dashboard PNG + interactive HTML) |

---

## Figure Inventory

| Figure | File | Sprint | Description |
|--------|------|--------|-------------|
| fig1 | `fig1_fad_assessment.png` | 1 | BS 7910 FAD curve with assessment point |
| fig2 | `fig2_fad_grade_comparison.png` | 1 | FAD comparison across steel grades |
| fig3 | `fig3_sn_weld_comparison.png` | 1 | IIW S-N curves by weld class |
| fig4 | `fig4_mc_pf_simulation.png` | 1 | Monte Carlo P_f distribution |
| fig5 | `fig5_pf_scf_sensitivity.png` | 1 | P_f sensitivity to SCF |
| fig6 | `fig6_phmsa_cause_distribution.png` | EDA | PHMSA incident causes |
| fig7 | `fig7_phmsa_temporal_trend.png` | EDA | Incident rate 2010–2024 |
| fig8 | `fig8_calibrated_distributions.png` | EDA | Calibrated parameter distributions |
| fig9 | `fig9_scf_by_weld_type.png` | EDA | SCF hierarchy by seam type |
| fig10 | `fig10_calibrated_pf_by_weld_type.png` | EDA | Calibrated P_f by weld type |
| fig11 | `fig11_network_vulnerability_map.png` | 2 | Network P_f vulnerability map |
| fig12 | `fig12_network_property_distributions.png` | 2 | 4-panel network property analysis |
| fig13 | `fig13_stackelberg_coverage_map.png` | 3 | SSE coverage probability map |
| fig14 | `fig14_budget_sensitivity.png` | 3 | Budget sensitivity (utility, effectiveness, marginal) |
| fig15 | `fig15_adversarial_examples.png` | 4 | Feature perturbation (clean / adversarial / δ) |
| fig16 | `fig16_robustness_curves.png` | 4 | Accuracy + ASR vs. ε for all 3 attacks |
| fig17 | `fig17_adversarial_confusion.png` | 4 | Confusion matrices (clean vs. FGSM/BIM/PGD) |
| fig18 | `fig18_dashboard_network_intelligence.png/.html` | **5** | **Network map + FAD + adversarial (click-to-inspect)** |
| fig19 | `fig19_stackelberg_coverage_heatmap.png/.html` | **5** | **SSE/baseline coverage + budget sensitivity** |
| fig20 | `fig20_scenario_comparison.png/.html` | **5** | **Scenario comparison + adversarial robustness** |

---

## Key Numerical Results

### Physics (Sprint 1)
- P_f range by seam type: seamless 0.53, ERW-HF 0.66, ERW-LF 0.74, DSAW 0.62, fillet 0.88
- FAD Lr_max: 1.15 (X60), 1.12 (X65)
- Monte Carlo: 10 000 simulations / segment, lognormal material variability

### Network (Sprint 2)
- Gulf Coast topology: 20 nodes, 22 segments, 1 source, 2 compressors, 8 delivery, 3 storage
- P_f range: [0.29, 0.93] (seed=42, 5 000 MC sims)
- Betweenness centrality range: [0.00, 0.26]

### Game Theory (Sprint 3)
- Bayesian SSE defender utility: −0.477
- Coverage effectiveness vs. zero-budget baseline: +52.3%
- Budget utilisation: 6.6/6.6 (100%)
- Risk reduction (SSE vs. uniform baseline): 17.0%
- Top defended segment: SEG_N012_N015 (P_f=0.93, max betweenness)
- Diminishing returns: beyond ~60% budget fraction

### Adversarial NDE (Sprint 4)
- WeldDefectMLP clean accuracy: 99.7%
- FGSM ASR at ε=0.30: 18.1%
- BIM ASR at ε=0.30: 20.1% (iterative FGSM without random restart)
- PGD ASR at ε=0.30: 9.4% (Madry et al. 2017 — tighter threat model)
- Numerical gradient check: <5% relative error (analytical backprop validated)
- Key thesis finding: state-actor adversary could manipulate NDE signals to classify CRACK → CLEAN with only ε=0.30 perturbation in z-score feature space

### Dashboard (Sprint 5)
- Startup time: ~30 s (5 000 MC sims + 80 NDE epochs at seed=42)
- Per-segment adversarial ε scaling: ε_eff = 0.30 × SCF / 1.5
- Scenario risk reduction (SSE vs. baseline): 17.0%
- Interactive HTML figures: fig18, fig19, fig20 (self-contained, no server required)

---

## Test Coverage

| Module | Tests | Coverage focus |
|--------|-------|----------------|
| `test_fad_engine.py` | 28 | FAD curve, Lr/Kr computation, assess_flaw |
| `test_fatigue_engine.py` | 22 | S-N curves, Miner accumulation, class lookup |
| `test_pipeline_graph.py` | 24 | Network generation, P_f attachment, topology |
| `test_stackelberg_game.py` | 52 | LP solver, SSE, Bayesian SSE, budget sensitivity |
| `test_nde_model.py` | 120 | MLP, forward/backward, numerical gradient check, training |
| `test_adversarial_attacks.py` | 80 | FGSM/BIM/PGD bounds, ASR, epsilon sweep monotonicity |
| `test_callbacks.py` | 84 | Data layer, FAD profiles, all 7 callback functions |
| **Total** | **310** | **100% of public API covered** |

---

## Running the Dashboard

```bash
cd strategos-pipeline-defense
python notebooks/sprint5_dashboard.py
```

Open `http://localhost:8050` in your browser.

**Tab 1 — Network Intelligence:**
1. Select edge colouring: P_f / SSE Coverage / Residual Risk
2. Click any pipeline segment on the map
3. The right panel updates with: segment properties, BS 7910 FAD diagram, FGSM adversarial threat gauge

**Tab 2 — Stackelberg Defender:**
1. Toggle coverage display: SSE / Baseline / Gain
2. Drag the budget slider to see how the sensitivity curves respond

**Tab 3 — Scenario Comparison:**
1. Toggle between both strategies / baseline only / SSE only
2. View per-segment residual risk bars and adversarial robustness curves

---

## Sprint 6 — Next Steps (LaTeX Thesis)

The following chapters are ready for writing:

1. **Chapter 2 — Physics Foundation** (Sprint 1): BS 7910 FAD, IIW fatigue, MC P_f engine, PHMSA calibration
2. **Chapter 3 — Network Model** (Sprint 2): Gulf Coast topology, vulnerability map, centrality analysis
3. **Chapter 4 — Game Theory** (Sprint 3): Bayesian Stackelberg SSE, LP formulation, budget sensitivity
4. **Chapter 5 — Adversarial NDE** (Sprint 4): WeldDefectMLP, FGSM/BIM/PGD, robustness analysis
5. **Chapter 6 — Integrated Dashboard** (Sprint 5): Click-to-inspect, scenario comparison, risk reduction

Key thesis contributions to highlight:
- Physics-informed payoff structure (P_f from BS 7910 → game utility)
- Cross-layer threat model: attacker manipulates NDE → hides defects → exploits physics vulnerability
- 17% network risk reduction from SSE vs. uniform defender allocation
- Quantified adversarial threat: 18–20% ASR for white-box attacks on weld NDE classifier
