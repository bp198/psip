"""
Sprint 1 Physics Engine Demonstration
=======================================
Generates publication-quality FAD and S-N curve plots.
Runs a sample Monte Carlo P_f simulation.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.zone_c.physics.fad_engine import (
    MaterialProperties, FlawGeometry, PipeGeometry, WeldJoint,
    fad_option1, compute_Lr_max, assess_flaw, plot_fad,
)
from src.zone_c.physics.fatigue_engine import (
    FatigueParameters, fatigue_life, plot_sn_curve,
)
from src.zone_c.physics.mc_failure_probability import (
    DistributionParams, PipelineSegmentConfig, monte_carlo_Pf,
    default_distributions_api5l_x65, plot_mc_on_fad,
)

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "docs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ===================================================================
# FIGURE 1: FAD Curve with Multiple Assessment Points
# ===================================================================
print("=" * 60)
print("FIGURE 1: BS 7910 Option 1 FAD with Assessment Points")
print("=" * 60)

mat_x65 = MaterialProperties(sigma_y=450, sigma_u=535, K_mat=100)
pipe_20in = PipeGeometry(outer_diameter=508.0, wall_thickness=12.7)

# Define several flaw scenarios
scenarios = [
    ("Small flaw, low P", FlawGeometry(1.0, 10.0), WeldJoint(scf=1.2), 5.0),
    ("Medium flaw, nominal P", FlawGeometry(3.0, 30.0), WeldJoint(scf=1.5), 7.0),
    ("Large flaw, high P", FlawGeometry(6.0, 60.0), WeldJoint(scf=2.0), 9.0),
    ("Critical flaw + residual", FlawGeometry(8.0, 80.0), WeldJoint(scf=2.5), 10.0),
]

results = []
labels = []
for name, flaw, weld, pressure in scenarios:
    sigma_res = 450.0 if "residual" in name else 0.0
    r = assess_flaw(mat_x65, flaw, pipe_20in, weld, pressure, sigma_residual=sigma_res)
    results.append(r)
    labels.append(name)
    status = "ACCEPT" if r.is_acceptable else "REJECT"
    print(f"  {name}: Kr={r.Kr:.3f}, Lr={r.Lr:.3f}, f(Lr)={r.f_Lr:.3f}, "
          f"RF={r.reserve_factor:.2f} [{status}]")

fig, ax = plot_fad(mat_x65, results, labels,
                   title="BS 7910 Option 1 FAD — API 5L X65, 20\" Pipeline",
                   save_path=os.path.join(OUTPUT_DIR, "fig1_fad_assessment.png"))
print(f"\n  Saved: {os.path.join(OUTPUT_DIR, 'fig1_fad_assessment.png')}")
plt.close(fig)

# ===================================================================
# FIGURE 2: Multi-Grade FAD Comparison
# ===================================================================
print("\n" + "=" * 60)
print("FIGURE 2: FAD Curves — X42 vs X52 vs X65 vs X80")
print("=" * 60)

grades = {
    "X42": MaterialProperties(sigma_y=290, sigma_u=414, K_mat=80),
    "X52": MaterialProperties(sigma_y=358, sigma_u=455, K_mat=90),
    "X65": MaterialProperties(sigma_y=450, sigma_u=535, K_mat=100),
    "X80": MaterialProperties(sigma_y=555, sigma_u=625, K_mat=110),
}

fig, ax = plt.subplots(figsize=(10, 8))
colors = ["#e74c3c", "#e67e22", "#2ecc71", "#3498db"]
for (name, mat), color in zip(grades.items(), colors):
    Lr_max = compute_Lr_max(mat.sigma_y, mat.sigma_u)
    Lr = np.linspace(0, Lr_max, 300)
    f = fad_option1(Lr, mat)
    ax.plot(Lr, f, linewidth=2.5, color=color,
            label=f"API 5L {name} (Lr_max={Lr_max:.3f})")
    print(f"  {name}: sigma_y={mat.sigma_y} MPa, Lr_max={Lr_max:.3f}")

ax.set_xlabel("$L_r$ (Load Ratio)", fontsize=14)
ax.set_ylabel("$K_r$ (Fracture Ratio)", fontsize=14)
ax.set_title("FAD Curve Comparison — API 5L Pipeline Steel Grades", fontsize=16, fontweight="bold")
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)
ax.set_xlim(0, 1.7)
ax.set_ylim(0, 1.05)
plt.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, "fig2_fad_grade_comparison.png"), dpi=300, bbox_inches="tight")
print(f"\n  Saved: {os.path.join(OUTPUT_DIR, 'fig2_fad_grade_comparison.png')}")
plt.close(fig)

# ===================================================================
# FIGURE 3: IIW S-N Curves for Different Weld Types
# ===================================================================
print("\n" + "=" * 60)
print("FIGURE 3: IIW S-N Curves — Weld Type Comparison")
print("=" * 60)

fig, ax = plt.subplots(figsize=(10, 7))
fat_classes = [
    (112, "Butt weld, ground flush", "#2ecc71"),
    (90, "Butt weld, as-welded (both sides)", "#3498db"),
    (71, "Butt weld, single-side (girth weld)", "#e67e22"),
    (56, "Socket weld", "#e74c3c"),
]

stresses = np.logspace(np.log10(20), np.log10(400), 300)
for fat, label, color in fat_classes:
    params = FatigueParameters(fat_class=fat)
    N = fatigue_life(stresses, params)
    ax.loglog(N, stresses, linewidth=2.5, color=color, label=f"FAT {fat} — {label}")
    print(f"  FAT {fat}: Life at 100 MPa = {fatigue_life(100.0, params):.0f} cycles")

# Mark the 2e6 reference line
ax.axvline(x=2e6, color="gray", linestyle=":", alpha=0.7, label="$N = 2 \\times 10^6$")
ax.set_xlabel("Fatigue Life N (cycles)", fontsize=14)
ax.set_ylabel("Stress Range Δσ (MPa)", fontsize=14)
ax.set_title("IIW S-N Curves — Impact of Weld Type on Fatigue Life", fontsize=16, fontweight="bold")
ax.legend(fontsize=10, loc="upper right")
ax.grid(True, which="both", alpha=0.3)
ax.set_xlim(1e4, 1e9)
ax.set_ylim(20, 500)
plt.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, "fig3_sn_weld_comparison.png"), dpi=300, bbox_inches="tight")
print(f"\n  Saved: {os.path.join(OUTPUT_DIR, 'fig3_sn_weld_comparison.png')}")
plt.close(fig)

# ===================================================================
# FIGURE 4: Monte Carlo P_f Simulation on FAD
# ===================================================================
print("\n" + "=" * 60)
print("FIGURE 4: Monte Carlo Failure Probability — 10,000 Simulations")
print("=" * 60)

defaults = default_distributions_api5l_x65()
config = PipelineSegmentConfig(
    segment_id="DEMO-001",
    pipe=pipe_20in,
    weld=WeldJoint(weld_type="butt", fat_class=71, scf=1.5),
    **defaults,
)

mc_result = monte_carlo_Pf(config, n_simulations=10_000, seed=42)

print(f"  Segment: {mc_result.segment_id}")
print(f"  Simulations: {mc_result.n_simulations:,}")
print(f"  Failures: {mc_result.n_failures:,}")
print(f"  P_f = {mc_result.P_f:.4f} [{mc_result.P_f_lower:.4f}, {mc_result.P_f_upper:.4f}] (95% CI)")
print(f"  Mean Kr = {mc_result.mean_Kr:.3f}")
print(f"  Mean Lr = {mc_result.mean_Lr:.3f}")
print(f"  Mean Reserve Factor = {mc_result.mean_reserve:.2f}")

fig, ax = plot_mc_on_fad(mc_result, mat_x65,
                         save_path=os.path.join(OUTPUT_DIR, "fig4_mc_pf_simulation.png"))
print(f"\n  Saved: {os.path.join(OUTPUT_DIR, 'fig4_mc_pf_simulation.png')}")
plt.close(fig)

# ===================================================================
# FIGURE 5: P_f Sensitivity to Weld Type (SCF)
# ===================================================================
print("\n" + "=" * 60)
print("FIGURE 5: P_f Sensitivity to Weld SCF")
print("=" * 60)

scf_values = [1.0, 1.5, 2.0, 2.5, 3.0, 3.5]
pf_values = []

for scf in scf_values:
    cfg = PipelineSegmentConfig(
        segment_id=f"SCF-{scf}",
        pipe=pipe_20in,
        weld=WeldJoint(weld_type="butt", fat_class=71, scf=scf),
        dist_defect_a=defaults["dist_defect_a"],
        dist_defect_2c=defaults["dist_defect_2c"],
        dist_K_mat=defaults["dist_K_mat"],
        dist_sigma_y=defaults["dist_sigma_y"],
        dist_sigma_u=defaults["dist_sigma_u"],
        dist_pressure=defaults["dist_pressure"],
        dist_scf=DistributionParams("uniform", scf * 0.9, scf * 1.1),  # tight around nominal
    )
    res = monte_carlo_Pf(cfg, n_simulations=5_000, seed=42)
    pf_values.append(res.P_f)
    print(f"  SCF={scf:.1f}: P_f = {res.P_f:.4f}")

fig, ax = plt.subplots(figsize=(10, 6))
ax.bar([str(s) for s in scf_values], pf_values, color=["#2ecc71", "#27ae60", "#f39c12", "#e67e22", "#e74c3c", "#c0392b"],
       edgecolor="black", linewidth=0.5)
ax.set_xlabel("Stress Concentration Factor (SCF)", fontsize=14)
ax.set_ylabel("Probability of Failure $P_f$", fontsize=14)
ax.set_title("Impact of Weld SCF on Failure Probability\n(This is the physics-informed payoff for game theory)",
             fontsize=14, fontweight="bold")
ax.grid(True, axis="y", alpha=0.3)
for i, (scf, pf) in enumerate(zip(scf_values, pf_values)):
    ax.text(i, pf + 0.005, f"{pf:.3f}", ha="center", fontsize=11, fontweight="bold")
plt.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, "fig5_pf_scf_sensitivity.png"), dpi=300, bbox_inches="tight")
print(f"\n  Saved: {os.path.join(OUTPUT_DIR, 'fig5_pf_scf_sensitivity.png')}")
plt.close(fig)

print("\n" + "=" * 60)
print("ALL FIGURES GENERATED SUCCESSFULLY")
print("=" * 60)
