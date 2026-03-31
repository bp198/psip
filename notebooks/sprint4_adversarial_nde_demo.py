"""
Sprint 4 Demo — Adversarial Robustness of Weld NDE Deep Learning Model
=======================================================================

Trains a WeldDefectMLP classifier on synthetic NDE signal features,
evaluates its clean accuracy, then applies FGSM, BIM, and PGD attacks
to demonstrate adversarial vulnerability.

Generates:
    fig15_adversarial_examples.png   – Feature perturbation radar plots
    fig16_robustness_curves.png      – Accuracy vs. ε for all three attacks
    fig17_adversarial_confusion.png  – Confusion matrices: clean vs. adversarial

Author: Babak Pirzadi (STRATEGOS Thesis — Zone A)
"""

import sys, os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyArrowPatch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.zone_a import (
    DefectClass, CLASS_NAMES, N_FEATURES, N_CLASSES,
    generate_nde_dataset, normalise_features,
    WeldDefectMLP, TrainerConfig, train_model,
    AttackConfig,
    fgsm_attack, bim_attack, pgd_attack, epsilon_sweep,
)

FIGURES_DIR = os.path.join(os.path.dirname(__file__), "..", "docs", "figures")
os.makedirs(FIGURES_DIR, exist_ok=True)
ETC_FIGS = "/sessions/magical-peaceful-curie/mnt/etc/figures"
os.makedirs(ETC_FIGS, exist_ok=True)

NAVY  = "#1a3a5c"
RED   = "#c0392b"
GREEN = "#1e8449"
AMBER = "#d4a017"
PURPLE = "#7d3c98"

FEATURE_GROUP_LABELS = [
    "Amplitude\n(0–7)", "Frequency\n(8–15)",
    "Geometry\n(16–23)", "Texture\n(24–31)",
]

# =========================================================================
# 1. Generate Dataset & Train Model
# =========================================================================
print("=" * 70)
print("SPRINT 4: Adversarial Robustness of Weld NDE Classifier")
print("=" * 70)

print("\n  Generating synthetic NDE dataset (overlapping distributions)...")
# noise_level=0.18 introduces realistic inter-class overlap that produces
# 88-95% clean accuracy — the realistic operating regime for NDE models.
dataset = generate_nde_dataset(n_samples_per_class=500, seed=42, noise_level=0.18)
print(f"  Total samples: {dataset.n_samples} | "
      f"Classes: {dataset.class_counts}")

train_ds, val_ds, test_ds = dataset.split(train_fraction=0.70,
                                           val_fraction=0.15, seed=42)
print(f"  Split: train={train_ds.n_samples}  "
      f"val={val_ds.n_samples}  test={test_ds.n_samples}")

# Z-score normalise
X_tr, X_te, mu_feat, sig_feat = normalise_features(train_ds.X, test_ds.X)
_, X_val, _, _                 = normalise_features(train_ds.X, val_ds.X)
y_tr  = train_ds.y
y_val = val_ds.y
y_te  = test_ds.y

print("\n  Training WeldDefectMLP (32→128→64→4, cosine LR, 80 epochs)...")
model  = WeldDefectMLP(n_input=N_FEATURES, n_hidden1=128, n_hidden2=64,
                       n_classes=N_CLASSES, seed=42)
config = TrainerConfig(n_epochs=80, batch_size=64, lr_max=0.05, lr_min=1e-4,
                       momentum=0.9, l2_lambda=1e-4, dropout_p=0.30,
                       seed=42, verbose=True, print_every=20)
model  = train_model(model, X_tr, y_tr, X_val, y_val, config)

clean_acc = model.accuracy(X_te, y_te)
print(f"\n  Clean test accuracy: {clean_acc*100:.1f}%")

# =========================================================================
# 2. Run Adversarial Attacks
# =========================================================================
print("\n" + "=" * 70)
print("  Running Adversarial Attacks  (ε = 0.30)")
print("=" * 70)

eps_nominal = 0.30
n_steps     = 20

cfg_fgsm = AttackConfig(epsilon=eps_nominal, n_steps=1,
                         clip_min=-5.0, clip_max=5.0)
cfg_bim  = AttackConfig(epsilon=eps_nominal, n_steps=n_steps,
                         clip_min=-5.0, clip_max=5.0)
cfg_pgd  = AttackConfig(epsilon=eps_nominal, n_steps=n_steps,
                         random_start=True, clip_min=-5.0, clip_max=5.0)

result_fgsm = fgsm_attack(model, X_te, y_te, cfg_fgsm)
result_bim  = bim_attack (model, X_te, y_te, cfg_bim)
result_pgd  = pgd_attack (model, X_te, y_te, cfg_pgd, seed=42)

for name, res in [("FGSM", result_fgsm), ("BIM", result_bim), ("PGD", result_pgd)]:
    print(f"  {name:<6}  acc={res.adv_acc*100:.1f}%  "
          f"ASR={res.attack_success_rate*100:.1f}%  "
          f"‖δ‖_∞={res.l_inf_norm:.3f}  "
          f"‖δ‖_2={res.l2_norm:.3f}")

# =========================================================================
# 3. Epsilon Sweep
# =========================================================================
print("\n  Running epsilon sweep (15 points, 0→1.0)...")
epsilons = np.linspace(0.0, 1.0, 16)
eps_arr, acc_fgsm = epsilon_sweep(model, X_te, y_te, epsilons, "fgsm", seed=42)
eps_arr, acc_bim  = epsilon_sweep(model, X_te, y_te, epsilons, "bim",  n_steps=15, seed=42)
eps_arr, acc_pgd  = epsilon_sweep(model, X_te, y_te, epsilons, "pgd",  n_steps=20, seed=42)
print("  Done.")

# =========================================================================
# 4. Figure 15 — Feature Perturbation Visualisation
# =========================================================================
print("\n" + "=" * 70)
print("Generating Figure 15: Adversarial Feature Perturbation Examples")
print("=" * 70)

# Select one crack sample for illustration
crack_mask = (y_te == int(DefectClass.CRACK))
idx_crack  = np.where(crack_mask)[0][:1]

X_sample   = X_te[idx_crack]
y_sample   = y_te[idx_crack]

# Generate adversarial version
cfg_vis = AttackConfig(epsilon=0.40, n_steps=20, random_start=True,
                        clip_min=-5.0, clip_max=5.0)
res_vis  = pgd_attack(model, X_sample, y_sample, cfg_vis, seed=1)

x_clean  = X_sample[0]
x_adv    = res_vis.X_adv[0]
delta    = res_vis.perturbation[0]

fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle(
    "Adversarial NDE Feature Perturbation — PGD Attack (ε = 0.40)\n"
    "Weld Defect: CRACK  →  Model prediction: "
    f"{'CLEAN (fooled!)' if res_vis.y_adv_pred[0] == 0 else CLASS_NAMES[res_vis.y_adv_pred[0]]}",
    fontsize=12, fontweight="bold",
)

x_pos = np.arange(N_FEATURES)
group_colors = ["#2980b9", "#27ae60", "#8e44ad", "#e67e22"]

def feature_colors(n=N_FEATURES):
    cols = []
    for i in range(n):
        cols.append(group_colors[i // 8])
    return cols

fc = feature_colors()

# (a) Clean features
ax = axes[0]
ax.bar(x_pos, x_clean, color=fc, alpha=0.85, edgecolor="white", linewidth=0.3)
ax.set_title("(a) Clean NDE Feature Vector", fontsize=10, fontweight="bold")
ax.set_xlabel("Feature index", fontsize=9)
ax.set_ylabel("Z-score normalised value", fontsize=9)
ax.set_xlim(-1, N_FEATURES)
ax.tick_params(labelsize=8)
for g, lab in enumerate(FEATURE_GROUP_LABELS):
    ax.axvspan(g*8 - 0.5, g*8 + 7.5, alpha=0.07,
               color=group_colors[g], label=lab)
ax.legend(fontsize=7.5, loc="upper right", title="Feature block")
ax.grid(True, axis="y", alpha=0.3)

# (b) Adversarial features
ax = axes[1]
ax.bar(x_pos, x_adv, color=fc, alpha=0.85, edgecolor="white", linewidth=0.3)
# Overlay clean reference
ax.step(x_pos, x_clean, where="mid", color="black",
        linewidth=1.0, alpha=0.5, linestyle="--", label="Clean reference")
ax.set_title(
    f"(b) Adversarial Feature Vector\nPred: "
    f"{CLASS_NAMES[res_vis.y_adv_pred[0]]} "
    f"({'✗ Fooled' if res_vis.y_adv_pred[0] != int(DefectClass.CRACK) else '✓ Correct'})",
    fontsize=10, fontweight="bold",
)
ax.set_xlabel("Feature index", fontsize=9)
ax.set_ylabel("Z-score normalised value", fontsize=9)
ax.set_xlim(-1, N_FEATURES)
ax.legend(fontsize=8)
ax.tick_params(labelsize=8)
ax.grid(True, axis="y", alpha=0.3)

# (c) Perturbation δ
ax = axes[2]
colors_delta = [RED if d > 0 else NAVY for d in delta]
ax.bar(x_pos, delta, color=colors_delta, alpha=0.85, edgecolor="white", linewidth=0.3)
ax.axhline(0, color="black", linewidth=0.8)
ax.axhline( 0.40, color="gray", linestyle=":", linewidth=1.2, label="±ε bound")
ax.axhline(-0.40, color="gray", linestyle=":", linewidth=1.2)
ax.set_title(
    f"(c) Perturbation δ = X_adv − X_clean\n"
    f"‖δ‖_∞ = {np.abs(delta).max():.3f}  ‖δ‖_2 = {np.linalg.norm(delta):.3f}",
    fontsize=10, fontweight="bold",
)
ax.set_xlabel("Feature index", fontsize=9)
ax.set_ylabel("Δ feature value", fontsize=9)
ax.set_xlim(-1, N_FEATURES)
ax.legend(fontsize=8)
ax.tick_params(labelsize=8)
ax.grid(True, axis="y", alpha=0.3)

# Group background shading
for g in range(4):
    ax.axvspan(g*8 - 0.5, g*8 + 7.5, alpha=0.07, color=group_colors[g])

plt.tight_layout()
for path in [os.path.join(FIGURES_DIR, "fig15_adversarial_examples.png"),
             os.path.join(ETC_FIGS, "fig15_adversarial_examples.png")]:
    plt.savefig(path, dpi=150, bbox_inches="tight")
plt.close()
print("  Saved fig15_adversarial_examples.png")

# =========================================================================
# 5. Figure 16 — Robustness Curves
# =========================================================================
print("\nGenerating Figure 16: Robustness Curves (Accuracy vs. ε)")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle(
    "Adversarial Robustness of WeldDefectMLP — NDE Security Analysis\n"
    "Accuracy vs. Perturbation Budget ε  (L∞ norm, z-score feature space)",
    fontsize=12, fontweight="bold",
)

# ── Panel (a): All 3 attacks ──────────────────────────────────────────────
ax = axes[0]
ax.plot(eps_arr, acc_fgsm * 100, "o-",  color=AMBER,  linewidth=2.0,
        markersize=5, label="FGSM (1-step)", zorder=3)
ax.plot(eps_arr, acc_bim  * 100, "s--", color=NAVY,   linewidth=2.0,
        markersize=5, label=f"BIM ({config.n_epochs//4}-step)", zorder=3)
ax.plot(eps_arr, acc_pgd  * 100, "D-",  color=RED,    linewidth=2.5,
        markersize=5, label=f"PGD ({n_steps}-step, random start)", zorder=4)
ax.axhline(clean_acc * 100, color="green", linestyle=":", linewidth=1.8,
           label=f"Clean accuracy ({clean_acc*100:.1f}%)")
ax.axhline(100 / N_CLASSES, color="gray", linestyle=":", linewidth=1.2,
           label=f"Random baseline ({100/N_CLASSES:.0f}%)", alpha=0.7)
ax.axvline(eps_nominal, color="lightgray", linestyle="--", linewidth=1.2,
           label=f"Nominal ε = {eps_nominal}")
ax.set_xlabel("Perturbation budget ε (L∞)", fontsize=10)
ax.set_ylabel("Test accuracy (%)", fontsize=10)
ax.set_title("(a) Accuracy vs. ε — All Attacks", fontsize=10, fontweight="bold")
ax.set_xlim(0, 1.0)
ax.set_ylim(0, 105)
ax.legend(fontsize=8.5, loc="upper right")
ax.grid(True, alpha=0.3)
ax.tick_params(labelsize=9)

# ── Panel (b): Attack Success Rate ───────────────────────────────────────
asr_fgsm, asr_bim, asr_pgd = [], [], []
for eps in epsilons:
    cfg_tmp = AttackConfig(epsilon=float(eps), n_steps=n_steps,
                           random_start=True, clip_min=-5.0, clip_max=5.0)
    for atype, fn, asr_list in [
        ("fgsm", fgsm_attack, asr_fgsm),
        ("bim",  lambda m,X,y,c: bim_attack(m,X,y,c), asr_bim),
        ("pgd",  lambda m,X,y,c: pgd_attack(m,X,y,c,seed=42), asr_pgd),
    ]:
        r = fn(model, X_te, y_te, cfg_tmp)
        asr_list.append(r.attack_success_rate * 100)

ax = axes[1]
ax.plot(epsilons, asr_fgsm, "o-",  color=AMBER,  linewidth=2.0, markersize=5,
        label="FGSM")
ax.plot(epsilons, asr_bim,  "s--", color=NAVY,   linewidth=2.0, markersize=5,
        label="BIM")
ax.plot(epsilons, asr_pgd,  "D-",  color=RED,    linewidth=2.5, markersize=5,
        label="PGD")
ax.axvline(eps_nominal, color="lightgray", linestyle="--", linewidth=1.2,
           label=f"Nominal ε = {eps_nominal}")
ax.set_xlabel("Perturbation budget ε (L∞)", fontsize=10)
ax.set_ylabel("Attack Success Rate (%)", fontsize=10)
ax.set_title("(b) Attack Success Rate vs. ε", fontsize=10, fontweight="bold")
ax.set_xlim(0, 1.0)
ax.set_ylim(0, 105)
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)
ax.tick_params(labelsize=9)

plt.tight_layout()
for path in [os.path.join(FIGURES_DIR, "fig16_robustness_curves.png"),
             os.path.join(ETC_FIGS, "fig16_robustness_curves.png")]:
    plt.savefig(path, dpi=150, bbox_inches="tight")
plt.close()
print("  Saved fig16_robustness_curves.png")

# =========================================================================
# 6. Figure 17 — Confusion Matrices
# =========================================================================
print("\nGenerating Figure 17: Confusion Matrices (Clean vs. Adversarial)")

from sklearn.metrics import confusion_matrix  # noqa: E402 (sklearn is installed)

labels_short = ["Clean", "Poros.", "Crack", "LoF"]

def plot_cm(ax, cm, title, total, cmap="Blues"):
    im = ax.imshow(cm, interpolation="nearest", cmap=cmap, vmin=0, vmax=total)
    ax.set_title(title, fontsize=9, fontweight="bold")
    ticks = np.arange(N_CLASSES)
    ax.set_xticks(ticks); ax.set_xticklabels(labels_short, fontsize=8)
    ax.set_yticks(ticks); ax.set_yticklabels(labels_short, fontsize=8, rotation=45)
    ax.set_xlabel("Predicted", fontsize=8)
    ax.set_ylabel("True", fontsize=8)
    thresh = cm.max() / 2
    for i in range(N_CLASSES):
        for j in range(N_CLASSES):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    fontsize=8, color="white" if cm[i,j] > thresh else "black")
    return im

fig, axes = plt.subplots(1, 4, figsize=(17, 4.5))
fig.suptitle(
    "Confusion Matrices — WeldDefectMLP NDE Classifier\n"
    f"Clean: {clean_acc*100:.1f}% acc  |  "
    f"FGSM (ε={eps_nominal}): {result_fgsm.adv_acc*100:.1f}%  |  "
    f"BIM: {result_bim.adv_acc*100:.1f}%  |  "
    f"PGD: {result_pgd.adv_acc*100:.1f}%",
    fontsize=11, fontweight="bold",
)

N_test = len(y_te)
configs_cm = [
    (y_te,                  result_fgsm.y_clean_pred, "Clean Inputs",  "Blues"),
    (y_te,                  result_fgsm.y_adv_pred,   f"FGSM (ε={eps_nominal})", "Oranges"),
    (y_te,                  result_bim.y_adv_pred,    f"BIM  (ε={eps_nominal})", "Purples"),
    (y_te,                  result_pgd.y_adv_pred,    f"PGD  (ε={eps_nominal})", "Reds"),
]

for ax, (yt, yp, title, cmap) in zip(axes, configs_cm):
    cm = confusion_matrix(yt, yp, labels=list(range(N_CLASSES)))
    plot_cm(ax, cm, title, N_test // N_CLASSES, cmap=cmap)

plt.tight_layout()
for path in [os.path.join(FIGURES_DIR, "fig17_adversarial_confusion.png"),
             os.path.join(ETC_FIGS, "fig17_adversarial_confusion.png")]:
    plt.savefig(path, dpi=150, bbox_inches="tight")
plt.close()
print("  Saved fig17_adversarial_confusion.png")

# =========================================================================
# 7. Summary
# =========================================================================
print("\n" + "=" * 70)
print("SPRINT 4 NDE DEMO — COMPLETE")
print("=" * 70)
print(f"""
  Model:  WeldDefectMLP (32→128→64→4)
  Data:   {dataset.n_samples} synthetic NDE samples, 4 defect classes
  Epochs: {config.n_epochs}  |  Final clean accuracy: {clean_acc*100:.1f}%

  Adversarial Results (ε = {eps_nominal}):
    FGSM  acc={result_fgsm.adv_acc*100:.1f}%  ASR={result_fgsm.attack_success_rate*100:.1f}%
    BIM   acc={result_bim.adv_acc*100:.1f}%  ASR={result_bim.attack_success_rate*100:.1f}%
    PGD   acc={result_pgd.adv_acc*100:.1f}%  ASR={result_pgd.attack_success_rate*100:.1f}%

  Key thesis finding:
    PGD reduces accuracy from {clean_acc*100:.1f}% → {result_pgd.adv_acc*100:.1f}%
    with ‖δ‖_∞ = {result_pgd.l_inf_norm:.3f} (imperceptible in normalised feature space).
    A sophisticated adversary (state-actor type from Sprint 3) could
    manipulate NDE inspection data to classify defective welds as CLEAN.

  Figures:
    fig15 → Feature perturbation bar charts (clean / adversarial / delta)
    fig16 → Robustness curves (accuracy + ASR vs. epsilon)
    fig17 → Confusion matrices (clean vs. all three attacks)
""")
