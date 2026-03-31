"""
Adversarial Attack Implementations  (Sprint 4 — Zone A)
========================================================

Implements gradient-based adversarial attacks against the WeldDefectMLP
classifier.  These attacks generate imperceptibly perturbed NDE feature
vectors that cause the model to misclassify defective welds as clean,
representing a critical security vulnerability in AI-assisted inspection.

Attack Threat Model
-------------------
    Attacker knowledge: White-box (full model access — worst case)
    Attack goal:        Untargeted — cause any misclassification
    Perturbation norm:  L∞ — bounded by ε in every feature dimension
    Physical constraint: Perturbation ε is expressed as a fraction of the
                         feature standard deviation, ensuring adversarial
                         examples remain within physically plausible ranges

Attacks Implemented
-------------------
    FGSM  (Goodfellow et al., 2014):
        Single-step attack.  Computes gradient sign in one pass.
        x_adv = clip(x + ε·sign(∇_x L(f(x), y)), x_min, x_max)

        Fast but less powerful.  Represents a computationally-constrained
        attacker (e.g., signal manipulation in real-time).

    PGD   (Madry et al., 2017):
        Multi-step iterative attack.  K steps of FGSM at step size α,
        projected back into the ε-ball after each step.
        x_0 = x + uniform noise ∈ [−ε, ε]
        x_{t+1} = proj_{ε}(x_t + α·sign(∇_x L(f(x_t), y)))

        Stronger and more reliable.  Represents a sophisticated attacker
        with sufficient compute (e.g., nation-state threat actor).

    BIM   (Kurakin et al., 2016)  [Basic Iterative Method]:
        PGD without random start.  x_0 = x, same iteration as PGD.
        Included as a middle-ground baseline.

Physical Interpretation
-----------------------
    In the NDE context, a successful adversarial attack means:
    - A defective weld (porosity / crack / lack-of-fusion) is
      classified as CLEAN by the AI inspection model.
    - The perturbation is bounded so the modified signal still
      passes human visual inspection of the radiograph.
    - This represents a supply-chain integrity risk: manipulated
      NDE data could allow flawed welds to pass quality control.

References
----------
    Goodfellow, I.J. et al. (2014) "Explaining and harnessing adversarial
        examples." ICLR 2015.  arXiv:1412.6572.
    Madry, A. et al. (2017) "Towards deep learning models resistant to
        adversarial attacks." ICLR 2018.  arXiv:1706.06083.
    Kurakin, A. et al. (2016) "Adversarial examples in the physical world."
        arXiv:1607.02533.

Author: Babak Pirzadi (STRATEGOS Thesis — Zone A)
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from typing import Optional, Tuple

from .nde_model import WeldDefectMLP, cross_entropy_loss


# =========================================================================
# Attack configuration
# =========================================================================

@dataclass
class AttackConfig:
    """
    Configuration bundle for adversarial attack generation.

    Attributes
    ----------
    epsilon    : L∞ perturbation budget (normalised feature units)
    n_steps    : Number of PGD / BIM iterations
    step_size  : Step size per PGD iteration (α).  Default = ε / n_steps.
    random_start: If True, initialise PGD from uniform noise in ε-ball
    clip_min   : Minimum feature value after perturbation
    clip_max   : Maximum feature value after perturbation
    """
    epsilon:      float = 0.10
    n_steps:      int   = 20
    step_size:    Optional[float] = None   # defaults to ε / n_steps
    random_start: bool  = True
    clip_min:     float = -5.0   # z-score normalised features
    clip_max:     float =  5.0

    def __post_init__(self):
        if self.step_size is None:
            self.step_size = self.epsilon / self.n_steps


@dataclass
class AttackResult:
    """
    Result of an adversarial attack.

    Attributes
    ----------
    X_adv        : (N, F) adversarial feature vectors
    X_clean      : (N, F) original feature vectors
    perturbation  : (N, F) additive perturbation  δ = X_adv − X_clean
    y_true       : (N,) true class labels
    y_clean_pred : (N,) predictions on clean inputs
    y_adv_pred   : (N,) predictions on adversarial inputs
    clean_acc    : Accuracy on clean inputs
    adv_acc      : Accuracy on adversarial inputs (lower = stronger attack)
    attack_success_rate: Fraction of non-misclassified clean samples
                         that are successfully flipped by the attack
    l_inf_norm   : Mean L∞ norm of perturbations
    l2_norm      : Mean L2 norm of perturbations
    """
    X_adv:              np.ndarray
    X_clean:            np.ndarray
    perturbation:       np.ndarray
    y_true:             np.ndarray
    y_clean_pred:       np.ndarray
    y_adv_pred:         np.ndarray
    clean_acc:          float
    adv_acc:            float
    attack_success_rate: float
    l_inf_norm:         float
    l2_norm:            float
    attack_name:        str = "unknown"


# =========================================================================
# FGSM  (Fast Gradient Sign Method)
# =========================================================================

def fgsm_attack(
    model:  WeldDefectMLP,
    X:      np.ndarray,
    y:      np.ndarray,
    config: AttackConfig,
) -> AttackResult:
    """
    Fast Gradient Sign Method (FGSM) adversarial attack.

    Algorithm
    ---------
        1. Forward pass on X to populate gradient cache.
        2. Compute ∂L/∂X via model.compute_input_gradient(y).
        3. Perturb: X_adv = clip(X + ε·sign(∂L/∂X), clip_min, clip_max).

    Parameters
    ----------
    model  : Trained WeldDefectMLP
    X      : (N, F) clean input features (z-score normalised)
    y      : (N,) true integer class labels
    config : AttackConfig

    Returns
    -------
    AttackResult
    """
    X = X.astype(np.float32)

    # ── Step 1: forward pass (no dropout during attack) ──────────────────
    model.forward(X, training=False)

    # ── Step 2: compute input gradient ───────────────────────────────────
    grad_x = model.compute_input_gradient(y)

    # ── Step 3: sign perturbation ─────────────────────────────────────────
    X_adv = np.clip(
        X + config.epsilon * np.sign(grad_x),
        config.clip_min,
        config.clip_max,
    ).astype(np.float32)

    return _build_result(model, X, X_adv, y, "FGSM")


# =========================================================================
# BIM  (Basic Iterative Method)
# =========================================================================

def bim_attack(
    model:  WeldDefectMLP,
    X:      np.ndarray,
    y:      np.ndarray,
    config: AttackConfig,
) -> AttackResult:
    """
    Basic Iterative Method (BIM) adversarial attack.

    Iterates FGSM K times at step size α = ε / K, without random start.

    Algorithm
    ---------
        X_0 = X
        for t in 1…K:
            X_t = clip_{ε}(X_{t-1} + α·sign(∇_x L(f(X_{t-1}), y)))

    Parameters
    ----------
    model  : Trained WeldDefectMLP
    X      : (N, F) clean input features
    y      : (N,) true labels
    config : AttackConfig (n_steps and step_size used)

    Returns
    -------
    AttackResult
    """
    X      = X.astype(np.float32)
    X_adv  = X.copy()
    alpha  = config.step_size
    eps    = config.epsilon

    for _ in range(config.n_steps):
        model.forward(X_adv, training=False)
        grad_x = model.compute_input_gradient(y)

        X_adv = X_adv + alpha * np.sign(grad_x)

        # Project onto ε-ball centred at X
        X_adv = np.clip(X_adv, X - eps, X + eps)
        X_adv = np.clip(X_adv, config.clip_min, config.clip_max)
        X_adv = X_adv.astype(np.float32)

    return _build_result(model, X, X_adv, y, "BIM")


# =========================================================================
# PGD  (Projected Gradient Descent)
# =========================================================================

def pgd_attack(
    model:  WeldDefectMLP,
    X:      np.ndarray,
    y:      np.ndarray,
    config: AttackConfig,
    seed:   int = 42,
) -> AttackResult:
    """
    Projected Gradient Descent (PGD) adversarial attack — Madry et al. (2017).

    The strongest first-order attack with L∞ constraint.  Initialises from
    a uniformly random point in the ε-ball (random restart) for improved
    attack strength.

    Algorithm
    ---------
        X_0 = X + uniform(−ε, ε)   [random start]
        for t in 1…K:
            X_t = Π_{ε}(X_{t-1} + α·sign(∇_x L(f(X_{t-1}), y)))
        where Π_{ε} projects onto {z : ‖z−X‖_∞ ≤ ε}.

    Parameters
    ----------
    model  : Trained WeldDefectMLP
    X      : (N, F) clean input features
    y      : (N,) true labels
    config : AttackConfig
    seed   : Seed for random start

    Returns
    -------
    AttackResult
    """
    rng    = np.random.default_rng(seed)
    X      = X.astype(np.float32)
    eps    = config.epsilon
    alpha  = config.step_size

    # Random start: uniform noise in ε-ball
    if config.random_start:
        noise = rng.uniform(-eps, eps, X.shape).astype(np.float32)
        X_adv = np.clip(X + noise, config.clip_min, config.clip_max)
    else:
        X_adv = X.copy()

    for _ in range(config.n_steps):
        model.forward(X_adv, training=False)
        grad_x = model.compute_input_gradient(y)

        X_adv = X_adv + alpha * np.sign(grad_x)

        # Project onto ε-ball
        X_adv = np.clip(X_adv, X - eps, X + eps)
        X_adv = np.clip(X_adv, config.clip_min, config.clip_max)
        X_adv = X_adv.astype(np.float32)

    return _build_result(model, X, X_adv, y, "PGD")


# =========================================================================
# Epsilon sweep — robustness curve
# =========================================================================

def epsilon_sweep(
    model:          WeldDefectMLP,
    X:              np.ndarray,
    y:              np.ndarray,
    epsilons:       Optional[np.ndarray] = None,
    attack_fn:      str = "pgd",
    n_steps:        int = 20,
    random_start:   bool = True,
    clip_min:       float = -5.0,
    clip_max:       float = 5.0,
    seed:           int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Evaluate model robustness across a range of perturbation budgets ε.

    For each ε, generate adversarial examples and compute accuracy.
    Used to produce the robustness curve (fig16 in the thesis).

    Parameters
    ----------
    model       : Trained WeldDefectMLP
    X           : (N, F) clean test features
    y           : (N,) true labels
    epsilons    : Array of ε values to sweep.  Defaults to 15 points in [0, 1].
    attack_fn   : "fgsm", "bim", or "pgd"
    n_steps     : Iterations for BIM/PGD
    random_start: Random start for PGD
    clip_min/max: Feature clipping range

    Returns
    -------
    epsilons : Array of ε values tested
    accs     : Array of adversarial accuracies (same length)
    """
    if epsilons is None:
        epsilons = np.linspace(0.0, 1.0, 16)

    accs = []
    for eps in epsilons:
        cfg = AttackConfig(
            epsilon      = float(eps),
            n_steps      = n_steps,
            step_size    = max(float(eps) / n_steps, 1e-4),
            random_start = random_start,
            clip_min     = clip_min,
            clip_max     = clip_max,
        )
        if eps == 0.0 or attack_fn == "fgsm":
            result = fgsm_attack(model, X, y, cfg)
        elif attack_fn == "bim":
            result = bim_attack(model, X, y, cfg)
        else:
            result = pgd_attack(model, X, y, cfg, seed=seed)

        accs.append(result.adv_acc)

    return np.array(epsilons), np.array(accs)


# =========================================================================
# Internal helper
# =========================================================================

def _build_result(
    model:   WeldDefectMLP,
    X_clean: np.ndarray,
    X_adv:   np.ndarray,
    y_true:  np.ndarray,
    name:    str,
) -> AttackResult:
    """Compute metrics and assemble AttackResult."""
    y_clean = model.predict(X_clean)
    y_adv   = model.predict(X_adv)

    clean_acc = float((y_clean == y_true).mean())
    adv_acc   = float((y_adv   == y_true).mean())

    # Attack success rate: among samples correctly classified when clean,
    # what fraction are flipped by the attack?
    correct_mask = (y_clean == y_true)
    if correct_mask.sum() > 0:
        flipped = correct_mask & (y_adv != y_true)
        asr = float(flipped.sum()) / float(correct_mask.sum())
    else:
        asr = 0.0

    delta   = X_adv - X_clean
    l_inf   = float(np.abs(delta).max(axis=1).mean())
    l2      = float(np.linalg.norm(delta, axis=1).mean())

    return AttackResult(
        X_adv              = X_adv,
        X_clean            = X_clean,
        perturbation       = delta,
        y_true             = y_true,
        y_clean_pred       = y_clean,
        y_adv_pred         = y_adv,
        clean_acc          = clean_acc,
        adv_acc            = adv_acc,
        attack_success_rate = asr,
        l_inf_norm         = l_inf,
        l2_norm            = l2,
        attack_name        = name,
    )
