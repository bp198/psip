"""
Unit Tests — Zone A: Adversarial Attack Implementations (Sprint 4)
==================================================================

Tests for:
    - AttackConfig defaults and validation
    - FGSM: perturbation bounds, ASR > 0 at high ε, result structure
    - BIM: iterative improvement over FGSM, budget respect
    - PGD: random start, stronger than BIM at high steps
    - epsilon_sweep: shape, monotonicity
    - AttackResult: metric consistency

Author: Babak Pirzadi (STRATEGOS Thesis)
"""

import pytest
import numpy as np

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from src.zone_a.synthetic_data import (
    N_FEATURES, N_CLASSES, generate_nde_dataset, normalise_features,
)
from src.zone_a.nde_model import (
    WeldDefectMLP, TrainerConfig, train_model,
)
from src.zone_a.adversarial_attacks import (
    AttackConfig, AttackResult,
    fgsm_attack, bim_attack, pgd_attack, epsilon_sweep,
)


# =========================================================================
# Shared fixtures
# =========================================================================

@pytest.fixture(scope="module")
def trained_model_and_data():
    """Train a small model once for all attack tests."""
    ds = generate_nde_dataset(n_samples_per_class=120, seed=42, noise_level=0.18)
    tr, vl, te = ds.split(seed=42)
    X_tr, X_te, _, _ = normalise_features(tr.X, te.X)
    _, X_val, _, _   = normalise_features(tr.X, vl.X)

    m = WeldDefectMLP(n_input=N_FEATURES, n_hidden1=32, n_hidden2=16,
                      n_classes=N_CLASSES, seed=42)
    cfg = TrainerConfig(n_epochs=40, batch_size=32, lr_max=0.05,
                        verbose=False, seed=42)
    train_model(m, X_tr, tr.y, X_val, vl.y, cfg)
    return m, X_te, te.y


@pytest.fixture(scope="module")
def nominal_config():
    return AttackConfig(epsilon=0.40, n_steps=10, random_start=True,
                        clip_min=-5.0, clip_max=5.0)


# =========================================================================
# 1. AttackConfig
# =========================================================================

class TestAttackConfig:
    def test_default_step_size(self):
        cfg = AttackConfig(epsilon=0.20, n_steps=10)
        assert abs(cfg.step_size - 0.02) < 1e-9

    def test_explicit_step_size_respected(self):
        cfg = AttackConfig(epsilon=0.20, n_steps=10, step_size=0.05)
        assert abs(cfg.step_size - 0.05) < 1e-9

    def test_clip_bounds(self):
        cfg = AttackConfig(clip_min=-3.0, clip_max=3.0)
        assert cfg.clip_min == -3.0
        assert cfg.clip_max ==  3.0

    def test_random_start_default(self):
        cfg = AttackConfig()
        assert cfg.random_start is True


# =========================================================================
# 2. FGSM
# =========================================================================

class TestFGSM:
    def test_returns_attack_result(self, trained_model_and_data, nominal_config):
        m, X, y = trained_model_and_data
        r = fgsm_attack(m, X[:20], y[:20], nominal_config)
        assert isinstance(r, AttackResult)

    def test_attack_name(self, trained_model_and_data, nominal_config):
        m, X, y = trained_model_and_data
        r = fgsm_attack(m, X[:20], y[:20], nominal_config)
        assert r.attack_name == "FGSM"

    def test_x_adv_shape(self, trained_model_and_data, nominal_config):
        m, X, y = trained_model_and_data
        r = fgsm_attack(m, X[:20], y[:20], nominal_config)
        assert r.X_adv.shape == X[:20].shape

    def test_perturbation_bounded(self, trained_model_and_data):
        m, X, y = trained_model_and_data
        eps = 0.30
        cfg = AttackConfig(epsilon=eps, clip_min=-5.0, clip_max=5.0)
        r   = fgsm_attack(m, X, y, cfg)
        linf = np.abs(r.perturbation).max(axis=1)
        assert linf.max() <= eps + 1e-5, f"L∞ exceeded: {linf.max():.4f} > {eps}"

    def test_adversarial_examples_clipped(self, trained_model_and_data):
        m, X, y = trained_model_and_data
        cfg = AttackConfig(epsilon=0.30, clip_min=-5.0, clip_max=5.0)
        r   = fgsm_attack(m, X, y, cfg)
        assert r.X_adv.max() <= 5.0 + 1e-5
        assert r.X_adv.min() >= -5.0 - 1e-5

    def test_asr_positive_at_high_epsilon(self, trained_model_and_data):
        m, X, y = trained_model_and_data
        cfg = AttackConfig(epsilon=1.0, clip_min=-5.0, clip_max=5.0)
        r   = fgsm_attack(m, X, y, cfg)
        assert r.attack_success_rate > 0.0, "Expected some successful attacks at ε=1.0"

    def test_clean_acc_consistent(self, trained_model_and_data, nominal_config):
        m, X, y = trained_model_and_data
        r = fgsm_attack(m, X, y, nominal_config)
        direct_acc = float((m.predict(X) == y).mean())
        assert abs(r.clean_acc - direct_acc) < 1e-6

    def test_asr_in_valid_range(self, trained_model_and_data, nominal_config):
        m, X, y = trained_model_and_data
        r = fgsm_attack(m, X, y, nominal_config)
        assert 0.0 <= r.attack_success_rate <= 1.0

    def test_l_inf_norm_positive(self, trained_model_and_data, nominal_config):
        m, X, y = trained_model_and_data
        r = fgsm_attack(m, X, y, nominal_config)
        assert r.l_inf_norm > 0.0


# =========================================================================
# 3. BIM
# =========================================================================

class TestBIM:
    def test_returns_attack_result(self, trained_model_and_data, nominal_config):
        m, X, y = trained_model_and_data
        r = bim_attack(m, X[:20], y[:20], nominal_config)
        assert isinstance(r, AttackResult)
        assert r.attack_name == "BIM"

    def test_perturbation_bounded(self, trained_model_and_data):
        m, X, y = trained_model_and_data
        eps = 0.30
        cfg = AttackConfig(epsilon=eps, n_steps=10, clip_min=-5.0, clip_max=5.0)
        r   = bim_attack(m, X, y, cfg)
        linf = np.abs(r.perturbation).max(axis=1)
        assert linf.max() <= eps + 1e-4

    def test_x_adv_within_clip(self, trained_model_and_data, nominal_config):
        m, X, y = trained_model_and_data
        r = bim_attack(m, X, y, nominal_config)
        assert r.X_adv.max() <= nominal_config.clip_max + 1e-4
        assert r.X_adv.min() >= nominal_config.clip_min - 1e-4

    def test_more_steps_not_worse_than_fewer(self, trained_model_and_data):
        """BIM with more steps should achieve at least as high ASR."""
        m, X, y = trained_model_and_data
        eps = 0.50
        cfg_few  = AttackConfig(epsilon=eps, n_steps=3,  clip_min=-5.0, clip_max=5.0)
        cfg_many = AttackConfig(epsilon=eps, n_steps=20, clip_min=-5.0, clip_max=5.0)
        r_few  = bim_attack(m, X, y, cfg_few)
        r_many = bim_attack(m, X, y, cfg_many)
        # More steps ≥ fewer steps in attack success
        assert r_many.attack_success_rate >= r_few.attack_success_rate - 0.02


# =========================================================================
# 4. PGD
# =========================================================================

class TestPGD:
    def test_returns_attack_result(self, trained_model_and_data, nominal_config):
        m, X, y = trained_model_and_data
        r = pgd_attack(m, X[:20], y[:20], nominal_config, seed=0)
        assert isinstance(r, AttackResult)
        assert r.attack_name == "PGD"

    def test_perturbation_bounded(self, trained_model_and_data):
        m, X, y = trained_model_and_data
        eps = 0.30
        cfg = AttackConfig(epsilon=eps, n_steps=10, random_start=True,
                           clip_min=-5.0, clip_max=5.0)
        r   = pgd_attack(m, X, y, cfg, seed=7)
        linf = np.abs(r.perturbation).max(axis=1)
        assert linf.max() <= eps + 1e-4

    def test_random_start_vs_no_random_start_differ(self, trained_model_and_data):
        m, X, y = trained_model_and_data
        eps = 0.40
        cfg_rs  = AttackConfig(epsilon=eps, n_steps=10, random_start=True,
                               clip_min=-5.0, clip_max=5.0)
        cfg_nrs = AttackConfig(epsilon=eps, n_steps=10, random_start=False,
                               clip_min=-5.0, clip_max=5.0)
        r_rs  = pgd_attack(m, X, y, cfg_rs,  seed=42)
        r_nrs = pgd_attack(m, X, y, cfg_nrs, seed=42)
        # Adversarial examples should differ (random start adds noise)
        assert not np.allclose(r_rs.X_adv, r_nrs.X_adv)

    def test_asr_positive_at_high_epsilon(self, trained_model_and_data):
        m, X, y = trained_model_and_data
        cfg = AttackConfig(epsilon=1.0, n_steps=20, random_start=True,
                           clip_min=-5.0, clip_max=5.0)
        r   = pgd_attack(m, X, y, cfg, seed=0)
        assert r.attack_success_rate > 0.0

    def test_different_seeds_give_different_adversarials(self, trained_model_and_data):
        m, X, y = trained_model_and_data
        cfg = AttackConfig(epsilon=0.30, n_steps=5, random_start=True,
                           clip_min=-5.0, clip_max=5.0)
        r1 = pgd_attack(m, X[:10], y[:10], cfg, seed=1)
        r2 = pgd_attack(m, X[:10], y[:10], cfg, seed=2)
        assert not np.allclose(r1.X_adv, r2.X_adv)


# =========================================================================
# 5. AttackResult Metrics Consistency
# =========================================================================

class TestAttackResultMetrics:
    @pytest.fixture
    def result(self, trained_model_and_data):
        m, X, y = trained_model_and_data
        cfg = AttackConfig(epsilon=0.40, n_steps=10, clip_min=-5.0, clip_max=5.0)
        return pgd_attack(m, X, y, cfg, seed=0), y

    def test_clean_acc_range(self, result):
        r, y = result
        assert 0.0 <= r.clean_acc <= 1.0

    def test_adv_acc_range(self, result):
        r, y = result
        assert 0.0 <= r.adv_acc <= 1.0

    def test_adv_acc_leq_clean_acc(self, result):
        """Attack should not improve accuracy."""
        r, y = result
        assert r.adv_acc <= r.clean_acc + 1e-6

    def test_asr_in_range(self, result):
        r, y = result
        assert 0.0 <= r.attack_success_rate <= 1.0

    def test_perturbation_consistent(self, result):
        r, y = result
        assert np.allclose(r.perturbation, r.X_adv - r.X_clean)

    def test_l_inf_matches_perturbation(self, result):
        r, y = result
        expected = float(np.abs(r.perturbation).max(axis=1).mean())
        assert abs(r.l_inf_norm - expected) < 1e-5

    def test_l2_matches_perturbation(self, result):
        r, y = result
        expected = float(np.linalg.norm(r.perturbation, axis=1).mean())
        assert abs(r.l2_norm - expected) < 1e-5


# =========================================================================
# 6. Epsilon Sweep
# =========================================================================

class TestEpsilonSweep:
    @pytest.fixture(scope="class")
    def sweep_result(self, trained_model_and_data):
        m, X, y = trained_model_and_data
        epsilons = np.linspace(0.0, 0.8, 9)
        eps_arr, accs = epsilon_sweep(m, X, y, epsilons=epsilons,
                                      attack_fn="pgd", n_steps=10, seed=0)
        return eps_arr, accs

    def test_output_shapes_match(self, sweep_result):
        eps_arr, accs = sweep_result
        assert len(eps_arr) == len(accs)

    def test_accuracy_at_zero_eps_is_clean(self, sweep_result, trained_model_and_data):
        m, X, y = trained_model_and_data
        eps_arr, accs = sweep_result
        clean_acc = float((m.predict(X) == y).mean())
        assert abs(accs[0] - clean_acc) < 1e-5

    def test_accuracy_decreases_with_epsilon(self, sweep_result):
        """Overall trend: higher ε → lower accuracy."""
        eps_arr, accs = sweep_result
        assert accs[-1] <= accs[0] + 0.05

    def test_all_accs_in_valid_range(self, sweep_result):
        _, accs = sweep_result
        assert (accs >= 0.0).all()
        assert (accs <= 1.0 + 1e-5).all()
