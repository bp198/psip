"""
Unit Tests — Zone A: WeldDefectMLP & Synthetic NDE Dataset (Sprint 4)
======================================================================

Tests for:
    - DefectClass enum and class metadata
    - NDEDataset construction, split, and class counts
    - generate_nde_dataset: shape, dtype, reproducibility
    - normalise_features: zero mean, unit variance, test leakage
    - WeldDefectMLP: construction, forward, backward, gradient, predict
    - train_model: accuracy improves, loss decreases

Author: Babak Pirzadi (STRATEGOS Thesis)
"""

import pytest
import numpy as np

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from src.zone_a.synthetic_data import (
    DefectClass, CLASS_NAMES, N_FEATURES, N_CLASSES,
    NDEDataset, generate_nde_dataset, normalise_features,
)
from src.zone_a.nde_model import (
    WeldDefectMLP, TrainerConfig, train_model,
    relu, relu_grad, softmax, cross_entropy_loss,
)


# =========================================================================
# Helpers
# =========================================================================

def small_dataset(seed=42):
    return generate_nde_dataset(n_samples_per_class=50, seed=seed, noise_level=0.15)

def tiny_model(seed=0):
    return WeldDefectMLP(n_input=N_FEATURES, n_hidden1=16, n_hidden2=8,
                         n_classes=N_CLASSES, seed=seed)


# =========================================================================
# 1. DefectClass & Metadata
# =========================================================================

class TestDefectClass:
    def test_n_classes(self):
        assert N_CLASSES == 4

    def test_n_features(self):
        assert N_FEATURES == 32

    def test_class_values(self):
        assert DefectClass.CLEAN          == 0
        assert DefectClass.POROSITY       == 1
        assert DefectClass.CRACK          == 2
        assert DefectClass.LACK_OF_FUSION == 3

    def test_class_names_cover_all(self):
        for cls in DefectClass:
            assert cls in CLASS_NAMES
            assert isinstance(CLASS_NAMES[cls], str)
            assert len(CLASS_NAMES[cls]) > 0


# =========================================================================
# 2. Synthetic Dataset Generation
# =========================================================================

class TestNDEDatasetGeneration:
    @pytest.fixture
    def ds(self):
        return small_dataset()

    def test_total_samples(self, ds):
        assert ds.n_samples == 50 * N_CLASSES

    def test_X_shape(self, ds):
        assert ds.X.shape == (ds.n_samples, N_FEATURES)

    def test_y_shape(self, ds):
        assert ds.y.shape == (ds.n_samples,)

    def test_X_dtype(self, ds):
        assert ds.X.dtype == np.float32

    def test_y_dtype(self, ds):
        assert ds.y.dtype == np.int32

    def test_class_counts_balanced(self, ds):
        for cls in range(N_CLASSES):
            assert ds.class_counts[cls] == 50

    def test_labels_in_valid_range(self, ds):
        assert ds.y.min() >= 0
        assert ds.y.max() < N_CLASSES

    def test_deterministic_with_same_seed(self):
        d1 = generate_nde_dataset(n_samples_per_class=30, seed=7)
        d2 = generate_nde_dataset(n_samples_per_class=30, seed=7)
        assert np.allclose(d1.X, d2.X)
        assert np.array_equal(d1.y, d2.y)

    def test_different_seeds_differ(self):
        d1 = generate_nde_dataset(n_samples_per_class=30, seed=1)
        d2 = generate_nde_dataset(n_samples_per_class=30, seed=2)
        assert not np.allclose(d1.X, d2.X)


# =========================================================================
# 3. Dataset Split
# =========================================================================

class TestNDEDatasetSplit:
    @pytest.fixture
    def split(self):
        ds = small_dataset()
        return ds.split(train_fraction=0.70, val_fraction=0.15, seed=42)

    def test_three_splits_returned(self, split):
        assert len(split) == 3

    def test_no_overlap(self, split):
        tr, vl, te = split
        # Spot check: no row from test should appear in train
        # (use a hash-based check on rounded values)
        tr_set = {tuple(x[:4].round(4)) for x in tr.X}
        for row in te.X:
            assert tuple(row[:4].round(4)) not in tr_set

    def test_sizes_sum_to_original(self, split):
        ds = small_dataset()
        tr, vl, te = split
        assert tr.n_samples + vl.n_samples + te.n_samples == ds.n_samples

    def test_all_splits_have_features(self, split):
        for ds in split:
            assert ds.X.shape[1] == N_FEATURES


# =========================================================================
# 4. Feature Normalisation
# =========================================================================

class TestNormaliseFeatures:
    @pytest.fixture
    def normalised(self):
        ds = small_dataset()
        tr, _, te = ds.split(seed=42)
        X_tr_n, X_te_n, mu, sig = normalise_features(tr.X, te.X)
        return X_tr_n, X_te_n, mu, sig, tr.X

    def test_train_zero_mean(self, normalised):
        X_tr_n, _, _, _, _ = normalised
        assert np.abs(X_tr_n.mean(axis=0)).max() < 1e-5

    def test_train_unit_std(self, normalised):
        X_tr_n, _, _, _, _ = normalised
        assert np.abs(X_tr_n.std(axis=0) - 1.0).max() < 1e-4

    def test_output_dtype_float32(self, normalised):
        X_tr_n, X_te_n, _, _, _ = normalised
        assert X_tr_n.dtype == np.float32
        assert X_te_n.dtype == np.float32

    def test_mu_sig_shapes(self, normalised):
        _, _, mu, sig, raw = normalised
        assert mu.shape == (N_FEATURES,)
        assert sig.shape == (N_FEATURES,)

    def test_sig_positive(self, normalised):
        _, _, _, sig, _ = normalised
        assert (sig > 0).all()


# =========================================================================
# 5. Activation Functions
# =========================================================================

class TestActivations:
    def test_relu_positive_unchanged(self):
        x = np.array([0.5, 1.0, 2.0])
        assert np.allclose(relu(x), x)

    def test_relu_negative_zeroed(self):
        x = np.array([-1.0, -0.5, 0.0])
        assert np.allclose(relu(x), np.zeros(3))

    def test_relu_grad_positive(self):
        x = np.array([1.0, 2.0])
        assert np.allclose(relu_grad(x), np.ones(2))

    def test_relu_grad_negative(self):
        x = np.array([-1.0, -0.001])
        assert np.allclose(relu_grad(x), np.zeros(2))

    def test_softmax_sums_to_one(self):
        z = np.array([[1.0, 2.0, 3.0, 0.5]])
        s = softmax(z)
        assert abs(s.sum() - 1.0) < 1e-6

    def test_softmax_all_positive(self):
        z = np.random.default_rng(0).normal(0, 1, (10, 4))
        s = softmax(z)
        assert (s > 0).all()

    def test_softmax_numerical_stability(self):
        """Softmax should not overflow for large inputs."""
        z = np.array([[1000.0, 999.0, 998.0, 997.0]])
        s = softmax(z)
        assert np.isfinite(s).all()
        assert abs(s.sum() - 1.0) < 1e-5

    def test_cross_entropy_perfect_prediction(self):
        """Zero loss when model is perfectly confident and correct."""
        probs = np.array([[1.0, 1e-12, 1e-12, 1e-12],
                          [1e-12, 1.0, 1e-12, 1e-12]])
        y = np.array([0, 1])
        loss = cross_entropy_loss(probs, y)
        assert loss < 0.01

    def test_cross_entropy_uniform_gives_log_k(self):
        """Cross-entropy of uniform prediction = log(K)."""
        K = 4
        probs = np.ones((5, K)) / K
        y = np.zeros(5, dtype=int)
        loss = cross_entropy_loss(probs, y)
        assert abs(loss - np.log(K)) < 1e-4


# =========================================================================
# 6. WeldDefectMLP Construction
# =========================================================================

class TestMLPConstruction:
    def test_weight_shapes(self):
        m = tiny_model()
        assert m.W1.shape == (N_FEATURES, 16)
        assert m.b1.shape == (16,)
        assert m.W2.shape == (16, 8)
        assert m.b2.shape == (8,)
        assert m.W3.shape == (8, N_CLASSES)
        assert m.b3.shape == (N_CLASSES,)

    def test_weights_not_zero(self):
        m = tiny_model()
        for W in [m.W1, m.W2, m.W3]:
            assert not np.allclose(W, 0)

    def test_biases_initially_zero(self):
        m = tiny_model()
        for b in [m.b1, m.b2, m.b3]:
            assert np.allclose(b, 0)

    def test_different_seeds_give_different_weights(self):
        m1 = tiny_model(seed=1)
        m2 = tiny_model(seed=2)
        assert not np.allclose(m1.W1, m2.W1)


# =========================================================================
# 7. Forward Pass
# =========================================================================

class TestForwardPass:
    @pytest.fixture
    def model_input(self):
        m = tiny_model()
        rng = np.random.default_rng(0)
        X = rng.normal(0, 1, (8, N_FEATURES)).astype(np.float32)
        return m, X

    def test_output_shape(self, model_input):
        m, X = model_input
        probs = m.forward(X)
        assert probs.shape == (8, N_CLASSES)

    def test_output_sums_to_one(self, model_input):
        m, X = model_input
        probs = m.forward(X)
        assert np.abs(probs.sum(axis=1) - 1.0).max() < 1e-5

    def test_output_all_positive(self, model_input):
        m, X = model_input
        probs = m.forward(X)
        assert (probs > 0).all()

    def test_cache_populated(self, model_input):
        m, X = model_input
        m.forward(X)
        for key in ["X", "z1", "a1", "z2", "a2", "z3", "probs"]:
            assert key in m._cache

    def test_predict_returns_integers(self, model_input):
        m, X = model_input
        y_pred = m.predict(X)
        assert y_pred.dtype in (np.int32, np.int64, int)
        assert y_pred.shape == (8,)
        assert y_pred.min() >= 0 and y_pred.max() < N_CLASSES


# =========================================================================
# 8. Backward Pass & Input Gradient
# =========================================================================

class TestBackwardPass:
    @pytest.fixture
    def setup(self):
        rng = np.random.default_rng(5)
        m = tiny_model(seed=5)
        X = rng.normal(0, 1, (16, N_FEATURES)).astype(np.float32)
        y = rng.integers(0, N_CLASSES, 16).astype(np.int32)
        m.forward(X, training=False)
        return m, X, y

    def test_grad_shapes(self, setup):
        m, X, y = setup
        grads = m.backward(y)
        assert grads["dW1"].shape == m.W1.shape
        assert grads["db1"].shape == m.b1.shape
        assert grads["dW2"].shape == m.W2.shape
        assert grads["dW3"].shape == m.W3.shape

    def test_grads_not_zero(self, setup):
        m, X, y = setup
        grads = m.backward(y)
        assert not np.allclose(grads["dW1"], 0)

    def test_input_gradient_shape(self, setup):
        m, X, y = setup
        dX = m.compute_input_gradient(y)
        assert dX.shape == X.shape

    def test_input_gradient_not_zero(self, setup):
        m, X, y = setup
        dX = m.compute_input_gradient(y)
        assert not np.allclose(dX, 0)

    def test_numerical_gradient_check(self):
        """Finite-difference check on ∂L/∂x for a single input."""
        rng = np.random.default_rng(99)
        m = tiny_model(seed=99)
        X = rng.normal(0, 0.5, (1, N_FEATURES)).astype(np.float32)
        y = np.array([2], dtype=np.int32)
        eps = 1e-3

        # Analytical gradient
        m.forward(X, training=False)
        dX_analytical = m.compute_input_gradient(y)

        # Numerical gradient (central differences, spot-check 8 features)
        indices = [0, 4, 8, 12, 16, 20, 24, 28]
        for idx in indices:
            X_p = X.copy(); X_p[0, idx] += eps
            X_m = X.copy(); X_m[0, idx] -= eps
            loss_p = cross_entropy_loss(m.forward(X_p, training=False), y)
            loss_m = cross_entropy_loss(m.forward(X_m, training=False), y)
            numerical = (loss_p - loss_m) / (2 * eps)
            analytical = dX_analytical[0, idx]
            rel_err = abs(numerical - analytical) / (abs(numerical) + 1e-8)
            assert rel_err < 0.05, (
                f"Feature {idx}: numerical={numerical:.6f}, "
                f"analytical={analytical:.6f}, rel_err={rel_err:.4f}"
            )


# =========================================================================
# 9. Training (Quick)
# =========================================================================

class TestTraining:
    @pytest.fixture(scope="class")
    def trained_model(self):
        ds = generate_nde_dataset(n_samples_per_class=100, seed=0, noise_level=0.15)
        tr, vl, te = ds.split(seed=0)
        X_tr, X_te, _, _ = normalise_features(tr.X, te.X)
        _, X_val, _, _   = normalise_features(tr.X, vl.X)
        m = WeldDefectMLP(seed=0)
        cfg = TrainerConfig(n_epochs=30, batch_size=32, lr_max=0.05,
                            verbose=False, seed=0)
        train_model(m, X_tr, tr.y, X_val, vl.y, cfg)
        return m, X_te, te.y

    def test_history_length(self, trained_model):
        m, _, _ = trained_model
        assert len(m.train_losses) == 30
        assert len(m.val_losses)   == 30
        assert len(m.train_accs)   == 30

    def test_loss_decreases(self, trained_model):
        m, _, _ = trained_model
        # Final loss should be lower than initial
        assert m.train_losses[-1] < m.train_losses[0]

    def test_accuracy_above_chance(self, trained_model):
        m, X_te, y_te = trained_model
        acc = m.accuracy(X_te, y_te)
        assert acc > 1.0 / N_CLASSES + 0.10, f"Accuracy {acc:.3f} barely above chance"

    def test_all_losses_finite(self, trained_model):
        m, _, _ = trained_model
        assert all(np.isfinite(l) for l in m.train_losses)
        assert all(np.isfinite(l) for l in m.val_losses)
