"""
Weld Defect NDE Classifier — NumPy Multi-Layer Perceptron  (Sprint 4 — Zone A)
===============================================================================

Implements a fully-connected neural network for multi-class weld defect
classification from NDE signal features.  The model is implemented from
scratch in NumPy to expose the full computational graph needed for
gradient-based adversarial attacks.

Architecture
------------
    Input (32) → Dense(128, ReLU) → Dropout(0.3) → Dense(64, ReLU)
               → Dense(4, Softmax)

    32 input features  →  4 output classes
    (NDE signal features)  (Clean, Porosity, Crack, Lack-of-fusion)

    Total parameters: 32×128 + 128 + 128×64 + 64 + 64×4 + 4 = 12,740

Training
--------
    Optimiser: Stochastic Gradient Descent with momentum (μ = 0.9)
    Loss:      Cross-entropy (∂L/∂z_i = p_i − y_i for softmax output)
    Reg:       L2 weight decay (λ = 1e-4)
    LR schedule: Cosine annealing  η(t) = η_min + ½(η_max−η_min)(1+cos(πt/T))

Key design: all forward and backward passes are implemented symbolically
so that ∂L/∂x (gradient of loss w.r.t. input) is analytically computable.
This is essential for gradient-based adversarial attack generation.

Author: Babak Pirzadi (STRATEGOS Thesis — Zone A)
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional, Tuple


# =========================================================================
# Activation functions (and their derivatives)
# =========================================================================

def relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(0.0, x)


def relu_grad(x: np.ndarray) -> np.ndarray:
    """Derivative of ReLU: 1 where x > 0, else 0."""
    return (x > 0).astype(x.dtype)


def softmax(z: np.ndarray) -> np.ndarray:
    """Numerically stable row-wise softmax."""
    z_shifted = z - z.max(axis=-1, keepdims=True)
    exp_z = np.exp(z_shifted)
    return exp_z / exp_z.sum(axis=-1, keepdims=True)


def cross_entropy_loss(probs: np.ndarray, y: np.ndarray) -> float:
    """
    Mean cross-entropy loss.

    Parameters
    ----------
    probs : (N, K) softmax output probabilities
    y     : (N,) integer class labels

    Returns
    -------
    float : mean loss
    """
    N = len(y)
    log_probs = np.log(probs[np.arange(N), y] + 1e-12)
    return -float(log_probs.mean())


# =========================================================================
# Model dataclass
# =========================================================================

@dataclass
class WeldDefectMLP:
    """
    Fully-connected MLP for weld defect classification.

    All weights and biases are stored as plain numpy arrays.
    The computational graph is retained through the cache
    populated during forward() for use in backward() and
    compute_input_gradient().

    Attributes
    ----------
    W1, b1 : First dense layer (input → 128)
    W2, b2 : Second dense layer (128 → 64)
    W3, b3 : Output layer (64 → 4)
    _cache  : Intermediate activations from last forward pass
    train_losses, val_losses : Training history
    """
    n_input:   int = 32
    n_hidden1: int = 128
    n_hidden2: int = 64
    n_classes: int = 4
    seed:      int = 42

    # Weights (initialised in __post_init__)
    W1: np.ndarray = field(init=False)
    b1: np.ndarray = field(init=False)
    W2: np.ndarray = field(init=False)
    b2: np.ndarray = field(init=False)
    W3: np.ndarray = field(init=False)
    b3: np.ndarray = field(init=False)

    # Momentum buffers
    _vW1: np.ndarray = field(init=False)
    _vb1: np.ndarray = field(init=False)
    _vW2: np.ndarray = field(init=False)
    _vb2: np.ndarray = field(init=False)
    _vW3: np.ndarray = field(init=False)
    _vb3: np.ndarray = field(init=False)

    # Computational graph cache
    _cache: dict = field(default_factory=dict)

    # Training history
    train_losses: List[float] = field(default_factory=list)
    val_losses:   List[float] = field(default_factory=list)
    train_accs:   List[float] = field(default_factory=list)
    val_accs:     List[float] = field(default_factory=list)

    def __post_init__(self):
        rng = np.random.default_rng(self.seed)
        # Xavier (Glorot) uniform initialisation
        def xavier(fan_in, fan_out):
            limit = np.sqrt(6.0 / (fan_in + fan_out))
            return rng.uniform(-limit, limit, (fan_in, fan_out)).astype(np.float32)

        self.W1 = xavier(self.n_input,   self.n_hidden1)
        self.b1 = np.zeros(self.n_hidden1, dtype=np.float32)
        self.W2 = xavier(self.n_hidden1, self.n_hidden2)
        self.b2 = np.zeros(self.n_hidden2, dtype=np.float32)
        self.W3 = xavier(self.n_hidden2, self.n_classes)
        self.b3 = np.zeros(self.n_classes, dtype=np.float32)

        self._vW1 = np.zeros_like(self.W1)
        self._vb1 = np.zeros_like(self.b1)
        self._vW2 = np.zeros_like(self.W2)
        self._vb2 = np.zeros_like(self.b2)
        self._vW3 = np.zeros_like(self.W3)
        self._vb3 = np.zeros_like(self.b3)

    # ── Forward pass ──────────────────────────────────────────────────────

    def forward(
        self,
        X:        np.ndarray,
        training: bool = False,
        dropout_p: float = 0.3,
        seed:     Optional[int] = None,
    ) -> np.ndarray:
        """
        Forward pass through the network.

        Parameters
        ----------
        X        : (N, n_input) input feature matrix
        training : If True, apply dropout (stochastic)
        dropout_p: Dropout probability (fraction of neurons zeroed)
        seed     : Optional seed for dropout mask reproducibility

        Returns
        -------
        probs : (N, n_classes) softmax output probabilities
        """
        rng = np.random.default_rng(seed)

        # Layer 1
        z1 = X @ self.W1 + self.b1         # (N, 128)
        a1 = relu(z1)                       # (N, 128)

        # Dropout (training only)
        if training and dropout_p > 0:
            mask1 = (rng.random(a1.shape) > dropout_p).astype(np.float32)
            a1 = a1 * mask1 / (1.0 - dropout_p)
        else:
            mask1 = np.ones_like(a1)

        # Layer 2
        z2 = a1 @ self.W2 + self.b2        # (N, 64)
        a2 = relu(z2)                       # (N, 64)

        # Output layer
        z3    = a2 @ self.W3 + self.b3     # (N, 4)
        probs = softmax(z3)                 # (N, 4)

        # Cache for backprop
        self._cache = {
            "X": X, "z1": z1, "a1": a1, "mask1": mask1,
            "z2": z2, "a2": a2, "z3": z3, "probs": probs,
        }
        return probs

    # ── Backward pass ─────────────────────────────────────────────────────

    def backward(
        self,
        y:        np.ndarray,
        l2_lambda: float = 1e-4,
    ) -> dict:
        """
        Backpropagation through the network.

        Uses the cached intermediate activations from the most recent
        forward() call.

        Parameters
        ----------
        y         : (N,) integer class labels
        l2_lambda : L2 regularisation coefficient

        Returns
        -------
        grads : dict of parameter gradients {dW1, db1, dW2, db2, dW3, db3}
        """
        cache = self._cache
        X      = cache["X"]
        a1     = cache["a1"]
        mask1  = cache["mask1"]
        z1     = cache["z1"]
        a2     = cache["a2"]
        z2     = cache["z2"]
        probs  = cache["probs"]
        N      = len(y)

        # Output layer: dL/dz3 = p - y_onehot (softmax + cross-entropy)
        dz3 = probs.copy()
        dz3[np.arange(N), y] -= 1.0
        dz3 /= N

        dW3 = a2.T @ dz3 + l2_lambda * self.W3
        db3 = dz3.sum(axis=0)

        # Layer 2
        da2 = dz3 @ self.W3.T
        dz2 = da2 * relu_grad(z2)

        dW2 = a1.T @ dz2 + l2_lambda * self.W2
        db2 = dz2.sum(axis=0)

        # Layer 1 (through dropout)
        da1 = dz2 @ self.W2.T
        da1 = da1 * mask1   # backprop through dropout
        dz1 = da1 * relu_grad(z1)

        dW1 = X.T @ dz1 + l2_lambda * self.W1
        db1 = dz1.sum(axis=0)

        return dict(dW1=dW1, db1=db1, dW2=dW2, db2=db2, dW3=dW3, db3=db3)

    def compute_input_gradient(self, y: np.ndarray) -> np.ndarray:
        """
        Compute ∂L/∂X — gradient of cross-entropy loss w.r.t. input features.

        This is required by gradient-based adversarial attacks (FGSM, PGD).
        The gradient flows backward through all layers to the input.

        Parameters
        ----------
        y : (N,) integer class labels (true labels for untargeted attacks)

        Returns
        -------
        dX : (N, n_input) gradient of loss w.r.t. input
        """
        cache = self._cache
        z1    = cache["z1"]
        z2    = cache["z2"]
        probs = cache["probs"]
        a1    = cache["a1"]
        mask1 = cache["mask1"]
        N     = len(y)

        # dL/dz3
        dz3 = probs.copy()
        dz3[np.arange(N), y] -= 1.0
        dz3 /= N

        # Layer 3 → 2
        da2 = dz3 @ self.W3.T
        dz2 = da2 * relu_grad(z2)

        # Layer 2 → 1
        da1 = dz2 @ self.W2.T
        da1 = da1 * mask1
        dz1 = da1 * relu_grad(z1)

        # Layer 1 → input
        dX = dz1 @ self.W1.T

        return dX

    # ── SGD update with momentum ──────────────────────────────────────────

    def update(
        self,
        grads:    dict,
        lr:       float = 0.01,
        momentum: float = 0.9,
    ) -> None:
        """Apply SGD + momentum update to all parameters."""
        self._vW1 = momentum * self._vW1 - lr * grads["dW1"]
        self._vb1 = momentum * self._vb1 - lr * grads["db1"]
        self._vW2 = momentum * self._vW2 - lr * grads["dW2"]
        self._vb2 = momentum * self._vb2 - lr * grads["db2"]
        self._vW3 = momentum * self._vW3 - lr * grads["dW3"]
        self._vb3 = momentum * self._vb3 - lr * grads["db3"]

        self.W1 += self._vW1
        self.b1 += self._vb1
        self.W2 += self._vW2
        self.b2 += self._vb2
        self.W3 += self._vW3
        self.b3 += self._vb3

    # ── Convenience methods ───────────────────────────────────────────────

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return predicted class labels (argmax of softmax)."""
        probs = self.forward(X, training=False)
        return probs.argmax(axis=1)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return softmax probability matrix (N, K)."""
        return self.forward(X, training=False)

    def accuracy(self, X: np.ndarray, y: np.ndarray) -> float:
        """Return classification accuracy ∈ [0, 1]."""
        return float((self.predict(X) == y).mean())


# =========================================================================
# Trainer
# =========================================================================

@dataclass
class TrainerConfig:
    """Hyperparameter bundle for WeldDefectMLP training."""
    n_epochs:    int   = 80
    batch_size:  int   = 64
    lr_max:      float = 0.05
    lr_min:      float = 1e-4
    momentum:    float = 0.90
    l2_lambda:   float = 1e-4
    dropout_p:   float = 0.30
    seed:        int   = 42
    verbose:     bool  = True
    print_every: int   = 10


def train_model(
    model:      WeldDefectMLP,
    X_train:    np.ndarray,
    y_train:    np.ndarray,
    X_val:      np.ndarray,
    y_val:      np.ndarray,
    config:     Optional[TrainerConfig] = None,
) -> WeldDefectMLP:
    """
    Train WeldDefectMLP using mini-batch SGD with cosine LR annealing.

    Cosine schedule:
        η(t) = η_min + ½(η_max − η_min)(1 + cos(π · t / T))

    Parameters
    ----------
    model    : Uninitialised or pre-trained WeldDefectMLP
    X_train  : (N_train, F) feature matrix (z-score normalised)
    y_train  : (N_train,) integer labels
    X_val    : (N_val, F) validation features
    y_val    : (N_val,) validation labels
    config   : TrainerConfig (uses defaults if None)

    Returns
    -------
    model : Trained model (modified in-place and returned)
    """
    if config is None:
        config = TrainerConfig()

    rng = np.random.default_rng(config.seed)
    N   = len(y_train)
    T   = config.n_epochs

    for epoch in range(T):
        # Cosine annealing
        lr = config.lr_min + 0.5 * (config.lr_max - config.lr_min) * (
            1 + np.cos(np.pi * epoch / T)
        )

        # Shuffle training data
        idx = rng.permutation(N)
        X_shuf = X_train[idx]
        y_shuf = y_train[idx]

        epoch_loss = 0.0
        n_batches  = 0

        for start in range(0, N, config.batch_size):
            end  = min(start + config.batch_size, N)
            X_b  = X_shuf[start:end]
            y_b  = y_shuf[start:end]

            probs = model.forward(X_b, training=True,
                                  dropout_p=config.dropout_p,
                                  seed=int(rng.integers(0, 2**31)))
            loss  = cross_entropy_loss(probs, y_b)
            grads = model.backward(y_b, l2_lambda=config.l2_lambda)
            model.update(grads, lr=lr, momentum=config.momentum)

            epoch_loss += loss
            n_batches  += 1

        # Record history
        train_loss = epoch_loss / n_batches
        val_probs  = model.forward(X_val, training=False)
        val_loss   = cross_entropy_loss(val_probs, y_val)
        train_acc  = model.accuracy(X_train, y_train)
        val_acc    = model.accuracy(X_val, y_val)

        model.train_losses.append(train_loss)
        model.val_losses.append(val_loss)
        model.train_accs.append(train_acc)
        model.val_accs.append(val_acc)

        if config.verbose and (epoch % config.print_every == 0
                               or epoch == T - 1):
            print(f"  Epoch {epoch+1:3d}/{T}  "
                  f"lr={lr:.5f}  "
                  f"loss={train_loss:.4f}/{val_loss:.4f}  "
                  f"acc={train_acc:.3f}/{val_acc:.3f}")

    return model
