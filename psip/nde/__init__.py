"""
psip.nde — Weld Defect NDE Classifier (WeldDefectMLP).

Re-exports the complete public API from the underlying nde_model module.
"""

from src.zone_a.nde_model import (
    TrainerConfig,
    WeldDefectMLP,
    cross_entropy_loss,
    relu,
    relu_grad,
    softmax,
    train_model,
)

__all__ = [
    "WeldDefectMLP",
    "TrainerConfig",
    "train_model",
    "relu",
    "relu_grad",
    "softmax",
    "cross_entropy_loss",
]
