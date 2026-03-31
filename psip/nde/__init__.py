"""
psip.nde — Weld Defect NDE Classifier (WeldDefectMLP).

Re-exports the complete public API from the underlying nde_model module.
"""
from src.zone_a.nde_model import (
    WeldDefectMLP,
    TrainerConfig,
    train_model,
    relu,
    relu_grad,
    softmax,
    cross_entropy_loss,
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
