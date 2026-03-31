"""Zone A — Adversarial Robustness of Deep Learning Models for Weld NDE (Sprint 4)."""

from .synthetic_data import (
    DefectClass, CLASS_NAMES, N_FEATURES, N_CLASSES,
    NDEDataset, generate_nde_dataset, normalise_features,
)
from .nde_model import (
    WeldDefectMLP, TrainerConfig, train_model,
    relu, relu_grad, softmax, cross_entropy_loss,
)
from .adversarial_attacks import (
    AttackConfig, AttackResult,
    fgsm_attack, bim_attack, pgd_attack, epsilon_sweep,
)
