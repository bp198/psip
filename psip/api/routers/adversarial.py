"""
/api/adversarial — Adversarial attack on WeldDefectMLP router.
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException

import psip.adversarial as adv_engine
import psip.nde as nde_engine
from psip.api.models import AdversarialRequest, AdversarialResponse

# Import generate_nde_dataset and normalise_features from source
from src.zone_a.synthetic_data import generate_nde_dataset, normalise_features

router = APIRouter(prefix="/adversarial", tags=["Adversarial Attacks"])

_CLASS_NAMES = {
    0: "Clean",
    1: "Porosity",
    2: "Crack",
    3: "Lack_of_Fusion",
}

_ATTACK_MAP = {
    "fgsm": adv_engine.fgsm_attack,
    "bim": adv_engine.bim_attack,
    "pgd": adv_engine.pgd_attack,
}


@router.post(
    "/attack",
    response_model=AdversarialResponse,
    summary="Run an adversarial attack on the WeldDefectMLP NDE classifier",
    description=(
        "Trains a WeldDefectMLP (32→128→64→4) on synthetic NDE feature data, "
        "then attacks it using FGSM, BIM, or PGD under an L∞ constraint. "
        "Optionally applies physics-informed ε scaling: ε_eff = ε × SCF/1.5, "
        "ensuring perturbations remain within physically plausible sensor ranges. "
        "Thesis results: FGSM ASR=18.1%, BIM ASR=20.1%, PGD ASR=9.4% at ε=0.30."
    ),
)
def attack(req: AdversarialRequest) -> AdversarialResponse:
    method = req.method.lower()
    if method not in _ATTACK_MAP:
        raise HTTPException(
            status_code=422,
            detail=f"Unknown method '{req.method}'. Valid: 'fgsm', 'bim', 'pgd'.",
        )

    # Physics-informed epsilon scaling
    epsilon_eff = req.epsilon * (req.scf / 1.5) if req.physics_scaled else req.epsilon

    try:
        # 1. Generate synthetic NDE dataset
        n_per_class = max(req.n_samples // 4, 50)
        dataset = generate_nde_dataset(
            n_samples_per_class=n_per_class * 4,
            seed=req.random_seed,
        )
        train_ds, val_ds, test_ds = dataset.split(
            train_fraction=0.70,
            val_fraction=0.15,
            seed=req.random_seed,
        )

        X_train, X_test, mean_, std_ = normalise_features(train_ds.X, test_ds.X)
        X_val = (val_ds.X - mean_) / std_
        y_train, y_test = train_ds.y, test_ds.y
        y_val = val_ds.y

        # 2. Train model
        model = nde_engine.WeldDefectMLP(input_dim=32, hidden1=128, hidden2=64, n_classes=4)
        trainer_cfg = nde_engine.TrainerConfig(n_epochs=30, seed=req.random_seed)
        nde_engine.train_model(model, X_train, y_train, X_val, y_val, config=trainer_cfg)

        # 3. Attack
        config = adv_engine.AttackConfig(
            epsilon=epsilon_eff,
            n_steps=req.n_steps,
        )
        # Use only req.n_samples from test set
        X_attack = X_test[: req.n_samples]
        y_attack = y_test[: req.n_samples]

        result = _ATTACK_MAP[method](model=model, X=X_attack, y=y_attack, config=config)

    except (ValueError, RuntimeError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    # Per-class attack success rate
    class_breakdown: dict[str, float] = {}
    for cls_id, cls_name in _CLASS_NAMES.items():
        mask = result.y_true == cls_id
        if mask.sum() == 0:
            class_breakdown[cls_name] = 0.0
            continue
        # Among correctly-classified clean samples of this class, how many flipped?
        correct_clean = result.y_clean_pred[mask] == result.y_true[mask]
        if correct_clean.sum() == 0:
            class_breakdown[cls_name] = 0.0
        else:
            flipped = result.y_adv_pred[mask][correct_clean] != result.y_true[mask][correct_clean]
            class_breakdown[cls_name] = round(float(flipped.mean()) * 100, 2)

    return AdversarialResponse(
        method=method.upper(),
        epsilon_requested=req.epsilon,
        epsilon_effective=round(epsilon_eff, 4),
        n_samples=len(X_attack),
        clean_accuracy=round(result.clean_acc * 100, 2),
        adversarial_accuracy=round(result.adv_acc * 100, 2),
        attack_success_rate=round(result.attack_success_rate * 100, 2),
        mean_l_inf=round(float(result.l_inf_norm), 4),
        mean_l2=round(float(result.l2_norm), 4),
        class_breakdown=class_breakdown,
    )
