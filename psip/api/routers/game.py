"""
/api/game — Bayesian Stackelberg Security Equilibrium router.
"""
from __future__ import annotations

from typing import Dict
from fastapi import APIRouter, HTTPException
from psip.api.models import GameRequest, GameResponse
import psip.game as game_engine
import psip.network as net_engine

router = APIRouter(prefix="/game", tags=["Stackelberg Game"])

# Attacker type string → enum mapping
_TYPE_MAP = {
    "strategic":     game_engine.AttackerType.STRATEGIC,
    "opportunistic": game_engine.AttackerType.OPPORTUNISTIC,
    "state_actor":   game_engine.AttackerType.STATE_ACTOR,
}

_DEFAULT_PRIORS: Dict[game_engine.AttackerType, float] = {
    game_engine.AttackerType.STRATEGIC:     0.50,
    game_engine.AttackerType.OPPORTUNISTIC: 0.30,
    game_engine.AttackerType.STATE_ACTOR:   0.20,
}


@router.post(
    "/solve",
    response_model=GameResponse,
    summary="Solve the Bayesian Stackelberg Security Equilibrium",
    description=(
        "Builds a synthetic pipeline network, attaches physics-informed P_f values, "
        "then solves the Bayesian Stackelberg Security Game via LP enumeration (DOBSS). "
        "Returns the optimal defender coverage allocation c_i* for each segment, "
        "the attacker's best-response mixed strategy, and the expected risk reduction. "
        "Thesis result: 17.0% risk reduction vs. uniform allocation at B=0.40."
    ),
)
def solve_game(req: GameRequest) -> GameResponse:
    # --- build / validate attacker priors ---
    if req.attacker_priors is not None:
        total = sum(p.prior for p in req.attacker_priors)
        if abs(total - 1.0) > 1e-6:
            raise HTTPException(
                status_code=422,
                detail=f"Attacker priors must sum to 1.0 (got {total:.4f}).",
            )
        unknown = [p.attacker_type for p in req.attacker_priors if p.attacker_type not in _TYPE_MAP]
        if unknown:
            raise HTTPException(
                status_code=422,
                detail=f"Unknown attacker_type(s): {unknown}. "
                       f"Valid values: {list(_TYPE_MAP.keys())}.",
            )
        priors = {_TYPE_MAP[p.attacker_type]: p.prior for p in req.attacker_priors}
    else:
        priors = _DEFAULT_PRIORS

    try:
        # Build synthetic network and attach P_f
        net = net_engine.PipelineNetwork(name="psip-api-network")
        net.generate_synthetic(
            n_nodes=req.n_nodes, n_segments=req.n_segments, seed=req.random_seed,
        )
        net.attach_pf_values()

        # Build game config  (takes PipelineNetwork instance, not net.graph)
        targets = game_engine.build_target_nodes_from_network(net)
        profiles = [
            game_engine.AttackerProfile(attacker_type=atype, prior_prob=prob)
            for atype, prob in priors.items()
        ]
        config = game_engine.GameConfig(
            targets=targets,
            attacker_profiles=profiles,
            budget_fraction=req.budget,
        )
        solution = game_engine.solve_bayesian_stackelberg(config)

    except (ValueError, RuntimeError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    # Top-3 defended segments
    top3 = sorted(
        solution.coverage_by_id.items(), key=lambda x: x[1], reverse=True
    )[:3]

    return GameResponse(
        equilibrium_type=solution.equilibrium_type,
        budget_used=round(solution.budget_used, 6),
        defender_utility=round(solution.defender_utility, 6),
        attacker_utility=round(solution.attacker_utility, 6),
        coverage_effectiveness=round(solution.coverage_effectiveness * 100, 2),
        coverage_by_segment={k: round(v, 4) for k, v in solution.coverage_by_id.items()},
        attacker_strategy={k: round(v, 4) for k, v in solution.attacker_strategy.items()},
        top_3_defended=[seg_id for seg_id, _ in top3],
        n_segments=len(solution.coverage_by_id),
    )
