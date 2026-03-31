"""
/api/mc — Monte Carlo failure probability router.
"""
from fastapi import APIRouter, HTTPException
from psip.api.models import MCRequest, MCResponse
import psip.fad as fad_engine
import psip.mc as mc_engine

router = APIRouter(prefix="/mc", tags=["Monte Carlo Simulation"])


def _risk_level(pf: float) -> str:
    if pf < 0.10:
        return "LOW"
    elif pf < 0.50:
        return "MEDIUM"
    return "HIGH"


@router.post(
    "/simulate",
    response_model=MCResponse,
    summary="Monte Carlo failure probability simulation",
    description=(
        "Runs N Monte Carlo trials over the BS 7910 FAD framework, "
        "sampling flaw size, material toughness, yield strength, and operating pressure "
        "from calibrated distributions (PHMSA 1,996-record dataset, 2010–2024). "
        "Returns P_f with 95% Wilson confidence interval. "
        "Thesis calibration: 10,000 simulations per segment, P_f range [0.29, 0.93]."
    ),
)
def simulate(req: MCRequest) -> MCResponse:
    try:
        defaults = mc_engine.default_distributions_api5l_x65()
        pipe = fad_engine.PipeGeometry(
            outer_diameter=req.outer_diameter, wall_thickness=req.wall_thickness,
        )
        weld = fad_engine.WeldJoint(weld_type="butt", fat_class=71, scf=req.scf)
        config = mc_engine.PipelineSegmentConfig(
            segment_id=req.segment_id,
            pipe=pipe,
            weld=weld,
            **defaults,
        )
        result = mc_engine.monte_carlo_Pf(
            config=config,
            n_simulations=req.n_simulations,
            seed=req.random_seed,
        )
    except (ValueError, RuntimeError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    return MCResponse(
        segment_id=result.segment_id,
        n_simulations=result.n_simulations,
        n_failures=result.n_failures,
        P_f=round(result.P_f, 6),
        P_f_lower=round(result.P_f_lower, 6),
        P_f_upper=round(result.P_f_upper, 6),
        mean_Kr=round(result.mean_Kr, 4),
        mean_Lr=round(result.mean_Lr, 4),
        mean_reserve=round(result.mean_reserve, 4),
        risk_level=_risk_level(result.P_f),
    )
