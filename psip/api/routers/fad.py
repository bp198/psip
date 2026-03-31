"""
/api/fad — BS 7910:2019 Level 2 Failure Assessment Diagram router.
"""

from fastapi import APIRouter, HTTPException

import psip.fad as fad_engine
from psip.api.models import FADRequest, FADResponse

router = APIRouter(prefix="/fad", tags=["FAD Assessment"])


@router.post(
    "/assess",
    response_model=FADResponse,
    summary="Run a BS 7910:2019 Level 2 FAD assessment",
    description=(
        "Accepts weld flaw geometry, material properties, and operating pressure. "
        "Returns the (Kr, Lr) assessment point on the Failure Assessment Diagram, "
        "whether the flaw is acceptable, and the reserve factor (safety margin). "
        "Based on BS 7910:2019 Option 1 FAD curve (Eq. 7.26–7.28)."
    ),
)
def assess_flaw(req: FADRequest) -> FADResponse:
    try:
        mat = fad_engine.MaterialProperties(
            sigma_y=req.sigma_y,
            sigma_u=req.sigma_u,
            E=req.E,
            K_mat=req.K_mat,
        )
        pipe = fad_engine.PipeGeometry(
            outer_diameter=req.outer_diameter,
            wall_thickness=req.wall_thickness,
        )
        flaw = fad_engine.FlawGeometry(a=req.flaw_depth, two_c=req.flaw_length)
        weld = fad_engine.WeldJoint(
            weld_type=req.weld_type,
            fat_class=req.fat_class,
            scf=req.scf,
        )
        result = fad_engine.assess_flaw(
            mat=mat, pipe=pipe, flaw=flaw, weld=weld, pressure=req.pressure
        )
    except (ValueError, ZeroDivisionError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    return FADResponse(
        is_acceptable=result.is_acceptable,
        Kr=round(result.Kr, 6),
        Lr=round(result.Lr, 6),
        f_Lr=round(result.f_Lr, 6),
        Lr_max=round(result.Lr_max, 6),
        reserve_factor=round(result.reserve_factor, 4),
        assessment_point={"Lr": round(result.Lr, 6), "Kr": round(result.Kr, 6)},
    )
