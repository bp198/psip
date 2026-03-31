"""
/api/network — Pipeline network summary router.
"""
from __future__ import annotations

import numpy as np
from fastapi import APIRouter, Query
from psip.api.models import NetworkSummaryResponse
import psip.network as net_engine

router = APIRouter(prefix="/network", tags=["Network"])


@router.get(
    "/summary",
    response_model=NetworkSummaryResponse,
    summary="Get a summary of the Gulf Coast synthetic pipeline network",
    description=(
        "Generates the synthetic Gulf Coast pipeline network (default: 20 nodes, 22 segments), "
        "computes physics-informed P_f values for each segment, and returns summary statistics. "
        "This is the reference network used throughout the STRATEGOS thesis."
    ),
)
def network_summary(
    n_nodes: int    = Query(20, ge=4,  le=50,  description="Number of nodes."),
    n_segments: int = Query(22, ge=3,  le=100, description="Number of segments."),
    seed: int       = Query(42,                description="Random seed for reproducibility."),
) -> NetworkSummaryResponse:
    net = net_engine.PipelineNetwork(name=f"gulf-coast-{n_nodes}n-{n_segments}s")
    net.generate_synthetic(n_nodes=n_nodes, n_segments=n_segments, seed=seed)
    net.attach_pf_values()

    summary = net.summary()
    pf_values = [
        float(data.get("P_f", 0.0))
        for _, _, data in net.graph.edges(data=True)
        if data.get("P_f") is not None
    ]

    segments_list = []
    for u, v, data in net.graph.edges(data=True):
        seg: net_engine.PipeSegment = data.get("segment")
        segments_list.append({
            "id":                data.get("segment_id", "%s-%s" % (u, v)),
            "from_node":         u,
            "to_node":           v,
            "length_km":         round(seg.length_km if seg else 0.0, 2),
            "P_f":               round(float(data.get("P_f", 0.0)), 4),
            "P_f_lower":         round(float(data.get("P_f_lower", 0.0)), 4),
            "P_f_upper":         round(float(data.get("P_f_upper", 0.0)), 4),
            "material":          seg.material if seg else "unknown",
            "outer_diameter_mm": seg.outer_diameter if seg else 0.0,
            "wall_thickness_mm": seg.wall_thickness if seg else 0.0,
        })

    return NetworkSummaryResponse(
        name=net.name,
        n_nodes=summary.get("n_nodes", n_nodes),
        n_segments=summary.get("n_edges", n_segments),
        total_length_km=round(summary.get("total_length_km", 0.0), 2),
        pf_min=round(float(np.min(pf_values)) if pf_values else 0.0, 4),
        pf_max=round(float(np.max(pf_values)) if pf_values else 0.0, 4),
        pf_mean=round(float(np.mean(pf_values)) if pf_values else 0.0, 4),
        segments=segments_list,
    )
