"""
psip.network.entsog — ENTSOG Transparency Platform network adapter.

Converts ENTSOG GeoJSON corridor data into a :class:`PipelineNetwork` ready
for the PSIP game-theory and FAD engines.

Two entry points:

    1. **Generic adapter** — parse any GeoJSON FeatureCollection whose features
       carry LineString (segments) or Point (stations) geometries::

           net = entsog_geojson_to_network(geojson_dict, name="TAP")

    2. **TAP convenience builder** — constructs the Trans-Adriatic Pipeline
       network from hardcoded authoritative parameters (no external data needed)::

           net = build_tap_network()
           net.attach_pf_values()   # runs Monte Carlo P_f on every segment

ENTSOG GeoJSON conventions
--------------------------
The ENTSOG Transparency Platform (https://transparency.entsog.eu) publishes
gas-network corridors as GeoJSON FeatureCollections.  Each Feature may carry
the following optional properties that this adapter maps to PipeSegment fields:

    pointKey / id / name        → segment_id / node names
    outerDiameter               → diameter_mm  (mm)
    wallThickness               → wall_mm      (mm)
    designPressure              → maop_mpa     (MPa)
    grade / steelGrade          → grade        (API 5L string)
    lengthKm / length           → length_km    (km)
    yearInstalled               → year_installed
    seamType                    → seam_type    (string → SeamType)
    classLocation               → class_location

If a property is absent a sensible TAP-calibrated default is used.

Author: Babak Pirzadi (STRATEGOS Thesis)
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from src.zone_c.network.pipeline_graph import (
    API5L_GRADES,
    PHMSA_SEAM_MAP,
    NodeType,
    PipelineNetwork,
    PipeSegment,
    SeamType,
)

# ---------------------------------------------------------------------------
# TAP authoritative parameters (ILF / TAP AG public data)
# ---------------------------------------------------------------------------

#: TAP pipeline physical defaults (all sections use X70 DSAW pipe)
TAP_DEFAULTS: Dict[str, Any] = {
    "diameter_mm": 1219.2,       # 48-inch OD
    "wall_mm": 18.3,             # minimum wall (onshore sections)
    "grade": "X70",
    "seam_type": SeamType.DSAW,
    "year_installed": 2020,
    "design_pressure_mpa": 9.5,
    "class_location": 1,
    "design_factor": 0.72,
}

#: TAP route nodes: (node_id, name, NodeType, lat, lon)
TAP_NODES: List[Tuple[str, str, NodeType, float, float]] = [
    ("TAP-N01", "Kipoi (IGB Interconnection)",    NodeType.SOURCE,     41.1290, 26.3180),
    ("TAP-N02", "Komotini Compressor Station",    NodeType.COMPRESSOR, 41.1225, 25.4100),
    ("TAP-N03", "Kavala Compressor Station",      NodeType.COMPRESSOR, 40.9393, 24.4020),
    ("TAP-N04", "Thessaloniki Metering Station",  NodeType.VALVE,      40.6830, 22.9440),
    ("TAP-N05", "Florina Junction",               NodeType.JUNCTION,   40.7790, 21.4100),
    ("TAP-N06", "Bilisht (GR/AL Border)",         NodeType.JUNCTION,   40.6267, 20.9927),
    ("TAP-N07", "Korce Junction",                 NodeType.JUNCTION,   40.6154, 20.7699),
    ("TAP-N08", "Gramsh Compressor Station",      NodeType.COMPRESSOR, 40.8668, 20.1786),
    ("TAP-N09", "Fier Junction",                  NodeType.JUNCTION,   40.7228, 19.5563),
    ("TAP-N10", "Seman (Onshore/Offshore)",       NodeType.VALVE,      40.7050, 19.4220),
    ("TAP-N11", "Adriatic Crossing Midpoint",     NodeType.JUNCTION,   40.6280, 17.9640),
    ("TAP-N12", "San Foca Landfall (Italy)",      NodeType.VALVE,      40.2910, 18.4230),
    ("TAP-N13", "Melendugno Terminal",            NodeType.DELIVERY,   40.2703, 18.3302),
]

#: TAP route segments: (from_node, to_node, segment_id, length_km, wall_mm, notes)
TAP_SEGMENTS: List[Tuple[str, str, str, float, float, str]] = [
    ("TAP-N01", "TAP-N02", "TAP-SEG-001",  95.0, 18.3, "Greece: Kipoi–Komotini"),
    ("TAP-N02", "TAP-N03", "TAP-SEG-002",  98.0, 18.3, "Greece: Komotini–Kavala"),
    ("TAP-N03", "TAP-N04", "TAP-SEG-003",  82.0, 18.3, "Greece: Kavala–Thessaloniki"),
    ("TAP-N04", "TAP-N05", "TAP-SEG-004",  89.0, 18.3, "Greece: Thessaloniki–Florina"),
    ("TAP-N05", "TAP-N06", "TAP-SEG-005",  61.0, 18.3, "Greece: Florina–Bilisht border"),
    ("TAP-N06", "TAP-N07", "TAP-SEG-006",  38.0, 18.3, "Albania: Bilisht–Korce"),
    ("TAP-N07", "TAP-N08", "TAP-SEG-007",  72.0, 19.1, "Albania: Korce–Gramsh"),
    ("TAP-N08", "TAP-N09", "TAP-SEG-008",  66.0, 19.1, "Albania: Gramsh–Fier"),
    ("TAP-N09", "TAP-N10", "TAP-SEG-009",  26.0, 19.1, "Albania: Fier–Seman"),
    ("TAP-N10", "TAP-N11", "TAP-SEG-010",  58.0, 22.2, "Offshore: Seman–Adriatic mid"),
    ("TAP-N11", "TAP-N12", "TAP-SEG-011",  47.0, 22.2, "Offshore: Adriatic mid–San Foca"),
    ("TAP-N12", "TAP-N13", "TAP-SEG-012",   8.0, 18.3, "Italy: San Foca–Melendugno"),
]


# ---------------------------------------------------------------------------
# ENTSOG property mapping helpers
# ---------------------------------------------------------------------------

_ENTSOG_SEAM_MAP: Dict[str, SeamType] = {
    **PHMSA_SEAM_MAP,
    # ENTSOG-specific strings
    "DSAW": SeamType.DSAW,
    "LSAW": SeamType.DSAW,
    "ERW": SeamType.ERW_HF,
    "SEAMLESS": SeamType.SEAMLESS,
    "SPIRAL": SeamType.SPIRAL,
    "UNKNOWN": SeamType.UNKNOWN,
}


def _parse_seam(value: Optional[str]) -> SeamType:
    if not value:
        return SeamType.DSAW  # TAP default
    return _ENTSOG_SEAM_MAP.get(str(value).strip().upper(), SeamType.UNKNOWN)


def _parse_grade(value: Optional[str]) -> Tuple[str, float, float]:
    """Return (grade_string, smys_mpa, uts_mpa)."""
    if not value:
        grade = "X70"
    else:
        grade = str(value).strip().upper().replace("API5L", "").strip()
        if not grade.startswith("X") and not grade.startswith("B"):
            grade = "X70"
    smys, uts = API5L_GRADES.get(grade, API5L_GRADES["X70"])
    return grade, smys, uts


def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance in km between two (lat, lon) points."""
    r = 6371.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlam = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlam / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))


def _coord_to_node_id(lon: float, lat: float, precision: int = 3) -> str:
    """Stable node ID from rounded coordinates."""
    return f"N_{lat:.{precision}f}_{lon:.{precision}f}".replace("-", "m").replace(".", "p")


def _linestring_midpoint(coords: List[List[float]]) -> Tuple[float, float]:
    """Return (lat, lon) midpoint of a LineString coordinate array."""
    lons = [c[0] for c in coords]
    lats = [c[1] for c in coords]
    return sum(lats) / len(lats), sum(lons) / len(lons)


def _linestring_length_km(coords: List[List[float]]) -> float:
    """Approximate great-circle length of a LineString in km."""
    total = 0.0
    for i in range(len(coords) - 1):
        lon1, lat1 = coords[i][0], coords[i][1]
        lon2, lat2 = coords[i + 1][0], coords[i + 1][1]
        total += _haversine_km(lat1, lon1, lat2, lon2)
    return total


# ---------------------------------------------------------------------------
# Generic GeoJSON → PipelineNetwork converter
# ---------------------------------------------------------------------------


def entsog_geojson_to_network(
    geojson: Dict[str, Any],
    name: str = "entsog_network",
    default_diameter_mm: float = 1219.2,
    default_wall_mm: float = 18.3,
    default_grade: str = "X70",
    default_pressure_mpa: float = 9.5,
    default_year: int = 2020,
) -> PipelineNetwork:
    """Convert an ENTSOG GeoJSON FeatureCollection to a :class:`PipelineNetwork`.

    The function handles two feature geometry types:

    * **LineString / MultiLineString** — treated as pipeline segments (edges).
      Endpoint coordinates become network nodes automatically.
    * **Point** — treated as named stations (nodes). If a point's coordinates
      coincide with a LineString endpoint it enriches that node's metadata.

    Parameters
    ----------
    geojson:
        Parsed GeoJSON dict (``{"type": "FeatureCollection", "features": [...]}``)
        as returned by ``json.load()``.
    name:
        Name assigned to the :class:`PipelineNetwork` instance.
    default_diameter_mm:
        Fallback outer diameter (mm) when the feature has no ``outerDiameter``
        property.
    default_wall_mm:
        Fallback wall thickness (mm).
    default_grade:
        Fallback API 5L grade string (e.g., ``"X70"``).
    default_pressure_mpa:
        Fallback design pressure / MAOP (MPa).
    default_year:
        Fallback installation year.

    Returns
    -------
    PipelineNetwork
        Populated network.  Call ``net.attach_pf_values()`` to compute Monte
        Carlo P_f for each segment.

    Raises
    ------
    ValueError
        If *geojson* is not a valid FeatureCollection.
    """
    if geojson.get("type") != "FeatureCollection":
        raise ValueError("geojson must be a GeoJSON FeatureCollection")

    net = PipelineNetwork(name=name)
    features = geojson.get("features", [])

    # --- Pass 1: register Point features as named nodes ---
    point_nodes: Dict[str, dict] = {}  # coord_key → {name, node_type}
    for feat in features:
        geom = feat.get("geometry", {}) or {}
        if geom.get("type") != "Point":
            continue
        props = feat.get("properties") or {}
        lon, lat = geom["coordinates"][:2]
        key = _coord_to_node_id(lon, lat)
        raw_name = (
            props.get("name")
            or props.get("pointKey")
            or props.get("id")
            or key
        )
        raw_type = str(props.get("nodeType") or props.get("type") or "junction").lower()
        node_type_map = {
            "source": NodeType.SOURCE,
            "compressor": NodeType.COMPRESSOR,
            "valve": NodeType.VALVE,
            "delivery": NodeType.DELIVERY,
            "storage": NodeType.STORAGE,
            "junction": NodeType.JUNCTION,
        }
        node_type = node_type_map.get(raw_type, NodeType.JUNCTION)
        point_nodes[key] = {"name": raw_name, "node_type": node_type, "lat": lat, "lon": lon}

    # --- Pass 2: build edges from LineString features ---
    segment_counter = 0
    for feat in features:
        geom = feat.get("geometry", {}) or {}
        gtype = geom.get("type", "")

        if gtype == "LineString":
            coord_lists = [geom["coordinates"]]
        elif gtype == "MultiLineString":
            coord_lists = geom["coordinates"]
        else:
            continue

        props = feat.get("properties") or {}

        # -- Parse physical properties from ENTSOG properties --
        diam = float(props.get("outerDiameter") or default_diameter_mm)
        wall = float(props.get("wallThickness") or default_wall_mm)
        maop = float(props.get("designPressure") or props.get("maop") or default_pressure_mpa)
        year = int(props.get("yearInstalled") or props.get("year") or default_year)
        seam = _parse_seam(props.get("seamType") or props.get("seam"))
        raw_grade = props.get("grade") or props.get("steelGrade") or default_grade
        grade, smys, uts = _parse_grade(raw_grade)

        for coords in coord_lists:
            if len(coords) < 2:
                continue

            segment_counter += 1
            seg_id = (
                str(props.get("pointKey") or props.get("id") or "")
                or f"ENTSOG-SEG-{segment_counter:04d}"
            )

            # Endpoint node IDs
            u_lon, u_lat = coords[0][0], coords[0][1]
            v_lon, v_lat = coords[-1][0], coords[-1][1]
            u_id = _coord_to_node_id(u_lon, u_lat)
            v_id = _coord_to_node_id(v_lon, v_lat)

            # Register nodes if not already present
            for nid, nlat, nlon in ((u_id, u_lat, u_lon), (v_id, v_lat, v_lon)):
                if nid not in net.graph:
                    pt = point_nodes.get(nid, {})
                    net.add_node(
                        nid,
                        node_type=pt.get("node_type", NodeType.JUNCTION),
                        lat=round(nlat, 4),
                        lon=round(nlon, 4),
                        name=pt.get("name", nid),
                    )

            # Segment length: from property or computed from geometry
            length_km = float(
                props.get("lengthKm") or props.get("length") or 0.0
            ) or _linestring_length_km(coords)

            # Midpoint coordinates
            mid_lat, mid_lon = _linestring_midpoint(coords)

            segment = PipeSegment(
                segment_id=seg_id,
                diameter_mm=diam,
                wall_mm=wall,
                smys_mpa=round(smys, 1),
                uts_mpa=round(uts, 1),
                length_km=round(length_km, 2),
                seam_type=seam,
                grade=grade,
                year_installed=year,
                class_location=int(props.get("classLocation") or 1),
                design_factor=float(props.get("designFactor") or 0.72),
                maop_mpa=maop,
                lat=round(mid_lat, 4),
                lon=round(mid_lon, 4),
            )
            net.add_segment(u_id, v_id, segment)

    return net


# ---------------------------------------------------------------------------
# File / URL loader
# ---------------------------------------------------------------------------


def load_entsog_geojson(source: Union[str, Path]) -> Dict[str, Any]:
    """Load a GeoJSON dict from a local file path or HTTPS URL.

    Parameters
    ----------
    source:
        File path (``str`` or :class:`pathlib.Path`) or an ``https://`` URL
        pointing to a GeoJSON resource on the ENTSOG Transparency Platform.

    Returns
    -------
    dict
        Parsed GeoJSON FeatureCollection.

    Raises
    ------
    FileNotFoundError
        If *source* is a local path and the file does not exist.
    ValueError
        If the fetched content cannot be parsed as JSON.
    """
    source = str(source)
    if source.startswith("http://") or source.startswith("https://"):
        import urllib.request

        with urllib.request.urlopen(source, timeout=30) as resp:  # noqa: S310
            raw = resp.read().decode("utf-8")
    else:
        path = Path(source)
        if not path.exists():
            raise FileNotFoundError(f"GeoJSON file not found: {path}")
        raw = path.read_text(encoding="utf-8")

    try:
        return json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Cannot parse GeoJSON from {source!r}: {exc}") from exc


# ---------------------------------------------------------------------------
# TAP convenience builder  (no external data required)
# ---------------------------------------------------------------------------


def build_tap_network(attach_pf: bool = False, n_simulations: int = 5_000) -> PipelineNetwork:
    """Build the Trans-Adriatic Pipeline network from authoritative parameters.

    Constructs a 13-node, 12-segment directed graph representing the TAP
    corridor from the IGB interconnection at Kipoi (Greece) to the Melendugno
    terminal (Italy), passing through Albania and the Adriatic Sea.

    Pipeline specifications (TAP AG / ILF engineering data, 2020):
        - Diameter:   1219.2 mm (48 in) throughout
        - Grade:      API 5L X70 (SMYS = 482.6 MPa, UTS = 565.4 MPa)
        - Seam type:  DSAW (longitudinal submerged arc weld)
        - Wall:       18.3 mm onshore, 19.1 mm inland Albania, 22.2 mm offshore
        - MAOP:       9.5 MPa onshore / 10.5 MPa offshore
        - Year:       2020

    Parameters
    ----------
    attach_pf:
        If ``True``, run Monte Carlo P_f for every segment before returning.
        Adds ~10–30 s depending on *n_simulations*.
    n_simulations:
        Number of Monte Carlo trials per segment (used only if
        *attach_pf* is ``True``).

    Returns
    -------
    PipelineNetwork
        Populated TAP network ready for game-theory analysis.

    Example
    -------
    >>> from psip.network.entsog import build_tap_network
    >>> net = build_tap_network(attach_pf=True)
    >>> print(net.summary())
    """
    smys, uts = API5L_GRADES["X70"]
    net = PipelineNetwork("TAP_Trans_Adriatic_Pipeline")

    # Build coord lookup: node_id → (lat, lon)
    _node_coords = {nid: (lat, lon) for nid, _, _, lat, lon in TAP_NODES}

    # --- Add nodes ---
    for node_id, node_name, node_type, lat, lon in TAP_NODES:
        net.add_node(node_id, node_type=node_type, lat=lat, lon=lon, name=node_name)

    # --- Add segments ---
    for u, v, seg_id, length_km, wall_mm, _notes in TAP_SEGMENTS:
        # Offshore segments use higher MAOP
        is_offshore = "Offshore" in _notes or "Adriatic" in _notes
        maop = 10.5 if is_offshore else 9.5

        segment = PipeSegment(
            segment_id=seg_id,
            diameter_mm=TAP_DEFAULTS["diameter_mm"],
            wall_mm=wall_mm,
            smys_mpa=round(smys, 1),
            uts_mpa=round(uts, 1),
            length_km=length_km,
            seam_type=TAP_DEFAULTS["seam_type"],
            grade="X70",
            year_installed=TAP_DEFAULTS["year_installed"],
            class_location=TAP_DEFAULTS["class_location"],
            design_factor=TAP_DEFAULTS["design_factor"],
            maop_mpa=maop,
            lat=round((_node_coords[u][0] + _node_coords[v][0]) / 2, 4),
            lon=round((_node_coords[u][1] + _node_coords[v][1]) / 2, 4),
        )
        net.add_segment(u, v, segment)

    if attach_pf:
        net.attach_pf_values(n_simulations=n_simulations)

    return net
