"""
Pipeline Network Graph Model
==============================

Builds a weighted, directed graph representing a gas transmission pipeline
network where:
    - **Nodes** = compressor stations, valve stations, delivery points, junctions
    - **Edges** = pipeline segments with physics-informed vulnerability attributes

Each edge carries:
    - Physical properties:  diameter, wall thickness, SMYS grade, seam type, length
    - Vulnerability:        P_f from the calibrated Monte Carlo engine (Sprint 1)
    - Game payoff:          P_f serves as the attacker payoff; (1 - P_f) as defender value

The graph supports:
    - Loading from PHMSA annual report mileage data (aggregated by state/operator)
    - Generation of realistic synthetic topologies calibrated to PHMSA fleet statistics
    - GeoJSON import for real geographic coordinates (when available)
    - Attachment of Sprint 1 calibrated P_f values based on segment properties

This module is Layer 4 in the thesis architecture:
    Physics (L1-L3) → Network (L4) → Game Engine (L5)

Author: Babak Pirzadi (STRATEGOS Thesis)
"""

from __future__ import annotations

import numpy as np
import networkx as nx
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Tuple
from enum import Enum


# ---------------------------------------------------------------------------
# Enumerations & Data Types
# ---------------------------------------------------------------------------

class NodeType(Enum):
    """Classification of network nodes."""
    SOURCE = "source"                   # Supply/production point
    COMPRESSOR = "compressor"           # Compressor station
    VALVE = "valve"                     # Block valve / mainline valve
    JUNCTION = "junction"               # Pipe tee / branch point
    DELIVERY = "delivery"               # City gate / delivery point
    STORAGE = "storage"                 # Storage facility


class SeamType(Enum):
    """Pipe seam type classification (from PHMSA PIPE_SEAM_TYPE)."""
    SEAMLESS = "seamless"
    DSAW = "dsaw"
    ERW_HF = "erw_hf"
    ERW_LF = "erw_lf"
    ERW_UNK = "erw_unknown"
    FLASH_WELDED = "flash_welded"
    LAP_WELDED = "lap_welded"
    SPIRAL = "spiral"
    SINGLE_SAW = "single_saw"
    FURNACE_BUTT = "furnace_butt"
    UNKNOWN = "unknown"


# Mapping from SeamType to the SCF key used in calibrated_params.py
SEAM_TO_SCF_KEY = {
    SeamType.SEAMLESS:      "seamless",
    SeamType.DSAW:          "dsaw_seam",
    SeamType.ERW_HF:        "erw_hf_seam",
    SeamType.ERW_LF:        "erw_lf_seam",
    SeamType.ERW_UNK:       "butt_weld_as_welded",
    SeamType.FLASH_WELDED:  "butt_weld_as_welded",
    SeamType.LAP_WELDED:    "lap_welded",
    SeamType.SPIRAL:        "erw_lf_seam",
    SeamType.SINGLE_SAW:    "dsaw_seam",
    SeamType.FURNACE_BUTT:  "lap_welded",
    SeamType.UNKNOWN:       "girth_weld_field",
}

# PHMSA seam type string → SeamType
PHMSA_SEAM_MAP = {
    "SEAMLESS":                             SeamType.SEAMLESS,
    "DSAW":                                 SeamType.DSAW,
    "LONGITUDINAL ERW - HIGH FREQUENCY":    SeamType.ERW_HF,
    "LONGITUDINAL ERW - LOW FREQUENCY":     SeamType.ERW_LF,
    "LONGITUDINAL ERW - UNKNOWN FREQUENCY": SeamType.ERW_UNK,
    "FLASH WELDED":                         SeamType.FLASH_WELDED,
    "LAP WELDED":                           SeamType.LAP_WELDED,
    "SPIRAL WELDED":                        SeamType.SPIRAL,
    "SINGLE SAW":                           SeamType.SINGLE_SAW,
    "FURNACE BUTT WELDED":                  SeamType.FURNACE_BUTT,
}

# Mapping SeamType → IIW FAT class (for weld_type in fad_engine)
SEAM_TO_FAT_CLASS = {
    SeamType.SEAMLESS:      125,
    SeamType.DSAW:          90,
    SeamType.ERW_HF:        90,
    SeamType.ERW_LF:        71,
    SeamType.ERW_UNK:       80,
    SeamType.FLASH_WELDED:  80,
    SeamType.LAP_WELDED:    63,
    SeamType.SPIRAL:        71,
    SeamType.SINGLE_SAW:    90,
    SeamType.FURNACE_BUTT:  63,
    SeamType.UNKNOWN:       71,
}

# SMYS grade mapping: API 5L grade string → (SMYS_MPa, UTS_MPa_approx)
API5L_GRADES = {
    "B":   (241.3, 413.7),
    "X42": (289.6, 413.7),
    "X46": (317.2, 434.4),
    "X52": (358.5, 455.1),
    "X56": (386.1, 489.5),
    "X60": (413.7, 517.1),
    "X65": (448.2, 530.9),
    "X70": (482.6, 565.4),
    "X80": (551.6, 620.5),
}


# ---------------------------------------------------------------------------
# Edge Attributes Data Class
# ---------------------------------------------------------------------------

@dataclass
class PipeSegment:
    """Physical and vulnerability attributes for a pipeline edge.

    Attributes:
        segment_id:     Unique identifier
        diameter_mm:    Outer diameter in mm
        wall_mm:        Wall thickness in mm
        smys_mpa:       Specified Minimum Yield Strength in MPa
        uts_mpa:        Ultimate Tensile Strength in MPa (estimated)
        length_km:      Segment length in kilometers
        seam_type:      Pipe seam classification
        grade:          API 5L grade string (e.g., "X52")
        year_installed: Year of manufacture/installation
        class_location: ASME B31.8 class (1, 2, 3, or 4)
        design_factor:  ASME B31.8 design factor (0.72 for Class 1)
        maop_mpa:       Estimated MAOP in MPa
        P_f:            Probability of failure (from MC engine)
        P_f_ci:         95% CI on P_f as (lower, upper)
        scf_key:        SCF distribution key from calibrated_params
        fat_class:      IIW FAT class number
        lat:            Midpoint latitude (optional, for GIS)
        lon:            Midpoint longitude (optional, for GIS)
    """
    segment_id: str
    diameter_mm: float
    wall_mm: float
    smys_mpa: float
    uts_mpa: float
    length_km: float
    seam_type: SeamType = SeamType.UNKNOWN
    grade: str = "X52"
    year_installed: int = 1970
    class_location: int = 1
    design_factor: float = 0.72
    maop_mpa: float = 0.0
    P_f: float = 0.0
    P_f_ci: Tuple[float, float] = (0.0, 0.0)
    scf_key: str = "girth_weld_field"
    fat_class: int = 71
    lat: Optional[float] = None
    lon: Optional[float] = None

    def __post_init__(self):
        """Compute derived properties."""
        # SCF key from seam type
        self.scf_key = SEAM_TO_SCF_KEY.get(self.seam_type, "girth_weld_field")
        self.fat_class = SEAM_TO_FAT_CLASS.get(self.seam_type, 71)
        # MAOP via Barlow: MAOP = 2 * SMYS * t * F / D
        if self.maop_mpa == 0.0 and self.diameter_mm > 0 and self.wall_mm > 0:
            self.maop_mpa = (
                2 * self.smys_mpa * self.wall_mm * self.design_factor
                / self.diameter_mm
            )

    def pipe_age(self, ref_year: int = 2025) -> int:
        """Return pipe age in years."""
        return ref_year - self.year_installed

    def to_dict(self) -> dict:
        """Serialize to dict for NetworkX edge data."""
        return {
            "segment_id": self.segment_id,
            "diameter_mm": self.diameter_mm,
            "wall_mm": self.wall_mm,
            "smys_mpa": self.smys_mpa,
            "uts_mpa": self.uts_mpa,
            "length_km": self.length_km,
            "seam_type": self.seam_type.value,
            "grade": self.grade,
            "year_installed": self.year_installed,
            "class_location": self.class_location,
            "design_factor": self.design_factor,
            "maop_mpa": round(self.maop_mpa, 2),
            "P_f": self.P_f,
            "P_f_lower": self.P_f_ci[0],
            "P_f_upper": self.P_f_ci[1],
            "scf_key": self.scf_key,
            "fat_class": self.fat_class,
            "lat": self.lat,
            "lon": self.lon,
        }


# ---------------------------------------------------------------------------
# PHMSA Fleet Statistics (for realistic synthetic generation)
# ---------------------------------------------------------------------------

# Empirical diameter distribution from PHMSA incident data (inches → mm)
PHMSA_DIAMETER_DIST = {
    # diameter_inch: (weight_fraction, typical_wall_inch)
    6:  (0.047, 0.188),
    8:  (0.112, 0.219),
    10: (0.041, 0.250),
    12: (0.146, 0.250),
    16: (0.111, 0.281),
    20: (0.107, 0.312),
    24: (0.071, 0.375),
    30: (0.085, 0.375),
    36: (0.048, 0.438),
}

# Empirical seam type distribution from PHMSA (fractions)
PHMSA_SEAM_DIST = {
    SeamType.SEAMLESS:      0.215,
    SeamType.ERW_UNK:       0.214,
    SeamType.DSAW:          0.144,
    SeamType.ERW_HF:        0.130,
    SeamType.ERW_LF:        0.101,
    SeamType.FLASH_WELDED:  0.060,
    SeamType.LAP_WELDED:    0.022,
    SeamType.SINGLE_SAW:    0.014,
}

# Empirical SMYS distribution from PHMSA (psi → weight)
PHMSA_SMYS_DIST = {
    # smys_psi: weight
    35000: 0.153,
    42000: 0.198,
    46000: 0.051,
    52000: 0.290,
    60000: 0.110,
    65000: 0.030,
    70000: 0.019,
}

# Empirical installation year distribution (decade weights)
PHMSA_DECADE_DIST = {
    1950: 0.12, 1960: 0.22, 1970: 0.20, 1980: 0.12,
    1990: 0.10, 2000: 0.10, 2010: 0.08, 2020: 0.06,
}


# ---------------------------------------------------------------------------
# Network Builder
# ---------------------------------------------------------------------------

class PipelineNetwork:
    """Pipeline network graph with physics-informed vulnerability attributes.

    The graph is a NetworkX DiGraph where:
        - Nodes have 'type' (NodeType), 'lat', 'lon', 'name' attributes
        - Edges have full PipeSegment attributes including P_f

    Usage:
        net = PipelineNetwork("Gulf_Coast_30_segment")
        net.generate_synthetic(n_nodes=20, n_segments=30, seed=42)
        net.attach_pf_values()  # Compute P_f for each segment
        G = net.graph  # NetworkX DiGraph
    """

    def __init__(self, name: str = "pipeline_network"):
        self.name = name
        self.graph = nx.DiGraph(name=name)
        self._segments: Dict[str, PipeSegment] = {}

    @property
    def n_nodes(self) -> int:
        return self.graph.number_of_nodes()

    @property
    def n_edges(self) -> int:
        return self.graph.number_of_edges()

    @property
    def total_length_km(self) -> float:
        return sum(d.get("length_km", 0) for _, _, d in self.graph.edges(data=True))

    @property
    def segments(self) -> Dict[str, PipeSegment]:
        return self._segments

    # --- Node management ---

    def add_node(
        self,
        node_id: str,
        node_type: NodeType = NodeType.JUNCTION,
        lat: Optional[float] = None,
        lon: Optional[float] = None,
        name: Optional[str] = None,
    ) -> None:
        """Add a node to the network."""
        self.graph.add_node(
            node_id,
            type=node_type.value,
            lat=lat,
            lon=lon,
            name=name or node_id,
        )

    # --- Edge management ---

    def add_segment(self, u: str, v: str, segment: PipeSegment) -> None:
        """Add a pipeline segment (edge) between two nodes."""
        self._segments[segment.segment_id] = segment
        self.graph.add_edge(u, v, **segment.to_dict())

    # --- Synthetic network generation ---

    def generate_synthetic(
        self,
        n_nodes: int = 20,
        n_segments: int = 30,
        seed: int = 42,
        region_bounds: Tuple[float, float, float, float] = (
            29.0, -96.0, 33.0, -88.0  # Gulf Coast: lat_min, lon_min, lat_max, lon_max
        ),
    ) -> None:
        """Generate a realistic synthetic pipeline network.

        Creates a connected graph with node types and edge properties sampled
        from PHMSA fleet statistics. The topology follows a spine-and-branch
        pattern typical of gas transmission networks.

        Args:
            n_nodes:        Number of nodes
            n_segments:     Number of pipeline segments (edges)
            seed:           Random seed
            region_bounds:  (lat_min, lon_min, lat_max, lon_max) for coordinates
        """
        rng = np.random.default_rng(seed)
        lat_min, lon_min, lat_max, lon_max = region_bounds

        # --- Generate nodes ---
        # Assign types: 2 sources, 3 compressors, rest split between junctions and deliveries
        node_types = []
        node_types.extend([NodeType.SOURCE] * 2)
        node_types.extend([NodeType.COMPRESSOR] * min(3, n_nodes - 2))
        remaining = n_nodes - len(node_types)
        n_delivery = max(2, remaining // 3)
        n_junction = remaining - n_delivery
        node_types.extend([NodeType.JUNCTION] * n_junction)
        node_types.extend([NodeType.DELIVERY] * n_delivery)
        rng.shuffle(node_types)

        # Generate coordinates along a rough east-west corridor
        lats = rng.uniform(lat_min, lat_max, n_nodes)
        lons = np.sort(rng.uniform(lon_min, lon_max, n_nodes))  # sort E→W for spine
        # Add some scatter
        lats = lats + rng.normal(0, 0.2, n_nodes)

        for i, (nt, lat, lon) in enumerate(zip(node_types, lats, lons)):
            node_id = f"N{i:03d}"
            self.add_node(node_id, nt, lat=round(lat, 4), lon=round(lon, 4),
                          name=f"{nt.value.title()}_{i}")

        node_ids = list(self.graph.nodes())

        # --- Generate spine (ensure connectivity) ---
        # Connect nodes in order of longitude to create a main transmission spine
        sorted_nodes = sorted(node_ids, key=lambda n: self.graph.nodes[n]["lon"])
        edges_added = set()

        for i in range(len(sorted_nodes) - 1):
            u, v = sorted_nodes[i], sorted_nodes[i + 1]
            seg = self._random_segment(u, v, rng, node_ids)
            self.add_segment(u, v, seg)
            edges_added.add((u, v))

        # --- Add branch connections up to n_segments ---
        attempts = 0
        while len(edges_added) < n_segments and attempts < n_segments * 5:
            u = rng.choice(node_ids)
            v = rng.choice(node_ids)
            if u != v and (u, v) not in edges_added and (v, u) not in edges_added:
                # Prefer shorter geographic distances (realistic)
                u_data = self.graph.nodes[u]
                v_data = self.graph.nodes[v]
                dist = np.sqrt(
                    (u_data["lat"] - v_data["lat"]) ** 2
                    + (u_data["lon"] - v_data["lon"]) ** 2
                )
                # Accept with probability inversely proportional to distance
                if rng.random() < np.exp(-dist * 1.5):
                    seg = self._random_segment(u, v, rng, node_ids)
                    self.add_segment(u, v, seg)
                    edges_added.add((u, v))
            attempts += 1

    def _random_segment(
        self,
        u: str,
        v: str,
        rng: np.random.Generator,
        node_ids: list,
    ) -> PipeSegment:
        """Create a random PipeSegment with PHMSA-calibrated properties."""
        seg_id = f"SEG_{u}_{v}"

        # Sample diameter
        diams = list(PHMSA_DIAMETER_DIST.keys())
        weights = [PHMSA_DIAMETER_DIST[d][0] for d in diams]
        weights = np.array(weights) / sum(weights)
        diam_inch = rng.choice(diams, p=weights)
        wall_inch = PHMSA_DIAMETER_DIST[diam_inch][1]
        # Add some variability
        wall_inch *= rng.uniform(0.9, 1.15)

        # Sample seam type
        seam_types = list(PHMSA_SEAM_DIST.keys())
        seam_weights = np.array([PHMSA_SEAM_DIST[s] for s in seam_types])
        seam_weights /= seam_weights.sum()
        seam_type = rng.choice(seam_types, p=seam_weights)

        # Sample SMYS
        smys_vals = list(PHMSA_SMYS_DIST.keys())
        smys_weights = np.array([PHMSA_SMYS_DIST[s] for s in smys_vals])
        smys_weights /= smys_weights.sum()
        smys_psi = rng.choice(smys_vals, p=smys_weights)
        smys_mpa = smys_psi * 0.00689476

        # UTS estimate
        uts_mpa = smys_mpa * rng.uniform(1.15, 1.30)

        # Grade string
        grade_map = {35000: "X42", 42000: "X42", 46000: "X46", 52000: "X52",
                     60000: "X60", 65000: "X65", 70000: "X70"}
        grade = grade_map.get(smys_psi, "X52")

        # Installation year
        decades = list(PHMSA_DECADE_DIST.keys())
        dec_weights = np.array([PHMSA_DECADE_DIST[d] for d in decades])
        dec_weights /= dec_weights.sum()
        decade = rng.choice(decades, p=dec_weights)
        year = int(decade + rng.integers(0, 10))

        # Segment length from geographic distance
        u_data = self.graph.nodes[u]
        v_data = self.graph.nodes[v]
        geo_dist_deg = np.sqrt(
            (u_data["lat"] - v_data["lat"]) ** 2
            + (u_data["lon"] - v_data["lon"]) ** 2
        )
        length_km = max(5.0, geo_dist_deg * 111.0 * rng.uniform(0.9, 1.3))

        # Class location (mostly Class 1 for transmission)
        class_loc = rng.choice([1, 1, 1, 2, 3], p=[0.6, 0.15, 0.10, 0.10, 0.05])
        design_factors = {1: 0.72, 2: 0.60, 3: 0.50, 4: 0.40}
        design_f = design_factors[class_loc]

        # Midpoint coordinates
        mid_lat = round((u_data["lat"] + v_data["lat"]) / 2, 4)
        mid_lon = round((u_data["lon"] + v_data["lon"]) / 2, 4)

        return PipeSegment(
            segment_id=seg_id,
            diameter_mm=round(diam_inch * 25.4, 1),
            wall_mm=round(wall_inch * 25.4, 2),
            smys_mpa=round(smys_mpa, 1),
            uts_mpa=round(uts_mpa, 1),
            length_km=round(length_km, 1),
            seam_type=seam_type,
            grade=grade,
            year_installed=year,
            class_location=class_loc,
            design_factor=design_f,
            lat=mid_lat,
            lon=mid_lon,
        )

    # --- P_f attachment ---

    def attach_pf_values(
        self,
        n_simulations: int = 10_000,
        seed: int = 42,
    ) -> None:
        """Compute and attach P_f to every edge using the Sprint 1 MC engine.

        For each segment, builds a PipelineSegmentConfig from the edge's
        physical properties and the calibrated distributions, then runs
        Monte Carlo to get P_f.
        """
        from ..physics.fad_engine import PipeGeometry, WeldJoint
        from ..physics.mc_failure_probability import (
            DistributionParams, PipelineSegmentConfig, monte_carlo_Pf,
        )
        from ..physics.calibrated_params import (
            DIST_DEFECT_A, DIST_DEFECT_2C, DIST_K_MAT,
            SCF_DISTRIBUTIONS,
        )

        for u, v, data in self.graph.edges(data=True):
            seg_id = data["segment_id"]
            diam = data["diameter_mm"]
            wall = data["wall_mm"]
            smys = data["smys_mpa"]
            uts = data["uts_mpa"]
            scf_key = data["scf_key"]
            fat = data["fat_class"]

            # Build configuration
            pipe = PipeGeometry(outer_diameter=diam, wall_thickness=wall)
            weld_type = "fillet" if fat <= 63 else "butt"
            weld = WeldJoint(weld_type=weld_type, fat_class=fat, scf=1.0)

            # Use calibrated SCF distribution; fall back to girth_weld_field
            scf_dist = SCF_DISTRIBUTIONS.get(
                scf_key,
                SCF_DISTRIBUTIONS.get("girth_weld_field",
                    DistributionParams(dist_type="uniform", param1=1.3, param2=2.5))
            )

            # Segment-specific yield/UTS distributions centered on this pipe's grade
            dist_sy = DistributionParams(
                dist_type="normal", param1=smys, param2=smys * 0.05,
                lower_bound=smys * 0.85,
            )
            dist_su = DistributionParams(
                dist_type="normal", param1=uts, param2=uts * 0.04,
                lower_bound=smys * 1.05,
            )

            # Pressure distribution from MAOP
            maop = data["maop_mpa"]
            dist_p = DistributionParams(
                dist_type="normal", param1=maop * 0.90,  # typical operating ~90% of MAOP
                param2=maop * 0.05,
                lower_bound=0.5,
            )

            config = PipelineSegmentConfig(
                segment_id=seg_id,
                pipe=pipe,
                weld=weld,
                dist_defect_a=DIST_DEFECT_A,
                dist_defect_2c=DIST_DEFECT_2C,
                dist_K_mat=DIST_K_MAT,
                dist_sigma_y=dist_sy,
                dist_sigma_u=dist_su,
                dist_pressure=dist_p,
                dist_scf=scf_dist,
            )

            result = monte_carlo_Pf(config, n_simulations=n_simulations, seed=seed)

            # Update edge attributes
            self.graph[u][v]["P_f"] = round(result.P_f, 4)
            self.graph[u][v]["P_f_lower"] = round(result.P_f_lower, 4)
            self.graph[u][v]["P_f_upper"] = round(result.P_f_upper, 4)

            # Update internal segment object
            if seg_id in self._segments:
                self._segments[seg_id].P_f = result.P_f
                self._segments[seg_id].P_f_ci = (result.P_f_lower, result.P_f_upper)

    # --- Summary & Export ---

    def summary(self) -> dict:
        """Return summary statistics of the network."""
        pf_values = [d["P_f"] for _, _, d in self.graph.edges(data=True) if d.get("P_f", 0) > 0]
        diameters = [d["diameter_mm"] for _, _, d in self.graph.edges(data=True)]
        seam_types = [d["seam_type"] for _, _, d in self.graph.edges(data=True)]
        grades = [d["grade"] for _, _, d in self.graph.edges(data=True)]

        node_types = [d["type"] for _, d in self.graph.nodes(data=True)]

        return {
            "name": self.name,
            "n_nodes": self.n_nodes,
            "n_edges": self.n_edges,
            "total_length_km": round(self.total_length_km, 1),
            "node_types": {t: node_types.count(t) for t in set(node_types)},
            "diameter_range_mm": (min(diameters), max(diameters)) if diameters else (0, 0),
            "seam_type_counts": {s: seam_types.count(s) for s in set(seam_types)},
            "grade_counts": {g: grades.count(g) for g in set(grades)},
            "P_f_range": (min(pf_values), max(pf_values)) if pf_values else (0, 0),
            "P_f_mean": round(np.mean(pf_values), 4) if pf_values else 0,
            "is_connected": nx.is_weakly_connected(self.graph),
        }

    def to_edge_dataframe(self):
        """Export edge data as a pandas DataFrame."""
        import pandas as pd
        records = []
        for u, v, d in self.graph.edges(data=True):
            row = {"source": u, "target": v}
            row.update(d)
            records.append(row)
        return pd.DataFrame(records)

    def to_node_dataframe(self):
        """Export node data as a pandas DataFrame."""
        import pandas as pd
        records = []
        for n, d in self.graph.nodes(data=True):
            row = {"node_id": n}
            row.update(d)
            records.append(row)
        return pd.DataFrame(records)
