"""
Unit Tests — Pipeline Network Graph Module (Sprint 2)
=====================================================

Tests for:
    - PipeSegment dataclass (construction, derived properties, serialization)
    - PipelineNetwork (node/edge management, synthetic generation, summary)
    - Mapping dictionaries (SEAM_TO_SCF_KEY, SEAM_TO_FAT_CLASS, PHMSA_SEAM_MAP, API5L_GRADES)
    - P_f attachment integration (lightweight, reduced MC sims)

Author: Babak Pirzadi (STRATEGOS Thesis)
"""

import pytest
import numpy as np
import networkx as nx

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from src.zone_c.network.pipeline_graph import (
    NodeType, SeamType,
    SEAM_TO_SCF_KEY, SEAM_TO_FAT_CLASS, PHMSA_SEAM_MAP, API5L_GRADES,
    PHMSA_DIAMETER_DIST, PHMSA_SEAM_DIST, PHMSA_SMYS_DIST, PHMSA_DECADE_DIST,
    PipeSegment, PipelineNetwork,
)


# =========================================================================
# 1. Enumeration & Mapping Tests
# =========================================================================

class TestEnumerations:
    """Tests for NodeType and SeamType enums."""

    def test_node_type_values(self):
        expected = {"source", "compressor", "valve", "junction", "delivery", "storage"}
        actual = {nt.value for nt in NodeType}
        assert actual == expected

    def test_seam_type_values(self):
        assert len(SeamType) == 11
        assert SeamType.SEAMLESS.value == "seamless"
        assert SeamType.DSAW.value == "dsaw"
        assert SeamType.ERW_HF.value == "erw_hf"
        assert SeamType.UNKNOWN.value == "unknown"

    def test_seam_to_scf_key_covers_all_seam_types(self):
        """Every SeamType must have an SCF key mapping."""
        for st in SeamType:
            assert st in SEAM_TO_SCF_KEY, f"Missing SCF key for {st}"

    def test_seam_to_fat_class_covers_all_seam_types(self):
        """Every SeamType must have a FAT class mapping."""
        for st in SeamType:
            assert st in SEAM_TO_FAT_CLASS, f"Missing FAT class for {st}"
            assert isinstance(SEAM_TO_FAT_CLASS[st], int)
            assert SEAM_TO_FAT_CLASS[st] > 0

    def test_phmsa_seam_map_returns_seam_types(self):
        """All PHMSA string keys must map to valid SeamType."""
        for key, val in PHMSA_SEAM_MAP.items():
            assert isinstance(val, SeamType), f"{key} maps to {type(val)}"

    def test_api5l_grades_structure(self):
        """API 5L grade table must have (SMYS, UTS) tuples with UTS > SMYS."""
        assert len(API5L_GRADES) >= 8  # B through X80
        for grade, (smys, uts) in API5L_GRADES.items():
            assert uts > smys, f"Grade {grade}: UTS ({uts}) must exceed SMYS ({smys})"
            assert smys > 200, f"Grade {grade}: SMYS ({smys}) too low"

    def test_api5l_x52_values(self):
        """Spot-check X52 grade values."""
        smys, uts = API5L_GRADES["X52"]
        assert abs(smys - 358.5) < 1.0
        assert abs(uts - 455.1) < 1.0


# =========================================================================
# 2. PHMSA Fleet Statistics Tests
# =========================================================================

class TestPHMSAFleetStats:
    """Validate empirical distribution dictionaries sum to ~1.0."""

    def test_diameter_dist_coverage(self):
        """Top NPS sizes should cover at least 75% of fleet (rare sizes excluded)."""
        total = sum(v[0] for v in PHMSA_DIAMETER_DIST.values())
        assert total > 0.75, f"Diameter weights sum to {total}, expected > 0.75"
        assert total <= 1.0, f"Diameter weights sum to {total}, exceeds 1.0"

    def test_seam_dist_sums_to_approx_one(self):
        total = sum(PHMSA_SEAM_DIST.values())
        assert abs(total - 1.0) < 0.15, f"Seam weights sum to {total}"

    def test_smys_dist_sums_to_approx_one(self):
        total = sum(PHMSA_SMYS_DIST.values())
        assert abs(total - 1.0) < 0.15, f"SMYS weights sum to {total}"

    def test_decade_dist_sums_to_one(self):
        total = sum(PHMSA_DECADE_DIST.values())
        assert abs(total - 1.0) < 0.01, f"Decade weights sum to {total}"

    def test_diameter_values_in_inches(self):
        """All diameter keys should be standard NPS sizes in inches."""
        for d in PHMSA_DIAMETER_DIST:
            assert 4 <= d <= 48, f"Unexpected diameter: {d} inches"

    def test_smys_values_in_psi(self):
        """SMYS keys should be in psi (>= 30000)."""
        for s in PHMSA_SMYS_DIST:
            assert 30000 <= s <= 100000, f"Unexpected SMYS: {s} psi"


# =========================================================================
# 3. PipeSegment Tests
# =========================================================================

class TestPipeSegment:
    """Tests for PipeSegment dataclass."""

    @pytest.fixture
    def basic_segment(self):
        return PipeSegment(
            segment_id="TEST_001",
            diameter_mm=508.0,
            wall_mm=9.525,
            smys_mpa=358.5,
            uts_mpa=455.1,
            length_km=50.0,
            seam_type=SeamType.DSAW,
            grade="X52",
            year_installed=1975,
            class_location=1,
            design_factor=0.72,
        )

    def test_basic_construction(self, basic_segment):
        assert basic_segment.segment_id == "TEST_001"
        assert basic_segment.diameter_mm == 508.0
        assert basic_segment.grade == "X52"

    def test_maop_computed_via_barlow(self, basic_segment):
        """MAOP = 2 * SMYS * t * F / D."""
        expected = 2 * 358.5 * 9.525 * 0.72 / 508.0
        assert abs(basic_segment.maop_mpa - expected) < 0.01

    def test_maop_not_overwritten_if_provided(self):
        seg = PipeSegment(
            segment_id="TEST_MAOP",
            diameter_mm=508.0, wall_mm=9.525,
            smys_mpa=358.5, uts_mpa=455.1,
            length_km=10.0, maop_mpa=5.0,
        )
        assert seg.maop_mpa == 5.0

    def test_scf_key_derived_from_seam_type(self, basic_segment):
        assert basic_segment.scf_key == "dsaw_seam"

    def test_fat_class_derived_from_seam_type(self, basic_segment):
        assert basic_segment.fat_class == 90

    def test_seamless_scf_and_fat(self):
        seg = PipeSegment(
            segment_id="SEAMLESS_001",
            diameter_mm=305.0, wall_mm=7.0,
            smys_mpa=289.6, uts_mpa=413.7,
            length_km=20.0, seam_type=SeamType.SEAMLESS,
        )
        assert seg.scf_key == "seamless"
        assert seg.fat_class == 125

    def test_erw_lf_scf_and_fat(self):
        seg = PipeSegment(
            segment_id="ERW_LF_001",
            diameter_mm=203.0, wall_mm=5.0,
            smys_mpa=289.6, uts_mpa=413.7,
            length_km=15.0, seam_type=SeamType.ERW_LF,
        )
        assert seg.scf_key == "erw_lf_seam"
        assert seg.fat_class == 71

    def test_pipe_age(self, basic_segment):
        assert basic_segment.pipe_age(2025) == 50
        assert basic_segment.pipe_age(2000) == 25

    def test_to_dict_keys(self, basic_segment):
        d = basic_segment.to_dict()
        required_keys = {
            "segment_id", "diameter_mm", "wall_mm", "smys_mpa", "uts_mpa",
            "length_km", "seam_type", "grade", "year_installed",
            "class_location", "design_factor", "maop_mpa",
            "P_f", "P_f_lower", "P_f_upper", "scf_key", "fat_class",
            "lat", "lon",
        }
        assert required_keys.issubset(set(d.keys()))

    def test_to_dict_seam_type_is_string(self, basic_segment):
        d = basic_segment.to_dict()
        assert isinstance(d["seam_type"], str)
        assert d["seam_type"] == "dsaw"

    def test_default_pf_is_zero(self, basic_segment):
        assert basic_segment.P_f == 0.0
        assert basic_segment.P_f_ci == (0.0, 0.0)


# =========================================================================
# 4. PipelineNetwork — Node & Edge Management
# =========================================================================

class TestNetworkManagement:
    """Tests for PipelineNetwork node/edge operations."""

    @pytest.fixture
    def empty_net(self):
        return PipelineNetwork("test_net")

    def test_empty_network(self, empty_net):
        assert empty_net.n_nodes == 0
        assert empty_net.n_edges == 0
        assert empty_net.total_length_km == 0.0

    def test_add_node(self, empty_net):
        empty_net.add_node("N001", NodeType.SOURCE, lat=30.0, lon=-90.0)
        assert empty_net.n_nodes == 1
        assert empty_net.graph.nodes["N001"]["type"] == "source"
        assert empty_net.graph.nodes["N001"]["lat"] == 30.0

    def test_add_segment(self, empty_net):
        empty_net.add_node("A", NodeType.SOURCE)
        empty_net.add_node("B", NodeType.DELIVERY)
        seg = PipeSegment(
            segment_id="SEG_A_B",
            diameter_mm=508.0, wall_mm=9.525,
            smys_mpa=358.5, uts_mpa=455.1,
            length_km=100.0, seam_type=SeamType.DSAW,
        )
        empty_net.add_segment("A", "B", seg)
        assert empty_net.n_edges == 1
        assert empty_net.total_length_km == 100.0
        assert "SEG_A_B" in empty_net.segments

    def test_graph_is_digraph(self, empty_net):
        assert isinstance(empty_net.graph, nx.DiGraph)

    def test_name_stored(self, empty_net):
        assert empty_net.name == "test_net"


# =========================================================================
# 5. Synthetic Network Generation
# =========================================================================

class TestSyntheticGeneration:
    """Tests for generate_synthetic()."""

    @pytest.fixture
    def synth_net(self):
        net = PipelineNetwork("synth_test")
        net.generate_synthetic(n_nodes=15, n_segments=20, seed=99)
        return net

    def test_node_count(self, synth_net):
        assert synth_net.n_nodes == 15

    def test_edge_count_at_least_spine(self, synth_net):
        """Must have at least n_nodes - 1 edges for connectivity."""
        assert synth_net.n_edges >= 14

    def test_weakly_connected(self, synth_net):
        """Spine generation must ensure weak connectivity."""
        assert nx.is_weakly_connected(synth_net.graph)

    def test_has_source_nodes(self, synth_net):
        types = [d["type"] for _, d in synth_net.graph.nodes(data=True)]
        assert "source" in types

    def test_has_delivery_nodes(self, synth_net):
        types = [d["type"] for _, d in synth_net.graph.nodes(data=True)]
        assert "delivery" in types

    def test_has_compressor_nodes(self, synth_net):
        types = [d["type"] for _, d in synth_net.graph.nodes(data=True)]
        assert "compressor" in types

    def test_edges_have_required_attributes(self, synth_net):
        required = {"segment_id", "diameter_mm", "wall_mm", "smys_mpa",
                     "seam_type", "grade", "year_installed", "maop_mpa"}
        for u, v, d in synth_net.graph.edges(data=True):
            for key in required:
                assert key in d, f"Edge ({u},{v}) missing attribute: {key}"

    def test_diameters_in_valid_range(self, synth_net):
        for _, _, d in synth_net.graph.edges(data=True):
            assert 100 < d["diameter_mm"] < 1200

    def test_wall_thickness_positive(self, synth_net):
        for _, _, d in synth_net.graph.edges(data=True):
            assert d["wall_mm"] > 0

    def test_smys_in_valid_range(self, synth_net):
        for _, _, d in synth_net.graph.edges(data=True):
            assert 200 < d["smys_mpa"] < 600

    def test_maop_positive(self, synth_net):
        for _, _, d in synth_net.graph.edges(data=True):
            assert d["maop_mpa"] > 0

    def test_year_installed_reasonable(self, synth_net):
        for _, _, d in synth_net.graph.edges(data=True):
            assert 1940 <= d["year_installed"] <= 2030

    def test_total_length_positive(self, synth_net):
        assert synth_net.total_length_km > 0

    def test_deterministic_with_same_seed(self):
        net1 = PipelineNetwork("a")
        net1.generate_synthetic(n_nodes=10, n_segments=15, seed=42)
        net2 = PipelineNetwork("b")
        net2.generate_synthetic(n_nodes=10, n_segments=15, seed=42)
        edges1 = sorted(net1.graph.edges())
        edges2 = sorted(net2.graph.edges())
        assert edges1 == edges2

    def test_different_seed_gives_different_network(self):
        net1 = PipelineNetwork("a")
        net1.generate_synthetic(n_nodes=10, n_segments=15, seed=42)
        net2 = PipelineNetwork("b")
        net2.generate_synthetic(n_nodes=10, n_segments=15, seed=123)
        # At least some edges should differ
        edges1 = set(net1.graph.edges())
        edges2 = set(net2.graph.edges())
        assert edges1 != edges2

    def test_coordinates_within_bounds(self, synth_net):
        for _, d in synth_net.graph.nodes(data=True):
            # Allow some latitude scatter (~0.5 deg) beyond strict bounds
            assert 25.0 < d["lat"] < 38.0
            assert -100.0 < d["lon"] < -84.0


# =========================================================================
# 6. Summary & Export
# =========================================================================

class TestSummaryAndExport:
    """Tests for summary() and DataFrame export."""

    @pytest.fixture
    def net_with_edges(self):
        net = PipelineNetwork("export_test")
        net.generate_synthetic(n_nodes=10, n_segments=12, seed=77)
        return net

    def test_summary_keys(self, net_with_edges):
        s = net_with_edges.summary()
        expected_keys = {
            "name", "n_nodes", "n_edges", "total_length_km",
            "node_types", "diameter_range_mm", "seam_type_counts",
            "grade_counts", "P_f_range", "P_f_mean", "is_connected",
        }
        assert expected_keys.issubset(set(s.keys()))

    def test_summary_node_count_matches(self, net_with_edges):
        s = net_with_edges.summary()
        assert s["n_nodes"] == net_with_edges.n_nodes

    def test_summary_connected(self, net_with_edges):
        s = net_with_edges.summary()
        assert s["is_connected"] is True

    def test_edge_dataframe_shape(self, net_with_edges):
        df = net_with_edges.to_edge_dataframe()
        assert len(df) == net_with_edges.n_edges
        assert "segment_id" in df.columns
        assert "diameter_mm" in df.columns
        assert "P_f" in df.columns

    def test_node_dataframe_shape(self, net_with_edges):
        df = net_with_edges.to_node_dataframe()
        assert len(df) == net_with_edges.n_nodes
        assert "node_id" in df.columns
        assert "type" in df.columns


# =========================================================================
# 7. P_f Attachment Integration (lightweight)
# =========================================================================

class TestPfAttachment:
    """Integration test: attach_pf_values with reduced MC sims."""

    @pytest.fixture(scope="class")
    def net_with_pf(self):
        net = PipelineNetwork("pf_test")
        net.generate_synthetic(n_nodes=5, n_segments=5, seed=42)
        net.attach_pf_values(n_simulations=500, seed=42)  # reduced for speed
        return net

    def test_all_edges_have_pf(self, net_with_pf):
        for u, v, d in net_with_pf.graph.edges(data=True):
            assert d["P_f"] > 0, f"Edge ({u},{v}) has P_f = 0"

    def test_pf_in_valid_range(self, net_with_pf):
        for u, v, d in net_with_pf.graph.edges(data=True):
            assert 0.0 < d["P_f"] <= 1.0, f"P_f out of range: {d['P_f']}"

    def test_pf_ci_ordered(self, net_with_pf):
        for u, v, d in net_with_pf.graph.edges(data=True):
            assert d["P_f_lower"] <= d["P_f"] <= d["P_f_upper"]

    def test_pf_varies_across_segments(self, net_with_pf):
        pf_vals = [d["P_f"] for _, _, d in net_with_pf.graph.edges(data=True)]
        assert max(pf_vals) > min(pf_vals), "All segments have identical P_f"

    def test_internal_segments_updated(self, net_with_pf):
        for seg_id, seg in net_with_pf.segments.items():
            assert seg.P_f > 0, f"Segment {seg_id} internal P_f not updated"
            assert seg.P_f_ci[0] <= seg.P_f <= seg.P_f_ci[1]
