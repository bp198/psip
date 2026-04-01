"""
Tests for psip.network.entsog — ENTSOG GeoJSON adapter and TAP builder.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from psip.network.entsog import (
    TAP_NODES,
    TAP_SEGMENTS,
    _coord_to_node_id,
    _haversine_km,
    _linestring_length_km,
    _linestring_midpoint,
    _parse_grade,
    _parse_seam,
    build_tap_network,
    entsog_geojson_to_network,
    load_entsog_geojson,
)
from src.zone_c.network.pipeline_graph import NodeType, PipelineNetwork, SeamType

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

MINIMAL_GEOJSON: dict = {
    "type": "FeatureCollection",
    "features": [
        {
            "type": "Feature",
            "geometry": {
                "type": "LineString",
                "coordinates": [[26.318, 41.129], [25.410, 41.122]],
            },
            "properties": {
                "pointKey": "TAP-TEST-SEG-001",
                "outerDiameter": 1219.2,
                "wallThickness": 18.3,
                "grade": "X70",
                "designPressure": 9.5,
                "yearInstalled": 2020,
            },
        },
        {
            "type": "Feature",
            "geometry": {
                "type": "LineString",
                "coordinates": [[25.410, 41.122], [24.402, 40.939]],
            },
            "properties": {
                "outerDiameter": 1219.2,
                "wallThickness": 18.3,
                "grade": "X70",
                "designPressure": 9.5,
                "yearInstalled": 2020,
            },
        },
    ],
}

POINT_GEOJSON: dict = {
    "type": "FeatureCollection",
    "features": [
        {
            "type": "Feature",
            "geometry": {"type": "Point", "coordinates": [26.318, 41.129]},
            "properties": {"name": "Kipoi Station", "nodeType": "source"},
        },
        {
            "type": "Feature",
            "geometry": {
                "type": "LineString",
                "coordinates": [[26.318, 41.129], [25.410, 41.122]],
            },
            "properties": {"grade": "X70"},
        },
    ],
}

MULTILINESTRING_GEOJSON: dict = {
    "type": "FeatureCollection",
    "features": [
        {
            "type": "Feature",
            "geometry": {
                "type": "MultiLineString",
                "coordinates": [
                    [[26.318, 41.129], [25.865, 41.126]],
                    [[25.865, 41.126], [25.410, 41.122]],
                ],
            },
            "properties": {"grade": "X65"},
        }
    ],
}


# ---------------------------------------------------------------------------
# Helper function tests
# ---------------------------------------------------------------------------


class TestHaversineKm:
    def test_zero_distance(self):
        assert _haversine_km(41.0, 26.0, 41.0, 26.0) == pytest.approx(0.0)

    def test_known_distance(self):
        # Kipoi → Komotini approx 95 km
        d = _haversine_km(41.129, 26.318, 41.122, 25.410)
        assert 60.0 < d < 120.0

    def test_symmetry(self):
        d1 = _haversine_km(41.0, 26.0, 40.0, 25.0)
        d2 = _haversine_km(40.0, 25.0, 41.0, 26.0)
        assert d1 == pytest.approx(d2, rel=1e-6)


class TestLinestringHelpers:
    def test_midpoint_two_points(self):
        coords = [[0.0, 0.0], [2.0, 2.0]]
        lat, lon = _linestring_midpoint(coords)
        assert lat == pytest.approx(1.0)
        assert lon == pytest.approx(1.0)

    def test_midpoint_three_points(self):
        coords = [[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]]
        lat, lon = _linestring_midpoint(coords)
        assert lat == pytest.approx(1.0)
        assert lon == pytest.approx(1.0)

    def test_length_positive(self):
        coords = [[26.318, 41.129], [25.410, 41.122]]
        length = _linestring_length_km(coords)
        assert length > 0

    def test_length_multipoint(self):
        coords = [[26.318, 41.129], [25.865, 41.125], [25.410, 41.122]]
        full = _linestring_length_km(coords)
        part1 = _linestring_length_km(coords[:2])
        part2 = _linestring_length_km(coords[1:])
        assert full == pytest.approx(part1 + part2, rel=1e-6)


class TestCoordToNodeId:
    def test_deterministic(self):
        id1 = _coord_to_node_id(26.318, 41.129)
        id2 = _coord_to_node_id(26.318, 41.129)
        assert id1 == id2

    def test_different_coords_different_ids(self):
        id1 = _coord_to_node_id(26.318, 41.129)
        id2 = _coord_to_node_id(25.410, 41.122)
        assert id1 != id2

    def test_no_special_chars(self):
        node_id = _coord_to_node_id(-26.318, -41.129)
        # Should not contain '-' or '.' (replaced by 'm' and 'p')
        assert "-" not in node_id
        assert "." not in node_id


class TestParseSeam:
    def test_dsaw(self):
        assert _parse_seam("DSAW") == SeamType.DSAW
        assert _parse_seam("dsaw") == SeamType.DSAW

    def test_none_returns_dsaw_default(self):
        assert _parse_seam(None) == SeamType.DSAW

    def test_unknown_string(self):
        assert _parse_seam("GOBBLEDYGOOK") == SeamType.UNKNOWN

    def test_erw(self):
        assert _parse_seam("ERW") == SeamType.ERW_HF


class TestParseGrade:
    def test_x70(self):
        grade, smys, uts = _parse_grade("X70")
        assert grade == "X70"
        assert smys == pytest.approx(482.6, rel=0.01)

    def test_none_defaults_to_x70(self):
        grade, smys, uts = _parse_grade(None)
        assert grade == "X70"

    def test_uts_greater_than_smys(self):
        for g in ["X52", "X65", "X70", "X80"]:
            _, smys, uts = _parse_grade(g)
            assert uts > smys


# ---------------------------------------------------------------------------
# GeoJSON adapter tests
# ---------------------------------------------------------------------------


class TestEntsogGeojsonToNetwork:
    def test_returns_pipeline_network(self):
        net = entsog_geojson_to_network(MINIMAL_GEOJSON)
        assert isinstance(net, PipelineNetwork)

    def test_segment_count(self):
        net = entsog_geojson_to_network(MINIMAL_GEOJSON)
        assert net.n_edges == 2

    def test_node_count_minimal(self):
        # 2 segments sharing one node = 3 unique nodes
        net = entsog_geojson_to_network(MINIMAL_GEOJSON)
        assert net.n_nodes == 3

    def test_segment_id_from_property(self):
        net = entsog_geojson_to_network(MINIMAL_GEOJSON)
        seg_ids = [d["segment_id"] for _, _, d in net.graph.edges(data=True)]
        assert "TAP-TEST-SEG-001" in seg_ids

    def test_diameter_mapped(self):
        net = entsog_geojson_to_network(MINIMAL_GEOJSON)
        for _, _, d in net.graph.edges(data=True):
            assert d["diameter_mm"] == pytest.approx(1219.2)

    def test_grade_mapped(self):
        net = entsog_geojson_to_network(MINIMAL_GEOJSON)
        for _, _, d in net.graph.edges(data=True):
            assert d["grade"] == "X70"

    def test_length_computed_from_geometry(self):
        net = entsog_geojson_to_network(MINIMAL_GEOJSON)
        for _, _, d in net.graph.edges(data=True):
            assert d["length_km"] > 0

    def test_invalid_geojson_raises(self):
        with pytest.raises(ValueError, match="FeatureCollection"):
            entsog_geojson_to_network({"type": "Feature"})

    def test_point_features_enrich_nodes(self):
        net = entsog_geojson_to_network(POINT_GEOJSON)
        # The source Point should have enriched the endpoint node
        source_nodes = [
            n for n, d in net.graph.nodes(data=True) if d.get("type") == NodeType.SOURCE.value
        ]
        assert len(source_nodes) >= 1

    def test_multilinestring_feature(self):
        net = entsog_geojson_to_network(MULTILINESTRING_GEOJSON)
        # MultiLineString with 2 parts → 2 edges
        assert net.n_edges == 2

    def test_custom_defaults_applied(self):
        bare_geojson = {
            "type": "FeatureCollection",
            "features": [
                {
                    "type": "Feature",
                    "geometry": {
                        "type": "LineString",
                        "coordinates": [[20.0, 40.0], [19.0, 40.5]],
                    },
                    "properties": {},
                }
            ],
        }
        net = entsog_geojson_to_network(
            bare_geojson, default_diameter_mm=914.4, default_grade="X65"
        )
        for _, _, d in net.graph.edges(data=True):
            assert d["diameter_mm"] == pytest.approx(914.4)
            assert d["grade"] == "X65"

    def test_network_name_set(self):
        net = entsog_geojson_to_network(MINIMAL_GEOJSON, name="test_net")
        assert net.name == "test_net"

    def test_maop_positive(self):
        net = entsog_geojson_to_network(MINIMAL_GEOJSON)
        for _, _, d in net.graph.edges(data=True):
            assert d["maop_mpa"] > 0


# ---------------------------------------------------------------------------
# File loader tests
# ---------------------------------------------------------------------------


class TestLoadEntsogGeojson:
    def test_load_from_file(self):
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".geojson", delete=False, encoding="utf-8"
        ) as f:
            json.dump(MINIMAL_GEOJSON, f)
            tmp_path = f.name

        loaded = load_entsog_geojson(tmp_path)
        assert loaded["type"] == "FeatureCollection"
        assert len(loaded["features"]) == 2

    def test_load_from_pathlib(self):
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".geojson", delete=False, encoding="utf-8"
        ) as f:
            json.dump(MINIMAL_GEOJSON, f)
            tmp_path = Path(f.name)

        loaded = load_entsog_geojson(tmp_path)
        assert loaded["type"] == "FeatureCollection"

    def test_missing_file_raises(self):
        with pytest.raises(FileNotFoundError):
            load_entsog_geojson("/nonexistent/path/file.geojson")

    def test_invalid_json_raises(self):
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".geojson", delete=False, encoding="utf-8"
        ) as f:
            f.write("this is not valid json {{{")
            tmp_path = f.name

        with pytest.raises(ValueError, match="Cannot parse GeoJSON"):
            load_entsog_geojson(tmp_path)


# ---------------------------------------------------------------------------
# TAP network builder tests
# ---------------------------------------------------------------------------


class TestBuildTapNetwork:
    def test_returns_pipeline_network(self):
        net = build_tap_network()
        assert isinstance(net, PipelineNetwork)

    def test_node_count(self):
        net = build_tap_network()
        assert net.n_nodes == len(TAP_NODES)

    def test_segment_count(self):
        net = build_tap_network()
        assert net.n_edges == len(TAP_SEGMENTS)

    def test_total_length_approx(self):
        net = build_tap_network()
        # Segment lengths sum ~740 km (straight-line node distances);
        # full TAP route is ~878 km following terrain
        assert 700.0 < net.total_length_km < 960.0

    def test_all_x70_grade(self):
        net = build_tap_network()
        for _, _, d in net.graph.edges(data=True):
            assert d["grade"] == "X70"

    def test_all_dsaw_seam(self):
        net = build_tap_network()
        for _, _, d in net.graph.edges(data=True):
            assert d["seam_type"] == SeamType.DSAW.value

    def test_diameter_1219mm(self):
        net = build_tap_network()
        for _, _, d in net.graph.edges(data=True):
            assert d["diameter_mm"] == pytest.approx(1219.2)

    def test_year_2020(self):
        net = build_tap_network()
        for _, _, d in net.graph.edges(data=True):
            assert d["year_installed"] == 2020

    def test_source_node_at_kipoi(self):
        net = build_tap_network()
        kipoi = net.graph.nodes["TAP-N01"]
        assert kipoi["type"] == NodeType.SOURCE.value

    def test_delivery_node_at_melendugno(self):
        net = build_tap_network()
        melendugno = net.graph.nodes["TAP-N13"]
        assert melendugno["type"] == NodeType.DELIVERY.value

    def test_offshore_segments_higher_maop(self):
        net = build_tap_network()
        offshore_ids = {"TAP-SEG-010", "TAP-SEG-011"}
        for _, _, d in net.graph.edges(data=True):
            if d["segment_id"] in offshore_ids:
                assert d["maop_mpa"] == pytest.approx(10.5)
            else:
                assert d["maop_mpa"] == pytest.approx(9.5)

    def test_offshore_segments_thicker_wall(self):
        net = build_tap_network()
        offshore_ids = {"TAP-SEG-010", "TAP-SEG-011"}
        for _, _, d in net.graph.edges(data=True):
            if d["segment_id"] in offshore_ids:
                assert d["wall_mm"] == pytest.approx(22.2)

    def test_network_is_connected(self):
        import networkx as nx

        net = build_tap_network()
        assert nx.is_weakly_connected(net.graph)

    def test_no_pf_without_attach(self):
        net = build_tap_network(attach_pf=False)
        for _, _, d in net.graph.edges(data=True):
            assert d["P_f"] == pytest.approx(0.0)

    def test_segment_ids_unique(self):
        net = build_tap_network()
        seg_ids = [d["segment_id"] for _, _, d in net.graph.edges(data=True)]
        assert len(seg_ids) == len(set(seg_ids))

    def test_node_coordinates_in_europe(self):
        net = build_tap_network()
        for _, d in net.graph.nodes(data=True):
            assert 35.0 < d["lat"] < 48.0
            assert 15.0 < d["lon"] < 30.0

    def test_summary_returns_dict(self):
        net = build_tap_network()
        summary = net.summary()
        assert isinstance(summary, dict)
        assert summary["n_nodes"] == len(TAP_NODES)
        assert summary["n_edges"] == len(TAP_SEGMENTS)

    def test_psip_network_package_import(self):
        """Verify the adapter is accessible via the psip.network package."""
        from psip.network import build_tap_network as btn
        from psip.network import entsog_geojson_to_network, load_entsog_geojson  # noqa: F401

        net = btn()
        assert net.n_edges == len(TAP_SEGMENTS)
