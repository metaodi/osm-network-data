"""Tests for map_to_segments helpers and the three-level matching strategy."""

import geopandas as gpd
import pandas as pd
import pytest
from shapely.geometry import LineString

from map_to_segments import (
    OUTPUT_COLUMNS,
    _explode_segments_by_osmid,
    _spatial_pick,
    map_matched_edges_to_segments,
)


# ---------------------------------------------------------------------------
# _explode_segments_by_osmid
# ---------------------------------------------------------------------------

class TestExplodeSegmentsByOsmid:
    def _make_segs(self, osmid_norm_list):
        return gpd.GeoDataFrame(
            {
                "segment_id": [f"s{i}" for i in range(len(osmid_norm_list))],
                "osmid_norm": osmid_norm_list,
                "geometry": [LineString([(i, 0), (i + 1, 0)]) for i in range(len(osmid_norm_list))],
            },
            geometry="geometry",
            crs="EPSG:4326",
        )

    def test_single_osmid_per_segment_unchanged_count(self):
        result = _explode_segments_by_osmid(self._make_segs([[100], [200]]))
        assert len(result) == 2
        assert set(result["osmid_single"]) == {100, 200}

    def test_multiple_osmids_per_segment_exploded(self):
        result = _explode_segments_by_osmid(self._make_segs([[10, 20, 30]]))
        assert len(result) == 3
        assert list(result["osmid_single"]) == [10, 20, 30]

    def test_non_numeric_osmids_dropped(self):
        result = _explode_segments_by_osmid(self._make_segs([["abc", None, 100]]))
        assert len(result) == 1
        assert result.iloc[0]["osmid_single"] == 100

    def test_osmid_single_dtype_is_int64(self):
        result = _explode_segments_by_osmid(self._make_segs([[42]]))
        assert result["osmid_single"].dtype == "int64"

    def test_segment_id_preserved_after_explode(self):
        result = _explode_segments_by_osmid(self._make_segs([[1, 2]]))
        assert list(result["segment_id"]) == ["s0", "s0"]


# ---------------------------------------------------------------------------
# _spatial_pick
# ---------------------------------------------------------------------------

class TestSpatialPick:
    def test_empty_candidates_returns_none(self):
        geom = LineString([(0, 0), (1, 0)])
        empty = gpd.GeoDataFrame({"segment_id": [], "geometry": []}, geometry="geometry")
        assert _spatial_pick(geom, empty) == (None, None)

    def test_none_geometry_returns_none(self):
        candidates = gpd.GeoDataFrame(
            {"segment_id": ["s1"], "geometry": [LineString([(0, 0), (1, 0)])]},
            geometry="geometry",
        )
        assert _spatial_pick(None, candidates) == (None, None)

    def test_picks_closest_of_two_candidates(self):
        candidates = gpd.GeoDataFrame(
            {
                "segment_id": ["near", "far"],
                "geometry": [
                    LineString([(0, 0), (100, 0)]),    # at y=0
                    LineString([(0, 100), (100, 100)]),  # at y=100
                ],
            },
            geometry="geometry",
        )
        query = LineString([(0, 10), (100, 10)])  # at y=10, closer to "near"
        seg_id, dist = _spatial_pick(query, candidates)
        assert seg_id == "near"
        assert dist == pytest.approx(10.0)

    def test_single_candidate_always_selected(self):
        candidates = gpd.GeoDataFrame(
            {"segment_id": ["only"], "geometry": [LineString([(0, 0), (100, 0)])]},
            geometry="geometry",
        )
        seg_id, dist = _spatial_pick(LineString([(0, 5), (100, 5)]), candidates)
        assert seg_id == "only"
        assert dist == pytest.approx(5.0)


# ---------------------------------------------------------------------------
# map_matched_edges_to_segments — the three-level matching strategy
# ---------------------------------------------------------------------------

class TestMapMatchedEdgesToSegments:
    def test_empty_matched_edges_returns_empty_df_with_correct_columns(
        self, minimal_segments
    ):
        empty = gpd.GeoDataFrame(
            columns=["match_edge_idx", "way_id", "geometry"],
            geometry="geometry",
            crs="EPSG:4326",
        )
        result = map_matched_edges_to_segments(empty, segments=minimal_segments)
        assert isinstance(result, pd.DataFrame)
        assert list(result.columns) == OUTPUT_COLUMNS
        assert len(result) == 0

    def test_way_id_exact_match_quality_and_segment(
        self, minimal_segments, minimal_matched_edges
    ):
        result = map_matched_edges_to_segments(
            minimal_matched_edges,
            segments=minimal_segments,
            trajectory_id="traj-1",
        )
        assert len(result) == 1
        row = result.iloc[0]
        assert row["segment_id"] == "v0.1.0-00000000"
        assert row["match_quality"] == "way_id"
        assert row["distance_to_segment_m"] == 0.0
        assert row["trajectory_id"] == "traj-1"
        assert row["highway"] == "primary"
        assert not row["is_roundabout"]

    def test_output_always_has_all_output_columns(
        self, minimal_segments, minimal_matched_edges
    ):
        result = map_matched_edges_to_segments(
            minimal_matched_edges, segments=minimal_segments
        )
        assert list(result.columns) == OUTPUT_COLUMNS

    def test_spatial_only_when_way_id_is_none(self, minimal_segments):
        edge = gpd.GeoDataFrame(
            {
                "match_edge_idx": [0],
                "way_id": [None],
                "geometry": [LineString([(9.0, 46.2), (9.01, 46.2)])],
            },
            geometry="geometry",
            crs="EPSG:4326",
        )
        result = map_matched_edges_to_segments(edge, segments=minimal_segments)
        assert result.iloc[0]["match_quality"] == "spatial_only"
        assert result.iloc[0]["segment_id"] is not None

    def test_spatial_only_when_way_id_unknown(self, minimal_segments):
        edge = gpd.GeoDataFrame(
            {
                "match_edge_idx": [0],
                "way_id": [99999],  # not present in any segment's osmid_norm
                "geometry": [LineString([(9.0, 46.2), (9.01, 46.2)])],
            },
            geometry="geometry",
            crs="EPSG:4326",
        )
        result = map_matched_edges_to_segments(edge, segments=minimal_segments)
        assert result.iloc[0]["match_quality"] == "spatial_only"

    def test_way_id_plus_spatial_when_multiple_candidates(self):
        # Two segments that both carry the same osmid force the spatial tiebreak.
        segments = gpd.GeoDataFrame(
            {
                "segment_id": ["seg-A", "seg-B"],
                "network_version": ["v0.1.0", "v0.1.0"],
                "osmid": [100, 100],
                "osmid_norm": [[100], [100]],
                "u": [1, 2],
                "v": [2, 3],
                "key": [0, 0],
                "name": ["Road", "Road"],
                "highway": ["primary", "primary"],
                "oneway": [False, False],
                "maxspeed": ["50", "50"],
                "junction": [None, None],
                "length": [100.0, 100.0],
                "is_roundabout": [False, False],
                "starts_or_ends_at_intersection": [True, True],
                "geometry": [
                    LineString([(9.0, 46.2), (9.01, 46.2)]),   # seg-A
                    LineString([(9.0, 46.21), (9.01, 46.21)]),  # seg-B
                ],
            },
            geometry="geometry",
            crs="EPSG:4326",
        )
        edge = gpd.GeoDataFrame(
            {
                "match_edge_idx": [0],
                "way_id": [100],
                "geometry": [LineString([(9.0, 46.2), (9.01, 46.2)])],  # on seg-A
            },
            geometry="geometry",
            crs="EPSG:4326",
        )
        result = map_matched_edges_to_segments(edge, segments=segments)
        assert result.iloc[0]["match_quality"] == "way_id+spatial"
        assert result.iloc[0]["segment_id"] == "seg-A"

    def test_multiple_edges_all_mapped(self, minimal_segments):
        edges = gpd.GeoDataFrame(
            {
                "match_edge_idx": [0, 1],
                "way_id": [100, 200],
                "geometry": [
                    LineString([(9.0, 46.2), (9.01, 46.2)]),
                    LineString([(9.01, 46.2), (9.01, 46.21)]),
                ],
            },
            geometry="geometry",
            crs="EPSG:4326",
        )
        result = map_matched_edges_to_segments(edges, segments=minimal_segments)
        assert len(result) == 2
        assert result.iloc[0]["segment_id"] == "v0.1.0-00000000"
        assert result.iloc[1]["segment_id"] == "v0.1.0-00000001"
        assert (result["match_quality"] == "way_id").all()
