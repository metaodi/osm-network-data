"""Tests for build_segments pure helpers and Parquet I/O round-trip."""

import math

import geopandas as gpd
import pandas as pd
import pytest
from shapely.geometry import LineString, Point

from build_segments import (
    _build_ordered_line,
    _classify_road_context,
    _compute_length_split_distances,
    _detect_special_nodes,
    _ensure_columns,
    _normalise_osmid,
    _split_linestring,
    _stringify_list,
    load_segments,
    save_segments,
)


# ---------------------------------------------------------------------------
# Shared test helpers
# ---------------------------------------------------------------------------

def _make_node_gdf(node_ids, **columns):
    """Build a minimal nodes GeoDataFrame as OSMnx would return."""
    data = {k: list(v) for k, v in columns.items()}
    data["geometry"] = [Point(0, 0)] * len(node_ids)
    return gpd.GeoDataFrame(data, index=pd.Index(node_ids), geometry="geometry")


def _make_way_edges(spec):
    """Build a way-edges DataFrame from ``(u, v, start_xy, end_xy)`` tuples."""
    return pd.DataFrame(
        [{"u": u, "v": v, "geometry": LineString([s, e])} for u, v, s, e in spec]
    )


class TestNormaliseOsmid:
    def test_none_returns_empty_list(self):
        assert _normalise_osmid(None) == []

    def test_scalar_int(self):
        assert _normalise_osmid(42) == [42]

    def test_list_of_ints(self):
        assert _normalise_osmid([1, 2, 3]) == [1, 2, 3]

    def test_list_with_none_entries_filtered(self):
        assert _normalise_osmid([1, None, 3]) == [1, 3]

    def test_tuple_input(self):
        assert _normalise_osmid((10, 20)) == [10, 20]

    def test_set_input(self):
        assert _normalise_osmid({5}) == [5]

    def test_string_cast_to_int(self):
        assert _normalise_osmid("99") == [99]

    def test_single_element_list(self):
        assert _normalise_osmid([7]) == [7]


class TestStringifyList:
    def test_scalar_string_passthrough(self):
        assert _stringify_list("primary") == "primary"

    def test_none_passthrough(self):
        assert _stringify_list(None) is None

    def test_int_passthrough(self):
        assert _stringify_list(42) == 42

    def test_list_joined_with_comma(self):
        assert _stringify_list(["primary", "secondary"]) == "primary, secondary"

    def test_list_filters_none(self):
        assert _stringify_list([None, "primary", None]) == "primary"

    def test_list_filters_nan_string(self):
        assert _stringify_list(["nan", "motorway"]) == "motorway"

    def test_all_null_returns_none(self):
        assert _stringify_list([None, "nan"]) is None

    def test_empty_list_returns_none(self):
        assert _stringify_list([]) is None

    def test_tuple_joined(self):
        assert _stringify_list(("a", "b")) == "a, b"

    def test_single_element_list(self):
        assert _stringify_list(["motorway"]) == "motorway"


class TestEnsureColumns:
    def test_adds_missing_columns_as_none(self):
        df = pd.DataFrame({"a": [1]})
        _ensure_columns(df, ["a", "b", "c"])
        assert "b" in df.columns
        assert "c" in df.columns
        assert df["b"].iloc[0] is None

    def test_does_not_overwrite_existing_column(self):
        df = pd.DataFrame({"a": [99]})
        _ensure_columns(df, ["a"])
        assert df["a"].iloc[0] == 99

    def test_noop_on_empty_column_list(self):
        df = pd.DataFrame({"a": [1]})
        _ensure_columns(df, [])
        assert list(df.columns) == ["a"]

    def test_idempotent_when_column_already_present(self):
        df = pd.DataFrame({"x": ["hello"]})
        _ensure_columns(df, ["x"])
        _ensure_columns(df, ["x"])
        assert df["x"].iloc[0] == "hello"


def test_save_load_segments_roundtrip(tmp_path, minimal_segments):
    path = tmp_path / "segments.parquet"
    save_segments(minimal_segments, path=path)
    loaded = load_segments(path=path)
    assert len(loaded) == len(minimal_segments)
    assert list(loaded["segment_id"]) == list(minimal_segments["segment_id"])
    assert list(loaded["network_version"]) == list(minimal_segments["network_version"])
    assert loaded.crs == minimal_segments.crs
    assert (loaded.geometry.geom_type == "LineString").all()


# ---------------------------------------------------------------------------
# _classify_road_context
# ---------------------------------------------------------------------------

class TestClassifyRoadContext:
    def test_maxspeed_50_is_urban(self):
        assert _classify_road_context("primary", "50") == "urban"

    def test_maxspeed_51_is_rural(self):
        assert _classify_road_context("primary", "51") == "rural"

    def test_maxspeed_30_is_urban(self):
        assert _classify_road_context(None, "30") == "urban"

    def test_maxspeed_with_unit_string(self):
        assert _classify_road_context(None, "50 km/h") == "urban"

    def test_maxspeed_30mph_is_urban(self):
        # 30 mph ≈ 48 km/h → urban
        assert _classify_road_context(None, "30 mph") == "urban"

    def test_maxspeed_70mph_is_rural(self):
        # 70 mph ≈ 113 km/h → rural
        assert _classify_road_context(None, "70 mph") == "rural"

    def test_nonnumeric_maxspeed_falls_through_to_highway(self):
        assert _classify_road_context("residential", "walk") == "urban"

    def test_none_maxspeed_uses_highway(self):
        assert _classify_road_context("residential", None) == "urban"

    def test_nan_maxspeed_uses_highway(self):
        assert _classify_road_context("living_street", float("nan")) == "urban"

    def test_residential_is_urban(self):
        assert _classify_road_context("residential", None) == "urban"

    def test_living_street_is_urban(self):
        assert _classify_road_context("living_street", None) == "urban"

    def test_motorway_is_rural(self):
        assert _classify_road_context("motorway", None) == "rural"

    def test_primary_is_rural(self):
        assert _classify_road_context("primary", None) == "rural"

    def test_both_none_defaults_to_rural(self):
        assert _classify_road_context(None, None) == "rural"

    def test_maxspeed_overrides_urban_highway(self):
        # residential road with 100 km/h limit → rural
        assert _classify_road_context("residential", "100") == "rural"

    def test_maxspeed_overrides_rural_highway(self):
        # motorway with 50 km/h limit → urban
        assert _classify_road_context("motorway", "50") == "urban"


# ---------------------------------------------------------------------------
# _detect_special_nodes
# ---------------------------------------------------------------------------

class TestDetectSpecialNodes:
    def test_empty_gdf_returns_empty_dict(self):
        gdf = _make_node_gdf([])
        assert _detect_special_nodes(gdf) == {}

    def test_crossing_via_highway_tag(self):
        gdf = _make_node_gdf([1], highway=["crossing"])
        assert _detect_special_nodes(gdf) == {1: frozenset({"crossing"})}

    def test_traffic_signals_detected(self):
        gdf = _make_node_gdf([2], highway=["traffic_signals"])
        assert _detect_special_nodes(gdf) == {2: frozenset({"traffic_signals"})}

    def test_stop_sign_detected(self):
        gdf = _make_node_gdf([3], highway=["stop"])
        assert _detect_special_nodes(gdf) == {3: frozenset({"stop"})}

    def test_give_way_detected(self):
        gdf = _make_node_gdf([4], highway=["give_way"])
        assert _detect_special_nodes(gdf) == {4: frozenset({"give_way"})}

    def test_mini_roundabout_detected(self):
        gdf = _make_node_gdf([5], highway=["mini_roundabout"])
        assert _detect_special_nodes(gdf) == {5: frozenset({"mini_roundabout"})}

    def test_crossing_via_crossing_tag(self):
        gdf = _make_node_gdf([6], crossing=["uncontrolled"])
        assert _detect_special_nodes(gdf) == {6: frozenset({"crossing"})}

    def test_crossing_no_not_detected(self):
        gdf = _make_node_gdf([7], crossing=["no"])
        assert _detect_special_nodes(gdf) == {}

    def test_crossing_none_string_not_detected(self):
        gdf = _make_node_gdf([8], crossing=["none"])
        assert _detect_special_nodes(gdf) == {}

    def test_ordinary_node_not_detected(self):
        gdf = _make_node_gdf([9], highway=[None])
        assert _detect_special_nodes(gdf) == {}

    def test_no_highway_column_uses_crossing_col(self):
        gdf = _make_node_gdf([10], crossing=["marked"])
        assert _detect_special_nodes(gdf) == {10: frozenset({"crossing"})}

    def test_both_tags_deduplicated_into_single_set(self):
        gdf = _make_node_gdf([11], highway=["crossing"], crossing=["zebra"])
        assert _detect_special_nodes(gdf) == {11: frozenset({"crossing"})}

    def test_values_are_frozensets(self):
        gdf = _make_node_gdf([12], highway=["stop"])
        result = _detect_special_nodes(gdf)
        assert isinstance(result[12], frozenset)

    def test_multiple_nodes_mixed_tags(self):
        gdf = _make_node_gdf([1, 2, 3], highway=["crossing", "stop", None])
        result = _detect_special_nodes(gdf)
        assert 1 in result
        assert 2 in result
        assert 3 not in result


# ---------------------------------------------------------------------------
# _build_ordered_line
# ---------------------------------------------------------------------------

class TestBuildOrderedLine:
    def test_single_edge_returns_linestring(self):
        edges = _make_way_edges([(1, 2, (0, 0), (10, 0))])
        result = _build_ordered_line(edges)
        assert result is not None
        assert list(result.coords) == [(0.0, 0.0), (10.0, 0.0)]

    def test_two_edges_ordered_path(self):
        edges = _make_way_edges([
            (1, 2, (0, 0), (10, 0)),
            (2, 3, (10, 0), (20, 0)),
        ])
        result = _build_ordered_line(edges)
        assert result is not None
        assert result.coords[0] == (0.0, 0.0)
        assert result.coords[-1] == (20.0, 0.0)

    def test_no_duplicate_coords_at_junctions(self):
        edges = _make_way_edges([
            (1, 2, (0, 0), (5, 0)),
            (2, 3, (5, 0), (10, 0)),
            (3, 4, (10, 0), (15, 0)),
        ])
        result = _build_ordered_line(edges)
        assert len(list(result.coords)) == 4  # 4 nodes, no duplicate junctions

    def test_reversed_input_order_same_length(self):
        # Edges given in reverse path order
        edges = _make_way_edges([
            (2, 3, (10, 0), (20, 0)),
            (1, 2, (0, 0), (10, 0)),
        ])
        result = _build_ordered_line(edges)
        assert result is not None
        assert result.length == pytest.approx(20.0)

    def test_two_way_road_both_directions_correct_length(self):
        # Both directed copies of each segment present (as OSMnx returns)
        edges = _make_way_edges([
            (1, 2, (0, 0), (10, 0)),
            (2, 1, (10, 0), (0, 0)),
            (2, 3, (10, 0), (20, 0)),
            (3, 2, (20, 0), (10, 0)),
        ])
        result = _build_ordered_line(edges)
        assert result is not None
        assert result.length == pytest.approx(20.0)

    def test_closed_loop_includes_all_segments(self):
        # Triangle A→B→C→A (e.g. a roundabout)
        a, b, c = (0, 0), (10, 0), (5, 10)
        edges = _make_way_edges([
            (1, 2, a, b),
            (2, 3, b, c),
            (3, 1, c, a),
        ])
        result = _build_ordered_line(edges)
        assert result is not None
        expected_length = (
            LineString([a, b]).length
            + LineString([b, c]).length
            + LineString([c, a]).length
        )
        assert result.length == pytest.approx(expected_length, rel=1e-6)

    def test_empty_dataframe_returns_none(self):
        edges = pd.DataFrame(columns=["u", "v", "geometry"])
        assert _build_ordered_line(edges) is None


# ---------------------------------------------------------------------------
# _compute_length_split_distances
# ---------------------------------------------------------------------------

class TestComputeLengthSplitDistances:
    def test_way_within_budget_returns_empty(self):
        assert _compute_length_split_distances(40.0, 20.0, 50.0) == []

    def test_exact_max_returns_empty(self):
        assert _compute_length_split_distances(50.0, 20.0, 50.0) == []

    def test_longer_way_returns_splits(self):
        result = _compute_length_split_distances(100.0, 20.0, 50.0)
        assert len(result) > 0

    def test_splits_strictly_inside_total(self):
        total = 100.0
        result = _compute_length_split_distances(total, 20.0, 50.0)
        assert all(0.0 < d < total for d in result)

    def test_result_is_sorted(self):
        result = _compute_length_split_distances(200.0, 50.0, 100.0)
        assert result == sorted(result)

    def test_resulting_segments_within_target_range(self):
        total = 150.0
        target_min, target_max = 20.0, 50.0
        splits = _compute_length_split_distances(total, target_min, target_max)
        boundaries = [0.0] + splits + [total]
        lengths = [boundaries[i + 1] - boundaries[i] for i in range(len(boundaries) - 1)]
        for seg_len in lengths:
            assert target_min <= seg_len <= target_max + 1.0  # +1 for rounding

    def test_very_short_way_no_splits(self):
        assert _compute_length_split_distances(5.0, 20.0, 50.0) == []

    def test_rural_300m_way_reasonable_split_count(self):
        splits = _compute_length_split_distances(300.0, 50.0, 100.0)
        n_segments = len(splits) + 1
        assert 3 <= n_segments <= 6


# ---------------------------------------------------------------------------
# _split_linestring
# ---------------------------------------------------------------------------

class TestSplitLinestring:
    @pytest.fixture
    def line100(self):
        return LineString([(0, 0), (100, 0)])

    def test_no_splits_returns_single_segment(self, line100):
        result = _split_linestring(line100, [])
        assert len(result) == 1
        assert result[0].length == pytest.approx(100.0)

    def test_split_at_midpoint_gives_two_equal_halves(self, line100):
        result = _split_linestring(line100, [50.0])
        assert len(result) == 2
        assert result[0].length == pytest.approx(50.0)
        assert result[1].length == pytest.approx(50.0)

    def test_total_length_preserved(self, line100):
        result = _split_linestring(line100, [25.0, 60.0])
        assert sum(s.length for s in result) == pytest.approx(100.0, rel=1e-6)

    def test_two_valid_splits_give_three_segments(self, line100):
        result = _split_linestring(line100, [33.0, 66.0])
        assert len(result) == 3

    def test_split_too_close_to_start_is_skipped(self, line100):
        result = _split_linestring(line100, [2.0], min_length=5.0)
        assert len(result) == 1

    def test_split_too_close_to_end_is_skipped(self, line100):
        result = _split_linestring(line100, [98.0], min_length=5.0)
        assert len(result) == 1

    def test_adjacent_splits_within_min_length_deduplicated(self, line100):
        # 50 and 52 are only 2m apart → second is dropped (min_length=5)
        result = _split_linestring(line100, [50.0, 52.0], min_length=5.0)
        assert len(result) == 2

    def test_all_segments_are_nonempty_linestrings(self, line100):
        result = _split_linestring(line100, [30.0, 70.0])
        assert all(isinstance(s, LineString) for s in result)
        assert all(not s.is_empty for s in result)
        assert all(s.length > 0 for s in result)
