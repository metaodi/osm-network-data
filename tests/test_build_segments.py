"""Tests for build_segments pure helpers and Parquet I/O round-trip."""

import pandas as pd
import pytest

from build_segments import (
    _ensure_columns,
    _normalise_osmid,
    _stringify_list,
    load_segments,
    save_segments,
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
