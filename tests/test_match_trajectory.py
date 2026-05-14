"""Tests for match_trajectory pure helpers and mocked network calls."""

import requests
import pytest
from unittest.mock import MagicMock, patch

from match_trajectory import (
    GpsPoint,
    ValhallaError,
    _check_valhalla_alive,
    _coerce_points,
    _decode_polyline6,
    _edge_geometry,
    call_valhalla,
    parse_valhalla_response,
)


# ---------------------------------------------------------------------------
# Reference encoder (test-only) — used to generate known polyline6 strings
# so that _decode_polyline6 can be verified against a roundtrip.
# ---------------------------------------------------------------------------

def _encode_polyline6(lonlat_pairs):
    """Minimal polyline6 encoder that mirrors _decode_polyline6's output format."""
    def encode_coord(value, prev_scaled):
        scaled = round(value * 1e6)
        delta = scaled - prev_scaled
        e = delta << 1
        if delta < 0:
            e = ~e
        chars = []
        while e >= 0x20:
            chars.append(chr((0x20 | (e & 0x1F)) + 63))
            e >>= 5
        chars.append(chr(e + 63))
        return "".join(chars), scaled

    result = []
    prev_lat, prev_lng = 0, 0
    for lon, lat in lonlat_pairs:
        lat_str, prev_lat = encode_coord(lat, prev_lat)
        lng_str, prev_lng = encode_coord(lon, prev_lng)
        result.extend([lat_str, lng_str])
    return "".join(result)


# ---------------------------------------------------------------------------
# _decode_polyline6
# ---------------------------------------------------------------------------

class TestDecodePolyline6:
    def test_empty_string_returns_empty_list(self):
        assert _decode_polyline6("") == []

    def test_known_single_unit_deltas(self):
        # 'A' = chr(2+63) → b=2, even → delta=+1.
        # "AAAA" encodes two points each with lat delta=1, lng delta=1.
        result = _decode_polyline6("AAAA")
        assert len(result) == 2
        assert result[0] == pytest.approx((1e-6, 1e-6))
        assert result[1] == pytest.approx((2e-6, 2e-6))

    def test_negative_delta(self):
        # '@' = chr(1+63) → b=1, odd → delta=~(0)=-1.
        # "AA@@" → (1e-6, 1e-6) then back to (0, 0).
        result = _decode_polyline6("AA@@")
        assert len(result) == 2
        assert result[0] == pytest.approx((1e-6, 1e-6))
        assert result[1] == pytest.approx((0.0, 0.0))

    def test_roundtrip_positive_coordinates(self):
        coords = [(8.5, 47.3), (8.6, 47.4), (8.7, 47.5)]
        decoded = _decode_polyline6(_encode_polyline6(coords))
        assert len(decoded) == len(coords)
        for (exp_lon, exp_lat), (got_lon, got_lat) in zip(coords, decoded):
            assert got_lon == pytest.approx(exp_lon, abs=1e-6)
            assert got_lat == pytest.approx(exp_lat, abs=1e-6)

    def test_roundtrip_negative_longitude(self):
        coords = [(-73.9857, 40.7484), (-74.006, 40.7128)]
        decoded = _decode_polyline6(_encode_polyline6(coords))
        for (exp_lon, exp_lat), (got_lon, got_lat) in zip(coords, decoded):
            assert got_lon == pytest.approx(exp_lon, abs=1e-6)
            assert got_lat == pytest.approx(exp_lat, abs=1e-6)

    def test_roundtrip_single_point(self):
        coords = [(9.02, 46.19)]
        decoded = _decode_polyline6(_encode_polyline6(coords))
        assert len(decoded) == 1
        assert decoded[0] == pytest.approx(coords[0], abs=1e-6)

    def test_output_is_lon_lat_order(self):
        # Verify the decoder returns (lon, lat) not (lat, lon).
        # Encode a point where lon != lat so we can distinguish.
        coords = [(8.0, 46.0)]  # lon=8, lat=46
        lon, lat = _decode_polyline6(_encode_polyline6(coords))[0]
        assert lon == pytest.approx(8.0, abs=1e-6)
        assert lat == pytest.approx(46.0, abs=1e-6)


# ---------------------------------------------------------------------------
# GpsPoint
# ---------------------------------------------------------------------------

class TestGpsPoint:
    def test_to_valhalla_without_time(self):
        p = GpsPoint(lat=46.0, lon=8.0)
        result = p.to_valhalla()
        assert result == {"lat": 46.0, "lon": 8.0}
        assert "time" not in result

    def test_to_valhalla_with_time(self):
        p = GpsPoint(lat=46.0, lon=8.0, time=1_000_000.0)
        result = p.to_valhalla()
        assert result == {"lat": 46.0, "lon": 8.0, "time": 1_000_000.0}

    def test_default_time_is_none(self):
        assert GpsPoint(lat=0.0, lon=0.0).time is None


# ---------------------------------------------------------------------------
# _coerce_points
# ---------------------------------------------------------------------------

class TestCoercePoints:
    def test_accepts_gps_point_objects(self):
        pts = [GpsPoint(lat=46.0, lon=8.0), GpsPoint(lat=46.1, lon=8.1)]
        result = _coerce_points(pts)
        assert len(result) == 2
        assert result[0].lat == 46.0
        assert result[1].lon == 8.1

    def test_accepts_dicts(self):
        pts = [{"lat": 46.0, "lon": 8.0}, {"lat": 46.1, "lon": 8.1}]
        result = _coerce_points(pts)
        assert result[0].lat == 46.0
        assert result[0].lon == 8.0

    def test_dict_with_optional_time(self):
        pts = [
            {"lat": 46.0, "lon": 8.0, "time": 1000.0},
            {"lat": 46.1, "lon": 8.1, "time": 1001.0},
        ]
        result = _coerce_points(pts)
        assert result[0].time == 1000.0
        assert result[1].time == 1001.0

    def test_dict_without_time_gives_none(self):
        pts = [{"lat": 46.0, "lon": 8.0}, {"lat": 46.1, "lon": 8.1}]
        assert _coerce_points(pts)[0].time is None

    def test_single_point_raises_value_error(self):
        with pytest.raises(ValueError, match="at least 2"):
            _coerce_points([GpsPoint(lat=46.0, lon=8.0)])

    def test_empty_list_raises_value_error(self):
        with pytest.raises(ValueError, match="at least 2"):
            _coerce_points([])

    def test_unsupported_type_raises_type_error(self):
        pts = [GpsPoint(lat=46.0, lon=8.0), "not_a_point"]
        with pytest.raises(TypeError):
            _coerce_points(pts)

    def test_mixed_gps_point_and_dict(self):
        pts = [GpsPoint(lat=46.0, lon=8.0), {"lat": 46.1, "lon": 8.1}]
        result = _coerce_points(pts)
        assert len(result) == 2
        assert isinstance(result[0], GpsPoint)
        assert isinstance(result[1], GpsPoint)


# ---------------------------------------------------------------------------
# _edge_geometry
# ---------------------------------------------------------------------------

class TestEdgeGeometry:
    def test_normal_case_returns_linestring(self):
        shape = [(0.0, 0.0), (1.0, 0.0), (2.0, 0.0)]
        edge = {"begin_shape_index": 0, "end_shape_index": 2}
        geom = _edge_geometry(edge, shape)
        assert geom is not None
        assert geom.geom_type == "LineString"
        assert list(geom.coords) == [(0.0, 0.0), (1.0, 0.0), (2.0, 0.0)]

    def test_sub_slice_of_shape(self):
        shape = [(0.0, 0.0), (1.0, 0.0), (2.0, 0.0), (3.0, 0.0)]
        edge = {"begin_shape_index": 1, "end_shape_index": 2}
        geom = _edge_geometry(edge, shape)
        assert list(geom.coords) == [(1.0, 0.0), (2.0, 0.0)]

    def test_missing_begin_index_returns_none(self):
        edge = {"end_shape_index": 2}
        assert _edge_geometry(edge, [(0.0, 0.0), (1.0, 0.0)]) is None

    def test_missing_end_index_returns_none(self):
        edge = {"begin_shape_index": 0}
        assert _edge_geometry(edge, [(0.0, 0.0), (1.0, 0.0)]) is None

    def test_single_coord_slice_returns_none(self):
        # begin == end → slice has only one point, not enough for a LineString
        edge = {"begin_shape_index": 1, "end_shape_index": 1}
        assert _edge_geometry(edge, [(0.0, 0.0), (1.0, 0.0)]) is None

    def test_empty_shape_returns_none(self):
        edge = {"begin_shape_index": 0, "end_shape_index": 1}
        assert _edge_geometry(edge, []) is None


# ---------------------------------------------------------------------------
# parse_valhalla_response
# ---------------------------------------------------------------------------

class TestParseValhallaResponse:
    def test_normal_response_columns_and_values(self, sample_valhalla_response):
        gdf = parse_valhalla_response(sample_valhalla_response)
        assert len(gdf) == 1
        row = gdf.iloc[0]
        assert row["match_edge_idx"] == 0
        assert row["way_id"] == 123
        assert row["names"] == "Test Road"
        assert row["road_class"] == "primary"
        assert row.geometry is not None
        assert row.geometry.geom_type == "LineString"

    def test_crs_is_wgs84(self, sample_valhalla_response):
        gdf = parse_valhalla_response(sample_valhalla_response)
        assert gdf.crs.to_epsg() == 4326

    def test_empty_edges_returns_empty_geodataframe(self):
        gdf = parse_valhalla_response({"edges": [], "shape": ""})
        assert len(gdf) == 0

    def test_missing_shape_key_gives_none_geometry(self):
        response = {
            "edges": [{"way_id": 1, "begin_shape_index": 0, "end_shape_index": 1}]
        }
        gdf = parse_valhalla_response(response)
        assert len(gdf) == 1
        assert gdf.iloc[0].geometry is None

    def test_names_list_joined_to_string(self):
        response = {
            "edges": [
                {
                    "way_id": 1,
                    "names": ["Road A", "Road B"],
                    "begin_shape_index": 0,
                    "end_shape_index": 1,
                }
            ],
            "shape": "AAAA",
        }
        assert parse_valhalla_response(response).iloc[0]["names"] == "Road A, Road B"

    def test_empty_names_list_gives_none(self):
        response = {
            "edges": [
                {"way_id": 1, "names": [], "begin_shape_index": 0, "end_shape_index": 1}
            ],
            "shape": "AAAA",
        }
        assert parse_valhalla_response(response).iloc[0]["names"] is None

    def test_multiple_edges_indexed_correctly(self):
        response = {
            "edges": [
                {"way_id": 10, "begin_shape_index": 0, "end_shape_index": 1},
                {"way_id": 20, "begin_shape_index": 1, "end_shape_index": 1},
            ],
            "shape": "AAAA",
        }
        gdf = parse_valhalla_response(response)
        assert len(gdf) == 2
        assert gdf.iloc[0]["match_edge_idx"] == 0
        assert gdf.iloc[1]["match_edge_idx"] == 1
        assert gdf.iloc[0]["way_id"] == 10
        assert gdf.iloc[1]["way_id"] == 20


# ---------------------------------------------------------------------------
# _check_valhalla_alive (mocked)
# ---------------------------------------------------------------------------

class TestCheckValhallaAlive:
    def test_raises_valhalla_error_on_connection_error(self):
        with patch("match_trajectory.requests.get") as mock_get:
            mock_get.side_effect = requests.ConnectionError("refused")
            with pytest.raises(ValhallaError, match="not reachable"):
                _check_valhalla_alive("http://localhost:9999")

    def test_raises_valhalla_error_on_http_error(self):
        with patch("match_trajectory.requests.get") as mock_get:
            mock_resp = MagicMock()
            mock_resp.raise_for_status.side_effect = requests.HTTPError("503")
            mock_get.return_value = mock_resp
            with pytest.raises(ValhallaError):
                _check_valhalla_alive("http://localhost:9999")

    def test_does_not_raise_when_service_is_alive(self):
        with patch("match_trajectory.requests.get") as mock_get:
            mock_resp = MagicMock()
            mock_resp.raise_for_status.return_value = None
            mock_get.return_value = mock_resp
            _check_valhalla_alive("http://localhost:8002")  # must not raise


# ---------------------------------------------------------------------------
# call_valhalla (mocked)
# ---------------------------------------------------------------------------

class TestCallValhalla:
    _two_points = [GpsPoint(46.0, 8.0), GpsPoint(46.1, 8.1)]

    def test_returns_parsed_json_on_success(self, sample_valhalla_response):
        with patch("match_trajectory.requests.post") as mock_post:
            mock_resp = MagicMock()
            mock_resp.status_code = 200
            mock_resp.json.return_value = sample_valhalla_response
            mock_post.return_value = mock_resp
            result = call_valhalla(self._two_points)
        assert result == sample_valhalla_response

    def test_raises_valhalla_error_on_non_200(self):
        with patch("match_trajectory.requests.post") as mock_post:
            mock_resp = MagicMock()
            mock_resp.status_code = 400
            mock_resp.text = "Bad Request"
            mock_post.return_value = mock_resp
            with pytest.raises(ValhallaError, match="HTTP 400"):
                call_valhalla(self._two_points)

    def test_raises_valhalla_error_on_request_exception(self):
        with patch("match_trajectory.requests.post") as mock_post:
            mock_post.side_effect = requests.RequestException("timeout")
            with pytest.raises(ValhallaError, match="failed"):
                call_valhalla(self._two_points)
