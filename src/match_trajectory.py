"""Call Valhalla's ``trace_attributes`` endpoint and parse the response."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Optional, Sequence

import geopandas as gpd
import requests
from shapely.geometry import LineString

from config import CRS_WGS84, VALHALLA_TIMEOUT_S, VALHALLA_URL

logger = logging.getLogger(__name__)


# Polyline6 precision used by Valhalla for the ``shape`` field.
_POLYLINE_PRECISION = 1e6

# Attribute filter we ask Valhalla to return.
TRACE_ATTRIBUTE_FILTER: list[str] = [
    "edge.way_id",
    "edge.names",
    "edge.road_class",
    "edge.speed",
    "edge.length",
    "edge.begin_shape_index",
    "edge.end_shape_index",
    "shape",
]


class ValhallaError(RuntimeError):
    """Raised when Valhalla cannot be reached or returns an error."""


@dataclass
class GpsPoint:
    """A single GPS sample."""

    lat: float
    lon: float
    time: Optional[float] = None  # epoch seconds; optional

    def to_valhalla(self) -> dict[str, Any]:
        payload: dict[str, Any] = {"lat": self.lat, "lon": self.lon}
        if self.time is not None:
            payload["time"] = self.time
        return payload


def _coerce_points(points: Iterable[Any]) -> list[GpsPoint]:
    """Accept dicts or GpsPoint objects and return a normalised list."""
    out: list[GpsPoint] = []
    for raw in points:
        if isinstance(raw, GpsPoint):
            out.append(raw)
        elif isinstance(raw, dict):
            out.append(GpsPoint(lat=float(raw["lat"]), lon=float(raw["lon"]), time=raw.get("time")))
        else:
            raise TypeError(f"Unsupported point type: {type(raw)!r}")
    if len(out) < 2:
        raise ValueError("Need at least 2 GPS points for map matching.")
    return out


def _decode_polyline6(encoded: str) -> list[tuple[float, float]]:
    """Decode a Valhalla polyline6 string into ``[(lon, lat), ...]``.

    Valhalla uses precision 6 (i.e. multiplied by 1e6) - distinct from
    Google's classic polyline5.
    """
    coords: list[tuple[float, float]] = []
    index = 0
    lat = 0
    lng = 0
    length = len(encoded)

    while index < length:
        shift = 0
        result = 0
        while True:
            b = ord(encoded[index]) - 63
            index += 1
            result |= (b & 0x1F) << shift
            shift += 5
            if b < 0x20:
                break
        dlat = ~(result >> 1) if result & 1 else (result >> 1)
        lat += dlat

        shift = 0
        result = 0
        while True:
            b = ord(encoded[index]) - 63
            index += 1
            result |= (b & 0x1F) << shift
            shift += 5
            if b < 0x20:
                break
        dlng = ~(result >> 1) if result & 1 else (result >> 1)
        lng += dlng

        coords.append((lng / _POLYLINE_PRECISION, lat / _POLYLINE_PRECISION))
    return coords


def _check_valhalla_alive(base_url: str = VALHALLA_URL) -> None:
    """Raise :class:`ValhallaError` if Valhalla is not reachable."""
    try:
        resp = requests.get(f"{base_url}/status", timeout=5)
        resp.raise_for_status()
    except requests.RequestException as exc:
        raise ValhallaError(
            f"Valhalla not reachable at {base_url}. "
            "Did you start `docker compose up`? "
        ) from exc


def call_valhalla(
    points: Sequence[GpsPoint],
    base_url: str = VALHALLA_URL,
    costing: str = "auto",
    shape_match: str = "map_snap",
) -> dict[str, Any]:
    """POST to ``trace_attributes`` and return the parsed JSON."""
    payload: dict[str, Any] = {
        "shape": [p.to_valhalla() for p in points],
        "costing": costing,
        "shape_match": shape_match,
        "filters": {
            "attributes": TRACE_ATTRIBUTE_FILTER,
            "action": "include",
        },
    }
    url = f"{base_url}/trace_attributes"
    logger.info("POST %s with %d points (costing=%s)", url, len(points), costing)
    try:
        resp = requests.post(url, json=payload, timeout=VALHALLA_TIMEOUT_S)
    except requests.RequestException as exc:
        raise ValhallaError(f"Request to Valhalla failed: {exc}") from exc

    if resp.status_code != 200:
        raise ValhallaError(
            f"Valhalla returned HTTP {resp.status_code}: {resp.text[:500]}"
        )
    return resp.json()


def _edge_geometry(
    edge: dict[str, Any], full_shape: list[tuple[float, float]]
) -> Optional[LineString]:
    """Slice the full matched shape by the edge's begin/end indices."""
    begin = edge.get("begin_shape_index")
    end = edge.get("end_shape_index")
    if begin is None or end is None:
        return None
    coords = full_shape[begin : end + 1]
    if len(coords) < 2:
        return None
    return LineString(coords)


def parse_valhalla_response(response: dict[str, Any]) -> gpd.GeoDataFrame:
    """Turn a ``trace_attributes`` response into a ``matched_edges`` GeoDataFrame."""
    edges = response.get("edges", []) or []
    encoded_shape = response.get("shape")
    full_shape: list[tuple[float, float]] = (
        _decode_polyline6(encoded_shape) if encoded_shape else []
    )

    rows: list[dict[str, Any]] = []
    for idx, edge in enumerate(edges):
        names = edge.get("names")
        if isinstance(names, list):
            names_value: Optional[str] = ", ".join(str(n) for n in names) if names else None
        else:
            names_value = names

        rows.append(
            {
                "match_edge_idx": idx,
                "way_id": edge.get("way_id"),
                "names": names_value,
                "road_class": edge.get("road_class"),
                "speed": edge.get("speed"),
                "matched_length": edge.get("length"),
                "geometry": _edge_geometry(edge, full_shape),
            }
        )

    if rows:
        matched_edges = gpd.GeoDataFrame(rows, geometry="geometry", crs=CRS_WGS84)
    else:
        matched_edges = gpd.GeoDataFrame({"geometry": gpd.GeoSeries(crs=CRS_WGS84)})
    logger.info("Parsed %d matched edges from Valhalla response", len(matched_edges))
    return matched_edges


def match_trajectory(
    points: Iterable[Any],
    base_url: str = VALHALLA_URL,
    costing: str = "auto",
    shape_match: str = "map_snap",
    debug_json_path: Optional[Path] = None,
) -> gpd.GeoDataFrame:
    """End-to-end: validate Valhalla, send points, return matched edges.

    Parameters
    ----------
    points:
        Iterable of :class:`GpsPoint` or ``{"lat": ..., "lon": ..., "time": ...}``.
    debug_json_path:
        If given, the raw Valhalla response is written there for inspection.
    """
    gps_points = _coerce_points(points)
    _check_valhalla_alive(base_url)
    response = call_valhalla(gps_points, base_url=base_url, costing=costing, shape_match=shape_match)

    if debug_json_path is not None:
        debug_json_path = Path(debug_json_path)
        debug_json_path.parent.mkdir(parents=True, exist_ok=True)
        debug_json_path.write_text(json.dumps(response, indent=2))
        logger.info("Wrote debug Valhalla response to %s", debug_json_path)

    return parse_valhalla_response(response)
