"""Shared pytest fixtures for osm-network-data tests."""

import geopandas as gpd
import pytest
from shapely.geometry import LineString


@pytest.fixture
def minimal_segments():
    """Two-segment GeoDataFrame covering a small area near Bellinzona."""
    return gpd.GeoDataFrame(
        {
            "segment_id": ["v0.1.0-00000000", "v0.1.0-00000001"],
            "network_version": ["v0.1.0", "v0.1.0"],
            "osmid": [100, 200],
            "osmid_norm": [[100], [200]],
            "u": [1, 2],
            "v": [2, 3],
            "key": [0, 0],
            "name": ["Main St", "Side St"],
            "highway": ["primary", "secondary"],
            "oneway": [False, False],
            "maxspeed": ["50", "30"],
            "junction": [None, None],
            "length": [100.0, 50.0],
            "is_roundabout": [False, False],
            "starts_or_ends_at_intersection": [True, False],
            "geometry": [
                LineString([(9.0, 46.2), (9.01, 46.2)]),
                LineString([(9.01, 46.2), (9.01, 46.21)]),
            ],
        },
        geometry="geometry",
        crs="EPSG:4326",
    )


@pytest.fixture
def minimal_matched_edges():
    """Single matched edge spatially aligned with the first segment."""
    return gpd.GeoDataFrame(
        {
            "match_edge_idx": [0],
            "way_id": [100],
            "names": ["Main St"],
            "road_class": ["primary"],
            "speed": [50],
            "matched_length": [0.1],
            "geometry": [LineString([(9.0, 46.2), (9.01, 46.2)])],
        },
        geometry="geometry",
        crs="EPSG:4326",
    )


@pytest.fixture
def sample_valhalla_response():
    """Minimal dict resembling a Valhalla trace_attributes response.

    The shape 'AAAA' decodes (polyline6) to two points:
      (lon=1e-6, lat=1e-6) and (lon=2e-6, lat=2e-6).
    With begin_shape_index=0 and end_shape_index=1 the edge geometry
    is a LineString between those two points.
    """
    return {
        "edges": [
            {
                "way_id": 123,
                "names": ["Test Road"],
                "road_class": "primary",
                "speed": 50,
                "length": 0.1,
                "begin_shape_index": 0,
                "end_shape_index": 1,
            }
        ],
        "shape": "AAAA",
    }
