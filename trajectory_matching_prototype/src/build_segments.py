"""Build the analysis segment model from OSM via OSMnx.

The output Parquet file is the canonical reference any downstream consumer
should use to talk about "segments". Every segment carries an internal
`segment_id` that is only stable within a given `network_version`.
"""

from __future__ import annotations

import argparse
import logging
from typing import Iterable

import geopandas as gpd
import networkx as nx
import osmnx as ox
import pandas as pd

from config import (
    CRS_WGS84,
    DEFAULT_NETWORK_TYPE,
    DEFAULT_PLACE_NAME,
    NETWORK_VERSION,
    SEGMENTS_PATH,
    ensure_directories,
)

logger = logging.getLogger(__name__)

# Columns we want to keep on every road segment.
SEGMENT_COLUMNS: list[str] = [
    "segment_id",
    "network_version",
    "osmid",
    "osmid_norm",
    "u",
    "v",
    "key",
    "name",
    "highway",
    "oneway",
    "maxspeed",
    "junction",
    "length",
    "is_roundabout",
    "starts_or_ends_at_intersection",
    "geometry",
]


def _normalise_osmid(value: object) -> list[int]:
    """Return OSM ids as a flat ``list[int]``.

    OSMnx may return ``osmid`` either as a single int or as a list of ints
    when an edge represents the merged result of several OSM ways.
    """
    if value is None:
        return []
    if isinstance(value, (list, tuple, set)):
        return [int(v) for v in value if v is not None]
    return [int(value)]


def _stringify_list(value: object) -> object:
    """Collapse list-valued OSM tags into a single comma separated string.

    OSMnx returns some attributes (e.g. ``name``, ``highway``, ``maxspeed``)
    as a list when the underlying ways disagree. Parquet handles those badly
    when the dtype is mixed, so we pick the first non-null value and join
    the rest into a string for traceability.
    """
    if isinstance(value, (list, tuple)):
        non_null = [str(v) for v in value if v is not None and str(v) != "nan"]
        if not non_null:
            return None
        return ", ".join(non_null)
    return value


def _ensure_columns(frame: pd.DataFrame, columns: Iterable[str]) -> None:
    """Add missing columns as ``None`` so the schema stays stable."""
    for column in columns:
        if column not in frame.columns:
            frame[column] = None


def build_segments(
    place_name: str = DEFAULT_PLACE_NAME,
    network_type: str = DEFAULT_NETWORK_TYPE,
    network_version: str = NETWORK_VERSION,
) -> gpd.GeoDataFrame:
    """Download an OSM street network and turn it into a segment GeoDataFrame.

    Parameters
    ----------
    place_name:
        Anything OSMnx can geocode. For production, swap this for the same
        bounding box / polygon that was used to cut the PBF.
    network_type:
        OSMnx network filter (``drive``, ``walk``, ``bike``, ``all``...).
    network_version:
        Version label stored alongside every segment. Bump it whenever the
        segment definition changes.
    """
    logger.info("Downloading OSM graph for %r (network_type=%s)", place_name, network_type)
    graph = ox.graph_from_place(place_name, network_type=network_type, simplify=True)

    logger.info("Graph has %d nodes and %d edges", graph.number_of_nodes(), graph.number_of_edges())

    nodes_gdf, edges_gdf = ox.graph_to_gdfs(graph, nodes=True, edges=True)
    edges_gdf = edges_gdf.reset_index()  # u, v, key become columns

    # Node degrees on the undirected graph - used to flag intersections.
    undirected = graph.to_undirected()
    degrees: dict[int, int] = dict(nx.degree(undirected))

    _ensure_columns(
        edges_gdf,
        ["name", "highway", "oneway", "maxspeed", "junction", "length"],
    )

    edges_gdf["osmid_norm"] = edges_gdf["osmid"].apply(_normalise_osmid)
    for col in ("name", "highway", "maxspeed", "junction"):
        edges_gdf[col] = edges_gdf[col].apply(_stringify_list)

    edges_gdf["is_roundabout"] = edges_gdf["junction"].apply(
        lambda v: isinstance(v, str) and "roundabout" in v
    )
    edges_gdf["starts_or_ends_at_intersection"] = edges_gdf.apply(
        lambda row: degrees.get(int(row["u"]), 0) >= 3 or degrees.get(int(row["v"]), 0) >= 3,
        axis=1,
    )

    edges_gdf["network_version"] = network_version
    edges_gdf = edges_gdf.reset_index(drop=True)
    edges_gdf["segment_id"] = [
        f"{network_version}-{idx:08d}" for idx in range(len(edges_gdf))
    ]

    _ensure_columns(edges_gdf, SEGMENT_COLUMNS)
    segments = edges_gdf.loc[:, SEGMENT_COLUMNS].copy()
    segments = gpd.GeoDataFrame(segments, geometry="geometry", crs=CRS_WGS84)

    logger.info("Built %d segments (network_version=%s)", len(segments), network_version)
    return segments


def save_segments(segments: gpd.GeoDataFrame, path=SEGMENTS_PATH) -> None:
    """Persist the segment GeoDataFrame as Parquet."""
    ensure_directories()
    # GeoPandas Parquet writer needs the geometry column.
    segments.to_parquet(path, index=False)
    logger.info("Wrote %d segments to %s", len(segments), path)


def load_segments(path=SEGMENTS_PATH) -> gpd.GeoDataFrame:
    """Read the segment GeoDataFrame from Parquet."""
    logger.info("Loading segments from %s", path)
    return gpd.read_parquet(path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build OSMnx-based road segments.")
    parser.add_argument("--place", default=DEFAULT_PLACE_NAME, help="Place name for OSMnx.")
    parser.add_argument(
        "--network-type", default=DEFAULT_NETWORK_TYPE, help="OSMnx network filter."
    )
    parser.add_argument(
        "--network-version",
        default=NETWORK_VERSION,
        help="Version label stored on every segment.",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    segments = build_segments(
        place_name=args.place,
        network_type=args.network_type,
        network_version=args.network_version,
    )
    save_segments(segments)


if __name__ == "__main__":
    main()
