"""Build the analysis segment model from OSM via OSMnx.

The graph is downloaded without simplification so that intermediate nodes
(crossings, traffic signals, stop signs, give-way signs) are preserved.
Each OSM way is then split into sub-segments at two kinds of breakpoints:

1. Special nodes  — crossings, traffic signals, stop/give-way signs.
2. Length budget  — 20–50 m in urban context, 50–100 m in rural context.

The output Parquet file is the canonical reference for downstream consumers.
Every segment carries a ``segment_id`` that is only stable within a given
``network_version``.
"""

from __future__ import annotations

import argparse
import logging
from collections import defaultdict
from typing import Iterable, Optional

import geopandas as gpd
import networkx as nx
import osmnx as ox
import pandas as pd
from shapely.geometry import LineString
from shapely.ops import substring

from config import (
    CRS_METRIC,
    CRS_WGS84,
    DEFAULT_NETWORK_TYPE,
    DEFAULT_PLACE_NAME,
    NETWORK_VERSION,
    SEGMENTS_PATH,
    ensure_directories,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Segment length targets (metres) per road context
# ---------------------------------------------------------------------------
TARGET_SEGMENT_M: dict[str, tuple[float, float]] = {
    "urban": (20.0, 50.0),
    "rural": (50.0, 100.0),
}
MIN_SEGMENT_M: float = 5.0  # discard slivers shorter than this

# Highway types that imply an urban context when maxspeed is unavailable.
URBAN_HIGHWAY_TYPES: frozenset[str] = frozenset(
    {
        "residential",
        "living_street",
        "unclassified",
        "pedestrian",
        "footway",
        "path",
        "cycleway",
        "service",
    }
)

# ---------------------------------------------------------------------------
# Output schema
# ---------------------------------------------------------------------------
SEGMENT_COLUMNS: list[str] = [
    "segment_id",
    "network_version",
    "osmid",
    "osmid_norm",
    "name",
    "highway",
    "oneway",
    "maxspeed",
    "junction",
    "length",
    "road_context",
    "is_roundabout",
    "starts_or_ends_at_intersection",
    "has_crossing",
    "has_traffic_signals",
    "has_stop_sign",
    "has_give_way",
    "cycleway",
    "surface",
    "lanes",
    "geometry",
]


# ---------------------------------------------------------------------------
# OSM tag normalisation helpers
# ---------------------------------------------------------------------------

def _normalise_osmid(value: object) -> list[int]:
    """Return OSM ids as a flat ``list[int]``."""
    if value is None:
        return []
    if isinstance(value, (list, tuple, set)):
        return [int(v) for v in value if v is not None]
    return [int(value)]


def _stringify_list(value: object) -> object:
    """Collapse list-valued OSM tags into a single comma-separated string."""
    if isinstance(value, (list, tuple)):
        non_null = [str(v) for v in value if v is not None and str(v) != "nan"]
        return ", ".join(non_null) if non_null else None
    return value


def _safe_str(value: object) -> Optional[str]:
    """Return string value or None for NaN / None."""
    if value is None:
        return None
    if isinstance(value, float) and pd.isna(value):
        return None
    return str(value)


def _ensure_columns(frame: pd.DataFrame, columns: Iterable[str]) -> None:
    """Add missing columns as ``None`` so the schema stays stable."""
    for col in columns:
        if col not in frame.columns:
            frame[col] = None


# ---------------------------------------------------------------------------
# Road context classification
# ---------------------------------------------------------------------------

def _classify_road_context(highway: object, maxspeed: object) -> str:
    """Return ``"urban"`` or ``"rural"`` based on maxspeed tag or highway type."""
    ms = _safe_str(maxspeed)
    if ms:
        speed_token = ms.strip().split()[0].replace(",", ".")
        try:
            speed = float(speed_token)
            if "mph" in ms.lower():
                speed *= 1.60934
            return "urban" if speed <= 50 else "rural"
        except ValueError:
            pass

    hw = _safe_str(highway)
    if hw:
        tokens = {t.strip().lower() for t in hw.split(",")}
        if tokens & URBAN_HIGHWAY_TYPES:
            return "urban"

    return "rural"


# ---------------------------------------------------------------------------
# Special node detection
# ---------------------------------------------------------------------------

def _detect_special_nodes(nodes_gdf: gpd.GeoDataFrame) -> dict[int, frozenset[str]]:
    """Return ``{node_id: frozenset_of_feature_names}`` for nodes with special OSM tags.

    Detected features: ``crossing``, ``traffic_signals``, ``stop``,
    ``give_way``, ``mini_roundabout``.
    """
    raw: dict[int, set[str]] = defaultdict(set)

    if "highway" in nodes_gdf.columns:
        hw_map = {
            "crossing": "crossing",
            "traffic_signals": "traffic_signals",
            "stop": "stop",
            "give_way": "give_way",
            "mini_roundabout": "mini_roundabout",
        }
        for tag_val, feature in hw_map.items():
            for node_id in nodes_gdf.index[nodes_gdf["highway"] == tag_val]:
                raw[int(node_id)].add(feature)

    if "crossing" in nodes_gdf.columns:
        mask = (
            nodes_gdf["crossing"].notna()
            & (nodes_gdf["crossing"] != "no")
            & (nodes_gdf["crossing"] != "none")
        )
        for node_id in nodes_gdf.index[mask]:
            raw[int(node_id)].add("crossing")

    return {k: frozenset(v) for k, v in raw.items()}


# ---------------------------------------------------------------------------
# Way geometry reconstruction
# ---------------------------------------------------------------------------

def _build_ordered_line(way_edges: pd.DataFrame) -> Optional[LineString]:
    """Reconstruct a single ordered LineString from (possibly unordered) way edges.

    ``way_edges`` must have integer ``u``, ``v`` columns and a ``geometry``
    column of LineString objects in the desired CRS.  For two-way roads both
    directed copies of each edge may be present; the function handles that by
    storing the geometry for every directed pair and picking the matching
    direction during path traversal.
    """
    # Store directed geometries and build undirected adjacency.
    edge_geoms: dict[tuple[int, int], LineString] = {}
    adjacency: dict[int, set[int]] = defaultdict(set)

    for _, row in way_edges.iterrows():
        u, v = int(row["u"]), int(row["v"])
        edge_geoms[(u, v)] = row.geometry
        adjacency[u].add(v)
        adjacency[v].add(u)

    if not adjacency:
        return None

    # Find an endpoint (degree-1 node) to start the traversal; fall back to
    # any node for closed loops (e.g. roundabouts).
    degrees = {n: len(nbrs) for n, nbrs in adjacency.items()}
    endpoints = [n for n, d in degrees.items() if d == 1]
    start = endpoints[0] if endpoints else next(iter(adjacency))

    # Traverse the path greedily.
    path: list[int] = [start]
    visited: set[int] = {start}
    current = start

    for _ in range(len(adjacency)):
        candidates = adjacency[current] - visited
        if not candidates:
            break
        nxt = next(iter(candidates))
        path.append(nxt)
        visited.add(nxt)
        current = nxt

    if len(path) < 2:
        return None

    # Build the ordered coordinate sequence from directed edge geometries.
    all_coords: list[tuple] = []
    for i in range(len(path) - 1):
        u, v = path[i], path[i + 1]
        if (u, v) in edge_geoms:
            coords = list(edge_geoms[(u, v)].coords)
        elif (v, u) in edge_geoms:
            coords = list(reversed(edge_geoms[(v, u)].coords))
        else:
            continue  # Edge missing; skip (gap in coverage)
        if all_coords:
            coords = coords[1:]  # Drop duplicated junction point
        all_coords.extend(coords)

    # For closed ways (loops) the greedy traversal stops before revisiting the
    # start node, so the closing segment is never processed above.
    if not endpoints and len(path) >= 2:
        last_n, first_n = path[-1], path[0]
        if (last_n, first_n) in edge_geoms:
            all_coords.extend(list(edge_geoms[(last_n, first_n)].coords)[1:])
        elif (first_n, last_n) in edge_geoms:
            all_coords.extend(list(reversed(edge_geoms[(first_n, last_n)].coords))[1:])

    return LineString(all_coords) if len(all_coords) >= 2 else None


# ---------------------------------------------------------------------------
# Splitting helpers
# ---------------------------------------------------------------------------

def _compute_length_split_distances(
    total_length: float,
    target_min: float,
    target_max: float,
) -> list[float]:
    """Return interior split distances so every chunk falls in [target_min, target_max].

    Returns an empty list when the way already fits in one segment.
    """
    if total_length <= target_max:
        return []
    target = (target_min + target_max) / 2.0
    n = max(2, round(total_length / target))
    step = total_length / n
    return [step * i for i in range(1, n)]


def _split_linestring(
    line: LineString,
    split_distances: list[float],
    min_length: float = MIN_SEGMENT_M,
) -> list[LineString]:
    """Split *line* (metric) at *split_distances*, dropping slivers < *min_length*.

    Split points that are closer than *min_length* to an endpoint or to each
    other are silently skipped to avoid degenerate segments.
    """
    total = line.length
    # Filter out splits that are too close to the endpoints or each other.
    filtered: list[float] = []
    for d in sorted(split_distances):
        if d <= min_length:
            continue
        if d >= total - min_length:
            continue
        if filtered and d - filtered[-1] < min_length:
            continue
        filtered.append(d)

    boundaries = [0.0] + filtered + [total]
    segments: list[LineString] = []
    for i in range(len(boundaries) - 1):
        seg = substring(line, boundaries[i], boundaries[i + 1])
        if not seg.is_empty and seg.length >= min_length / 2:
            segments.append(seg)

    return segments if segments else [line]


def _node_distances_along_line(
    node_ids: set[int],
    nodes_metric: gpd.GeoDataFrame,
    line: LineString,
    feature_map: dict[int, frozenset[str]],
    total_length: float,
    margin: float = 1.0,
) -> dict[float, frozenset[str]]:
    """Project special nodes onto *line* and return ``{distance: features}``.

    Nodes at the endpoints (within *margin* metres) are excluded because the
    endpoint flags are handled separately.
    """
    result: dict[float, frozenset[str]] = {}
    for nid in node_ids:
        if nid not in feature_map:
            continue
        if nid not in nodes_metric.index:
            continue
        pt = nodes_metric.loc[nid, "geometry"]
        d = line.project(pt)
        if d <= margin or d >= total_length - margin:
            continue
        if d in result:
            result[d] = result[d] | feature_map[nid]
        else:
            result[d] = feature_map[nid]
    return result


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def build_segments(
    place_name: str = DEFAULT_PLACE_NAME,
    network_type: str = DEFAULT_NETWORK_TYPE,
    network_version: str = NETWORK_VERSION,
) -> gpd.GeoDataFrame:
    """Download an OSM street network and build sub-segment analysis units.

    Parameters
    ----------
    place_name:
        Anything OSMnx can geocode.
    network_type:
        OSMnx network filter (``drive``, ``walk``, ``bike``, ``all``...).
    network_version:
        Version label stored alongside every segment.  Bump whenever the
        segment definition changes.

    The graph is downloaded **without** simplification (``simplify=False``) so
    that intermediate OSM nodes — crossings, traffic signals, stop signs — are
    preserved as natural split points within each way.
    """
    logger.info(
        "Downloading OSM graph (simplify=False) for %r (network_type=%s)",
        place_name,
        network_type,
    )
    graph = ox.graph_from_place(place_name, network_type=network_type, simplify=False)
    logger.info("Graph: %d nodes, %d edges", graph.number_of_nodes(), graph.number_of_edges())

    nodes_gdf, edges_gdf = ox.graph_to_gdfs(graph, nodes=True, edges=True)
    edges_gdf = edges_gdf.reset_index()  # u, v, key become regular columns

    # Intersection detection on the undirected graph (degree >= 3).
    undirected = graph.to_undirected()
    degrees: dict[int, int] = dict(nx.degree(undirected))
    intersection_nodes: set[int] = {n for n, d in degrees.items() if d >= 3}

    # Special node detection (crossings, signals, stops, give-way).
    special_nodes = _detect_special_nodes(nodes_gdf)
    logger.info("Found %d nodes with special infrastructure tags", len(special_nodes))

    # Project to metric CRS for all distance arithmetic.
    nodes_metric = nodes_gdf.to_crs(CRS_METRIC)
    edges_metric = edges_gdf.copy()
    edges_metric["geometry"] = edges_gdf.to_crs(CRS_METRIC)["geometry"]

    # Ensure all expected tag columns exist (some may be absent for small graphs).
    _ensure_columns(
        edges_metric,
        ["name", "highway", "oneway", "maxspeed", "junction", "cycleway", "surface", "lanes"],
    )

    # Normalise osmid to a scalar so we can group by it.
    # With simplify=False each edge belongs to exactly one OSM way.
    edges_metric["_osmid_scalar"] = edges_metric["osmid"].apply(
        lambda v: int(v[0]) if isinstance(v, (list, tuple)) and v else int(v)
    )

    grouped = edges_metric.groupby("_osmid_scalar", sort=False)
    logger.info("Processing %d unique OSM ways → building sub-segments …", len(grouped))

    all_rows: list[dict] = []

    for osmid_int, way_edges in grouped:
        # Reconstruct the ordered way geometry from directed edge geometries.
        merged = _build_ordered_line(way_edges)
        if merged is None or merged.is_empty or merged.length < MIN_SEGMENT_M:
            continue
        total_length = merged.length  # metres (metric CRS)

        # --- Metadata from the first edge (uniform across the OSM way) ---
        first = way_edges.iloc[0]
        highway = _stringify_list(first.get("highway"))
        maxspeed = _stringify_list(first.get("maxspeed"))
        junction = _stringify_list(first.get("junction"))
        name = _stringify_list(first.get("name"))
        oneway = first.get("oneway")
        cycleway = _stringify_list(first.get("cycleway"))
        surface = _stringify_list(first.get("surface"))
        lanes = _stringify_list(first.get("lanes"))

        road_context = _classify_road_context(highway, maxspeed)
        is_roundabout = isinstance(junction, str) and "roundabout" in junction

        # --- Node IDs that appear along this way ---
        way_node_ids: set[int] = (
            set(way_edges["u"].astype(int)) | set(way_edges["v"].astype(int))
        )

        # --- Project special nodes onto the merged line ---
        special_dist_map = _node_distances_along_line(
            way_node_ids, nodes_metric, merged, special_nodes, total_length
        )

        # --- Intersection node distances (for starts_or_ends_at_intersection) ---
        intersection_dists: set[float] = set()
        for nid in way_node_ids:
            if nid in intersection_nodes and nid in nodes_metric.index:
                d = merged.project(nodes_metric.loc[nid, "geometry"])
                intersection_dists.add(d)

        # --- Length-based split distances ---
        target_min, target_max = TARGET_SEGMENT_M[road_context]
        length_splits = _compute_length_split_distances(total_length, target_min, target_max)

        # Combine: special nodes are natural breakpoints; length splits fill gaps.
        all_split_distances = sorted(set(special_dist_map.keys()) | set(length_splits))

        # --- Geometry splitting ---
        sub_lines_metric = _split_linestring(merged, all_split_distances)

        # --- Build one output row per sub-segment ---
        cumulative = 0.0
        for sub_line in sub_lines_metric:
            sub_len = sub_line.length
            seg_start = cumulative
            seg_end = cumulative + sub_len
            cumulative = seg_end

            TOL = 2.0  # metres — tolerance for snapping distances to split points
            has_crossing = False
            has_traffic_signals = False
            has_stop_sign = False
            has_give_way = False

            for d, features in special_dist_map.items():
                if abs(d - seg_start) <= TOL or abs(d - seg_end) <= TOL:
                    has_crossing |= "crossing" in features
                    has_traffic_signals |= "traffic_signals" in features
                    has_stop_sign |= "stop" in features
                    has_give_way |= "give_way" in features

            starts_or_ends_at_intersection = any(
                abs(d - seg_start) <= TOL or abs(d - seg_end) <= TOL
                for d in intersection_dists
            )

            # Convert back to WGS84 for storage.
            sub_wgs84 = (
                gpd.GeoSeries([sub_line], crs=CRS_METRIC).to_crs(CRS_WGS84).iloc[0]
            )

            all_rows.append(
                {
                    "osmid": osmid_int,
                    "osmid_norm": [osmid_int],
                    "name": name,
                    "highway": highway,
                    "oneway": oneway,
                    "maxspeed": maxspeed,
                    "junction": junction,
                    "length": sub_len,
                    "road_context": road_context,
                    "is_roundabout": is_roundabout,
                    "starts_or_ends_at_intersection": starts_or_ends_at_intersection,
                    "has_crossing": has_crossing,
                    "has_traffic_signals": has_traffic_signals,
                    "has_stop_sign": has_stop_sign,
                    "has_give_way": has_give_way,
                    "cycleway": cycleway,
                    "surface": surface,
                    "lanes": lanes,
                    "network_version": network_version,
                    "geometry": sub_wgs84,
                }
            )

    if not all_rows:
        raise RuntimeError(
            "No segments were built — check the place name and network type."
        )

    segments = gpd.GeoDataFrame(all_rows, geometry="geometry", crs=CRS_WGS84)
    segments = segments.reset_index(drop=True)
    segments["segment_id"] = [
        f"{network_version}-{i:08d}" for i in range(len(segments))
    ]

    _ensure_columns(segments, SEGMENT_COLUMNS)
    result = segments.loc[:, SEGMENT_COLUMNS].copy()

    logger.info("Built %d segments (network_version=%s)", len(result), network_version)
    return result


def save_segments(segments: gpd.GeoDataFrame, path=SEGMENTS_PATH) -> None:
    """Persist the segment GeoDataFrame as Parquet."""
    ensure_directories()
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

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    segments = build_segments(
        place_name=args.place,
        network_type=args.network_type,
        network_version=args.network_version,
    )
    save_segments(segments)


if __name__ == "__main__":
    main()
