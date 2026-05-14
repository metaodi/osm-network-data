"""Join Valhalla matched edges onto the OSMnx segment model."""

from __future__ import annotations

import logging
from typing import Optional

import geopandas as gpd
import pandas as pd

from build_segments import load_segments
from config import CRS_METRIC, NETWORK_VERSION, SEGMENTS_PATH

logger = logging.getLogger(__name__)


# Final output columns - keep this list as the public contract.
OUTPUT_COLUMNS: list[str] = [
    "trajectory_id",
    "network_version",
    "match_edge_idx",
    "way_id",
    "segment_id",
    "match_quality",
    "distance_to_segment_m",
    "segment_name",
    "highway",
    "is_roundabout",
    "starts_or_ends_at_intersection",
]


def _explode_segments_by_osmid(segments: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Explode the ``osmid_norm`` list column into one row per OSM way id."""
    exploded = segments.explode(column="osmid_norm", ignore_index=True)
    exploded = exploded.rename(columns={"osmid_norm": "osmid_single"})
    exploded["osmid_single"] = pd.to_numeric(exploded["osmid_single"], errors="coerce")
    exploded = exploded.dropna(subset=["osmid_single"])
    exploded["osmid_single"] = exploded["osmid_single"].astype("int64")
    return exploded


def _spatial_pick(
    matched_edge_geom,
    candidates_metric: gpd.GeoDataFrame,
) -> tuple[Optional[str], Optional[float]]:
    """Pick the closest candidate segment for a matched edge geometry.

    Both inputs are expected to be in :data:`config.CRS_METRIC` already.
    Returns ``(segment_id, distance_m)`` or ``(None, None)``.
    """
    if matched_edge_geom is None or matched_edge_geom.is_empty or candidates_metric.empty:
        return None, None

    distances = candidates_metric.geometry.distance(matched_edge_geom)
    if distances.empty:
        return None, None
    best_idx = distances.idxmin()
    best_distance = float(distances.loc[best_idx])
    best_segment_id = candidates_metric.loc[best_idx, "segment_id"]
    return best_segment_id, best_distance


def map_matched_edges_to_segments(
    matched_edges: gpd.GeoDataFrame,
    segments: Optional[gpd.GeoDataFrame] = None,
    trajectory_id: str = "trajectory",
    network_version: str = NETWORK_VERSION,
) -> pd.DataFrame:
    """Map every matched edge to exactly one segment id.

    Strategy:
    1. Primary join on ``way_id`` (from Valhalla) ↔ ``osmid_norm`` (OSMnx).
    2. If the join yields multiple segments for the same ``way_id``, fall
       back to a spatial pick by reprojecting both sides to ``CRS_METRIC``
       and choosing the segment with the smallest distance to the matched
       edge geometry.
    3. If ``way_id`` is missing or unknown, do a pure spatial nearest-
       neighbour lookup against all segments.
    """
    if segments is None:
        segments = load_segments(SEGMENTS_PATH)

    if matched_edges.empty:
        logger.warning("No matched edges to map.")
        return pd.DataFrame(columns=OUTPUT_COLUMNS)

    segments_exploded = _explode_segments_by_osmid(segments)
    segments_metric = segments.to_crs(CRS_METRIC)
    matched_metric = matched_edges.to_crs(CRS_METRIC)

    rows: list[dict] = []
    for idx, edge in matched_edges.iterrows():
        way_id = edge.get("way_id")
        edge_geom_metric = matched_metric.geometry.iloc[idx]

        segment_id: Optional[str] = None
        match_quality: str = "unmatched"
        distance_m: Optional[float] = None

        candidates = (
            segments_exploded[segments_exploded["osmid_single"] == int(way_id)]
            if way_id is not None
            else segments_exploded.iloc[0:0]
        )

        if len(candidates) == 1:
            segment_id = candidates.iloc[0]["segment_id"]
            match_quality = "way_id"
            distance_m = 0.0
        elif len(candidates) > 1:
            candidates_metric = segments_metric[
                segments_metric["segment_id"].isin(candidates["segment_id"])
            ]
            segment_id, distance_m = _spatial_pick(edge_geom_metric, candidates_metric)
            match_quality = "way_id+spatial" if segment_id else "unmatched"
        else:
            # No way_id hit at all - last resort: pure spatial nearest neighbour.
            segment_id, distance_m = _spatial_pick(edge_geom_metric, segments_metric)
            match_quality = "spatial_only" if segment_id else "unmatched"

        if segment_id is None:
            seg_row = None
        else:
            seg_match = segments[segments["segment_id"] == segment_id]
            seg_row = seg_match.iloc[0] if not seg_match.empty else None

        rows.append(
            {
                "trajectory_id": trajectory_id,
                "network_version": network_version,
                "match_edge_idx": int(edge["match_edge_idx"]),
                "way_id": int(way_id) if way_id is not None else None,
                "segment_id": segment_id,
                "match_quality": match_quality,
                "distance_to_segment_m": distance_m,
                "segment_name": seg_row["name"] if seg_row is not None else None,
                "highway": seg_row["highway"] if seg_row is not None else None,
                "is_roundabout": bool(seg_row["is_roundabout"]) if seg_row is not None else None,
                "starts_or_ends_at_intersection": (
                    bool(seg_row["starts_or_ends_at_intersection"])
                    if seg_row is not None
                    else None
                ),
            }
        )

    result = pd.DataFrame(rows, columns=OUTPUT_COLUMNS)
    matched_count = result["segment_id"].notna().sum()
    logger.info(
        "Mapped %d / %d matched edges to segments (%.1f %%)",
        matched_count,
        len(result),
        100.0 * matched_count / max(len(result), 1),
    )
    return result
