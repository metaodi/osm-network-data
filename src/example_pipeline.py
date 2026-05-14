"""End-to-end demo:

GPS points -> Valhalla map matching -> OSMnx segments -> Parquet.

Run with::

    uv run python src/example_pipeline.py
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

from build_segments import build_segments, load_segments, save_segments
from config import (
    MATCHES_PATH,
    NETWORK_VERSION,
    SEGMENTS_PATH,
    VALHALLA_URL,
    ensure_directories,
)
from map_to_segments import map_matched_edges_to_segments
from match_trajectory import GpsPoint, ValhallaError, match_trajectory

logger = logging.getLogger(__name__)

# A short demo trajectory through Bellinzona (CH).
# Coordinates lifted from OSM along Viale Stazione - feel free to swap them
# for whatever your local PBF actually covers.
DEMO_TRAJECTORY: list[GpsPoint] = [
    GpsPoint(lat=46.19447, lon=9.02436, time=0.0),
    GpsPoint(lat=46.19478, lon=9.02478, time=5.0),
    GpsPoint(lat=46.19510, lon=9.02527, time=10.0),
    GpsPoint(lat=46.19551, lon=9.02585, time=15.0),
    GpsPoint(lat=46.19594, lon=9.02646, time=20.0),
    GpsPoint(lat=46.19641, lon=9.02711, time=25.0),
    GpsPoint(lat=46.19686, lon=9.02775, time=30.0),
]
DEMO_TRAJECTORY_ID = "demo-001"


def _ensure_segments_available():
    """Build segments on demand if the Parquet file isn't there yet."""
    if SEGMENTS_PATH.exists():
        return load_segments(SEGMENTS_PATH)
    logger.info("No segments file at %s - building from OSMnx now.", SEGMENTS_PATH)
    segments = build_segments(network_version=NETWORK_VERSION)
    save_segments(segments)
    return segments


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    ensure_directories()

    segments = _ensure_segments_available()

    debug_path = Path(MATCHES_PATH).with_name("example_valhalla_response.json")
    try:
        matched_edges = match_trajectory(
            DEMO_TRAJECTORY,
            base_url=VALHALLA_URL,
            debug_json_path=debug_path,
        )
    except ValhallaError as exc:
        logger.error("Map matching failed: %s", exc)
        return 1

    matched_segments = map_matched_edges_to_segments(
        matched_edges,
        segments=segments,
        trajectory_id=DEMO_TRAJECTORY_ID,
        network_version=NETWORK_VERSION,
    )

    matched_segments.to_parquet(MATCHES_PATH, index=False)
    logger.info("Saved matched segments to %s", MATCHES_PATH)

    # ---- Demo summary (intentionally uses print) ----
    n_points = len(DEMO_TRAJECTORY)
    n_matched_edges = len(matched_edges)
    n_mapped_segments = int(matched_segments["segment_id"].notna().sum())
    ratio = n_mapped_segments / n_matched_edges if n_matched_edges else 0.0
    first_segments = (
        matched_segments["segment_id"].dropna().head(5).tolist()
    )

    print()
    print("=" * 60)
    print("Map matching demo summary")
    print("=" * 60)
    print(f"GPS points:                {n_points}")
    print(f"Valhalla matched edges:    {n_matched_edges}")
    print(f"Mapped to OSMnx segments:  {n_mapped_segments}")
    print(f"Mapped fraction:           {ratio:.1%}")
    print(f"First matched segment ids: {first_segments}")
    print(f"Output Parquet:            {MATCHES_PATH}")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
