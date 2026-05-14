"""Central configuration for the map-matching prototype.

All paths, URLs and CRS constants used across the pipeline live here so that
individual modules stay free of hard-coded values.
"""

from __future__ import annotations

from pathlib import Path

# ---------------------------------------------------------------------------
# Project layout
# ---------------------------------------------------------------------------
PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent
DATA_DIR: Path = PROJECT_ROOT / "data"

OSM_DIR: Path = DATA_DIR / "osm"
SEGMENTS_DIR: Path = DATA_DIR / "segments"
TRAJECTORIES_DIR: Path = DATA_DIR / "trajectories"
MATCHES_DIR: Path = DATA_DIR / "matches"

# ---------------------------------------------------------------------------
# Files
# ---------------------------------------------------------------------------
PBF_PATH: Path = OSM_DIR / "example.osm.pbf"
SEGMENTS_PATH: Path = SEGMENTS_DIR / "road_segments.parquet"
MATCHES_PATH: Path = MATCHES_DIR / "example_matched_segments.parquet"

# ---------------------------------------------------------------------------
# Valhalla
# ---------------------------------------------------------------------------
VALHALLA_URL: str = "http://localhost:8002"
VALHALLA_TIMEOUT_S: float = 30.0

# ---------------------------------------------------------------------------
# Network / segment versioning
# ---------------------------------------------------------------------------
# Bump this whenever the underlying segment definition changes (PBF version,
# OSMnx filter, simplification settings, ...). segment_id values are only
# stable within a single network_version.
NETWORK_VERSION: str = "v0.1.0"

# ---------------------------------------------------------------------------
# Coordinate reference systems
# ---------------------------------------------------------------------------
CRS_WGS84: str = "EPSG:4326"
# EPSG:2056 = CH1903+ / LV95 - metric CRS for Switzerland.
CRS_METRIC: str = "EPSG:2056"

# ---------------------------------------------------------------------------
# OSMnx defaults for the prototype
# ---------------------------------------------------------------------------
# A small place name keeps the demo download fast. Swap this for the real
# AOI of your PBF in production.
DEFAULT_PLACE_NAME: str = "Bellinzona, Switzerland"
DEFAULT_NETWORK_TYPE: str = "drive"


def ensure_directories() -> None:
    """Create the data subdirectories if they don't exist yet."""
    for directory in (OSM_DIR, SEGMENTS_DIR, TRAJECTORIES_DIR, MATCHES_DIR):
        directory.mkdir(parents=True, exist_ok=True)
