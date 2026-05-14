# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

**Install dependencies:**
```bash
uv sync --dev
```

**Run all tests:**
```bash
uv run pytest --cov=src --cov-report=term-missing -v
```

**Run a single test file or test:**
```bash
uv run pytest tests/test_build_segments.py -v
uv run pytest tests/test_match_trajectory.py::TestDecodePolyline6::test_roundtrip_positive_coordinates -v
```

**Run the end-to-end demo** (requires Valhalla running and a PBF in `data/osm/`):
```bash
uv run python src/example_pipeline.py
```

**Build segments only:**
```bash
uv run python src/build_segments.py --place "Bellinzona, Switzerland"
```

**Start Valhalla:**
```bash
docker compose up -d valhalla
```

## Architecture

The pipeline maintains two independent views of the same OSM data, both fed from a single `*.osm.pbf` file:

- **Valhalla** (Docker service on port 8002) handles the actual map-matching of noisy GPS traces via its `trace_attributes` endpoint. It returns `matched_edges` — one row per OSM edge the trace was snapped to, including `way_id`, road class, speed, and geometry.
- **OSMnx/GeoPandas** generates the canonical `road_segments.parquet` — the stable segment model that all downstream analytics consumers reference. Each segment carries a `segment_id` and `network_version`.

The join between these two worlds is `map_to_segments.py`, which uses a three-level matching strategy:
1. **`way_id` exact match**: Valhalla's `way_id` joined against `osmid_norm` (a list column that OSMnx sometimes populates with multiple OSM ids when edges are merged). `match_quality = "way_id"`.
2. **`way_id` + spatial tiebreak**: When a `way_id` matches multiple segments, the geometrically closest one is picked. `match_quality = "way_id+spatial"`.
3. **Pure spatial nearest-neighbour**: When `way_id` is absent or unknown, the closest segment by geometry in metric CRS (EPSG:2056) is used. `match_quality = "spatial_only"`. Unresolvable edges become `"unmatched"`.

## Key Invariants

- **`segment_id` is only stable within a `network_version`**. Any change to the OSMnx filter, simplification settings, or the underlying PBF requires bumping `NETWORK_VERSION` in `src/config.py`. Old IDs must not be reused.
- **Valhalla tiles and the OSMnx graph must come from the exact same PBF version.** Drift between them forces the spatial fallback path more often and degrades match quality.
- All geographic data is stored and exchanged in `EPSG:4326` (WGS84). Spatial distance operations reproject to `EPSG:2056` (CH1903+/LV95, metric) internally before computing distances.
- `src/` is on `PYTHONPATH` (set in `pyproject.toml`), so modules import each other with bare names (e.g. `from config import ...`, `from build_segments import ...`).

## Data Flow

```
data/osm/example.osm.pbf
    ├── → Valhalla tiles (built on first docker compose up)
    │       └── match_trajectory.py → matched_edges GeoDataFrame
    └── → build_segments.py (via OSMnx)
            └── data/segments/road_segments.parquet
                    └── map_to_segments.py (joins matched_edges onto segments)
                            └── data/matches/*.parquet
```

## Testing Approach

Tests are unit-level and do not require a running Valhalla or network access. Network calls (`requests.get/post`) are mocked with `unittest.mock.patch`. The `conftest.py` provides shared fixtures (`minimal_segments`, `minimal_matched_edges`, `sample_valhalla_response`) that cover the Bellinzona area and are reused across all three test files.

The Parquet I/O round-trip test in `test_build_segments.py` uses `tmp_path` and is the only test that touches the filesystem.
