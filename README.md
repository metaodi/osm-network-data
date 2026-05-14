# Trajectory Matching Prototype

A small, production-near scaffold for matching GPS trajectories against an OSM
street network. It deliberately keeps two views on the same OSM data:

- **Valhalla** is responsible for the actual map matching of noisy GPS traces.
- **OSMnx / GeoPandas** generate a separate analysis segment model that
  downstream consumers (analytics, BI, reporting) can rely on.

Both sides are fed from **the same `*.osm.pbf` file** so that segment ids and
matched edges talk about the same underlying OSM features.

## Architecture

```
Raw GPS  ─►  Valhalla (trace_attributes)  ─►  matched_edges
                                                  │
                                                  ▼
                                       map_to_segments.py
                                                  │
                              ┌───────────────────┘
                              ▼
        OSMnx segment model  ─►  matched_segments  ─►  Parquet
```

- `matched_edges` is what Valhalla returns: one row per OSM edge it snapped
  the trajectory to, with `way_id`, `road_class`, `speed`, geometry, etc.
- `matched_segments` is the join onto our own segment model. Every row carries
  a `segment_id`, the `network_version` it was produced under, and a
  `match_quality` flag (`way_id`, `way_id+spatial`, `spatial_only`,
  `unmatched`).

## Project layout

```
├── docker-compose.yml         # local Valhalla service
├── pyproject.toml
├── uv.lock
├── README.md
├── data/
│   ├── osm/                   # *.osm.pbf lives here (shared with Valhalla)
│   ├── segments/              # OSMnx segment Parquet output
│   ├── trajectories/          # raw GPS traces (you provide)
│   └── matches/               # final matched_segments Parquet output
└── src/
    ├── config.py              # paths, URLs, CRS, network version
    ├── build_segments.py      # OSMnx -> road_segments.parquet
    ├── match_trajectory.py    # GPS -> Valhalla -> matched_edges
    ├── map_to_segments.py     # matched_edges -> matched_segments
    └── example_pipeline.py    # runnable end-to-end demo
```

## Installation

This project uses [`uv`](https://docs.astral.sh/uv/) for dependency
management.

```bash
uv sync
```

Activate the environment if you want plain `python` to find the packages:

```bash
source .venv/bin/activate
```

Or use `uv run` to execute scripts without activating:

```bash
uv run python src/example_pipeline.py
```

## 1. Provide an OSM PBF

Drop a `*.osm.pbf` extract into `data/osm/` and name it `example.osm.pbf` (or
adjust `PBF_PATH` in `src/config.py`). Geofabrik publishes ready-made extracts:

```bash
# Example: Ticino canton
curl -L https://download.geofabrik.de/europe/switzerland/ticino-latest.osm.pbf \
     -o data/osm/example.osm.pbf
```

For the demo trajectory in `example_pipeline.py` you will want an extract that
covers Bellinzona; for production, use the same bounding box / polygon that
your downstream OSMnx code relies on so that Valhalla and OSMnx see the same
roads.

## 2. Start Valhalla

```bash
docker compose up -d valhalla
```

The first start builds Valhalla tiles from the PBF in `data/osm/`. This can
take a few minutes for a small extract; large extracts take longer. Watch the
logs with:

```bash
docker compose logs -f valhalla
```

When the service is ready, `http://localhost:8002/status` returns a JSON
status payload. Subsequent restarts re-use the cached tiles.

If you replace the PBF, set `force_rebuild=True` in `docker-compose.yml` once
or remove the cached `data/osm/valhalla_tiles*` files.

## 3. Build segments

```bash
uv run python src/build_segments.py --place "Bellinzona, Switzerland"
```

This writes `data/segments/road_segments.parquet`.

## 4. Run the demo pipeline

```bash
uv run python src/example_pipeline.py
```

The demo:

1. Defines a hard-coded GPS trace (7 points through Bellinzona).
2. Builds segments on the fly if the Parquet file is missing.
3. Calls Valhalla `trace_attributes`.
4. Joins the matched edges onto OSMnx segments.
5. Writes `data/matches/example_matched_segments.parquet`.
6. Prints a short summary (counts + first matched `segment_id`s).

## Notes & caveats

- **`segment_id` is only stable within a single `network_version`.** If you
  change the OSMnx filter, the simplification, or the underlying PBF, bump
  `NETWORK_VERSION` in `src/config.py`. Old segment ids must not be reused.
- For production, make sure the **Valhalla tiles and the OSMnx graph come
  from the exact same PBF version**. Otherwise Valhalla can return `way_id`s
  that simply don't exist in your segment table, forcing the spatial fallback
  more often than necessary.
- Each step is runnable on its own:
  - `python src/build_segments.py --place "..."`
  - `python -c "from match_trajectory import match_trajectory; ..."`
  - `python -c "from map_to_segments import map_matched_edges_to_segments; ..."`
- The HTTP client raises `ValhallaError` if the service is not reachable;
  the demo pipeline turns that into a non-zero exit code.
