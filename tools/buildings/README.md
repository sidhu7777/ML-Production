# Buildings

This tool extracts building geometry for a project polygon and saves the result for downstream map/RF workflows.

## Public API

Base prefix:

```text
/api/buildings
```

| Endpoint | Method | Purpose |
| --- | --- | --- |
| `/generate` | `POST` | Extracts and saves buildings for an input polygon. |
| `/test` | `GET` | Runs a small built-in test polygon. |

## Generate Request

Required JSON fields:

| Field | Meaning |
| --- | --- |
| `WKT` / `wkt` | Polygon WKT. A bare `((...))` polygon body is also accepted. |
| `Name` | Area/name label. |
| `project_id` | Project id. |

Optional:

| Field | Default | Meaning |
| --- | --- | --- |
| `region` | `india` | Region/database selector. |

The route detects likely lat/lon coordinate order and swaps to lon/lat for geometry processing when needed. The service returns GeoJSON plus extracted/saved counts.

## Code Map

| File | Role |
| --- | --- |
| `routes.py` | Request validation, WKT parsing, coordinate swap detection. |
| `services.py` | Building extraction and persistence. |
| `app.py` | Standalone helper entry point used by this tool. |
