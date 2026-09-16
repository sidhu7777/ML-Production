# Area Breakup

This tool processes an input polygon into grid blocks, AI zones, and building-cluster polygons, then saves the generated layers.

## Public API

Base prefix:

```text
/api/area-breakup
```

| Endpoint | Method | Purpose |
| --- | --- | --- |
| `/process` | `POST` | Processes an input polygon into area-breakup layers. |
| `/fetch/<project_id>` | `GET` | Fetches saved area-breakup data for a project. |

## Process Request

Required JSON fields:

| Field | Meaning |
| --- | --- |
| `WKT` | Polygon WKT. A bare `((...))` polygon body is accepted. |
| `project_id` | Project id. |

Optional fields:

| Field | Default | Meaning |
| --- | --- | --- |
| `Name` | `output` | Output/layer label. |
| `grid` | `100` | Grid block size. |
| `min_samples` | `10` | Building-cluster minimum samples. |

The route detects likely lat/lon coordinate order and swaps to lon/lat before processing when needed.

## Code Map

| File | Role |
| --- | --- |
| `routes.py` | API validation, WKT parsing, process/fetch endpoints. |
| `services.py` | Grid, AI-zone, building-cluster, database, and export helpers. |
