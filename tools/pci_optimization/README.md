# PCI Optimization

This tool detects PCI collision/confusion/mod-rule issues and can run PCI optimization for selected project/site scopes.

## Public API

Base prefix:

```text
/api/pci-optimization
```

| Endpoint | Method | Purpose |
| --- | --- | --- |
| `/run` | `POST` | Starts a PCI optimization job. |
| `/status/<job_id>` | `GET` | Returns in-memory job status. |
| `/result/<job_id>` | `GET` | Returns job result payload. |
| `/download?file=<path>` | `GET` | Downloads a result file inside the PCI output root. |

## Run Request

Required JSON fields:

| Field | Meaning |
| --- | --- |
| `project_id` | Project id. |

Common optional fields:

| Field | Default | Meaning |
| --- | --- | --- |
| `region` | `india` | Region/database selector. |
| `operator` | `all` | Operator filter. |
| `primary_only` | `true` | Uses primary serving logs only. |
| `filter_sites_to_polygon` | `true` | Restricts sites to project polygon. |
| `filter_logs_to_polygon` | `false` | Restricts logs to project polygon. |
| `neighbor_distance_m` | `500` | Neighbor search distance. |
| `rules` | collision/confusion enabled | Collision, confusion, mod, grouped, and co-centric rules. |
| `run_optimizer` | `true` | Runs optimizer after detection. |
| `site_ids` | empty | Optional site filter. |
| `max_sites` | `0` | Optional site processing cap. |

Only MOD rule values in `{1, 3, 6, 7, 8, 9}` are accepted; invalid values are dropped before the engine runs.

## Code Map

| File | Role |
| --- | --- |
| `routes.py` | API validation, MOD sanitization, job/download endpoints. |
| `services.py` | Job orchestration and output handling. |
| `engine.py` | PCI issue detection and optimization logic. |
| `db.py` | Database access helpers. |
| `schema.py`, `init_schema.py` | Result schema definitions/setup. |
| `export.py` | Result export helpers. |
