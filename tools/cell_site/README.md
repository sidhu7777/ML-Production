# Cell Site

This tool infers cell-site locations and azimuth information from uploaded drive data or existing session logs.

## Public API

Base prefix:

```text
/api/cell-site
```

| Endpoint | Method | Purpose |
| --- | --- | --- |
| `/health` | `GET` | Health check. |
| `/upload` | `POST` | Uploads a CSV/XLS-style file for cell-site inference. |
| `/process-session` | `POST` | Builds a temporary CSV from network-log session ids and processes it. |
| `/verify-project/<project_id>` | `GET` | Checks project existence in the selected region DB. |
| `/download/<output_dir>/<filename>` | `GET` | Downloads generated output. |
| `/outputs/<output_dir>` | `GET` | Lists generated files in an output directory. |
| `/update-project-id` | `POST` | Updates project id for a saved prediction file. |
| `/site-noml/<project_id>` | `GET` | Fetches non-ML inferred site rows for a project. |

## Common Inputs

`/upload` expects multipart form data with `file`. Common form fields include:

| Field | Default | Meaning |
| --- | --- | --- |
| `region` | `india` | Region/database selector. |
| `project_id` | none | Project id to attach results. |
| `method` | `noml` | Inference method. |
| `min_samples` | `30` | Minimum samples. |
| `bin_size` | `5` | Bearing/bin size. |
| `soft_spacing` | `false` | Enables soft spacing mode. |
| `use_ta` | `false` | Uses timing advance where available. |
| `make_map` | `false` | Generates map outputs. |

`/process-session` requires JSON `session_ids` and `project_id`, then reads regional network logs and processes the generated CSV.

## Code Map

| File | Role |
| --- | --- |
| `routes.py` | Upload/session/result endpoints. |
| `services.py` | Regional DB engine and orchestration helpers. |
| `cell_site_processing.py`, `cell_processing.py` | Cell-site inference logic. |
