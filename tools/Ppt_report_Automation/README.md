# PowerPoint Report Automation

This tool generates Mobility DT PowerPoint reports and exposes them through the ML Flask app.

## Public API

Base prefix:

```text
/api/ppt-report
```

| Endpoint | Method | Purpose |
| --- | --- | --- |
| `/health` | `GET` | Checks PowerPoint report API health. |
| `/generate` | `POST` | Generates a project PowerPoint file. |
| `/download/<project_id>` | `GET` | Downloads the generated `Mobility_DT_Project_<project_id>.pptx`. |

## Generate Request

Required JSON fields:

| Field | Meaning |
| --- | --- |
| `project_id` | Project id. |

Optional fields:

| Field | Default | Meaning |
| --- | --- | --- |
| `user_id` | `0` | User/request owner id. |
| `session_ids` | none | Session ids as list or comma-separated string. |
| `country_code` | `taiwan` | Country code. |
| `region` | `country_code` | Region selector. |
| `locked_bands` | none | String or string list of locked bands. |

Output files are written under the configured `OUTPUT_FOLDER/ppt_reports`.

## Code Map

| File | Role |
| --- | --- |
| `routes.py` | Flask API for generation/download. |
| `report_ppt_generator.py` | Main project PPT generation orchestration. |
| `ppt_map_generator.py` | Map rendering helpers. |
| `ppt_automation.py` | PowerPoint automation utilities. |
| `generate_ppt.py`, `main.py`, `api_server.py` | CLI/standalone entry helpers. |
| `API.md` | Additional API notes. |
