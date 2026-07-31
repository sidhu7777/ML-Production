# Report API

This folder contains the Flask route layer for project report generation. It owns request validation, job status, server-sent events, render health, and PDF download. The actual report build runs in `tools/report_engine`.

## Public API

Base prefix:

```text
/api/report
```

| Endpoint | Method | Purpose |
| --- | --- | --- |
| `/generate` | `POST` | Starts report generation in a background thread. |
| `/render-health` | `GET` | Checks Playwright/Chromium rendering. |
| `/status/<report_id>` | `GET` | Returns report status. |
| `/events/<report_id>` | `GET` | Streams report status as server-sent events. |
| `/download/<report_id>` | `GET` | Downloads the final PDF. |

## Generate Request

Required fields:

| Field | Meaning |
| --- | --- |
| `project_id` or `Project_id` | Project id. |
| `user_id` or `User_id` | User requesting the report. |

Successful start response:

```json
{
  "message": "Report generation started",
  "status": "processing",
  "project_id": 123,
  "user_id": 45,
  "report_id": "uuid"
}
```

The route returns HTTP `202` after queueing the background thread.

## Route Flow

```text
POST /api/report/generate
  -> validate project_id and user_id
  -> create report_id
  -> save queued status in memory and status.json
  -> start background_report_task()
  -> call tools.report_engine.main.main()
  -> publish status events
  -> mark ready or failed
```

## Job Status

Status is stored in two places:

| Storage | Meaning |
| --- | --- |
| `REPORT_JOBS` | In-memory state for the current Python process. |
| `data/reports/<report_id>/status.json` | Persisted status file for known report ids. |

`/status/<report_id>` checks memory first, then `status.json`. If no status exists but `data/reports/<report_id>/report.pdf` exists, it returns `ready` with the download URL.

Status payload can include:

```text
status
report_id
project_id
user_id
download_url
message
error
```

## Server-Sent Events

`/events/<report_id>` sends events named:

```text
report_status
```

If the report is already `ready` or `failed`, the endpoint sends that terminal status and closes. Otherwise it subscribes to the in-process queue and sends keep-alive comments during quiet periods.

## Download Behavior

The final PDF path is:

```text
data/reports/<report_id>/report.pdf
```

The download endpoint sends it as:

```text
drive_test_report.pdf
```

## Render Health

`/render-health` calls `check_chromium_rendering()` from `tools/report_engine/playwright_utils.py`. It returns HTTP `200` when Chromium rendering works and `503` when rendering is unavailable.

## Engine Link

The route layer delegates report content generation to:

```text
tools.report_engine.main.main(project_id, user_id, report_id, db_engine)
```

See [../report_engine/README.md](../report_engine/README.md) for data loading, maps, KPI analysis, metadata, LLM text generation, PDF output, download path update, and email behavior.
