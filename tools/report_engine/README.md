# Report Engine

This tool generates PDF reports for a project and exposes status/download behavior through the `tools/report` Flask routes.

## Where It Lives

| File/folder | Role |
| --- | --- |
| `../report/routes.py` | Flask endpoints under `/api/report` |
| `main.py` | Report generation orchestration |
| `db.py` | Database helpers for project/user/report metadata |
| `llm_integration.py` | Report text generation |
| `pdf_generator.py` | PDF generation |
| `email_service.py` | Email notification when a report is ready |
| `playwright_utils.py` | HTML-to-image rendering with Playwright/Chromium |

## Code Architecture Map

| Layer | Code reference | What it does |
| --- | --- | --- |
| In-memory job store | `tools/report/routes.py:13` | Stores `REPORT_JOBS` per report id |
| Background task | `tools/report/routes.py:81` | Runs report generation in a background thread |
| Generate route | `tools/report/routes.py:115` | Registers `POST /generate` |
| Status route | `tools/report/routes.py:153` | Registers `GET /status/<report_id>` |
| SSE route | `tools/report/routes.py:175` | Registers `GET /events/<report_id>` |
| Download route | `tools/report/routes.py:211` | Registers `GET /download/<report_id>` |
| Report orchestrator | `tools/report_engine/main.py:43` | Main report generation function |
| LLM/text generation | `tools/report_engine/llm_integration.py:80` | Creates report narrative JSON |
| PDF generator | `tools/report_engine/pdf_generator.py:491` | Writes final `report.pdf` |
| Email sender | `tools/report_engine/email_service.py:29` | Sends report-ready email |
| HTML screenshot helper | `tools/report_engine/playwright_utils.py:53` | Converts HTML into PNG using Playwright |

## Report Architecture

```text
POST /api/report/generate
  -> tools/report/routes.py:115
  -> create report_id
  -> REPORT_JOBS[report_id] = processing
  -> background_report_task()
  -> report_engine.main()
  -> collect DB/project data
  -> create processed metadata JSON
  -> generate_report_text()
  -> generate_pdf_report()
  -> update project Download_path
  -> optionally send_report_ready_email()
  -> REPORT_JOBS[report_id] = ready
```

The route layer and engine layer are separate:

| Layer | Responsibility |
| --- | --- |
| `tools/report/routes.py` | API validation, in-memory status, SSE events, background thread, file download |
| `tools/report_engine/main.py` | Report data preparation, output folders, text/PDF generation, DB download path update, email trigger |
| `tools/report_engine/pdf_generator.py` | Final PDF rendering |
| `tools/report_engine/playwright_utils.py` | Browser-backed HTML/image rendering |

## API

### `POST /api/report/generate`

Required fields:

| Field | Meaning |
| --- | --- |
| `project_id` / `Project_id` | Project id |
| `user_id` | User requesting the report |

Response:

```json
{
  "message": "Report generation started",
  "status": "processing",
  "project_id": 123,
  "report_id": "uuid"
}
```

### `GET /api/report/status/<report_id>`

Returns current in-memory report status.

### `GET /api/report/events/<report_id>`

Server-sent events stream for status changes. Events use the `report_status` event name.

### `GET /api/report/download/<report_id>`

Downloads:

```text
data/reports/<report_id>/report.pdf
```

## Current Production Flow

1. The route validates `project_id` and `user_id`.
2. A `report_id` UUID is created.
3. In-memory job state is set to `processing`.
4. A background thread starts `tools.report_engine.main.main()`.
5. The report engine creates temp folders:
   - `data/tmp/<report_id>/html`
   - `data/tmp/<report_id>/images`
   - `data/tmp/<report_id>/processed`
6. The report output folder is created:
   - `data/reports/<report_id>`
7. Project data is collected through DB helpers.
8. HTML/images/intermediate artifacts are prepared.
9. `generate_report_text()` creates the structured report narrative.
10. `generate_pdf_report()` writes `report.pdf`.
11. The project `Download_path` is updated to:
   - `/api/report/download/<report_id>`
12. If a user email exists, `send_report_ready_email()` sends the report link.
13. Temp files are cleaned after a successful run.
14. Job state becomes `ready` or `failed`.

## Output Files

| Path | Meaning |
| --- | --- |
| `data/reports/<report_id>/report.pdf` | Final downloadable PDF |
| `data/tmp/<report_id>/processed/report_metadata.json` | Intermediate report metadata |
| `data/tmp/<report_id>/processed/report_text.json` | Generated report text |
| `data/tmp/<report_id>/images/` | Intermediate rendered images |
| `data/tmp/<report_id>/html/` | Intermediate HTML |

The temp folder is cleaned after a successful report generation. If a run fails, temp artifacts may remain and can help debugging.

## Email Behavior

`email_service.py` sends a simple notification email when:

1. `user_id` is provided.
2. The DB returns a user row.
3. The user row has an `email`.
4. SMTP configuration is valid.

The email download link uses the current report download endpoint.

## Playwright / Chromium

`playwright_utils.py` renders HTML to images. If Chromium is missing, the error message points to:

```powershell
ML\venv\Scripts\python.exe -m playwright install chromium
```

This is required only for report rendering paths that need browser screenshots.

## In-Memory Job State

Report status is stored in process memory in `tools/report/routes.py`.

Operational implication:

- If the Flask/Python process restarts, old in-memory report job state is lost.
- The generated PDF can still exist on disk under `data/reports/<report_id>/report.pdf`.

## Common Debug Checks

If report generation fails:

1. Check `/api/report/status/<report_id>` for the error.
2. Confirm `project_id` exists.
3. Confirm `user_id` exists if email is expected.
4. Check Playwright/Chromium installation if image rendering fails.
5. Check that `data/reports/` and `data/tmp/` are writable.
6. Check SMTP environment/configuration if the report is generated but email is not sent.

## Production Notes

- Report generation is asynchronous.
- The API returns immediately with `status = processing`.
- Frontend should poll status or subscribe to `/api/report/events/<report_id>`.
- The final PDF is served through the Python API, not directly from the filesystem.
