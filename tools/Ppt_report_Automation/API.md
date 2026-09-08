# PPT report API — frontend handoff

The tool is registered in `ML/app.py`. Start/restart the main ML backend normally; no separate PPT process is required. Use the same ML API origin as the other Python tools. When running `python app.py`, the default origin is `http://localhost:8080`; `PORT` can override it. Remote clients must use the backend hostname instead of localhost.

| Method | Path | Result |
| --- | --- | --- |
| POST | `/api/ppt-report/generate` | Generate a PPT and return JSON after completion |
| GET | `/api/ppt-report/download/210` | Download the latest generated PPT for project 210 |
| GET | `/api/ppt-report/health` | Route health, not a database/rendering readiness check |

Send `Content-Type: application/json`:

```json
{
  "project_id": 210,
  "session_ids": [4479, 4478],
  "country_code": "taiwan",
  "region": "taiwan",
  "user_id": 0
}
```

Replace example IDs with the selected project and its sessions.

| Field | Required | Accepted value/default |
| --- | --- | --- |
| `project_id` | Yes | Positive integer |
| `session_ids` | No | Non-empty array of positive IDs or comma-separated string such as `"4479,4478"`; omit or null for project sessions |
| `country_code` | No | Non-empty string; defaults to `"taiwan"` |
| `region` | No | Non-empty string; defaults to country_code |
| `user_id` | No | Non-negative integer; defaults to 0 |
| `locked_bands` | No | String or array of band strings; omit or null to use pipeline/project defaults |

Successful generation returns HTTP 200:

```json
{
  "status": "success",
  "message": "PowerPoint presentation generated successfully",
  "project_id": 210,
  "output_file": "Mobility_DT_Project_210.pptx",
  "download_url": "/api/ppt-report/download/210"
}
```

Resolve `download_url` against the ML backend origin, not the frontend origin. The GET returns binary PPTX with an attachment filename, not JSON. Files are saved under `ML/outputs/ppt_reports` (or the configured OUTPUT_FOLDER). The latest successful generation replaces the previous file for that project; download links are not immutable per-request artifacts.

Generation is synchronous and may take several minutes. Show a loading state until POST completes, prevent duplicate submissions, and allow sufficient frontend/proxy request time. There is no job-status or progress endpoint.

```javascript
const mlOrigin = 'http://localhost:8080'; // use the existing ML backend origin
const response = await fetch(`${mlOrigin}/api/ppt-report/generate`, {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({
    project_id: 210,
    session_ids: [4479, 4478],
    country_code: 'taiwan',
    region: 'taiwan',
    user_id: 0
  })
});
const result = await response.json();
if (!response.ok) throw new Error(result.message || 'PPT request failed');
const downloadLink = new URL(result.download_url, mlOrigin).href;
// Set your Download button/link href to downloadLink.
```

Errors return JSON with `status: "error"` and `message`: 400 for invalid payload/JSON, 415 for non-JSON content type, 404 for a missing generated file, and 500 for pipeline failure. Detailed generation errors are logged by the ML backend.

The optional `api_server.py` uses these same paths, with standalone default port 5050 (also overridden by PORT). The old `/api/generate-ppt` and `/api/download-ppt/...` paths are replaced.

Runtime prerequisites are the existing ML database configuration, dependencies in `ML/requirements.txt`, the bundled PPT template, and Playwright Chromium used by the report renderer. API tests use a stub generator; they do not verify live database data or end-to-end rendering.
