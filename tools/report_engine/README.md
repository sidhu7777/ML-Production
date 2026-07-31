# Report Engine

This folder builds project PDF reports. The public routes live in `tools/report`; this engine loads project/session data, filters the report population, renders maps and charts, builds metadata, generates narrative text, writes the PDF, updates the project download path, and optionally sends an email.

## Code Map

| File | Current role |
| --- | --- |
| `main.py` | Full report orchestration. |
| `load_data_db.py` | Project/session loading, primary row filtering, polygon filtering, known-band filtering, handover all-cell filtering. |
| `db.py` | Bridge/direct DB helpers for project, sessions, logs, regions, thresholds, users, and download path update. |
| `map_generator.py` | KPI maps, categorical maps, poor-region maps, base route map, handover map. |
| `playwright_utils.py` | HTML-to-PNG rendering and Chromium health checks. |
| `kpi_analysis.py` | KPI summaries, band charts/tables, PCI, app/QoS, drive summary, native PDF table data. |
| `metadata_generator.py` | Report metadata JSON. |
| `cdf_kpi.py` | KPI CDF plots. |
| `llm_integration.py` | Report text generation and fallback section synthesis. |
| `pdf_generator.py` | ReportLab PDF assembly. |
| `email_service.py` | Optional report-ready email. |
| `threshold_resolver.py` | User/default KPI threshold ranges. |
| `kpi_config.py` | KPI config used for maps and analysis. |
| `s3_uploader.py` | Optional upload helper, not the primary report delivery path. |

## Entry Point

The route layer calls:

```python
tools.report_engine.main.main(
    project_id=project_id,
    user_id=user_id,
    report_id=report_id,
    db_engine=db.engine,
)
```

When `db_engine` is supplied, `db.py` stores it with `init_engine()` so the report job can reuse the Flask app SQLAlchemy engine. If bridge mode is configured, data helpers prefer PythonBridge and do not require opening direct MySQL first.

## Production Flow

```text
main(project_id, user_id, report_id)
  -> create data/tmp/<report_id>/ folders
  -> create data/reports/<report_id>/
  -> load_project_data(project_id)
  -> choose report_df
  -> render base route map
  -> render KPI maps in parallel
  -> render poor-region maps
  -> render handover map
  -> save processed/filtered_data.csv
  -> generate KPI CDF plots
  -> run KPI analysis and build native PDF tables
  -> build/write report_metadata.json
  -> generate_report_text()
  -> generate_pdf_report()
  -> update tbl_project.Download_path or bridge equivalent
  -> send optional report-ready email
  -> clean temp directory unless REPORT_KEEP_TMP=1
```

## Output Paths

| Path | Meaning |
| --- | --- |
| `data/reports/<report_id>/report.pdf` | Final PDF served by `/api/report/download/<report_id>`. |
| `data/tmp/<report_id>/html/` | Temporary Folium/HTML render sources. |
| `data/tmp/<report_id>/images/kpi_maps/` | Rendered map PNGs. |
| `data/tmp/<report_id>/images/kpi_analysis/` | KPI charts, CDFs, and table images. |
| `data/tmp/<report_id>/processed/filtered_data.csv` | Final report dataframe. |
| `data/tmp/<report_id>/processed/report_metadata.json` | Metadata used by LLM/PDF. |
| `data/tmp/<report_id>/processed/report_text.json` | Generated/fallback report text. |

The API route also persists:

```text
data/reports/<report_id>/status.json
```

Temp files are removed after a successful run unless:

```text
REPORT_KEEP_TMP=1
```

## Data Loading

`load_project_data()` reads:

| Data | Bridge mode | Direct DB mode |
| --- | --- | --- |
| Project metadata | `GetProject` | `tbl_project` |
| Network logs | `GetDriveTestRows` | `tbl_network_log` |
| Project regions | `GetProjectRegions` | `map_regions` |
| Sessions | `GetSessions` | `defaultdb.tbl_session` |
| User thresholds | `GetThresholds` | `thresholds` |
| User email/name | `GetUser` | `tbl_user` |
| Download path update | `UpdateProjectDownloadPath` | `tbl_project.Download_path` |

The project `ref_session_id` string is parsed into session ids. Project provider/date fields are passed to bridge report log loading when available.

## Report Data Population

The engine uses different populations for different report purposes, by design.

| Dataframe | How it is built | Used for |
| --- | --- | --- |
| `raw_df` | All fetched session rows. In bridge mode it is already report-prefiltered by the bridge contract. | Handover source and fallback/session context. |
| `filtered_df` | Bridge mode: same as bridge-prefiltered rows. Direct DB mode: primary-serving rows inside project polygon. | Intermediate report candidate rows. |
| `report_df` | Bridge mode: bridge-prefiltered rows. Direct DB mode: `filtered_df` with unknown bands removed. | Maps, KPI analysis, CDFs, metadata, LLM text, PDF. |
| `handover_df` | Bridge mode: bridge-prefiltered rows. Direct DB mode: all-cell rows inside polygon, without primary-only filtering. | Handover map/counts. |

Primary row filtering uses either:

```text
primary_cell_info_1 contains mRegistered=YES
primary in yes/y/true/1
```

Direct DB polygon filtering uses vectorized Shapely. If the first polygon pass returns zero rows, it retries with swapped polygon coordinates.

Known-band filtering uses `normalize_band_name()` and drops the `Unknown` bucket so band tables, pie charts, metadata, LLM percentages, maps, and CDFs use the same denominator.

## Map And Image Rendering

The report renders:

- Base route map.
- KPI maps from `KPI_CONFIG`.
- Categorical KPI maps where configured.
- Poor-region maps.
- Handover map.
- KPI analysis charts and tables.
- CDF plots.

HTML maps are converted to PNG through Playwright/Chromium:

```text
REPORT_RENDER_WIDTH = 1200
REPORT_RENDER_HEIGHT = 900
REPORT_DEVICE_SCALE = 1
```

KPI map rendering uses up to four workers, reduced for large report datasets.

Use the route below to verify Chromium availability:

```text
GET /api/report/render-health
```

## Metadata, LLM Text, And PDF

The engine writes metadata from the same `report_df` used for maps/charts. It then calls `generate_report_text()` to create the narrative report JSON. `llm_integration.py` validates and fills missing sections from metadata when needed.

`generate_pdf_report()` builds the final ReportLab PDF using:

- `report_metadata.json`
- `report_text.json`
- Rendered map/chart images
- Native PDF table data from `build_native_table_data()`
- Scratch image optimization directory under `data/tmp/<report_id>/_img_opt`

## Download Path And Email

After PDF generation, the engine builds the download link:

```text
<BASE_URL>/api/report/download/<report_id>
```

If `BASE_URL` is not set, it stores:

```text
/api/report/download/<report_id>
```

Then it updates the project download path through PythonBridge `UpdateProjectDownloadPath` or direct `tbl_project.Download_path`.

If `user_id` is supplied and the user has an email address, `send_report_ready_email()` sends the report link. Email failure is logged but does not remove the generated PDF.

## Operational Notes

- Report generation is asynchronous at the route layer.
- The API route owns `status.json`; the engine owns report artifacts and PDF creation.
- The final PDF is served by Python API, not direct S3/browser file access.
- Bridge mode fixes MySQL datetime object shapes before pandas/report code sees them.
- Direct DB mode still supports raw-table reporting when bridge is not configured.

## Common Failure Checks

1. Check `/api/report/status/<report_id>` for the stored error.
2. Check `/api/report/render-health` for Chromium/Playwright issues.
3. Confirm `tbl_project.ref_session_id` has valid session ids.
4. Confirm network logs have usable `lat`, `lon`, KPI, band, and timestamp fields.
5. Check polygon filtering logs, including swapped-coordinate fallback.
6. Check whether known-band filtering removed all rows.
7. Check SMTP configuration only after `report.pdf` exists but no email was sent.
