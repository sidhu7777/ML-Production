# New PDF Report (Per Technology)

This folder is the production home for the "Per Technology" report -- the
expanded, ~38-page drive-test PDF with per-technology Coverage/Mobility/
Handover/Service KPI breakdown, alongside the existing "Combined" report
built by `tools/report_engine`. Its logic originated from and was promoted
out of `tests/new_pdf_report`, a step-based test harness that is still
maintained separately and left untouched by this module -- it keeps
running as a standalone dev/test tool (`tests/new_pdf_report/test_new_pdf_report.py`)
for re-testing this report format directly against a hardcoded project id.

## Code Map

| File | Role |
| --- | --- |
| `main.py` | Report orchestration entry point (`main(project_id, user_id, report_id, db_engine, region, country_code)`), mirroring `tools/report_engine/main.py`'s signature/directory convention. |
| `new_report_sections.py` | Section builders and the `NewFormatPDFReport` PDF class (subclasses `tools.report_engine.pdf_generator.PDFReportGenerator`). |
| `grid_maps.py` | Grid-lattice/aggregation/rendering helpers for polygon projects (mirrors the frontend's own grid view). |
| `local_tiles.py` | Local tile caching + verified `html_to_png` used by every map render in this report. |
| `google_tiles.py` | Gray-basemap / Google tile overlay helpers for grid maps. |
| `routes.py` | Flask blueprint `new_pdf_report_bp`. |

## Dependency On `tools/report_engine`

This module reuses production's shared rendering/threshold/PDF-style code
and never modifies it -- only imports from it, the same way the original
test harness did:

- `pdf_generator` -- base `PDFReportGenerator`, native table styles.
- `metadata_generator` -- `build_metadata`, `write_metadata_file`, spatial grid, geocoding, haversine.
- `map_generator` -- `generate_kpi_map`, `generate_categorical_kpi_map`, `has_valid_numeric_data`/`has_valid_categorical_data`, map primitives.
- `kpi_analysis` -- band/PCI/QoS chart + table generation, drive summary images.
- `kpi_config` -- KPI color functions (`rsrp_colour_manual`, etc).
- `threshold_resolver` -- `resolve_kpi_ranges` (DB-configured KPI color ranges).
- `load_data_db` -- `load_project_data`, `filter_known_band_rows`, `polygon_filter_all_cells`.
- `db` -- `init_engine`/`get_engine` (single shared DB connection layer; this module does not open a second one).
- `playwright_utils` -- Chromium render health check, reused directly by `/render-health`.

## Endpoints

Base prefix:

```text
/api/new-pdf-report
```

| Endpoint | Method | Purpose |
| --- | --- | --- |
| `/generate` | `POST` | Starts Per-Technology report generation in a background thread. Body: `project_id`, `user_id` (both required), optional `region`/`country_code`. |
| `/render-health` | `GET` | Checks Playwright/Chromium rendering (delegates to `tools.report_engine.playwright_utils.check_chromium_rendering`). |
| `/status/<report_id>` | `GET` | Returns report status. |
| `/events/<report_id>` | `GET` | Streams report status as server-sent events. |
| `/download/<report_id>` | `GET` | Downloads the final PDF as `drive_test_report_per_technology.pdf`. |

## Output Namespace

Kept entirely separate from `tools/report_engine` so the two report types
never collide on the same `report_id`:

| Path | Meaning |
| --- | --- |
| `data/tmp/<report_id>/` | Working html/images/processed files (same subfolder layout as `tools/report_engine`: `html/`, `images/kpi_maps/`, `images/kpi_analysis/`, `processed/`). |
| `data/new_pdf_reports/<report_id>/report.pdf` | Final PDF. |
| `data/new_pdf_reports/<report_id>/status.json` | Persisted job status. |

Temp files under `data/tmp/<report_id>/` are removed after a successful run
unless `REPORT_KEEP_TMP=1` is set, matching `tools/report_engine/main.py`'s
own convention.

## Scope Of This Pass

This module makes the existing test-case report callable as a real
endpoint for a given `project_id`/`user_id`/`report_id` -- it does not add
technology-selection or grid/raw-selection request parameters, and it does
not update `tbl_project.Download_path` or send a report-ready email (unlike
`tools/report_engine/main.py`). Those remain candidates for a later pass.
