# S-Tracer ML Backend

This folder contains the Python ML/RF services used by S-Tracer. The active production LTE flow is:

```text
baseline LTE prediction
  -> RF tilt recommendation
  -> recommendation/manual optimized LTE prediction
  -> optional coverage Model 1-4 forecasting and report generation
```

The code is exposed through Flask blueprints registered in `app.py`. Most long-running endpoints return a `job_id` immediately and keep job status in process memory.

## Active Tool Documentation

| Tool | Production purpose | README |
| --- | --- | --- |
| LTE Prediction | Builds the baseline LTE RF surface and geo-feature layer from site, drive-test, building, polygon, grid, and DEM context. | [tools/lte_prediction/README.md](tools/lte_prediction/README.md) |
| LTE Prediction Offset | Production physical/offset-calibrated LTE/5G prediction path using Phase 27 terrain/building features, Phase 36 antenna/reference corrections, Phase 37 quality estimates, and Phase 48 DT residual calibration. | [tools/lte_prediction_offset/README.md](tools/lte_prediction_offset/README.md) |
| LTE Prediction Optimised | Runs manual or recommendation-driven after-change LTE RF prediction and stores scenario results. | [tools/lte_prediction_optimised/README.md](tools/lte_prediction_optimised/README.md) |
| LTE Tilt Recommendation | Finds bad RF areas/cells, RF-validates candidate antenna changes, and saves accepted recommendations. | [tools/lte_tilt_recommandation/README.md](tools/lte_tilt_recommandation/README.md) |
| Coverage Prediction | Exposes Model 1-4 coverage, demand/capacity, current recommendation, and future recommendation APIs. | [tools/coverage_prediction/README.md](tools/coverage_prediction/README.md) |
| Report API | Starts report jobs, persists report status, streams status, checks rendering health, and serves PDFs. | [tools/report/README.md](tools/report/README.md) |
| Report Engine | Loads report data, renders maps/charts, creates metadata/text, writes PDF, updates download path, and sends optional email. | [tools/report_engine/README.md](tools/report_engine/README.md) |
| Buildings | Extracts building geometry for a project polygon and saves it for RF/geospatial consumers. | [tools/buildings/README.md](tools/buildings/README.md) |
| PowerPoint Report Automation | Generates downloadable Mobility DT PowerPoint reports. | [tools/Ppt_report_Automation/README.md](tools/Ppt_report_Automation/README.md) |
| PCI Optimization | Detects PCI collision/confusion/mod-rule issues and can run PCI optimization jobs. | [tools/pci_optimization/README.md](tools/pci_optimization/README.md) |
| Area Breakup | Builds grid blocks, AI zones, and building-cluster polygons for an input WKT area. | [tools/area_breakup/README.md](tools/area_breakup/README.md) |
| Cell Site | Infers cell-site locations/azimuths from uploaded or session-derived drive data. | [tools/cell_site/README.md](tools/cell_site/README.md) |
| Prediction | Legacy prediction wrapper API. | [tools/prediction/README.md](tools/prediction/README.md) |

## API Entry Points

| Module | Endpoint |
| --- | --- |
| LTE baseline prediction | `POST /api/lte-prediction/run` |
| LTE physical/offset prediction | `POST /api/lte-prediction-offset/run` |
| Manual optimized LTE prediction | `POST /api/lte-prediction-optimised/run` or `POST /api/lte-prediction-optimised/optimized` |
| Recommendation optimized LTE prediction | `POST /api/lte-prediction-optimised/recommendation-optimized` |
| LTE tilt recommendation | `POST /api/lte-tilt-recommandation/optimize` |
| Coverage Model 1 | `POST /api/lte-model1-coverage/run` |
| Coverage Model 2 | `POST /api/lte-model2-demand-capacity/run` |
| Coverage Model 3 | `POST /api/lte-model3-current-recommendation/run` |
| Coverage Model 4 | `POST /api/lte-model4-future-recommendation/run` |
| Report generation | `POST /api/report/generate` |
| PowerPoint report generation | `POST /api/ppt-report/generate` |
| Building extraction | `POST /api/buildings/generate` |
| PCI optimization | `POST /api/pci-optimization/run` |

## Production LTE Flow

1. `tools/lte_prediction` builds the baseline RF surface.
   - Reads site rows from `site_prediction`.
   - Reads drive-test rows from frontend payload, PythonBridge, or direct DB.
   - Reads buildings, project polygons, frontend grid analytics, and DEM context when available.
   - Saves `lte_prediction_baseline_results`.
   - Saves `lte_prediction_geo_features`.

2. `tools/lte_prediction_offset` runs the physical/offset LTE prediction path.
   - Uses the same site/drive/building/grid/DEM input sources as baseline LTE prediction.
   - Builds a per-cell directional COST-231 candidate surface.
   - Applies Phase 27 terrain/building context and Phase 36 physical corrections.
   - Uses `.pap` antenna pattern files under `tools/lte_prediction_offset/antenna_patterns`.
   - Phase 48 calibration only applies when drive-test measurements match the same technology and band as deployed site rows.
   - Saves baseline-style prediction rows and geo features through the existing baseline persistence path.

3. `tools/lte_tilt_recommandation` builds RF recommendations.
   - Reads latest baseline rows from `lte_prediction_baseline_results`.
   - Reads antenna rows from `site_prediction`.
   - Reads `lte_prediction_geo_features` and optional `grid_analytics_results`.
   - Runs RF-validated candidate search.
   - Saves accepted rows into `rf_optimization_results`.

4. `tools/lte_prediction_optimised` creates after-change RF results.
   - Manual path reads saved frontend edits from `site_prediction_optimized`.
   - Recommendation path reads `rf_optimization_results`, applies supported changes to site rows, and saves those rows into `site_prediction_optimized`.
   - Creates/updates `lte_optimization_scenarios`.
   - Saves after-change RF rows into `lte_prediction_optimised_results`.

5. `tools/coverage_prediction` exposes separate Model 1-4 forecast/recommendation APIs.
   - Model 1 predicts grid RF metrics.
   - Model 2 forecasts demand/capacity.
   - Model 3 saves current congestion recommendations and after-RF rows.
   - Model 4 runs future recommendations using upstream forecast outputs.

6. `tools/report` and `tools/report_engine` generate project PDFs.
   - Report API owns job/status/download endpoints.
   - Report engine loads project/session data, renders maps/charts, builds metadata/text, writes `report.pdf`, updates `tbl_project.Download_path`, and optionally emails the user.

## Scenario ID Rules

Several tables use scenario ids, but they do not all mean the same thing.

| Table | Column | Meaning |
| --- | --- | --- |
| `rf_optimization_results` | `scenario_id` | RF tilt recommendation scenario id |
| `lte_optimization_scenarios` | `id` | Internal optimized scenario row id |
| `lte_optimization_scenarios` | `scenario_id` | Public frontend optimized scenario id |
| `lte_prediction_optimised_results` | `scenario_id` | Internal optimized scenario row id |
| `lte_prediction_optimised_results` | `public_scenario_id` | Public frontend optimized scenario id |
| `site_prediction_optimized` | `scenario` | Public frontend optimized scenario id |

Example:

```text
rf_optimization_results.scenario_id = 19
lte_optimization_scenarios.id = 69
lte_optimization_scenarios.scenario_id = 1
lte_prediction_optimised_results.scenario_id = 69
lte_prediction_optimised_results.public_scenario_id = 1
site_prediction_optimized.scenario = 1
```

Frontend scenario selection should use the public scenario id. Internal optimized result queries should use `lte_optimization_scenarios.id`.

## PythonBridge And Database Behavior

The shared bridge client is enabled when either of these is configured:

```text
PYTHON_BRIDGE_BASE_URL
SIGNAL_TRACKERS_BRIDGE_URL
```

When bridge mode is enabled, supported reads/writes go through `/api/PythonBridge` or related .NET endpoints. Without bridge mode, the services fall back to SQLAlchemy/direct DB where that module supports it.

Common environment variables:

| Variable | Used for |
| --- | --- |
| `DATABASE_URL` | Default/India database connection |
| `DATABASE_URL_Taiwan` | Taiwan database connection |
| `PYTHON_BRIDGE_BASE_URL` / `SIGNAL_TRACKERS_BRIDGE_URL` | .NET bridge base URL |
| `PYTHON_BRIDGE_API_KEY` | Optional bridge auth key |
| `PYTHON_BRIDGE_TIMEOUT_SECONDS` | Bridge request timeout |
| `BASE_URL` | Absolute report download link when available |
| SMTP variables | Optional report-ready email delivery |

## Reading Order

1. Start with this file.
2. Read [LTE Prediction](tools/lte_prediction/README.md) for baseline creation.
3. Read [LTE Prediction Offset](tools/lte_prediction_offset/README.md) for the physical/offset-calibrated path.
4. Read [LTE Tilt Recommendation](tools/lte_tilt_recommandation/README.md) for RF recommendation generation.
5. Read [LTE Prediction Optimised](tools/lte_prediction_optimised/README.md) for manual/recommendation after-change scenarios.
6. Read [Coverage Prediction](tools/coverage_prediction/README.md) for Model 1-4 forecasting.
7. Read [Report API](tools/report/README.md), [Report Engine](tools/report_engine/README.md), and [PowerPoint Report Automation](tools/Ppt_report_Automation/README.md) for reports.
