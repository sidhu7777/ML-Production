# Coverage Prediction

This folder exposes the production-facing LTE coverage and capacity model APIs registered in `ML/app.py`. The four models are separate Flask blueprints with their own background job stores and result tables.

## Public APIs

| Model | URL prefix | Main endpoint | Purpose |
| --- | --- | --- | --- |
| Model 1 | `/api/lte-model1-coverage` | `POST /run` | Grid-level RF coverage prediction. |
| Model 2 | `/api/lte-model2-demand-capacity` | `POST /run` | Cell-level demand/capacity forecast. |
| Model 3 | `/api/lte-model3-current-recommendation` | `POST /run` | Current congestion recommendation plus after-RF surface. |
| Model 4 | `/api/lte-model4-future-recommendation` | `POST /run` | Future capacity recommendation plus after-RF surface. |

Each model also exposes:

```text
GET /status/<job_id>
GET /result/<job_id>
```

Job state is in memory inside each model service module. Result rows are persisted in MySQL tables defined by each model's `schema.py`.

## Folder Map

| Path | Role |
| --- | --- |
| `future_coverage_kpi_prediction/routes.py`, `future_coverage_kpi_prediction/services.py`, `future_coverage_kpi_prediction/db.py`, `future_coverage_kpi_prediction/feature_builder.py`, `future_coverage_kpi_prediction/model_registry.py` | Model 1 API, orchestration, DB/bridge access, feature frame, and weight loading. |
| `future_demand_capacity_forecast/routes.py`, `future_demand_capacity_forecast/services.py`, `future_demand_capacity_forecast/db.py`, `future_demand_capacity_forecast/feature_builder.py`, `future_demand_capacity_forecast/model_registry.py` | Model 2 API, orchestration, DB/bridge access, feature frame, and weight loading. |
| `current_capacity_recommendation/routes.py`, `current_capacity_recommendation/services.py`, `current_capacity_recommendation/db.py` | Model 3 API wrapper, current recommendation runner, and persistence. |
| `future_capacity_recommendation/routes.py`, `future_capacity_recommendation/services.py`, `future_capacity_recommendation/db.py` | Model 4 API wrapper, upstream Model 1/2 orchestration, future recommendation runner, and persistence. |
| `model*/schema.py` | Result table definitions and expected columns. |
| `model*/init_schema.py` | Schema creation/verification scripts. |

## Shared Database And Bridge Behavior

Coverage models load environment from `ML/.env`.

| Region | DB variable |
| --- | --- |
| `india` or default | `DATABASE_URL` |
| `taiwan` | `DATABASE_URL_Taiwan`, falling back to `DATABASE_URL` |

Model 1 and Model 2 fetch baseline and geo rows through PythonBridge when available:

| Operation | Bridge endpoint |
| --- | --- |
| Latest baseline job id | `GetLatestLteBaselineJobId` |
| Baseline rows | `GetLteBaselineRows` |
| Geo feature rows | `GetLtePredictionGeoFeatures` |

Direct DB fallback reads:

```text
lte_prediction_baseline_results
lte_prediction_geo_features
```

Result persistence is direct SQLAlchemy table writes. Before saving, each model creates its result schema if needed and deletes existing rows for the same `(project_id, baseline_job_id, model_run_id)` before inserting replacement rows.

## Model 1 - Coverage Prediction

Model 1 predicts grid-level RF metrics from the latest/specified LTE baseline and geo-feature rows.

Endpoint:

```text
POST /api/lte-model1-coverage/run
```

Input fields:

| Field | Required | Meaning |
| --- | --- | --- |
| `project_id` | Yes | Project to process. |
| `region` | No | Defaults to `india`. |
| `operator` | No | Optional operator filter. |
| `baseline_job_id` | No | Specific LTE baseline job. If omitted, latest baseline is resolved. |
| `model_run_id` | No | Custom output run id. Defaults to `model1_<project_id>_<job>`. |

Flow:

```text
load latest Model 1 weights
fetch baseline rows
fetch geo rows for resolved baseline job
build grid feature frame
predict pred_rsrp, pred_rsrq, pred_sinr
save lte_future_coverage_kpi_prediction_results
```

Weights are loaded from:

```text
ML/models/model1
```

The registry rejects production-unsafe weights that still contain forbidden training-only features such as bucket/time sequence fields.

Result table:

```text
lte_future_coverage_kpi_prediction_results
```

## Model 2 - Demand And Capacity Forecast

Model 2 forecasts current/future congestion and demand/capacity values at cell level.

Endpoint:

```text
POST /api/lte-model2-demand-capacity/run
```

Input fields:

| Field | Required | Meaning |
| --- | --- | --- |
| `project_id` | Yes | Project to process. |
| `region` | No | Defaults to `india`. |
| `operator` | No | Optional operator filter. |
| `baseline_job_id` | No | Specific LTE baseline job. |
| `model_run_id` | No | Custom output run id. Defaults to `future_demand_capacity_<project_id>_<job>`. |
| `input_excel_path` | No | Overrides the default Model 2 cell-input Excel. |

Default input:

```text
tools/coverage_prediction/future_demand_capacity_forecast/data/project_196_model2_demand_capacity_input.xlsx
```

Flow:

```text
load Model2_Cell_Input sheet
filter rows to project_id when column exists
load production-clean Model 2 weights
fetch baseline rows
fetch geo rows for resolved baseline job
build cell feature frame
predict demand_index, active_users_est, traffic_demand_est
derive future PRB/RRC/users/traffic/congested fields
save lte_future_demand_capacity_forecast_results
```

Weights are loaded from candidate roots:

```text
ML/models/model2
ML/models/model2_hybrid_target_experiment
```

The registry rejects weights that contain forbidden production features such as `bucket_seq` or `time_bucket`.

Result table:

```text
lte_future_demand_capacity_forecast_results
```

See [future_demand_capacity_forecast/README.md](future_demand_capacity_forecast/README.md) for the current dataset note.

## Model 3 - Current Recommendation

Model 3 creates current congestion recommendations and an after-RF surface. It currently wraps the existing dynamic recommendation runner under `ML/tests/coverage_prediction` and then persists production result tables.

Endpoint:

```text
POST /api/lte-model3-current-recommendation/run
```

Input fields:

| Field | Required | Meaning |
| --- | --- | --- |
| `project_id` | Yes | Project to process. |
| `region` | No | Defaults to `india`. |
| `operator` | No | Optional operator filter. |
| `baseline_job_id` | No | Specific LTE baseline job. If omitted, latest baseline is resolved. |
| `model_run_id` | No | Custom output run id. Defaults to `current_capacity_recommendation_<project_id>_<job>`. |
| `source_model2_run_id` | No | Links Model 3 output to a Model 2 run. |
| `input_excel_path` | No | Overrides the Model 3 input Excel. |
| `congestion_threshold` | No | Defaults to `70.0`. |
| `max_congested_cells` | No | Optional processing cap. |
| `carrier_reselection_hysteresis_db` | No | Defaults to `0.0`. |
| `rf_workers` | No | Defaults to `2`. |
| `max_interference_sites` | No | Defaults to `10`. |
| `action_neighbor_cells` | No | Defaults to `2`. |
| `sector_parallelism` | No | Defaults to `1`. |
| `stop_on_partial` | No | Defaults to `false`. |

Default input:

```text
ML/models/model3_project196_input/project_196_model3_input.xlsx
```

Flow:

```text
resolve baseline job id
load tests.coverage_prediction.model3_current_recommendation_test runner
run dynamic current recommendation logic
read summary/artifact paths
read recommendations CSV
read after-RF surface CSV
prepare DB rows
delete/replace rows for same project/baseline/model_run_id
save recommendation and after-RF result tables
```

Result tables:

```text
lte_current_capacity_recommendation_results
lte_current_capacity_recommendation_rf_results
```

## Model 4 - Future Recommendation

Model 4 creates future capacity recommendations. By default it submits upstream Model 1 and Model 2 jobs first, waits for them to complete, converts Model 2 future forecast rows into a CSV for the future recommendation runner, and then saves recommendation plus after-RF rows.

Endpoint:

```text
POST /api/lte-model4-future-recommendation/run
```

Input fields:

| Field | Required | Meaning |
| --- | --- | --- |
| `project_id` | Yes | Project to process. |
| `region` | No | Defaults to `india`. |
| `operator` | No | Optional operator filter. |
| `baseline_job_id` | No | Specific LTE baseline job. If omitted, latest baseline is resolved. |
| `model_run_id` | No | Custom output run id. Defaults to `future_capacity_recommendation_<project_id>_<job>`. |
| `source_model1_run_id` | No | Upstream Model 1 run id. Generated when omitted. |
| `source_model2_run_id` | No | Upstream Model 2 run id. Generated when omitted. |
| `model3_input_excel_path` | No | Overrides the default Model 3-style input Excel. |
| `model2_input_excel_path` | No | Overrides the default Model 2 input Excel. |
| `congestion_threshold` | No | Defaults to `70.0`. |
| `max_congested_cells` | No | Optional processing cap. |
| `carrier_reselection_hysteresis_db` | No | Defaults to `0.0`. |
| `rf_workers` | No | Defaults to `2`. |
| `max_interference_sites` | No | Defaults to `10`. |
| `action_neighbor_cells` | No | Defaults to `2`. |
| `sector_parallelism` | No | Defaults to `1`. |
| `skip_model1` | No | Defaults to `false`; skips upstream Model 1 submission when true. |

Flow:

```text
resolve baseline job id
run upstream Model 1 unless skip_model1=true
run upstream Model 2
fetch Model 2 forecast rows by source_model2_run_id
write model2_future_for_model4.csv
load future recommendation runner from ML/tests/coverage_prediction
run future recommendation logic
read recommendation CSV and after-RF surface CSV
prepare DB rows
delete/replace rows for same project/baseline/model_run_id
save recommendation and after-RF result tables
```

Result tables:

```text
lte_future_capacity_recommendation_results
lte_future_capacity_recommendation_rf_results
```

## Model 1-4 Relationship

| Model | Depends on | Produces |
| --- | --- | --- |
| Model 1 | Baseline RF + geo features + Model 1 weights | Grid RF predictions. |
| Model 2 | Cell input Excel + baseline RF + geo features + Model 2 weights | Current/future cell demand and congestion forecast. |
| Model 3 | Latest baseline + Model 3 input Excel + dynamic current recommendation runner | Current recommendations and after-RF surface. |
| Model 4 | Latest baseline + upstream Model 1/2 + future recommendation runner | Future recommendations and after-RF surface. |

## Operational Notes

- These APIs are asynchronous.
- Status is in process memory; restarting Python loses job status.
- Result rows are persisted in MySQL and can be queried by `project_id`, `baseline_job_id`, and `model_run_id`.
- Model 3 and Model 4 intentionally call existing runner modules from `ML/tests/coverage_prediction`; that is part of the current implementation, not just test-only code.
- Model 3/4 output artifacts are written under `ML/outputs/` and stable output folders under `ML/models/`.
