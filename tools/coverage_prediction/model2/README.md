# Coverage Model 2 - Demand And Capacity Forecast

This folder contains the production Model 2 service for cell-level demand/capacity forecasting.

Model 2 is not bucket based. Production input granularity is:

```text
project_id + site_id + sector_id + Node_Cell_ID + band + operator/context fields
```

## API

Model 2 is exposed by the parent Flask app at:

```text
POST /api/lte-model2-demand-capacity/run
GET  /api/lte-model2-demand-capacity/status/<job_id>
GET  /api/lte-model2-demand-capacity/result/<job_id>
```

Required request field:

```text
project_id
```

Optional fields include `region`, `operator`, `baseline_job_id`, `model_run_id`, and `input_excel_path`.

## Default Cell Input

Default Excel input:

```text
tools/coverage_prediction/model2/data/project_196_model2_demand_capacity_input.xlsx
```

The workbook keeps only:

```text
Model2_Cell_Input
README
Summary
```

Baseline RF rows and geo rows are not stored in the workbook. Production fetches them dynamically through PythonBridge first and direct DB fallback second.

## Current Dataset Note

The Project 196 starter cell dataset is:

```text
data/project_196_model2_demand_capacity_dataset.csv
```

The cell input is derived from the corrected Project 196 Model 3 cell input, keeping the same 102 current cells and 18 current congested cells, then adding future forecast label columns:

```text
future_prb_utilization_pct
future_rrc_utilization_pct
future_rrc_connected_users
future_estimated_offered_traffic_mbps
future_congested_flag
```

Do not use `time_bucket`, `bucket_seq`, or `PART_1/PART_2/PART_3` for this production dataset.

## Production Flow

```text
load Model2_Cell_Input
filter rows to project_id when project_id column exists
load production-clean Model 2 weights
fetch baseline rows from GetLteBaselineRows or lte_prediction_baseline_results
fetch geo rows from GetLtePredictionGeoFeatures or lte_prediction_geo_features
build Model 2 feature frame
predict demand_index, active_users_est, traffic_demand_est
derive future PRB/RRC/users/traffic/congested fields
save lte_model2_demand_capacity_forecast_results
```

## Result Table

Model 2 writes:

```text
lte_model2_demand_capacity_forecast_results
```

Rows are replaced for the same `(project_id, baseline_job_id, model_run_id)` before insert.
