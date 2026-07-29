# Model 2 - Demand & Capacity Forecast Model

This folder holds the production-side Model 2 dataset assets.

Model 2 is cell-level, not bucket-based. The input granularity is:

```text
project_id + site_id + sector_id + Node_Cell_ID + band + operator/context fields
```

The Project 196 starter cell dataset is:

```text
data/project_196_model2_demand_capacity_dataset.csv
```

The Excel input is:

```text
data/project_196_model2_demand_capacity_input.xlsx
```

The Excel keeps only `Model2_Cell_Input`, `README`, and `Summary`. Baseline RF rows and geo rows are not stored inside the workbook; production fetches them dynamically through PythonBridge first and direct DB fallback second.

The cell input is derived from the corrected Project 196 Model 3 cell input, so it keeps the same 102 current cells and 18 current congested cells, but adds future forecast label columns:

```text
future_prb_utilization_pct
future_rrc_utilization_pct
future_rrc_connected_users
future_estimated_offered_traffic_mbps
future_congested_flag
```

Do not use `time_bucket`, `bucket_seq`, or `PART_1/PART_2/PART_3` for this production dataset.
