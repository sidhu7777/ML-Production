# LTE Prediction

This tool builds the baseline LTE RF prediction for a project. Its output is the reference layer used by optimized prediction and tilt recommendation.

## Where It Lives

| File | Role |
| --- | --- |
| `routes.py` | Flask endpoints under `/api/lte-prediction` |
| `services.py` | Background job orchestration, DB writes, bridge writes |
| `ml_engine.py` | Site/drive/grid sampling and RF prediction logic |
| `geo_correction_pipeline.py` | Production geo correction and display correction |
| `dem_utils.py` | DEM/elevation support |
| `grid_sampling.py` | Frontend grid analytics sampling support |

## Code Architecture Map

Use this section as the implementation map when reviewing or maintaining the module.

| Layer | Code reference | What it does |
| --- | --- | --- |
| API route | `tools/lte_prediction/routes.py:8` | Registers `POST /run` and converts request JSON into service config |
| Status route | `tools/lte_prediction/routes.py:48` | Returns in-memory job status |
| Result route | `tools/lte_prediction/routes.py:53` | Returns the same job payload for result polling |
| Service class | `tools/lte_prediction/services.py:139` | `LTEPredictionService` owns job lifecycle |
| Job submit | `tools/lte_prediction/services.py:141` | Creates `job_id`, stores queued state, starts background thread |
| Main job runner | `tools/lte_prediction/services.py:160` | Orchestrates site fetch, drive fetch, RF prediction, correction, and DB writes |
| Site fetch/model input | `tools/lte_prediction/ml_engine.py:367` | `fetch_site_data()` reads and normalizes `site_prediction` |
| Drive fetch/model input | `tools/lte_prediction/ml_engine.py:484` | `fetch_drive_data()` reads session drive-test rows or frontend rows |
| Source RF model | `tools/lte_prediction/ml_engine.py:788` | `run_rf_prediction_fast()` runs the sector-wise RF prediction engine |
| Production correction | `tools/lte_prediction/ml_engine.py:982` | `run_ml_fast()` applies production display/geo correction |
| Geo correction | `tools/lte_prediction/geo_correction_pipeline.py:2205` | `apply_full_display_correction()` builds corrected display-ready RF output |
| Geo weights | `tools/lte_prediction/geo_correction_pipeline.py:37` | `DEFAULT_GEO_WEIGHTS` controls weighted geo correction defaults |
| Metrics helper | `tools/lte_prediction/geo_correction_pipeline.py:189` | `_metric_bundle()` computes MAE/RMSE/R2/bias style validation metrics |
| Frontend grid source | `tools/lte_prediction/grid_sampling.py:33` | `fetch_frontend_grid_cells()` loads frontend grid analytics cells |
| Baseline DB write | `tools/lte_prediction/services.py:853` | `_save_baseline_results()` writes `lte_prediction_baseline_results` |
| Geo DB write | `tools/lte_prediction/services.py:1024` | `_save_geo_features()` writes `lte_prediction_geo_features` |

## Model Architecture

This module is a hybrid RF prediction pipeline, not one isolated ML model.

```text
Request payload
  -> routes.py validates/builds config
  -> LTEPredictionService.submit()
  -> LTEPredictionService._run()
  -> fetch_site_data()
  -> fetch_drive_data()
  -> run_rf_prediction_fast()
  -> run_ml_fast()
  -> apply_full_display_correction()
  -> _save_baseline_results()
  -> _save_geo_features()
```

The model stack has three main parts:

| Part | Code | Explanation |
| --- | --- | --- |
| RF propagation engine | `ml_engine.py:788` | Generates base LTE RSRP/RSRQ/SINR using site geometry, antenna configuration, drive data, buildings, and interference settings |
| Geo/display correction layer | `geo_correction_pipeline.py:2205` | Adjusts raw prediction with clutter, morphology, terrain, LOS/NLOS, building density, serving distance, azimuth alignment, and drive-test proximity |
| Persistence layer | `services.py:853`, `services.py:1024` | Saves the baseline prediction surface and geo feature context for downstream modules |

The production output is the baseline reference for downstream workflows. Tilt recommendation and optimized prediction both depend on the tables written by this module.

## API

### `POST /api/lte-prediction/run`

Required payload fields:

| Field | Meaning |
| --- | --- |
| `project_id` | Project id to predict |
| `session_ids` | Drive-test session ids used as the RF source |
| `radius` | Prediction radius in meters |
| `grid_resolution` | Grid size/resolution |

Optional fields include:

| Field | Meaning |
| --- | --- |
| `region` | Defaults to `india`; selects the DB engine |
| `operator` | Optional operator filter |
| `n_workers` | Worker count for RF computation |
| `polygon_ids` / `polygonIds` | Restrict site and prediction work to selected map regions |
| `dem_raster_path` | Optional DEM raster path |
| `drive_rows` | Frontend-supplied drive rows instead of DB fetch |
| `grid_analytics_scenario_id` | Reuse a frontend grid analytics scenario |
| `use_frontend_grid_sampling` | Whether to sample frontend grid cells |
| `max_interference_sites` | Limit interfering sites per grid/cell calculation |

Response returns a `job_id`. Use:

```text
GET /api/lte-prediction/status/<job_id>
GET /api/lte-prediction/result/<job_id>
```

## Current Production Flow

1. The route builds a normalized config and submits a background job.
2. `LTEPredictionService._run()` loads project sites from `site_prediction`.
3. Site rows are normalized into consistent columns such as `Node_Cell_ID`, `node_b_id`, `cell_id`, `lat`, `lon`, `azimuth`, `electrical_tilt`, `mechanical_tilt`, `tx_power`, and `antenna_height`.
4. Drive-test data is loaded from bridge/DB using `session_ids`, or from `drive_rows` if supplied by the frontend.
5. Building, polygon, terrain, and frontend grid context are prepared.
6. The RF engine predicts LTE metrics (`pred_rsrp`, `pred_rsrq`, `pred_sinr`) over the selected grid/population.
7. The geo correction pipeline adjusts/display-corrects the output using terrain, clutter, polygons, and drive-test replacement/blending settings.
8. Baseline rows are written to `lte_prediction_baseline_results`.
9. Geo feature rows are written to `lte_prediction_geo_features`.
10. The job status is updated in memory.

## Input Tables

| Table | Purpose |
| --- | --- |
| `site_prediction` | Current site/sector antenna configuration |
| `tbl_network_log` | Serving drive-test records |
| `tbl_network_log_neighbour` | Neighbor drive-test records when requested |
| `map_regions` / polygon tables | Region filters and polygon context |
| `grid_analytics_results` | Optional frontend grid sampling source |

When the Python bridge is configured, reads and writes are routed through the .NET API instead of direct SQL for supported operations.

## Output Tables

### `lte_prediction_baseline_results`

This is the main baseline RF output. It stores the predicted LTE metrics and cell identity fields.

Important columns:

| Column | Meaning |
| --- | --- |
| `project_id` | Project id |
| `job_id` | Baseline job id generated by this ML service |
| `lat`, `lon` | Prediction point |
| `pred_rsrp`, `pred_rsrq`, `pred_sinr` | Predicted LTE RF metrics |
| `node_b_id`, `cell_id`, `nodeb_id_cell_id` | Cell identity |
| `operator` | Operator used for filtering/display |
| `created_at` | Write timestamp |

The service uses a delta/upsert style write. It avoids blindly duplicating unchanged baseline rows when bridge/direct DB helpers support that behavior.

### `lte_prediction_geo_features`

This stores the geo/environment features used by downstream modules.

Important columns include:

| Column | Meaning |
| --- | --- |
| `baseline_job_id` | Links feature rows to the baseline run |
| `project_id`, `region`, `operator` | Scope |
| `grid_id` | Frontend or generated grid id |
| `nodeb_id_cell_id` | Serving/target cell identity |
| `clutter_class`, `morphology_cluster` | Environment classification |
| `building_count`, `building_area_ratio` | Building density context |
| `los_blocked_ratio`, `nlos_flag` | LOS/NLOS context |
| `terrain_elevation_m`, `terrain_slope_deg` | DEM context |
| `serving_distance_m`, `nearest_site_distance_m` | Distance features |
| `azimuth_delta_deg`, `polygon_alignment` | Geometry alignment features |

## Bridge Behavior

If `PYTHON_BRIDGE_BASE_URL` or `SIGNAL_TRACKERS_BRIDGE_URL` is configured, `utils/python_bridge.py` returns a bridge client. In this mode:

| Operation | Bridge endpoint |
| --- | --- |
| Save baseline rows | `SaveLtePredictionBaselineResults` |
| Save geo features | `SaveLtePredictionGeoFeatures` |

If the bridge is not configured, the service writes directly with SQLAlchemy using `DATABASE_URL` or `DATABASE_URL_Taiwan`.

## Why This Module Matters

Optimized prediction and tilt recommendation both depend on a successful baseline run:

- Optimized prediction reads `lte_prediction_baseline_results` to compare before/after RF.
- Tilt recommendation reads the latest baseline job for the project/operator.
- Geo features from this run provide blockage, terrain, clutter, and azimuth context for recommendation quality.

## Common Debug Checks

If LTE prediction returns unexpected results:

1. Check the job status endpoint for the actual error.
2. Confirm `session_ids` exist and contain serving records.
3. Confirm `site_prediction` has rows for the project and selected operator.
4. Confirm the operator string matches between site rows and drive rows.
5. Check whether `polygon_ids` narrowed the work area too much.
6. Check logs for:
   - `[LTE][JOB_START]`
   - `[LTE][BASELINE_DB_WRITE]`
   - `[LTE][BASELINE_DB_WRITE_DONE]`
   - `[LTE][GEO_DB_WRITE]`
   - `[LTE][GEO_DB_WRITE_DONE]`

## Production Notes

- `job_id` is the baseline run id and is later used by tilt recommendation and optimized prediction.
- Keep baseline logic stable before debugging optimized runs; optimized output is only as reliable as the baseline source.
- Avoid deleting or overwriting `lte_prediction_geo_features` without assessing the downstream impact on tilt recommendation and optimized prediction workflows.
