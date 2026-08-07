from __future__ import annotations

TABLE_NAME = "lte_future_coverage_kpi_prediction_results"

CREATE_TABLE_SQL = f"""
CREATE TABLE IF NOT EXISTS {TABLE_NAME} (
    id BIGINT UNSIGNED NOT NULL AUTO_INCREMENT,
    project_id BIGINT NOT NULL,
    baseline_job_id VARCHAR(100) NOT NULL,
    model_run_id VARCHAR(100) NOT NULL,
    grid_id BIGINT NOT NULL,
    grid_row INT NULL,
    grid_col INT NULL,
    grid_centroid_lat DOUBLE NOT NULL,
    grid_centroid_lon DOUBLE NOT NULL,
    current_rsrp DOUBLE NULL,
    current_rsrq DOUBLE NULL,
    current_sinr DOUBLE NULL,
    delta_rsrp DOUBLE NULL,
    delta_rsrq DOUBLE NULL,
    delta_sinr DOUBLE NULL,
    pred_rsrp DOUBLE NULL,
    pred_rsrq DOUBLE NULL,
    pred_sinr DOUBLE NULL,
    model_name VARCHAR(100) NOT NULL DEFAULT 'future_coverage_kpi_prediction',
    model_version VARCHAR(100) NULL,
    weights_rsrp_path VARCHAR(500) NULL,
    weights_rsrq_path VARCHAR(500) NULL,
    weights_sinr_path VARCHAR(500) NULL,
    input_source VARCHAR(100) NOT NULL DEFAULT 'python_bridge_baseline',
    status VARCHAR(50) NOT NULL DEFAULT 'completed',
    created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    PRIMARY KEY (id),
    UNIQUE KEY uq_future_coverage_kpi_project_baseline_run_grid (
        project_id,
        baseline_job_id,
        model_run_id,
        grid_id
    ),
    KEY idx_future_coverage_kpi_project_id (project_id),
    KEY idx_future_coverage_kpi_project_baseline (project_id, baseline_job_id),
    KEY idx_future_coverage_kpi_model_run_id (model_run_id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;
"""

RESULT_COLUMNS = [
    "project_id",
    "baseline_job_id",
    "model_run_id",
    "grid_id",
    "grid_row",
    "grid_col",
    "grid_centroid_lat",
    "grid_centroid_lon",
    "current_rsrp",
    "current_rsrq",
    "current_sinr",
    "delta_rsrp",
    "delta_rsrq",
    "delta_sinr",
    "pred_rsrp",
    "pred_rsrq",
    "pred_sinr",
    "model_name",
    "model_version",
    "weights_rsrp_path",
    "weights_rsrq_path",
    "weights_sinr_path",
    "input_source",
    "status",
]

EXPECTED_COLUMNS = [
    "id",
    *RESULT_COLUMNS,
    "created_at",
    "updated_at",
]
