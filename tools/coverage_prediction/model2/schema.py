from __future__ import annotations

TABLE_NAME = "lte_model2_demand_capacity_forecast_results"

CREATE_TABLE_SQL = f"""
CREATE TABLE IF NOT EXISTS {TABLE_NAME} (
    id BIGINT UNSIGNED NOT NULL AUTO_INCREMENT,
    project_id BIGINT NOT NULL,
    baseline_job_id VARCHAR(100) NOT NULL,
    model_run_id VARCHAR(100) NOT NULL,
    site_id VARCHAR(100) NOT NULL,
    sector_id VARCHAR(150) NOT NULL,
    node_cell_id VARCHAR(200) NOT NULL,
    canonical_physical_cell_id VARCHAR(200) NULL,
    band VARCHAR(50) NOT NULL,
    operator VARCHAR(100) NULL,
    current_prb_utilization_pct DOUBLE NULL,
    current_rrc_utilization_pct DOUBLE NULL,
    current_rrc_connected_users DOUBLE NULL,
    current_estimated_dl_capacity_mbps DOUBLE NULL,
    current_estimated_offered_traffic_mbps DOUBLE NULL,
    current_congested_flag TINYINT(1) NOT NULL DEFAULT 0,
    future_prb_utilization_pct DOUBLE NULL,
    future_rrc_utilization_pct DOUBLE NULL,
    future_rrc_connected_users DOUBLE NULL,
    future_estimated_offered_traffic_mbps DOUBLE NULL,
    future_congested_flag TINYINT(1) NULL,
    model_name VARCHAR(100) NOT NULL DEFAULT 'model2_demand_capacity_forecast',
    model_version VARCHAR(100) NULL,
    weights_demand_path VARCHAR(500) NULL,
    weights_users_path VARCHAR(500) NULL,
    weights_traffic_path VARCHAR(500) NULL,
    input_source VARCHAR(100) NOT NULL DEFAULT 'python_bridge_baseline',
    status VARCHAR(50) NOT NULL DEFAULT 'completed',
    created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    PRIMARY KEY (id),
    UNIQUE KEY uq_model2_project_baseline_run_cell (
        project_id,
        baseline_job_id,
        model_run_id,
        node_cell_id,
        band
    ),
    KEY idx_model2_project_id (project_id),
    KEY idx_model2_project_baseline (project_id, baseline_job_id),
    KEY idx_model2_model_run_id (model_run_id),
    KEY idx_model2_cell (site_id, sector_id, node_cell_id, band)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;
"""

RESULT_COLUMNS = [
    "project_id",
    "baseline_job_id",
    "model_run_id",
    "site_id",
    "sector_id",
    "node_cell_id",
    "canonical_physical_cell_id",
    "band",
    "operator",
    "current_prb_utilization_pct",
    "current_rrc_utilization_pct",
    "current_rrc_connected_users",
    "current_estimated_dl_capacity_mbps",
    "current_estimated_offered_traffic_mbps",
    "current_congested_flag",
    "future_prb_utilization_pct",
    "future_rrc_utilization_pct",
    "future_rrc_connected_users",
    "future_estimated_offered_traffic_mbps",
    "future_congested_flag",
    "model_name",
    "model_version",
    "weights_demand_path",
    "weights_users_path",
    "weights_traffic_path",
    "input_source",
    "status",
]

EXPECTED_COLUMNS = [
    "id",
    *RESULT_COLUMNS,
    "created_at",
    "updated_at",
]
