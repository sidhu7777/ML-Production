from __future__ import annotations


RECOMMENDATION_TABLE_NAME = "lte_model3_current_recommendation_results"
RF_SURFACE_TABLE_NAME = "lte_model3_current_recommendation_rf_results"


CREATE_RECOMMENDATION_TABLE_SQL = f"""
CREATE TABLE IF NOT EXISTS {RECOMMENDATION_TABLE_NAME} (
    id BIGINT UNSIGNED NOT NULL AUTO_INCREMENT,
    project_id BIGINT NOT NULL,
    baseline_job_id VARCHAR(100) NOT NULL,
    model_run_id VARCHAR(100) NOT NULL,
    recommendation_id VARCHAR(100) NOT NULL,
    source_model2_run_id VARCHAR(100) NULL,
    region VARCHAR(50) NOT NULL DEFAULT 'india',
    operator VARCHAR(100) NULL,

    site_id VARCHAR(100) NOT NULL,
    sector_id VARCHAR(150) NOT NULL,
    node_cell_id VARCHAR(200) NOT NULL,
    canonical_physical_cell_id VARCHAR(200) NULL,
    sector_congested_node_cell_ids TEXT NULL,
    band VARCHAR(50) NOT NULL,
    earfcn VARCHAR(50) NULL,

    grid_count BIGINT NULL,
    congested_grid_count BIGINT NULL,
    prb_before_pct DOUBLE NULL,
    rrc_before_pct DOUBLE NULL,
    rrc_users_before DOUBLE NULL,
    pressure_before_pct DOUBLE NULL,

    recommended_action VARCHAR(100) NOT NULL,
    recommendation_level VARCHAR(50) NOT NULL,
    status VARCHAR(50) NOT NULL,
    resolved_flag TINYINT(1) NOT NULL DEFAULT 0,
    decision_path TEXT NULL,
    attempted_actions TEXT NULL,
    action_reason TEXT NULL,
    next_step TEXT NULL,
    priority_score DOUBLE NULL,

    load_balance_possible TINYINT(1) NULL,
    selected_peer_node_cell_id VARCHAR(200) NULL,
    selected_peer_band VARCHAR(50) NULL,
    recommended_band_to_add VARCHAR(50) NULL,
    available_bands_to_add TEXT NULL,
    carrier_addition_possible TINYINT(1) NULL,
    carrier_addition_blocked TINYINT(1) NULL,
    max_supported_carriers INT NULL,
    existing_carrier_count INT NULL,
    existing_carriers TEXT NULL,

    new_sector_value VARCHAR(200) NULL,
    new_site_value VARCHAR(200) NULL,
    projected_prb_after_pct DOUBLE NULL,
    projected_rrc_after_pct DOUBLE NULL,
    projected_rrc_users_after DOUBLE NULL,
    pressure_after_pct DOUBLE NULL,

    resimulation_required TINYINT(1) NOT NULL DEFAULT 0,
    resimulation_flow TEXT NULL,
    after_rf_job_id VARCHAR(100) NULL,
    after_rf_rows BIGINT NULL,
    affected_cells_count INT NULL,
    affected_sites_count INT NULL,
    rf_runtime_sec DOUBLE NULL,

    model_name VARCHAR(100) NOT NULL DEFAULT 'model3_current_congestion_recommendation',
    model_version VARCHAR(100) NULL,
    input_source VARCHAR(100) NOT NULL DEFAULT 'model2_current_input',
    artifact_recommendations_path VARCHAR(500) NULL,
    artifact_after_rf_path VARCHAR(500) NULL,
    created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,

    PRIMARY KEY (id),
    UNIQUE KEY uq_model3_recommendation (
        project_id,
        baseline_job_id,
        model_run_id,
        recommendation_id
    ),
    KEY idx_model3_project_baseline (project_id, baseline_job_id),
    KEY idx_model3_run (model_run_id),
    KEY idx_model3_cell (site_id, sector_id, node_cell_id, band),
    KEY idx_model3_status (status, resolved_flag),
    KEY idx_model3_action (recommended_action, recommendation_level)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;
"""


CREATE_RF_SURFACE_TABLE_SQL = f"""
CREATE TABLE IF NOT EXISTS {RF_SURFACE_TABLE_NAME} (
    id BIGINT UNSIGNED NOT NULL AUTO_INCREMENT,
    project_id BIGINT NOT NULL,
    baseline_job_id VARCHAR(100) NOT NULL,
    model_run_id VARCHAR(100) NOT NULL,
    recommendation_id VARCHAR(100) NULL,
    rf_stage VARCHAR(20) NOT NULL DEFAULT 'after',
    region VARCHAR(50) NOT NULL DEFAULT 'india',
    operator VARCHAR(100) NULL,

    grid_id BIGINT NULL,
    grid_row INT NULL,
    grid_col INT NULL,
    lat DOUBLE NOT NULL,
    lon DOUBLE NOT NULL,
    lat_6dp DOUBLE NULL,
    lon_6dp DOUBLE NULL,

    site_id VARCHAR(100) NULL,
    sector_id VARCHAR(150) NULL,
    node_cell_id VARCHAR(200) NULL,
    canonical_physical_cell_id VARCHAR(200) NULL,
    band VARCHAR(50) NULL,
    earfcn VARCHAR(50) NULL,

    pred_rsrp DOUBLE NULL,
    pred_rsrq DOUBLE NULL,
    pred_sinr DOUBLE NULL,
    pred_rsrp_smoothed DOUBLE NULL,
    pred_rsrq_smoothed DOUBLE NULL,
    pred_sinr_smoothed DOUBLE NULL,
    best_server_flag TINYINT(1) NULL,
    affected_flag TINYINT(1) NULL,

    rf_source VARCHAR(100) NOT NULL DEFAULT 'model3_affected_rf_rerun',
    artifact_path VARCHAR(500) NULL,
    created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,

    PRIMARY KEY (id),
    UNIQUE KEY uq_model3_rf_surface (
        project_id,
        baseline_job_id,
        model_run_id,
        rf_stage,
        lat,
        lon,
        node_cell_id,
        band
    ),
    KEY idx_model3_rf_project_baseline (project_id, baseline_job_id),
    KEY idx_model3_rf_run_stage (model_run_id, rf_stage),
    KEY idx_model3_rf_cell (site_id, sector_id, node_cell_id, band),
    KEY idx_model3_rf_grid (grid_id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;
"""


RECOMMENDATION_RESULT_COLUMNS = [
    "project_id",
    "baseline_job_id",
    "model_run_id",
    "recommendation_id",
    "source_model2_run_id",
    "region",
    "operator",
    "site_id",
    "sector_id",
    "node_cell_id",
    "canonical_physical_cell_id",
    "sector_congested_node_cell_ids",
    "band",
    "earfcn",
    "grid_count",
    "congested_grid_count",
    "prb_before_pct",
    "rrc_before_pct",
    "rrc_users_before",
    "pressure_before_pct",
    "recommended_action",
    "recommendation_level",
    "status",
    "resolved_flag",
    "decision_path",
    "attempted_actions",
    "action_reason",
    "next_step",
    "priority_score",
    "load_balance_possible",
    "selected_peer_node_cell_id",
    "selected_peer_band",
    "recommended_band_to_add",
    "available_bands_to_add",
    "carrier_addition_possible",
    "carrier_addition_blocked",
    "max_supported_carriers",
    "existing_carrier_count",
    "existing_carriers",
    "new_sector_value",
    "new_site_value",
    "projected_prb_after_pct",
    "projected_rrc_after_pct",
    "projected_rrc_users_after",
    "pressure_after_pct",
    "resimulation_required",
    "resimulation_flow",
    "after_rf_job_id",
    "after_rf_rows",
    "affected_cells_count",
    "affected_sites_count",
    "rf_runtime_sec",
    "model_name",
    "model_version",
    "input_source",
    "artifact_recommendations_path",
    "artifact_after_rf_path",
]


RF_SURFACE_RESULT_COLUMNS = [
    "project_id",
    "baseline_job_id",
    "model_run_id",
    "recommendation_id",
    "rf_stage",
    "region",
    "operator",
    "grid_id",
    "grid_row",
    "grid_col",
    "lat",
    "lon",
    "lat_6dp",
    "lon_6dp",
    "site_id",
    "sector_id",
    "node_cell_id",
    "canonical_physical_cell_id",
    "band",
    "earfcn",
    "pred_rsrp",
    "pred_rsrq",
    "pred_sinr",
    "pred_rsrp_smoothed",
    "pred_rsrq_smoothed",
    "pred_sinr_smoothed",
    "best_server_flag",
    "affected_flag",
    "rf_source",
    "artifact_path",
]


RECOMMENDATION_EXPECTED_COLUMNS = [
    "id",
    *RECOMMENDATION_RESULT_COLUMNS,
    "created_at",
    "updated_at",
]


RF_SURFACE_EXPECTED_COLUMNS = [
    "id",
    *RF_SURFACE_RESULT_COLUMNS,
    "created_at",
    "updated_at",
]
