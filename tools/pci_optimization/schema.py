from __future__ import annotations


RESULT_TABLE_NAME = "lte_pci_optimization_results"


# One row per (sector, rule) finding -- covers both a detected conflict
# (Collision/Confusion/Mod-N/Grouped/Co-centric) and, where the optimizer
# ran, that same sector's recommendation (current_pci -> suggested_pci).
# Mirrors the shape ML/tests/Pci_optimization/pci_map_dashboard.py already
# produces via build_conflict_summary_table()/build_recommendation_table(),
# plus the closed-loop verification fields (iterations/converged/
# verified_clean/stopped_reason/infeasible_proof) so a caller can tell
# "fully resolved" apart from "best effort, mathematically impossible"
# without re-deriving it. Production-only: every row here is a real
# project/site/PCI value, so there's no synthetic/demo/data-source
# provenance tracking -- that's a test-dashboard-only concept.
CREATE_RESULT_TABLE_SQL = f"""
CREATE TABLE IF NOT EXISTS {RESULT_TABLE_NAME} (
    id BIGINT UNSIGNED NOT NULL AUTO_INCREMENT,
    project_id BIGINT NOT NULL,
    job_id VARCHAR(100) NOT NULL,
    region VARCHAR(50) NOT NULL DEFAULT 'india',
    operator VARCHAR(100) NULL,

    site_id VARCHAR(150) NOT NULL,
    sector_id VARCHAR(150) NOT NULL DEFAULT '',
    earfcn VARCHAR(50) NOT NULL DEFAULT '',
    band VARCHAR(50) NULL,
    lat DOUBLE NULL,
    lon DOUBLE NULL,

    rule_type VARCHAR(20) NOT NULL,
    severity VARCHAR(20) NULL,
    current_pci INT NOT NULL,
    suggested_pci INT NULL,
    changed_flag TINYINT(1) NULL,
    conflict_cost_before DOUBLE NULL,
    conflict_cost_after DOUBLE NULL,
    same_earfcn_sectors_considered INT NULL,

    neighbor_distance_m DOUBLE NULL,
    iterations INT NULL,
    converged TINYINT(1) NULL,
    verified_clean TINYINT(1) NULL,
    remaining_conflicts INT NULL,
    stopped_reason VARCHAR(30) NULL,
    infeasible_proof TEXT NULL,

    created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,

    PRIMARY KEY (id),
    UNIQUE KEY uq_pci_optimization_row (
        project_id,
        job_id,
        site_id,
        sector_id,
        earfcn,
        rule_type,
        current_pci
    ),
    KEY idx_pci_optimization_project_job (project_id, job_id),
    KEY idx_pci_optimization_site (site_id, sector_id, earfcn),
    KEY idx_pci_optimization_rule (rule_type, severity)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;
"""


RESULT_COLUMNS = [
    "project_id",
    "job_id",
    "region",
    "operator",
    "site_id",
    "sector_id",
    "earfcn",
    "band",
    "lat",
    "lon",
    "rule_type",
    "severity",
    "current_pci",
    "suggested_pci",
    "changed_flag",
    "conflict_cost_before",
    "conflict_cost_after",
    "same_earfcn_sectors_considered",
    "neighbor_distance_m",
    "iterations",
    "converged",
    "verified_clean",
    "remaining_conflicts",
    "stopped_reason",
    "infeasible_proof",
]


RESULT_EXPECTED_COLUMNS = [
    "id",
    *RESULT_COLUMNS,
    "created_at",
    "updated_at",
]
