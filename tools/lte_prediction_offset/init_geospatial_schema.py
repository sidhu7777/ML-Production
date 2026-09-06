"""Create the additive project-geospatial schema in India and Taiwan DBs.

This migration preserves existing geometry.  It only adds provenance columns
to ``tbl_savepolygon`` and creates tables used to cache validated project data.
"""
from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv
from sqlalchemy import create_engine, text


ROOT = Path(__file__).resolve().parents[2]
load_dotenv(ROOT / ".env")

DATABASES = {
    "india": "DATABASE_URL",
    "taiwan": "DATABASE_URL_Taiwan",
}


TABLES = (
    """
    CREATE TABLE IF NOT EXISTS tbl_project_geo_dataset (
        id BIGINT NOT NULL AUTO_INCREMENT,
        project_id BIGINT NOT NULL,
        dataset_type VARCHAR(64) NOT NULL,
        source_name VARCHAR(128) NOT NULL,
        source_version VARCHAR(128) NULL,
        boundary_hash CHAR(64) NOT NULL,
        resolution_m DECIMAL(10,3) NULL,
        checksum CHAR(64) NULL,
        storage_uri VARCHAR(1024) NULL,
        metadata_json JSON NULL,
        fetched_at DATETIME(6) NOT NULL DEFAULT CURRENT_TIMESTAMP(6),
        validated_at DATETIME(6) NULL,
        is_active TINYINT(1) NOT NULL DEFAULT 1,
        PRIMARY KEY (id),
        KEY ix_project_geo_dataset_lookup (project_id, dataset_type, is_active),
        KEY ix_project_geo_dataset_source (source_name, source_version)
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
    """,
    """
    CREATE TABLE IF NOT EXISTS tbl_project_building_profile (
        id BIGINT NOT NULL AUTO_INCREMENT,
        project_id BIGINT NOT NULL,
        building_geometry_id VARCHAR(128) NOT NULL,
        geo_dataset_id BIGINT NULL,
        height_m DECIMAL(8,3) NULL,
        height_method VARCHAR(64) NOT NULL,
        height_source VARCHAR(128) NULL,
        confidence DECIMAL(5,4) NULL,
        is_active TINYINT(1) NOT NULL DEFAULT 1,
        created_at DATETIME(6) NOT NULL DEFAULT CURRENT_TIMESTAMP(6),
        updated_at DATETIME(6) NOT NULL DEFAULT CURRENT_TIMESTAMP(6) ON UPDATE CURRENT_TIMESTAMP(6),
        PRIMARY KEY (id),
        UNIQUE KEY uq_project_building_profile (project_id, building_geometry_id, geo_dataset_id),
        KEY ix_project_building_profile_active (project_id, is_active),
        CONSTRAINT fk_project_building_profile_dataset
            FOREIGN KEY (geo_dataset_id) REFERENCES tbl_project_geo_dataset(id)
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
    """,
    """
    CREATE TABLE IF NOT EXISTS tbl_project_clutter_tile (
        id BIGINT NOT NULL AUTO_INCREMENT,
        project_id BIGINT NOT NULL,
        geo_dataset_id BIGINT NOT NULL,
        grid_id VARCHAR(128) NOT NULL,
        geometry_wkt LONGTEXT NOT NULL,
        clutter_class VARCHAR(64) NOT NULL,
        land_cover_class VARCHAR(128) NULL,
        resolution_m DECIMAL(10,3) NOT NULL,
        is_active TINYINT(1) NOT NULL DEFAULT 1,
        created_at DATETIME(6) NOT NULL DEFAULT CURRENT_TIMESTAMP(6),
        PRIMARY KEY (id),
        UNIQUE KEY uq_project_clutter_tile (project_id, geo_dataset_id, grid_id),
        KEY ix_project_clutter_tile_active (project_id, is_active),
        CONSTRAINT fk_project_clutter_tile_dataset
            FOREIGN KEY (geo_dataset_id) REFERENCES tbl_project_geo_dataset(id)
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
    """,
    """
    CREATE TABLE IF NOT EXISTS tbl_project_dem_asset (
        id BIGINT NOT NULL AUTO_INCREMENT,
        project_id BIGINT NOT NULL,
        geo_dataset_id BIGINT NULL,
        source_name VARCHAR(128) NOT NULL,
        storage_uri VARCHAR(1024) NOT NULL,
        checksum CHAR(64) NOT NULL,
        crs VARCHAR(128) NOT NULL,
        resolution_m DECIMAL(10,3) NOT NULL,
        selected_elevation_band INT NOT NULL,
        nodata_value DOUBLE NULL,
        is_active TINYINT(1) NOT NULL DEFAULT 1,
        created_at DATETIME(6) NOT NULL DEFAULT CURRENT_TIMESTAMP(6),
        PRIMARY KEY (id),
        UNIQUE KEY uq_project_dem_asset (project_id, checksum),
        KEY ix_project_dem_asset_active (project_id, is_active),
        CONSTRAINT fk_project_dem_asset_dataset
            FOREIGN KEY (geo_dataset_id) REFERENCES tbl_project_geo_dataset(id)
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
    """,
)


SAVE_POLYGON_COLUMNS = {
    "geo_dataset_id": "BIGINT NULL",
    "source_name": "VARCHAR(128) NULL",
    "source_version": "VARCHAR(128) NULL",
    "geometry_hash": "CHAR(64) NULL",
    "is_active": "TINYINT(1) NOT NULL DEFAULT 1",
}


def _table_exists(connection, name: str) -> bool:
    return bool(connection.execute(text("""
        SELECT COUNT(*) FROM information_schema.TABLES
        WHERE TABLE_SCHEMA = DATABASE() AND TABLE_NAME = :name
    """), {"name": name}).scalar())


def _columns(connection, table: str) -> set[str]:
    return set(connection.execute(text("""
        SELECT COLUMN_NAME FROM information_schema.COLUMNS
        WHERE TABLE_SCHEMA = DATABASE() AND TABLE_NAME = :table
    """), {"table": table}).scalars())


def _index_exists(connection, table: str, index: str) -> bool:
    return bool(connection.execute(text("""
        SELECT COUNT(*) FROM information_schema.STATISTICS
        WHERE TABLE_SCHEMA = DATABASE() AND TABLE_NAME = :table AND INDEX_NAME = :index
    """), {"table": table, "index": index}).scalar())


def _preflight():
    engines = {}
    for label, env_key in DATABASES.items():
        url = os.getenv(env_key)
        if not url:
            raise RuntimeError(f"{env_key} is not configured")
        engine = create_engine(url, pool_pre_ping=True)
        with engine.connect() as connection:
            database = connection.execute(text("SELECT DATABASE()")).scalar()
            if not _table_exists(connection, "tbl_savepolygon"):
                raise RuntimeError(f"{label} ({database}) is missing tbl_savepolygon")
            print(f"[GEO_SCHEMA] preflight region={label} database={database} ok")
        engines[label] = engine
    return engines


def _migrate(label: str, engine) -> None:
    with engine.begin() as connection:
        for ddl in TABLES:
            connection.execute(text(ddl))

        existing = _columns(connection, "tbl_savepolygon")
        for column, definition in SAVE_POLYGON_COLUMNS.items():
            if column not in existing:
                connection.execute(text(f"ALTER TABLE tbl_savepolygon ADD COLUMN `{column}` {definition}"))

        if not _index_exists(connection, "tbl_savepolygon", "ix_savepolygon_project_dataset_active"):
            connection.execute(text("""
                CREATE INDEX ix_savepolygon_project_dataset_active
                ON tbl_savepolygon (project_id, geo_dataset_id, is_active)
            """))

        tables = [row[0] for row in connection.execute(text("""
            SELECT TABLE_NAME FROM information_schema.TABLES
            WHERE TABLE_SCHEMA = DATABASE()
              AND TABLE_NAME IN (
                'tbl_project_geo_dataset',
                'tbl_project_building_profile',
                'tbl_project_clutter_tile',
                'tbl_project_dem_asset'
              )
            ORDER BY TABLE_NAME
        """))]
        added = sorted(SAVE_POLYGON_COLUMNS.keys() & _columns(connection, "tbl_savepolygon"))
        print(f"[GEO_SCHEMA] migrated region={label} tables={tables} savepolygon_columns={added}")


def main() -> None:
    for label, engine in _preflight().items():
        _migrate(label, engine)


if __name__ == "__main__":
    main()
