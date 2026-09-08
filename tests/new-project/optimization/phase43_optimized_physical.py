"""Phase 43 experimental physical scorer.

This file is intentionally outside production code. It keeps the production
terrain/building formulas but changes the hot loop to collect arrays and assign
the result columns once instead of mutating a DataFrame row by row.
"""
from __future__ import annotations

from pathlib import Path
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

from tools.lte_prediction.dem_utils import ensure_project_dem
from tools.lte_prediction.geo_correction_pipeline import building_df_to_gdf
from tools.lte_prediction_offset import phase27_physical as prod


def _score_chunk(args):
    chunk, site_df, building_df, project_id, region, dem_raster_path, clutter_by_grid, allow_auto_dem = args
    return score_candidates_phase43(
        chunk,
        site_df,
        building_df,
        project_id,
        region,
        dem_raster_path=dem_raster_path,
        clutter_by_grid=clutter_by_grid,
        allow_auto_dem=allow_auto_dem,
    )


def score_candidates_phase43_v2(
    candidates: pd.DataFrame,
    site_df: pd.DataFrame,
    building_df: pd.DataFrame,
    project_id: int,
    region: str,
    dem_raster_path: str | Path | None = None,
    clutter_by_grid: dict | None = None,
    allow_auto_dem: bool = True,
    workers: int = 4,
) -> pd.DataFrame:
    """Parallel Phase 43 scorer.

    Candidate rows are independent after raw COST231 selection, so chunks can be
    scored separately and concatenated back in original order. Each worker opens
    its own DEM handle, preserving the production terrain/building formulas.
    """
    workers = max(1, int(workers or 1))
    if workers == 1 or len(candidates) < 20_000:
        return score_candidates_phase43(
            candidates,
            site_df,
            building_df,
            project_id,
            region,
            dem_raster_path=dem_raster_path,
            clutter_by_grid=clutter_by_grid,
            allow_auto_dem=allow_auto_dem,
        )

    chunk_positions = [part for part in np.array_split(np.arange(len(candidates)), workers) if len(part)]
    chunks = [candidates.iloc[part].copy() for part in chunk_positions]
    print(f"[LTE_OFFSET][PHASE43_V2] workers={workers} chunks={len(chunks)} rows={len(candidates)}", flush=True)
    args = [
        (chunk, site_df, building_df, project_id, region, dem_raster_path, clutter_by_grid or {}, allow_auto_dem)
        for chunk in chunks
    ]
    with ProcessPoolExecutor(max_workers=workers) as executor:
        scored = list(executor.map(_score_chunk, args))
    return pd.concat(scored, ignore_index=False).sort_index()


def score_candidates_phase43(
    candidates: pd.DataFrame,
    site_df: pd.DataFrame,
    building_df: pd.DataFrame,
    project_id: int,
    region: str,
    dem_raster_path: str | Path | None = None,
    clutter_by_grid: dict | None = None,
    allow_auto_dem: bool = True,
) -> pd.DataFrame:
    """Apply production-equivalent physical corrections with batched writes."""
    out = candidates.copy()
    sites = site_df.drop_duplicates("strict_cell_key").set_index("strict_cell_key")
    buildings = building_df_to_gdf(building_df)
    if not buildings.empty:
        buildings, building_alignment = prod._align_buildings_to_prediction_extent(buildings, out)
        print(f"[LTE_OFFSET][PHASE43_BUILDINGS] rows={len(buildings)} alignment={building_alignment}", flush=True)
        heights = pd.to_numeric(buildings.get("building_height_m"), errors="coerce")
        levels = pd.to_numeric(buildings.get("building_levels"), errors="coerce")
        buildings["height_m"] = heights.fillna(levels * 3.0).fillna(prod.DEFAULT_BUILDING_HEIGHT_M).clip(3.0, 120.0)
        sindex = buildings.sindex
    else:
        sindex = None
        print("[LTE_OFFSET][PHASE43_BUILDINGS] rows=0 alignment=empty", flush=True)

    dem = None
    explicit_dem = Path(dem_raster_path).expanduser() if dem_raster_path else None
    try:
        if explicit_dem is not None:
            if not explicit_dem.is_file():
                raise FileNotFoundError(f"Configured DEM does not exist: {explicit_dem}")
            dem_path = explicit_dem
        else:
            if not allow_auto_dem:
                raise FileNotFoundError("No approved project terrain DEM resolved")
            dem_path = ensure_project_dem(int(project_id), str(region), site_df)
        dem = prod._DemSampler(dem_path)
        print(f"[LTE_OFFSET][PHASE43_DEM] enabled=True path={dem_path} selected_band={dem.band}", flush=True)
    except Exception as exc:
        if explicit_dem is not None:
            raise RuntimeError(f"Configured terrain DEM is unusable: {exc}") from exc
        print(f"[LTE_OFFSET][PHASE43_DEM] enabled=False reason={exc}", flush=True)

    n = len(out)
    building_loss_values = np.zeros(n, dtype=float)
    terrain_loss_values = np.zeros(n, dtype=float)
    terrain_excess_values = np.full(n, np.nan, dtype=float)
    terrain_peak_values = np.full(n, np.nan, dtype=float)
    terrain_decisions = np.empty(n, dtype=object)
    branches = np.empty(n, dtype=object)
    clutters = np.empty(n, dtype=object)
    terrain_decisions[:] = "not_evaluated"
    branches[:] = "clear"
    clutters[:] = "Open"

    clutter_by_grid = clutter_by_grid or {}
    positions = {index: pos for pos, index in enumerate(out.index)}
    lat_col = out["lat"].to_numpy(dtype=float)
    lon_col = out["lon"].to_numpy(dtype=float)
    freq_col = pd.to_numeric(out["serving_frequency_mhz"], errors="coerce").to_numpy(dtype=float)
    grid_col = out["grid_id"].astype(str).to_numpy()

    for key, index_values in out.groupby("strict_cell_key", sort=False, dropna=False).groups.items():
        if key not in sites.index:
            continue
        site = sites.loc[key]
        if isinstance(site, pd.DataFrame):
            site = site.iloc[0]
        tx_lat = float(site.lat)
        tx_lon = float(site.lon)
        tx_height = float(site.Height)
        for row_index in index_values:
            pos = positions[row_index]
            freq = float(freq_col[pos])
            if sindex is not None:
                building_loss, branch, clutter = prod._building_loss(
                    buildings, sindex, tx_lat, tx_lon, tx_height, float(lat_col[pos]), float(lon_col[pos]), freq
                )
            else:
                building_loss, branch, clutter = 0.0, "clear", "Open"
            source_clutter = str(clutter_by_grid.get(str(grid_col[pos]), "")).strip()
            if source_clutter and source_clutter.lower() not in {"nan", "none", "unknown"}:
                clutter = source_clutter if branch != "indoor" else "Indoor"
            if dem is None:
                terrain, excess, peak, decision = 0.0, np.nan, np.nan, "dem_disabled"
            elif str(clutter).lower() == "water":
                terrain, excess, peak, decision = 0.0, np.nan, np.nan, "water_land_cover"
            else:
                terrain, excess, peak, decision = prod._terrain_loss_details(
                    dem, tx_lat, tx_lon, tx_height, float(lat_col[pos]), float(lon_col[pos]), freq
                )
            building_loss_values[pos] = building_loss
            terrain_loss_values[pos] = terrain
            terrain_excess_values[pos] = excess
            terrain_peak_values[pos] = peak
            terrain_decisions[pos] = decision
            branches[pos] = branch
            clutters[pos] = clutter

    if dem is not None:
        dem.close()

    out["building_obstruction_loss_db"] = building_loss_values
    out["terrain_diffraction_loss_db"] = terrain_loss_values
    out["terrain_fresnel_excess_m"] = terrain_excess_values
    out["terrain_peak_clearance_m"] = terrain_peak_values
    out["terrain_decision"] = terrain_decisions
    out["obstruction_branch"] = branches
    out["clutter_class"] = clutters
    terrain_counts = out["terrain_decision"].value_counts(dropna=False).to_dict()
    print(
        "[LTE_OFFSET][PHASE43_TERRAIN] "
        f"policy=fresnel_gate_{prod.TERRAIN_FRESNEL_CLEARANCE_FRACTION:.2f} "
        f"nonzero={int((out['terrain_diffraction_loss_db'] > 0).sum())} decisions={terrain_counts}",
        flush=True,
    )
    out["physical_rsrp_unclipped"] = (
        pd.to_numeric(out["raw_cost231_rsrp"], errors="coerce")
        + pd.to_numeric(out["building_obstruction_loss_db"], errors="coerce")
        - pd.to_numeric(out["terrain_diffraction_loss_db"], errors="coerce")
    )
    return out


