"""
Coverage-redundancy (overlapping) run: cached project data -> physical antennas -> drive-test
calibrated RF surface -> one-at-a-time removal test -> one verdict per site (or cell).

Verdicts
  COVERAGE_REDUNDANT  coverage and quality are kept without it. Capacity is NOT verified (no PRB/load
                      data yet): check the absorbing cells' load before any switch-off.
  KEEP                removing it loses coverage/quality; `reason` names the failed gate.
                      NEEDED_AFTER_REMOVALS(...) = redundant on its own, but not once the higher-ranked
                      removals are applied (they were each other's backup)
  DATA_DUPLICATE      the same physical antenna stored twice in site_prediction
  NOT_TESTABLE        config not trustworthy, outside the polygon, too little of it in the area, or
                      RF_SURFACE_NOT_VALIDATED: the calibrated model does not reproduce this operator's
                      drive test on held-out sessions, so no RSRP-based verdict is published
                      (the model's own answer stays in `model_verdict` / `model_reason`)

Outputs, output/project_<id>/<operator>_<level>/:
  candidates.csv       one row per site/cell: verdict, reason, rank, metrics, absorbers, confidence
  iterations.csv       removal order with network coverage after each step
  antennas.csv         cleaned antennas (duplicates, config issues, calibration offset)
  calibration.csv      per-antenna drive-test rows and offsets
  dt_validation.csv    drive-test rows: measured vs held-out prediction
  grid_states.parquet  every analysis point before / after all accepted removals
  candidate_footprints.parquet  per evaluated candidate: its footprint points before / after removing it alone
  summary.json         data counts, calibration + validation, thresholds, verdict counts, config

Run from the ML/ directory (after tests.overlapping.fetch_data):
    venv\\Scripts\\python.exe -m tests.overlapping.run_overlap --project-id 193 --operator all --level site
"""
from __future__ import annotations

import argparse
import json
import math
import re
import time
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Callable, Sequence

import numpy as np
import pandas as pd

from tests.overlapping.config import (
    DATA_DIR,
    LEVEL_CELL,
    LEVEL_SITE,
    OUTPUT_DIR,
    VERDICT_DUPLICATE,
    VERDICT_KEEP,
    VERDICT_NOT_TESTABLE,
    VERDICT_REDUNDANT,
    DecisionParams,
    OverlapConfig,
)
from tests.overlapping.geo import orient_lonlat, union_area
from tests.overlapping.removal_model import Candidate, ExtraGate, RedundancyResult, design_thresholds, run_redundancy
from tests.overlapping.rf_matrix import VALIDATION_FAIL, DriveTestCalibration, NetworkBuild, build_network, grid_point_ids
from tests.overlapping.site_cleaning import clean_sites, normalise_operator

VERDICT_ORDER = {VERDICT_REDUNDANT: 0, VERDICT_KEEP: 1, VERDICT_DUPLICATE: 2, VERDICT_NOT_TESTABLE: 3}


@dataclass
class InputData:
    site_rows: pd.DataFrame
    polygon_wkts: list[str]
    dt_serving: pd.DataFrame | None = None
    forecast_users: pd.DataFrame | None = None
    building_wkts: list[str] = field(default_factory=list)
    source: str = "cache"


def load_cached_inputs(project_id: int, data_dir: Path = DATA_DIR) -> InputData:
    folder = Path(data_dir) / f"project_{int(project_id)}"
    site_csv = folder / "site_prediction.csv"
    if not site_csv.exists():
        raise FileNotFoundError(
            f"{site_csv} is missing. Run first: venv\\Scripts\\python.exe -m tests.overlapping.fetch_data --project-id {project_id}"
        )

    def parquet(name: str) -> pd.DataFrame | None:
        path = folder / name
        return pd.read_parquet(path) if path.exists() else None

    polygons = json.loads((folder / "polygons.json").read_text())
    buildings = parquet("buildings.parquet")
    return InputData(
        site_rows=pd.read_csv(site_csv, low_memory=False),
        polygon_wkts=[p["wkt"] for p in polygons if p.get("wkt")],
        dt_serving=parquet("dt_serving.parquet"),
        forecast_users=parquet("forecast_users.parquet"),
        building_wkts=buildings["wkt"].dropna().tolist() if buildings is not None else [],
        source=str(folder),
    )


def build_candidates(antennas: pd.DataFrame, level: str, site_buffer_m: float) -> tuple[list[Candidate], pd.DataFrame]:
    considered = antennas[antennas["distance_to_area_m"] <= site_buffer_m]
    if level == LEVEL_SITE:
        groups = list(considered.groupby(["operator", "site_key"], sort=True))
    elif level == LEVEL_CELL:
        groups = [((r.operator, r.antenna_key), considered.loc[[i]]) for i, r in considered.iterrows()]
    else:
        raise ValueError(f"level must be '{LEVEL_SITE}' or '{LEVEL_CELL}'")

    candidates, meta = [], []
    for (operator, key), g in groups:
        cid = f"{operator}|{key}" if level == LEVEL_SITE else str(key)
        duplicate_of = sorted({d.rsplit("|", 1)[0] if level == LEVEL_SITE else d for d in g["duplicate_of"] if d})
        issue = next((s for s in g["site_config_issue"] if s), "")
        meta.append(
            {
                "candidate_id": cid,
                "level": level,
                "operator": operator,
                "site_key": str(g["site_key"].iat[0]),
                "antenna_keys": "|".join(g["antenna_key"]),
                "azimuths": ",".join(f"{a:.0f}" for a in g["azimuth"]),
                "lat": float(g["lat"].iat[0]),
                "lon": float(g["lon"].iat[0]),
                "in_polygon": bool(g["in_polygon"].any()),
                "distance_to_area_m": float(g["distance_to_area_m"].min()),
                "raw_rows": int(g["raw_rows"].sum()),
                "pcis": ",".join(p for p in g["pcis"] if p),
                "tx_power_defaulted": bool(g["tx_power_filled"].any()),
                "site_config_issue": issue,
                "duplicate_of": "|".join(duplicate_of),
                "duplicate_rule": "|".join(sorted({r for r in g["duplicate_rule"] if r})),
            }
        )
        cells = g.loc[g["net_index"] >= 0, "net_index"].to_numpy(dtype=int)
        if g["is_duplicate"].all():
            candidates.append(Candidate(cid, cells, VERDICT_DUPLICATE, f"DUPLICATE_OF({'|'.join(duplicate_of)})"))
        elif issue:
            candidates.append(Candidate(cid, cells, VERDICT_NOT_TESTABLE, f"CONFIG_AMBIGUOUS({issue})"))
        elif not g["in_polygon"].any():
            candidates.append(Candidate(cid, cells, VERDICT_NOT_TESTABLE, "OUTSIDE_POLYGON"))
        else:
            candidates.append(Candidate(cid, cells))
    return candidates, pd.DataFrame(meta)


def _confidence(verdict: str, footprint_points, dt_points: int, attributed_rows: int, decision: DecisionParams) -> str:
    if verdict not in (VERDICT_REDUNDANT, VERDICT_KEEP):
        return ""
    footprint_points = 0 if footprint_points is None or not np.isfinite(footprint_points) else footprint_points
    if footprint_points >= 20 and dt_points >= 20 and attributed_rows >= 50:
        return "HIGH"
    if footprint_points >= decision.min_footprint_points and attributed_rows >= 50:
        return "MEDIUM"
    return "LOW"


def enrich_candidates(
    result: RedundancyResult, meta: pd.DataFrame, candidates: Sequence[Candidate], build: NetworkBuild, cfg: OverlapConfig
) -> pd.DataFrame:
    df = meta.merge(result.candidates, on="candidate_id", how="left")
    net = build.network
    cells_of = {c.candidate_id: np.asarray(c.cells, dtype=int) for c in candidates}

    dt_per_cell = np.zeros(len(net.cells), dtype=int)
    dt_table = build.drive_test.table
    if not dt_table.empty:
        pid = grid_point_ids(net.points, cfg.grid_resolution_m, build.projection, dt_table["lat"], dt_table["lon"])
        serving = result.initial_state.serving[pid[pid >= 0]]
        dt_per_cell = np.bincount(serving[serving >= 0], minlength=len(net.cells))
    attributed_rows = net.cells["dt_rows"].to_numpy(dtype=int)
    pairs = build.drive_test.handover_site_pairs
    sites_seen_in_handovers = {s for pair in pairs for s in pair}

    dt_points, attributed, ho_share, confidence = [], [], [], []
    for row in df.itertuples(index=False):
        cells = cells_of.get(row.candidate_id, np.array([], dtype=int))
        n_dt = int(dt_per_cell[cells].sum()) if cells.size else 0
        n_attr = int(attributed_rows[cells].sum()) if cells.size else 0
        share = float("nan")
        absorbers = getattr(row, "absorbers", None)
        if row.site_key in sites_seen_in_handovers and isinstance(absorbers, str) and absorbers not in ("", "[]"):
            items = [a for a in json.loads(absorbers) if a["site_key"] != row.site_key]
            total = sum(a["share"] for a in items)
            if total > 0:
                share = sum(a["share"] for a in items if tuple(sorted((row.site_key, a["site_key"]))) in pairs) / total
        dt_points.append(n_dt)
        attributed.append(n_attr)
        ho_share.append(share)
        confidence.append(_confidence(row.verdict, getattr(row, "footprint_points", 0), n_dt, n_attr, cfg.decision))
    df["dt_points_in_footprint"] = dt_points
    df["attributed_dt_rows"] = attributed
    df["absorber_handover_neighbour_share"] = ho_share
    df["confidence"] = confidence
    return df


def withhold_unvalidated(table: pd.DataFrame, drive_test: DriveTestCalibration) -> pd.DataFrame:
    """RSRP-based verdicts are only published when the RF surface passed drive-test validation."""
    table = table.copy()
    table["model_verdict"] = table["verdict"]
    table["model_reason"] = table["reason"]
    if drive_test.validation_status != VALIDATION_FAIL:
        return table
    rf_based = table["verdict"].isin([VERDICT_REDUNDANT, VERDICT_KEEP])
    table.loc[rf_based, "verdict"] = VERDICT_NOT_TESTABLE
    table.loc[rf_based, "reason"] = "RF_SURFACE_NOT_VALIDATED(" + ";".join(drive_test.validation_failures) + ")"
    table.loc[rf_based, "confidence"] = ""
    return table


def _sorted(table: pd.DataFrame) -> pd.DataFrame:
    rank = table["removal_rank"] if "removal_rank" in table else pd.Series(np.nan, index=table.index)
    order = table["model_verdict"].map(VERDICT_ORDER) + 10 * table["verdict"].map(VERDICT_ORDER)
    return (
        table.assign(_order=order, _rank=rank.fillna(1e9))
        .sort_values(["_order", "_rank", "candidate_id"])
        .drop(columns=["_order", "_rank"])
        .reset_index(drop=True)
    )


@dataclass
class RunResult:
    config: OverlapConfig
    antennas: pd.DataFrame
    candidates: pd.DataFrame
    build: NetworkBuild
    redundancy: RedundancyResult
    summary: dict


def _clean_json(value):
    if isinstance(value, dict):
        return {str(k): _clean_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_clean_json(v) for v in value]
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def run_pipeline(
    cfg: OverlapConfig,
    inputs: InputData,
    extra_gates: Sequence[ExtraGate] = (),
    log: Callable[[str], None] = print,
) -> RunResult:
    if not cfg.operator or cfg.operator.lower() == "all":
        raise ValueError("Run one operator at a time: operators are separate networks and never absorb each other")
    started = time.time()
    antennas, cleaning_report = clean_sites(inputs.site_rows, cfg.operator, cfg.cleaning, cfg.rf)
    if antennas.empty:
        raise ValueError(f"No usable {cfg.operator} rows in site_prediction")
    area, polygon_swapped = orient_lonlat(union_area(inputs.polygon_wkts), antennas["lat"], antennas["lon"])
    log(f"[OVERLAP][DATA] operator={cfg.operator} level={cfg.decision.level} antennas={len(antennas)} "
        f"polygon_swapped={polygon_swapped} source={inputs.source}")

    build = build_network(cfg, antennas, area, inputs.dt_serving, inputs.forecast_users, inputs.building_wkts, log)
    thresholds = design_thresholds(cfg.decision, build.network.sigma_rsrp_db)
    candidates, meta = build_candidates(build.antennas, cfg.decision.level, cfg.site_buffer_m)
    redundancy = run_redundancy(build.network, candidates, cfg.decision, thresholds, extra_gates, log)
    table = _sorted(withhold_unvalidated(enrich_candidates(redundancy, meta, candidates, build, cfg), build.drive_test))

    dt = build.drive_test
    total_weight = float(build.network.points["weight"].sum())
    removed = redundancy.iterations["removed_candidate"].tolist() if not redundancy.iterations.empty else []
    summary = _clean_json(
        {
            "project_id": cfg.project_id,
            "operator": cfg.operator,
            "level": cfg.decision.level,
            "run_seconds": round(time.time() - started, 1),
            "capacity_checked": False,
            "note": "COVERAGE_REDUNDANT is not a switch-off decision: capacity/load is not verified yet.",
            "rf_surface_validation": {"status": dt.validation_status, "failures": dt.validation_failures},
            "verdicts": table["verdict"].value_counts().to_dict(),
            "model_verdicts": table["model_verdict"].value_counts().to_dict(),
            "removed_in_order": removed,
            "data": {
                "source": inputs.source,
                "polygon_swapped_lat_lon": polygon_swapped,
                "cleaning": cleaning_report,
                "points": build.report,
                "dt_serving_rows": 0 if inputs.dt_serving is None else len(inputs.dt_serving),
            },
            "network": {
                "antennas_in_rf": len(build.network.cells),
                "sites_in_rf": int(build.network.cells["site_key"].nunique()),
                "expected_coverage_initial": redundancy.covered_weight_initial / max(total_weight, 1e-12),
                "expected_coverage_after_removals": redundancy.covered_weight_final / max(total_weight, 1e-12),
            },
            "calibration": dt.summary(),
            "thresholds": {
                "kpi_rsrp_dbm": thresholds.kpi_rsrp_dbm,
                "sigma_rsrp_db": build.network.sigma_rsrp_db,
                "location_probability": thresholds.location_probability,
                "equivalent_design_rsrp_dbm": thresholds.rsrp_dbm,
                "min_probability_drop": cfg.decision.min_probability_drop,
                "kpi_sinr_db": thresholds.kpi_sinr_db,
                "interference_load_factor": build.network.interference_load_factor,
                "qrxlevmin_dbm": thresholds.qrxlevmin_dbm,
            },
            "config": cfg.to_dict(),
        }
    )
    log(f"[OVERLAP][DONE] rf_surface={dt.validation_status} verdicts={summary['verdicts']} "
        f"model_verdicts={summary['model_verdicts']} seconds={summary['run_seconds']}")
    return RunResult(cfg, build.antennas, table, build, redundancy, summary)


def write_outputs(run: RunResult, out_dir: Path) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    run.candidates.to_csv(out_dir / "candidates.csv", index=False)
    run.antennas.to_csv(out_dir / "antennas.csv", index=False)
    run.build.drive_test.antenna_table.to_csv(out_dir / "calibration.csv", index=False)
    run.build.drive_test.table.to_csv(out_dir / "dt_validation.csv", index=False)
    run.redundancy.iterations.to_csv(out_dir / "iterations.csv", index=False)

    net = run.build.network
    names = net.cells["antenna_key"].to_numpy(dtype=object)
    grid = net.points[["point_id", "lat", "lon", "row", "col", "in_polygon", "indoor", "users", "weight"]].copy()
    for tag, st in (("before", run.redundancy.initial_state), ("after", run.redundancy.final_state)):
        grid[f"serving_{tag}"] = np.where(st.serving >= 0, names[np.maximum(st.serving, 0)], "")
        grid[f"rsrp_{tag}"] = np.where(st.serving >= 0, st.rsrp, np.nan)
        grid[f"sinr_{tag}"] = np.where(np.isfinite(st.sinr), st.sinr, np.nan)
        grid[f"coverage_prob_{tag}"] = st.coverage_prob
        grid[f"covered_{tag}"] = st.covered
    grid.to_parquet(out_dir / "grid_states.parquet", index=False)

    point_ids = net.points["point_id"].to_numpy()
    frames = []
    for cid, pts in run.redundancy.standalone_points.items():
        if pts is None:
            continue
        served = pts["serving_after"] >= 0
        frames.append(
            pd.DataFrame(
                {
                    "candidate_id": cid,
                    "point_id": point_ids[pts["point_idx"]],
                    "rsrp_before": pts["rsrp_before"],
                    "rsrp_after": np.where(served, pts["rsrp_after"], np.nan),
                    "prob_before": pts["prob_before"],
                    "prob_after": pts["prob_after"],
                    "lost": pts["lost"],
                    "serving_after": np.where(served, names[np.maximum(pts["serving_after"], 0)], ""),
                }
            )
        )
    if frames:
        pd.concat(frames, ignore_index=True).to_parquet(out_dir / "candidate_footprints.parquet", index=False)
    (out_dir / "summary.json").write_text(json.dumps(run.summary, indent=2))
    return out_dir


def run_dir(project_id: int, operator: str, level: str) -> Path:
    slug = re.sub(r"[^A-Za-z0-9]+", "_", operator).strip("_").lower()
    return OUTPUT_DIR / f"project_{int(project_id)}" / f"{slug}_{level}"


def _parse_args() -> argparse.Namespace:
    d = DecisionParams()
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--project-id", type=int, default=193)
    p.add_argument("--operator", default="Airtel", help="Airtel | 'JIO 4G' | 'Vi India' | all (one run per operator)")
    p.add_argument("--level", choices=[LEVEL_SITE, LEVEL_CELL], default=LEVEL_SITE)
    p.add_argument("--grid", type=float, default=OverlapConfig.grid_resolution_m)
    p.add_argument("--rsrp-threshold", type=float, default=d.rsrp_threshold_dbm)
    p.add_argument("--sinr-threshold", type=float, default=d.sinr_threshold_db)
    p.add_argument("--location-probability", type=float, default=d.location_probability)
    p.add_argument("--min-retention", type=float, default=d.min_retention)
    p.add_argument("--max-hole-m2", type=float, default=d.max_hole_m2)
    p.add_argument("--max-network-loss", type=float, default=d.max_cumulative_coverage_loss)
    p.add_argument("--max-removals", type=int, default=None)
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    inputs = load_cached_inputs(args.project_id)
    if args.operator.lower() == "all":
        operators = sorted({normalise_operator(v) for v in inputs.site_rows["cluster"]} - {""})
    else:
        operators = [normalise_operator(args.operator)]
    for operator in operators:
        decision = replace(
            DecisionParams(),
            level=args.level,
            rsrp_threshold_dbm=args.rsrp_threshold,
            sinr_threshold_db=args.sinr_threshold,
            location_probability=args.location_probability,
            min_retention=args.min_retention,
            max_hole_m2=args.max_hole_m2,
            max_cumulative_coverage_loss=args.max_network_loss,
            max_removals=args.max_removals,
        )
        cfg = OverlapConfig(project_id=args.project_id, operator=operator, grid_resolution_m=args.grid, decision=decision)
        run = run_pipeline(cfg, inputs)
        out = write_outputs(run, run_dir(args.project_id, operator, args.level))
        print(f"\n[{operator}] rf_surface={run.summary['rf_surface_validation']} verdicts={run.summary['verdicts']} -> {out}")
        redundant = run.candidates[run.candidates["model_verdict"] == VERDICT_REDUNDANT]
        if not redundant.empty:
            cols = ["removal_rank", "candidate_id", "verdict", "retention", "largest_hole_m2", "footprint_points",
                    "absorber_sites", "confidence"]
            print(redundant[cols].to_string(index=False))


if __name__ == "__main__":
    main()
