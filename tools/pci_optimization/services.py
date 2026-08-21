from __future__ import annotations

import datetime as dt
import json
import threading
import uuid
from pathlib import Path
from typing import Any

import pandas as pd

from . import db
from .engine import run_pci_optimization, severity_label
from .export import build_reason, export_pci_optimization_excel


ML_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_ROOT = ML_ROOT / "outputs" / "pci_optimization"

JOBS: dict[str, dict[str, Any]] = {}


def _build_result_rows(result: dict[str, Any], project_id: int, job_id: str, region: str, operator: str) -> pd.DataFrame:
    """Flattens the engine's conflict/recommendation output into one row
    per (site, rule) finding, matching schema.RESULT_COLUMNS. Every row
    is real -- no synthetic data is ever produced by run_pci_optimization."""
    findings: list[dict[str, Any]] = []
    for c in result["collision_conflicts"]:
        for site in c["sites"]:
            findings.append({"site_id": site, "current_pci": c["pci"], "earfcn": c["earfcn"], "rule_type": "Collision"})
    for c in result["confusion_conflicts"]:
        for site in c["neighbor_sites"]:
            findings.append({"site_id": site, "current_pci": c["pci"], "earfcn": c["earfcn"], "rule_type": "Confusion"})
    for mod_n, conflicts in result["mod_conflicts_by_n"].items():
        for c in conflicts:
            for site, pci in c["members"]:
                findings.append({"site_id": site, "current_pci": pci, "earfcn": c["earfcn"], "rule_type": f"Mod{mod_n}"})
    for c in result["grouped_conflicts"]:
        for site, pci in c["members"]:
            findings.append({"site_id": site, "current_pci": pci, "earfcn": c["earfcn"], "rule_type": "Grouped"})
    for g in result["co_centric_groups"]:
        for earfcn, pci in g["members"]:
            findings.append({"site_id": g["site"], "current_pci": pci, "earfcn": earfcn, "rule_type": "Co-centric"})

    if not findings:
        return pd.DataFrame()

    findings_df = pd.DataFrame(findings).drop_duplicates()

    sites_df = result["selected_sites_df"]
    sector_lookup = sites_df.set_index(["site_id_inferred", "site_pci", "site_earfcn"])[
        [c for c in ["site_cell_id_representative", "band", "site_lat", "site_lon"] if c in sites_df.columns]
    ].rename(columns={"site_cell_id_representative": "sector_id", "site_lat": "lat", "site_lon": "lon"})

    recs = result.get("recommendations")
    verification = result.get("verification") or {}
    infeasible_proof = json.dumps(verification.get("infeasible_rules", [])) if verification else None

    rows: list[dict[str, Any]] = []
    for finding in findings_df.itertuples(index=False):
        site_id, current_pci, earfcn, rule_type = finding.site_id, finding.current_pci, finding.earfcn, finding.rule_type
        lookup_key = (site_id, current_pci, earfcn)
        sector = sector_lookup.loc[lookup_key] if lookup_key in sector_lookup.index else None
        if isinstance(sector, pd.DataFrame):
            sector = sector.iloc[0]

        rec_row = None
        if recs is not None and not recs.empty:
            match = recs[(recs["site"] == site_id) & (recs["current_pci"] == current_pci) & (recs["earfcn"] == earfcn)]
            if not match.empty:
                rec_row = match.iloc[0]

        # Per-row, dynamic, per-RULE outcome -- NOT the job-level
        # stopped_reason/verified_clean fields (those describe the whole
        # combined run), and NOT the sector's combined after_cost either
        # (a sector touched by both Collision and Mod3 can genuinely clear
        # Collision while Mod3 stays capacity-limited on that same
        # sector -- judging by the combined total would call the resolved
        # rule unresolved too). Same build_reason logic the Excel export
        # uses, so DB and Excel never disagree about what happened.
        if rec_row is not None:
            rec_dict = rec_row.to_dict()
            rule_cost = (rec_dict.get("rule_costs") or {}).get(rule_type, rec_dict["after_cost"])
            resolved, reason = build_reason(
                rule_type, rule_cost, rec_dict["current_pci"], rec_dict["suggested_pci"], rec_dict["earfcn"],
                verification.get("infeasible_rules", []),
            )
        else:
            resolved, reason = None, (
                "Not evaluated: flagged by detection but the optimizer did not produce a recommendation "
                "for this exact (site, PCI, EARFCN) combination."
            )

        rows.append(
            {
                "project_id": project_id,
                "job_id": job_id,
                "region": region,
                "operator": operator,
                "site_id": str(site_id),
                "sector_id": str(sector.get("sector_id")) if sector is not None and pd.notna(sector.get("sector_id")) else "",
                "earfcn": str(int(earfcn)) if pd.notna(earfcn) else "",
                "band": str(sector.get("band")) if sector is not None and pd.notna(sector.get("band")) else None,
                "lat": float(sector.get("lat")) if sector is not None and pd.notna(sector.get("lat")) else None,
                "lon": float(sector.get("lon")) if sector is not None and pd.notna(sector.get("lon")) else None,
                "rule_type": rule_type,
                "severity": severity_label({rule_type}),
                "current_pci": int(current_pci),
                "suggested_pci": int(rec_row["suggested_pci"]) if rec_row is not None else None,
                "changed_flag": bool(rec_row["current_pci"] != rec_row["suggested_pci"]) if rec_row is not None else None,
                "resolved": resolved,
                "reason": reason,
                "conflict_cost_before": float(rec_row["before_cost"]) if rec_row is not None else None,
                "conflict_cost_after": float(rec_row["after_cost"]) if rec_row is not None else None,
                "same_earfcn_sectors_considered": int(rec_row["num_same_earfcn_sectors"]) if rec_row is not None else None,
                "neighbor_distance_m": result.get("neighbor_distance_m"),
                "iterations": verification.get("iterations"),
                "converged": verification.get("converged"),
                "verified_clean": verification.get("verified_clean"),
                "remaining_conflicts": verification.get("remaining_conflicts"),
                "stopped_reason": verification.get("stopped_reason"),
                "infeasible_proof": infeasible_proof,
            }
        )
    return pd.DataFrame(rows)


class PciOptimizationService:
    def submit(self, cfg: dict[str, Any]) -> dict[str, Any]:
        job_id = str(uuid.uuid4())
        JOBS[job_id] = {
            "job_id": job_id,
            "status": "queued",
            "progress": "queued",
            "created_at": dt.datetime.utcnow().isoformat() + "Z",
            "updated_at": dt.datetime.utcnow().isoformat() + "Z",
            "config": cfg,
        }
        thread = threading.Thread(target=self._run_job, args=(job_id, dict(cfg)), daemon=True)
        thread.start()
        return {"job_id": job_id, "status": "queued"}

    def get(self, job_id: str) -> dict[str, Any]:
        return JOBS.get(job_id, {"job_id": job_id, "status": "not_found"})

    def _update(self, job_id: str, status: str, progress: str, **extra: Any) -> None:
        job = JOBS.setdefault(job_id, {"job_id": job_id})
        job.update(extra)
        job["status"] = status
        job["progress"] = progress
        job["updated_at"] = dt.datetime.utcnow().isoformat() + "Z"
        print(f"[PCI_OPTIMIZATION][{job_id[:8]}] {status}: {progress}", flush=True)

    def _run_job(self, job_id: str, cfg: dict[str, Any]) -> None:
        started = dt.datetime.utcnow()
        try:
            self._update(job_id, "running", "Fetching real site/sector data")
            result = run_pci_optimization(cfg)

            project_id = result["project_id"]
            region = result["region"]
            operator = result["operator"]

            self._update(job_id, "running", "Saving PCI optimization results")
            result_df = _build_result_rows(result, project_id, job_id, region, operator)
            saved_rows = db.save_results(result_df, region=region)

            self._update(job_id, "running", "Exporting Excel report")
            # Job-scoped, not project-scoped -- same reason tilt recommendation
            # writes to outputs/temp_<job_id>/: a fixed, reused-per-project
            # path collides with itself if the user still has last run's file
            # open in Excel (confirmed: Windows PermissionError on overwrite).
            # A fresh folder per run means every run's file is untouched by
            # whatever the user is doing with a previous one.
            output_dir = OUTPUT_ROOT / f"project_{project_id}" / f"job_{job_id}"
            excel_path = None
            excel_error = None
            try:
                excel_path = export_pci_optimization_excel(result, output_dir)
            except OSError as exc:
                # DB results are already saved by this point -- a file-write
                # problem (locked file, permissions, disk full) shouldn't
                # fail the whole job when the actual optimization result is
                # already safely persisted. Surface it, don't hide it.
                excel_error = str(exc)
                print(f"[PCI_OPTIMIZATION][{job_id[:8]}] Excel export failed (non-fatal): {excel_error}", flush=True)

            runtime_sec = (dt.datetime.utcnow() - started).total_seconds()
            verification = result.get("verification") or {}
            self._update(
                job_id,
                "completed",
                "PCI optimization completed",
                project_id=project_id,
                region=region,
                operator=operator,
                site_count=result["site_count"],
                sector_count=result["sector_count"],
                neighbor_edge_count=result["graph_edge_count"],
                collision_count=len(result["collision_conflicts"]),
                confusion_count=len(result["confusion_conflicts"]),
                mod_conflict_count=sum(len(v) for v in result["mod_conflicts_by_n"].values()),
                grouped_conflict_count=len(result["grouped_conflicts"]),
                co_centric_count=len(result["co_centric_groups"]),
                order_conflict_counts=result["order_conflict_counts"],
                recommendation_count=int(len(result["recommendations"])),
                verified_clean=verification.get("verified_clean"),
                iterations=verification.get("iterations"),
                stopped_reason=verification.get("stopped_reason"),
                infeasible_rules=verification.get("infeasible_rules"),
                saved_rows=saved_rows,
                excel_path=str(excel_path) if excel_path else None,
                excel_error=excel_error,
                runtime_sec=round(runtime_sec, 3),
            )
        except Exception as exc:
            self._update(job_id, "error", str(exc), error=str(exc))
