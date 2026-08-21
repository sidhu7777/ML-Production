from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
from openpyxl import Workbook
from openpyxl.styles import Font, PatternFill
from openpyxl.utils import get_column_letter

from .engine import severity_label


def build_reason(
    rule_label: str, rule_cost: float, current_pci: int, suggested_pci: int, earfcn: Any, infeasible_rules: list[dict[str, Any]]
) -> tuple[bool, str]:
    """Per-RULE outcome, not per-sector. A sector touched by two rules at
    once (e.g. Collision AND Mod3 on the same physical sector) can clear
    one while the other stays capacity-limited -- judging by the combined
    cost would wrongly call the resolved side unresolved too, just
    because a different, co-occurring rule couldn't be satisfied. Always
    a genuine reason, backed by the same pigeonhole proof the closed-loop
    optimizer itself used wherever it applies, never a silent blank."""
    resolved = rule_cost == 0

    if resolved:
        if current_pci != suggested_pci:
            return True, f"Resolved: PCI changed {current_pci} -> {suggested_pci} to clear {rule_label}."
        return True, f"Already optimal for {rule_label} -- no PCI change needed."

    matches = [p for p in infeasible_rules if p["rule"] == rule_label and p["earfcn"] == earfcn]
    if matches:
        p = matches[0]
        return False, (
            f"Cannot be resolved: {p['clique_size']} sectors are mutual neighbors on EARFCN {p['earfcn']}, "
            f"but {p['rule']} only offers {p['capacity']} distinct value(s) -- mathematically proven "
            "(pigeonhole principle), not a search limitation."
        )
    return False, f"Not resolved: no better PCI found for {rule_label} given the current neighbor state."


def build_export_rows(result: dict[str, Any]) -> pd.DataFrame:
    """One row per (sector, rule) actually touched by a checked rule --
    never the whole project's sector list, and never one row combining
    multiple rules' outcomes into a single misleading total. Every row
    states whether THAT rule was resolved on THAT sector, and what
    changed or exactly why it could not be -- same shape as the DB save
    (services.py._build_result_rows), so Excel and DB never disagree."""
    recs = result.get("recommendations")
    if recs is None or recs.empty:
        return pd.DataFrame()

    verification = result.get("verification") or {}
    infeasible_rules = verification.get("infeasible_rules", [])

    rows = []
    for _, r in recs.iterrows():
        row = r.to_dict()
        rule_costs = row.get("rule_costs") or {}
        for rule_label in sorted(row.get("conflict_types") or []):
            rule_cost = rule_costs.get(rule_label, row["after_cost"])
            resolved, reason = build_reason(rule_label, rule_cost, row["current_pci"], row["suggested_pci"], row["earfcn"], infeasible_rules)
            rows.append(
                {
                    "Site": row["site"],
                    "EARFCN": row["earfcn"],
                    "Rule": rule_label,
                    "Severity": severity_label({rule_label}),
                    "Current PCI": row["current_pci"],
                    "Suggested PCI": row["suggested_pci"],
                    "Changed": bool(row["current_pci"] != row["suggested_pci"]),
                    "Resolved": resolved,
                    "Reason": reason,
                    "Same-EARFCN Sectors Considered": row["num_same_earfcn_sectors"],
                }
            )
    df = pd.DataFrame(rows)
    return df.sort_values(by=["Resolved", "Severity"], ascending=[True, False]).reset_index(drop=True)


def export_pci_optimization_excel(result: dict[str, Any], output_dir: Path) -> Path | None:
    """Local Excel report, same tools/*-established pattern as tilt
    recommendation's RF_Optimization_Report.xlsx (ML/outputs/<feature>/...).
    Returns None (no file written) if nothing was touched -- an empty
    report isn't useful and shouldn't be produced."""
    export_df = build_export_rows(result)
    if export_df.empty:
        return None

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "PCI_Optimization_Report.xlsx"

    verification = result.get("verification") or {}
    summary_df = pd.DataFrame(
        [
            {"Metric": "Project ID", "Value": result.get("project_id")},
            {"Metric": "Region", "Value": result.get("region")},
            {"Metric": "Operator", "Value": result.get("operator")},
            {"Metric": "Neighbor distance (m)", "Value": result.get("neighbor_distance_m")},
            {"Metric": "Sites considered", "Value": result.get("site_count")},
            {"Metric": "Sectors considered", "Value": result.get("sector_count")},
            {"Metric": "Sectors touched by checked rules", "Value": len(export_df)},
            {"Metric": "Sectors resolved", "Value": int(export_df["Resolved"].sum())},
            {"Metric": "Sectors NOT resolved", "Value": int((~export_df["Resolved"]).sum())},
            {"Metric": "Closed-loop iterations", "Value": verification.get("iterations")},
            {"Metric": "Verified clean (0 remaining)", "Value": verification.get("verified_clean")},
            {"Metric": "Stopped reason", "Value": verification.get("stopped_reason")},
        ]
    )

    wb = Workbook()
    wb.remove(wb.active)

    header_fill = PatternFill(fill_type="solid", fgColor="D9EAF7")
    resolved_fill = PatternFill(fill_type="solid", fgColor="C6EFCE")
    not_resolved_fill = PatternFill(fill_type="solid", fgColor="FFC7CE")
    bold = Font(bold=True)

    def _write_sheet(name: str, df: pd.DataFrame, highlight_resolved: bool) -> None:
        ws = wb.create_sheet(name)
        ws.append(list(df.columns))
        for cell in ws[1]:
            cell.font = bold
            cell.fill = header_fill
        for _, row in df.iterrows():
            ws.append(list(row))
            if highlight_resolved:
                fill = resolved_fill if row["Resolved"] else not_resolved_fill
                for cell in ws[ws.max_row]:
                    cell.fill = fill
        ws.freeze_panes = "A2"
        for idx, col in enumerate(df.columns, start=1):
            longest = df[col].astype(str).str.len().max() if len(df) else len(col)
            ws.column_dimensions[get_column_letter(idx)].width = max(12, min(60, int(longest) + 2))

    _write_sheet("Summary", summary_df, highlight_resolved=False)
    _write_sheet("Recommendations", export_df, highlight_resolved=True)

    wb.save(output_path)
    return output_path
