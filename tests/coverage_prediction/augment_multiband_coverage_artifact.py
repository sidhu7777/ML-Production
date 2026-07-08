from __future__ import annotations

import argparse
import hashlib
import shutil
import tarfile
import sys
from pathlib import Path

import pandas as pd

ML_ROOT = Path(__file__).resolve().parents[2]
if str(ML_ROOT) not in sys.path:
    sys.path.insert(0, str(ML_ROOT))

from tools.lte_tilt_recommandation.cell_identity import (
    build_rf_identity,
    build_sector_identity,
    build_site_sector_band_identity,
)


BAND_POOL = [700, 850, 900, 2100, 2300]
BAND_RSRP_BIAS = {
    700: 0.45,
    850: 0.25,
    900: 0.15,
    2100: -0.05,
    2300: -0.15,
}


def _hash_order(value: str) -> str:
    return hashlib.sha1(value.encode("utf-8")).hexdigest()


def _choose_bands(base_band: int, key: str, count: int) -> list[int]:
    candidates = [band for band in BAND_POOL if int(band) != int(base_band)]
    ordered = sorted(candidates, key=lambda band: _hash_order(f"{key}|{band}"))
    return ordered[:count]


def _band_jitter(grid_key: str, band: int, scale: float = 0.45) -> float:
    digest = hashlib.sha1(f"{grid_key}|{band}".encode("utf-8")).hexdigest()
    raw = int(digest[:8], 16) / 0xFFFFFFFF
    return (raw - 0.5) * 2.0 * scale


def _retag_row(row: pd.Series, new_band: int, new_node_cell_id: str) -> pd.Series:
    out = row.copy()
    sector_value = row.get("sector")
    if pd.isna(sector_value) and "original_node_cell_id" in row.index:
        sector_value = str(row["original_node_cell_id"]).rsplit("_", 1)[-1]
    site_value = row.get("Site ID")
    base_cell = row.get("original_node_cell_id", row.get("Node_Cell_ID", new_node_cell_id))
    out["Node_Cell_ID"] = new_node_cell_id
    if "cell_id" in out.index:
        out["cell_id"] = new_node_cell_id
    if "original_node_cell_id" in out.index:
        out["original_node_cell_id"] = base_cell
    if "original_cell_id" in out.index:
        out["original_cell_id"] = base_cell
    if "band" in out.index:
        out["band"] = float(new_band)
    if "earfcn" in out.index:
        out["earfcn"] = float({700: 700, 850: 850, 900: 900, 1800: 1750, 2100: 2100, 2300: 2300}.get(int(new_band), int(new_band)))
    if "site_sector_band_key" in out.index:
        out["site_sector_band_key"] = build_site_sector_band_identity(site_value, sector_value, new_band)
    if "rf_identity_key" in out.index:
        out["rf_identity_key"] = build_rf_identity(site_value, new_node_cell_id, sector_value, new_band, fallback=new_node_cell_id)
    if "sector_identity_key" in out.index:
        out["sector_identity_key"] = build_sector_identity(site_value, new_node_cell_id, sector_value, fallback=new_node_cell_id)
    if "PCI" in out.index and pd.notna(out["PCI"]):
        out["PCI"] = float((int(out["PCI"]) + int(new_band)) % 504)
    return out


def _augment_sites(df: pd.DataFrame, dual_share: float, triple_share: float) -> tuple[pd.DataFrame, dict[str, int]]:
    if df.empty:
        return df.copy(), {"dual": 0, "triple": 0}

    work = df.copy()
    work["Node_Cell_ID"] = work["Node_Cell_ID"].astype(str)
    ordered = sorted(work["Node_Cell_ID"].dropna().unique().tolist(), key=_hash_order)
    triple_count = min(len(ordered), max(0, int(round(len(ordered) * triple_share))))
    dual_count = min(len(ordered) - triple_count, max(0, int(round(len(ordered) * dual_share))))
    triple_cells = ordered[:triple_count]
    dual_cells = ordered[triple_count : triple_count + dual_count]

    rows = [work]
    for cell in dual_cells:
        base_row = work.loc[work["Node_Cell_ID"] == cell].iloc[0]
        base_band = int(pd.to_numeric(base_row.get("band"), errors="coerce") or 1800)
        new_band = _choose_bands(base_band, f"{cell}|dual", 1)[0]
        new_row = _retag_row(base_row, new_band, f"{cell}__MB{new_band}")
        rows.append(pd.DataFrame([new_row]))

    for cell in triple_cells:
        base_row = work.loc[work["Node_Cell_ID"] == cell].iloc[0]
        base_band = int(pd.to_numeric(base_row.get("band"), errors="coerce") or 1800)
        new_bands = _choose_bands(base_band, f"{cell}|triple", 2)
        for idx, new_band in enumerate(new_bands, start=2):
            new_row = _retag_row(base_row, new_band, f"{cell}__MB{new_band}")
            rows.append(pd.DataFrame([new_row]))

    augmented = pd.concat(rows, ignore_index=True)
    return augmented, {"dual": int(len(dual_cells)), "triple": int(len(triple_cells))}


def _augment_prediction_grid(df: pd.DataFrame, dual_share: float, triple_share: float) -> tuple[pd.DataFrame, dict[str, int]]:
    if df.empty:
        return df.copy(), {"dual": 0, "triple": 0}

    work = df.copy()
    base_key_col = "original_node_cell_id" if "original_node_cell_id" in work.columns else ("Node_Cell_ID" if "Node_Cell_ID" in work.columns else None)
    if base_key_col is None:
        return work, {"dual": 0, "triple": 0}

    work[base_key_col] = work[base_key_col].astype(str)
    ordered = sorted(work[base_key_col].dropna().unique().tolist(), key=_hash_order)
    triple_count = min(len(ordered), max(0, int(round(len(ordered) * triple_share))))
    dual_count = min(len(ordered) - triple_count, max(0, int(round(len(ordered) * dual_share))))
    triple_cells = ordered[:triple_count]
    dual_cells = ordered[triple_count : triple_count + dual_count]

    rows = [work]
    selected = {cell: "triple" for cell in triple_cells}
    selected.update({cell: "dual" for cell in dual_cells})
    for cell, mode in selected.items():
        base_rows = work.loc[work[base_key_col] == cell]
        if base_rows.empty:
            continue
        base_row = base_rows.iloc[0]
        base_band = int(pd.to_numeric(base_row.get("band"), errors="coerce") or 1800)
        add_count = 2 if mode == "triple" else 1
        new_bands = _choose_bands(base_band, f"{cell}|{mode}", add_count)
        for new_band in new_bands:
            clone = base_rows.copy()
            new_node = f"{base_row.get('Node_Cell_ID', cell)}__MB{new_band}"
            clone["Node_Cell_ID"] = new_node
            if "cell_id" in clone.columns:
                clone["cell_id"] = new_node
            if "band" in clone.columns:
                clone["band"] = float(new_band)
            if "earfcn" in clone.columns:
                clone["earfcn"] = float({700: 700, 850: 850, 900: 900, 1800: 1750, 2100: 2100, 2300: 2300}.get(int(new_band), int(new_band)))
            if "site_sector_band_key" in clone.columns:
                clone["site_sector_band_key"] = clone.apply(
                    lambda r: build_site_sector_band_identity(r.get("Site ID"), r.get("sector"), new_band), axis=1
                )
            if "rf_identity_key" in clone.columns:
                clone["rf_identity_key"] = clone.apply(
                    lambda r: build_rf_identity(r.get("Site ID"), new_node, r.get("sector"), new_band, fallback=new_node), axis=1
                )
            if "sector_identity_key" in clone.columns:
                clone["sector_identity_key"] = clone.apply(
                    lambda r: build_sector_identity(r.get("Site ID"), new_node, r.get("sector"), fallback=new_node), axis=1
                )
            if "original_node_cell_id" in clone.columns:
                clone["original_node_cell_id"] = cell
            if "original_cell_id" in clone.columns:
                clone["original_cell_id"] = cell
            if "pred_rsrp" in clone.columns or "pred_rsrq" in clone.columns or "pred_sinr" in clone.columns:
                grid_keys = (
                    clone["grid_id"].astype(str)
                    if "grid_id" in clone.columns
                    else pd.Series(clone.index.astype(str), index=clone.index)
                )
                time_keys = clone["time_bucket"].astype(str) if "time_bucket" in clone.columns else pd.Series("NA", index=clone.index)
                offsets = [
                    BAND_RSRP_BIAS.get(int(new_band), 0.0) + _band_jitter(f"{cell}|{grid}|{tb}", int(new_band))
                    for grid, tb in zip(grid_keys, time_keys)
                ]
                offset_series = pd.Series(offsets, index=clone.index)
                if "pred_rsrp" in clone.columns:
                    clone["pred_rsrp"] = pd.to_numeric(clone["pred_rsrp"], errors="coerce") + offset_series
                if "pred_rsrp_geo" in clone.columns:
                    clone["pred_rsrp_geo"] = pd.to_numeric(clone["pred_rsrp_geo"], errors="coerce") + offset_series
                if "pred_rsrp_calibrated" in clone.columns:
                    clone["pred_rsrp_calibrated"] = pd.to_numeric(clone["pred_rsrp_calibrated"], errors="coerce") + offset_series
                if "pred_rsrp_demo" in clone.columns:
                    clone["pred_rsrp_demo"] = pd.to_numeric(clone["pred_rsrp_demo"], errors="coerce") + offset_series
                if "pred_rsrq" in clone.columns:
                    clone["pred_rsrq"] = pd.to_numeric(clone["pred_rsrq"], errors="coerce") + offset_series * 0.45
                if "pred_rsrq_geo" in clone.columns:
                    clone["pred_rsrq_geo"] = pd.to_numeric(clone["pred_rsrq_geo"], errors="coerce") + offset_series * 0.45
                if "pred_rsrq_calibrated" in clone.columns:
                    clone["pred_rsrq_calibrated"] = pd.to_numeric(clone["pred_rsrq_calibrated"], errors="coerce") + offset_series * 0.45
                if "pred_rsrq_demo" in clone.columns:
                    clone["pred_rsrq_demo"] = pd.to_numeric(clone["pred_rsrq_demo"], errors="coerce") + offset_series * 0.45
                if "pred_sinr" in clone.columns:
                    clone["pred_sinr"] = pd.to_numeric(clone["pred_sinr"], errors="coerce") + offset_series * 0.35
                if "pred_sinr_geo" in clone.columns:
                    clone["pred_sinr_geo"] = pd.to_numeric(clone["pred_sinr_geo"], errors="coerce") + offset_series * 0.35
                if "pred_sinr_calibrated" in clone.columns:
                    clone["pred_sinr_calibrated"] = pd.to_numeric(clone["pred_sinr_calibrated"], errors="coerce") + offset_series * 0.35
                if "pred_sinr_demo" in clone.columns:
                    clone["pred_sinr_demo"] = pd.to_numeric(clone["pred_sinr_demo"], errors="coerce") + offset_series * 0.35
            if "carrier_load_share" in clone.columns:
                clone["carrier_load_share"] = 1.0
            rows.append(clone)

    augmented = pd.concat(rows, ignore_index=True)
    return augmented, {"dual": int(len(dual_cells)), "triple": int(len(triple_cells))}


def _copy_tree(source_run_dir: Path, output_run_dir: Path) -> None:
    if output_run_dir.exists():
        shutil.rmtree(output_run_dir)
    shutil.copytree(source_run_dir, output_run_dir)


def _repack_archive(run_dir: Path, archive_path: Path) -> None:
    if archive_path.exists():
        archive_path.unlink()
    with tarfile.open(archive_path, "w") as tf:
        for path in sorted(run_dir.rglob("*")):
            if path.is_file():
                tf.add(path, arcname=str(path.relative_to(run_dir.parent)))


def main() -> None:
    parser = argparse.ArgumentParser(description="Augment an existing coverage run with synthetic triple-band sectors.")
    parser.add_argument("--source-run-dir", required=True)
    parser.add_argument("--output-run-dir", required=True)
    parser.add_argument("--archive-out", required=True)
    parser.add_argument("--dual-share", type=float, default=0.04)
    parser.add_argument("--triple-share", type=float, default=0.01)
    args = parser.parse_args()

    source_run_dir = Path(args.source_run_dir)
    output_run_dir = Path(args.output_run_dir)
    archive_out = Path(args.archive_out)

    _copy_tree(source_run_dir, output_run_dir)

    sites_path = output_run_dir / "project_sites.csv"
    baseline_path = output_run_dir / "baseline_prediction_grid.csv"
    corrected_path = output_run_dir / "bucket_corrected_prediction_grid.csv"

    sites_df = pd.read_csv(sites_path)
    baseline_df = pd.read_csv(baseline_path)
    corrected_df = pd.read_csv(corrected_path)

    sites_aug, sites_counts = _augment_sites(sites_df, args.dual_share, args.triple_share)
    baseline_aug, base_counts = _augment_prediction_grid(baseline_df, args.dual_share, args.triple_share)
    corrected_aug, corr_counts = _augment_prediction_grid(corrected_df, args.dual_share, args.triple_share)

    sites_aug.to_csv(sites_path, index=False)
    baseline_aug.to_csv(baseline_path, index=False)
    corrected_aug.to_csv(corrected_path, index=False)

    _repack_archive(output_run_dir, archive_out)

    summary = {
        "source_run_dir": str(source_run_dir),
        "output_run_dir": str(output_run_dir),
        "archive_out": str(archive_out),
        "sites_dual": sites_counts["dual"],
        "sites_triple": sites_counts["triple"],
        "baseline_dual": base_counts["dual"],
        "baseline_triple": base_counts["triple"],
        "corrected_dual": corr_counts["dual"],
        "corrected_triple": corr_counts["triple"],
    }
    print(summary)


if __name__ == "__main__":
    main()
