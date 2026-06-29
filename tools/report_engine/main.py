# src/main.py

import os
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from .threshold_resolver import resolve_kpi_ranges
from .load_data_db import (
    load_project_data,
    filter_known_band_rows,
    polygon_filter_all_cells,
)
from .kpi_config import KPI_CONFIG
from .map_generator import (
    generate_kpi_map,
    generate_categorical_kpi_map,
    generate_poor_region_maps,
    generate_base_route_map,
    has_valid_numeric_data,
    has_valid_categorical_data,
)
from .map_generator import generate_handover_map, detect_handover_events
from .playwright_utils import html_to_png
from .kpi_analysis import run_kpi_analysis, build_native_table_data
from .metadata_generator import (build_metadata, write_metadata_file,)
from .cdf_kpi import generate_all_cdf_plots_from_df
from .llm_integration import generate_report_text
from .pdf_generator import generate_pdf_report
from .email_service import send_report_ready_email
from .db import get_user_by_id, init_engine, update_project_download_path

import shutil
import os

# Report map render settings.  1200x900 @ scale 1 keeps the route filling the
# frame and the PNGs small (the old 1920x1200 @ scale 2 produced ~4 MB maps and
# bloated the PDF to 40-60 MB).
REPORT_RENDER_WIDTH = 1200
REPORT_RENDER_HEIGHT = 900
REPORT_DEVICE_SCALE = 1
REPORT_MAP_WORKERS = 4

def clean_directory(path):
    if os.path.exists(path):
        for f in os.listdir(path):
            fp = os.path.join(path, f)
            if os.path.isfile(fp):
                os.remove(fp)
            else:
                shutil.rmtree(fp)


DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data")
REPORTS_DIR = os.path.join(DATA_DIR, "reports")


def main(
    project_id: int,
    user_id: int | None = None,
    report_id: str | None = None,
    db_engine=None
):
    if db_engine is not None:
        init_engine(db_engine)
    report_id = report_id or str(uuid.uuid4())

    report_tmp_dir = f"{DATA_DIR}/tmp/{report_id}"
    html_dir = f"{report_tmp_dir}/html"
    images_dir = f"{report_tmp_dir}/images"
    kpi_maps_dir = f"{images_dir}/kpi_maps"
    processed_dir = f"{report_tmp_dir}/processed"
    report_out_dir = f"{REPORTS_DIR}/{report_id}"

    os.makedirs(html_dir, exist_ok=True)
    os.makedirs(kpi_maps_dir, exist_ok=True)
    os.makedirs(processed_dir, exist_ok=True)
    os.makedirs(report_out_dir, exist_ok=True)
    # --------------------------------------------------
    # 1. LOAD DATA FROM DB (RAW + FILTERED)
    # --------------------------------------------------
    raw_df, filtered_df, project_meta = load_project_data(project_id)

    polygon_wkt = project_meta.get("region")

    print("Raw rows:", len(raw_df))
    print("Filtered rows (primary + polygon):", len(filtered_df))

    if filtered_df.empty:
        print("No data after polygon filtering — stopping.")
        return

    # report_df = primary + polygon + KNOWN band.  Every report visual (maps,
    # KPI analysis, CDF, metadata) is computed over this single population so
    # band %, table and LLM text all agree.
    report_df = filter_known_band_rows(filtered_df)
    print("Report rows (primary + polygon + known band):", len(report_df))
    if report_df.empty:
        print("No rows with a known band after filtering — stopping.")
        return

    # handover_df = polygon-filtered ALL cells (no primary-cell restriction) so
    # the handover count matches the frontend's GetNetworkLog backend.
    handover_df = polygon_filter_all_cells(raw_df, polygon_wkt)
    print("Handover rows (all cells, polygon):", len(handover_df))

    # In production, remove temp files after successful report generation.
    # Keep during local debugging by setting REPORT_KEEP_TMP=1
    keep_tmp = os.getenv("REPORT_KEEP_TMP", "0") == "1"
    
    # Debug: Check if polygon is loaded
    print("\n==============================================")
    if polygon_wkt:
        print(f" POLYGON LOADED: {polygon_wkt[:100]}...")
    else:
        print(" WARNING: No polygon boundary found!")
    print("==============================================\n")

    # --------------------------------------------------
    # 2.0 BASE ROUTE MAP (DRIVE ROUTE + POLYGON) - create before KPI maps
    # --------------------------------------------------
    print("\n==============================================")
    print("GENERATING BASE ROUTE MAP")
    print("==============================================")
    base_html_path = f"{html_dir}/base_route.html"
    base_png_path = f"{kpi_maps_dir}/base_route_map.png"
    try:
        generate_base_route_map(report_df, polygon_wkt, base_html_path)
        html_to_png(
            base_html_path, base_png_path,
            width=REPORT_RENDER_WIDTH, height=REPORT_RENDER_HEIGHT,
            device_scale_factor=REPORT_DEVICE_SCALE,
        )
        print("Generated base route map")
    except Exception as e:
        print(f" Warning: failed to generate base route map: {e}")

    # --------------------------------------------------
    # 2. KPI MAP GENERATION (parallel — one worker per KPI)
    # --------------------------------------------------
    def _render_kpi_map(kpi, cfg):
        col = cfg["column"]
        if col not in report_df.columns:
            return kpi, False, "column missing"

        df_kpi = report_df[
            report_df[col].notna()
            & report_df["lat"].notna()
            & report_df["lon"].notna()
        ]
        if df_kpi.empty:
            return kpi, False, "no valid data"

        html_path = f"{html_dir}/{kpi}.html"
        png_path = f"{kpi_maps_dir}/{cfg['map_name']}"

        try:
            if cfg["type"] == "range":
                if not has_valid_numeric_data(report_df, col):
                    return kpi, False, "invalid numeric data"
                ranges = resolve_kpi_ranges(kpi_name=kpi, user_id=user_id, values=report_df[col])
                generate_kpi_map(
                    df=df_kpi, kpi_column=col, color_func=cfg["color_func"],
                    ranges=ranges, output_html=html_path, polygon_wkt=polygon_wkt,
                )
            elif cfg["type"] == "categorical":
                if not has_valid_categorical_data(report_df, col):
                    return kpi, False, "invalid categorical data"
                generate_categorical_kpi_map(
                    df=df_kpi, kpi_column=col, output_html=html_path, polygon_wkt=polygon_wkt,
                )
            else:
                return kpi, False, f"unknown type {cfg['type']}"

            html_to_png(
                html_path, png_path,
                width=REPORT_RENDER_WIDTH, height=REPORT_RENDER_HEIGHT,
                device_scale_factor=REPORT_DEVICE_SCALE,
            )
            return kpi, True, None
        except Exception as exc:
            return kpi, False, str(exc)

    with ThreadPoolExecutor(max_workers=REPORT_MAP_WORKERS) as pool:
        futures = {pool.submit(_render_kpi_map, kpi, cfg): kpi for kpi, cfg in KPI_CONFIG.items()}
        for fut in as_completed(futures):
            kpi, ok, err = fut.result()
            print(f"Generated map for {kpi}" if ok else f" Skipped/failed {kpi}: {err}")

    # --------------------------------------------------
    # 2.5 POOR REGION MAPS (PNG OUTPUT)
    # --------------------------------------------------
    print("\n==============================================")
    print("GENERATING POOR REGION MAPS")
    print("==============================================")
    generate_poor_region_maps(
        report_df,
        output_dir=f"{images_dir}/kpi_maps",
        tmp_dir=html_dir,
        polygon_wkt=polygon_wkt,
        render_width=REPORT_RENDER_WIDTH,
        render_height=REPORT_RENDER_HEIGHT,
        device_scale_factor=REPORT_DEVICE_SCALE,
    )

    # --------------------------------------------------
    # HANDOVER MAP
    # --------------------------------------------------
    print("\n==============================================")
    print("GENERATING HANDOVER MAP")
    print("==============================================")
    try:
        handover_html = f"{html_dir}/handover_map.html"
        handover_png = f"{kpi_maps_dir}/handover_map.png"
        # Handover uses all-cells polygon data so the count matches the frontend.
        events = detect_handover_events(handover_df)
        print(f"Handover events selected for map: {len(events)}")
        generate_handover_map(handover_df, events, handover_html, polygon_wkt=polygon_wkt)
        html_to_png(
            handover_html, handover_png,
            width=REPORT_RENDER_WIDTH, height=REPORT_RENDER_HEIGHT,
            device_scale_factor=REPORT_DEVICE_SCALE,
        )
        print("Generated handover map")
    except Exception as e:
        print(f" Warning: failed to generate handover map: {e}")

    # --------------------------------------------------
    # 3. SAVE FILTERED DATA
    # --------------------------------------------------
    report_df.to_csv(
        f"{processed_dir}/filtered_data.csv",
        index=False,
    )

    # --------------------------------------------------
    # 3.5 GENERATE CDF DISTRIBUTION GRAPHS (from report dataframe)
    # --------------------------------------------------
    # Parse session IDs from project metadata
    ref_session_id = project_meta.get("ref_session_id", "")
    session_ids = [
        int(s.strip())
        for s in str(ref_session_id).split(",")
        if s.strip().isdigit()
    ]

    # CDFs are computed from the exact report dataframe (not the session API)
    # so every distribution matches the rest of the report.
    generate_all_cdf_plots_from_df(report_df, output_dir=f"{images_dir}/kpi_analysis")

    # --------------------------------------------------
    # 4. KPI ANALYSIS + METADATA
    # --------------------------------------------------
    kpi_metadata, drive_summary_metadata = run_kpi_analysis(
        report_df,
        user_id,
        KPI_CONFIG,
        session_ids=session_ids,
        image_dir=f"{images_dir}/kpi_analysis"
    )
    metadata = build_metadata(
        report_df,
        kpi_analysis_results=kpi_metadata,
        drive_summary_data=drive_summary_metadata,
    )
    write_metadata_file(
        metadata,
        f"{processed_dir}/report_metadata.json",
    )

    # Build native ReportLab table data (crisp, tiny) for the PDF.
    native_tables = build_native_table_data(
        report_df,
        metadata=metadata,
        band_summary=metadata.get("band_summary"),
        drive_summary=drive_summary_metadata,
    )

    # --------------------------------------------------
    # 5. GENERATE REPORT TEXT USING LLM
    # --------------------------------------------------
    print("\n==============================================")
    print("GENERATING REPORT TEXT WITH LLM")
    print("==============================================")
    
    report_text = generate_report_text(
        metadata=metadata,
        output_path=f"{processed_dir}/report_text.json",
        verbose=True
    )

    # --------------------------------------------------
    # 6. GENERATE PDF REPORT
    # --------------------------------------------------
    print("\n==============================================")
    print("GENERATING PDF REPORT")
    print("==============================================")
    
    pdf_path = generate_pdf_report(
        metadata_path=f"{processed_dir}/report_metadata.json",
        report_text_path=f"{processed_dir}/report_text.json",
        output_path=f"{report_out_dir}/report.pdf",
        images_dir=images_dir,
        verbose=True,
        native_tables=native_tables,
        scratch_dir=f"{report_tmp_dir}/_img_opt",
    )
    
    print(f"\nPDF Report generated: {pdf_path}")
    print("Pipeline completed successfully.")

    # Keep report download local. The frontend downloads through the Python API,
    # so the browser never needs direct S3 access.
    base_url = os.getenv("BASE_URL", "").rstrip("/")
    if base_url:
        download_link = f"{base_url}/api/report/download/{report_id}"
    else:
        download_link = f"/api/report/download/{report_id}"
    try:
        update_project_download_path(project_id, download_link)
        print(f"Updated tbl_project.Download_path: {download_link}")
    except Exception as e:
        print(f"Warning: failed to update Download_path: {e}")

    if user_id is not None:
        user_row = get_user_by_id(user_id)
        if user_row and user_row.get("email"):
            print(f"[Email] Sending report link to: {user_row.get('email')}")
            user_name = (
                user_row.get("name")
                or user_row.get("user_name")
                or user_row.get("username")
                or "User"
            )
            project_name = project_meta.get("project_name") or "Project"
            try:
                send_report_ready_email(
                    to_email=user_row["email"],
                    user_name=user_name,
                    project_name=project_name,
                    report_id=report_id,
                    download_url=download_link,
                )
                print("[Email] Sent successfully.")
            except Exception as e:
                print(f"[Email] Failed to send: {e}")
        else:
            print(f"Warning: No email found for user_id={user_id}, skipping email send.")
    else:
        print("Warning: user_id not provided, skipping email send.")     
     


    if not keep_tmp:
        clean_directory(report_tmp_dir)

    return kpi_metadata
    
    



