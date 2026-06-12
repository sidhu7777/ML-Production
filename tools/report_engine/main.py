# src/main.py

import os
import uuid
from .threshold_resolver import resolve_kpi_ranges
from .load_data_db import load_project_data
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
from .kpi_analysis import run_kpi_analysis
from .metadata_generator import (build_metadata, write_metadata_file,)
from .cdf_kpi import generate_all_cdf_plots
from .llm_integration import generate_report_text
from .pdf_generator import generate_pdf_report
from .email_service import send_report_ready_email
from .db import get_user_by_id, init_engine, update_project_download_path

import shutil
import os

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
    


    print("Raw rows:", len(raw_df))
    print("Filtered rows:", len(filtered_df))

    if filtered_df.empty:
        print("No data after polygon filtering — stopping.")
        return
    
    # In production, remove temp files after successful report generation.
    # Keep during local debugging by setting REPORT_KEEP_TMP=1
    keep_tmp = os.getenv("REPORT_KEEP_TMP", "0") == "1"


    polygon_wkt = None
    if "region" in project_meta:
        polygon_wkt = project_meta["region"]
    
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
        generate_base_route_map(filtered_df, polygon_wkt, base_html_path)
        html_to_png(base_html_path, base_png_path)
        print("Generated base route map")
    except Exception as e:
        print(f" Warning: failed to generate base route map: {e}")

    # --------------------------------------------------
    # 2. KPI MAP GENERATION
    # --------------------------------------------------
    for kpi, cfg in KPI_CONFIG.items():
        col = cfg["column"]
        

        if col not in filtered_df.columns:
            continue

        # Filter rows with valid KPI values AND valid lat/lon
        df_kpi = filtered_df[
            filtered_df[col].notna() & 
            filtered_df["lat"].notna() & 
            filtered_df["lon"].notna()
        ]
        if df_kpi.empty:
            continue

        html_path = f"{html_dir}/{kpi}.html"
        png_path = f"{kpi_maps_dir}/{cfg['map_name']}"

        if cfg["type"] == "range":
            print("\n==============================================")
            print(f"PROCESSING KPI: {kpi}")
            print(f"DATA COLUMN  : {col}")
            print(f"Total filtered rows: {len(filtered_df)}")
            print(f"Rows with valid {col}: {len(df_kpi)}")
            print("==============================================")

            ranges = resolve_kpi_ranges(kpi_name=kpi, user_id=user_id,values=filtered_df[col])
            
            # Debug: Print ranges for troubleshooting
            print(f"Ranges for {kpi}:")
            for idx, r in enumerate(ranges):
                print(f"  [{idx}] {r['range']} | min={r['min']}, max={r['max']} | {r['source']}")

            if not has_valid_numeric_data(filtered_df, col):
                print(f"Skipping KPI {kpi} due to invalid numeric data.")
                continue
            
            generate_kpi_map(
                df=df_kpi,
                kpi_column=col,
                color_func=cfg["color_func"],
                ranges=ranges,
                output_html=html_path,
                polygon_wkt=polygon_wkt,
            )

        elif cfg["type"] == "categorical":
            if not has_valid_categorical_data(filtered_df, col):
                print(f"Skipping KPI {kpi} due to invalid categorical data.")
                continue

            generate_categorical_kpi_map(
                df=df_kpi,
                kpi_column=col,
                output_html=html_path,
                polygon_wkt=polygon_wkt,
            )

        html_to_png(html_path, png_path)
        print(f"Generated map for {kpi}")

    # --------------------------------------------------
    # 2.5 POOR REGION MAPS (PNG OUTPUT)
    # --------------------------------------------------
    print("\n==============================================")
    print("GENERATING POOR REGION MAPS")
    print("==============================================")
    generate_poor_region_maps(
        filtered_df,
        output_dir=f"{images_dir}/kpi_maps",
        tmp_dir=html_dir,
        polygon_wkt=polygon_wkt
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
        # Use stable run length to avoid noisy events (use 10 as tuned in test)
        events = detect_handover_events(filtered_df, use_global_detection=True, min_run_length=10)
        generate_handover_map(filtered_df, events, handover_html, polygon_wkt=polygon_wkt)
        html_to_png(handover_html, handover_png)
        print("Generated handover map")
    except Exception as e:
        print(f" Warning: failed to generate handover map: {e}")

    # --------------------------------------------------
    # 3. SAVE FILTERED DATA
    # --------------------------------------------------
    filtered_df.to_csv(
        f"{processed_dir}/filtered_data.csv",
        index=False,
    )

    # --------------------------------------------------
    # 3.5 GENERATE CDF DISTRIBUTION GRAPHS FROM API
    # --------------------------------------------------
    # Parse session IDs from project metadata
    ref_session_id = project_meta.get("ref_session_id", "")
    session_ids = [
        int(s.strip())
        for s in str(ref_session_id).split(",")
        if s.strip().isdigit()
    ]
    
    if session_ids:
        generate_all_cdf_plots(
            session_ids=session_ids,
            output_dir=f"{images_dir}/kpi_analysis"
        )
    else:
        print(" Warning: No session IDs found, skipping CDF generation")

    # --------------------------------------------------
    # 4. KPI ANALYSIS + METADATA
    # --------------------------------------------------
    kpi_metadata, drive_summary_metadata = run_kpi_analysis(
        filtered_df, 
        user_id, 
        KPI_CONFIG,
        session_ids=session_ids,
        image_dir=f"{images_dir}/kpi_analysis"
    )
    metadata = build_metadata(
        filtered_df,
        kpi_analysis_results=kpi_metadata,
        drive_summary_data=drive_summary_metadata,
    )
    write_metadata_file(
        metadata,
        f"{processed_dir}/report_metadata.json",
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
        verbose=True
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
    
    



