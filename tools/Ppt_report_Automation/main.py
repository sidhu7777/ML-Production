"""
Main Entrypoint for PowerPoint Report Generation
Generates Mobility DT report PPTX for a specified project_id, session_ids, and region/country_code (e.g. Taiwan).

Usage:
    python main.py --project-id 210 --country-code taiwan
"""

import sys
import io
import os
import argparse
from dotenv import load_dotenv

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace', line_buffering=True)

# Load local .env file
LOCAL_ENV_PATH = os.path.join(os.path.dirname(__file__), ".env")
if os.path.exists(LOCAL_ENV_PATH):
    load_dotenv(LOCAL_ENV_PATH)

from report_ppt_generator import generate_ppt_for_project


def main():
    parser = argparse.ArgumentParser(description="Generate Mobility DT PowerPoint Report from Database Project ID and Session ID")
    parser.add_argument(
        "--project-id",
        type=int,
        required=False,
        help="Project ID to load data from DB (e.g. 210)",
    )
    parser.add_argument(
        "--session-ids",
        type=str,
        default=None,
        help="Comma-separated session IDs (optional)",
    )
    parser.add_argument(
        "--country-code",
        type=str,
        default=None,
        help="Country code / region database (e.g. taiwan, india, default)",
    )
    parser.add_argument(
        "--region",
        type=str,
        default=None,
        help="Region name override (e.g. taiwan)",
    )
    parser.add_argument(
        "--template",
        type=str,
        default=None,
        help="Path to PPTX template file (default: Mobility DT-美濃區(After)-20260729.pptx)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path for generated PPTX output file",
    )
    parser.add_argument(
        "--locked-bands",
        type=str,
        default=None,
        help="Comma-separated LTE bands locked during test (e.g. '1800,2100', 'L1800,L2600', 'B3,B1', 'all')",
    )
    args = parser.parse_args()

    if args.project_id is None:
        print("[Notice] No --project-id passed. Run with --project-id 210 --country-code taiwan")
        print("Executing demonstration run using sample template...")
        curr_dir = os.path.dirname(os.path.abspath(__file__))
        template_path = os.path.join(curr_dir, "Mobility DT-美濃區(After)-20260729.pptx")
        output_path = os.path.join(curr_dir, "Mobility_DT_Automated_Output.pptx")
        from ppt_automation import PPTAutomator
        automator = PPTAutomator(template_path)
        automator.update_cover_metadata(
            test_type="Mobility Test",
            region="SEO",
            area="美濃區",
            completion_date="2026/08/24",
            total_km="40.43 km"
        )
        automator.save(output_path)
        print(f"Sample presentation generated successfully at: {output_path}")
        return

    country = args.country_code or args.region
    print(f"Starting PPT generation for Project ID={args.project_id}, Sessions={args.session_ids}, Country/Region={country}...")
    try:
        out_file = generate_ppt_for_project(
            project_id=args.project_id,
            session_ids=args.session_ids,
            country_code=country,
            region=args.region or country,
            template_path=args.template,
            output_path=args.output,
            locked_bands=args.locked_bands,
        )
        print(f"\n[Success] Generated PowerPoint presentation at: {out_file}")
    except Exception as e:
        print(f"\n[Error] PPT generation failed for project_id={args.project_id}: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
