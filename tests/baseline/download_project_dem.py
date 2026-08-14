cd m __future__ import annotations

import argparse
from pathlib import Path

from tools.lte_prediction.dem_utils import ensure_project_dem
from tools.lte_prediction.ml_engine import fetch_site_data


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-id", type=int, required=True)
    parser.add_argument("--region", type=str, default="india")
    parser.add_argument("--output-path", type=str, default=None)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    site_df, _ = fetch_site_data(args.project_id, region=args.region)
    output_path = Path(args.output_path) if args.output_path else PROJECT_ROOT / "data" / "dem" / f"project_{args.project_id}_dem.tif"
    final_path = ensure_project_dem(
        project_id=args.project_id,
        region=args.region,
        site_df=site_df,
        output_path=output_path,
        force=bool(args.force),
    )
    print(final_path)


if __name__ == "__main__":
    main()
