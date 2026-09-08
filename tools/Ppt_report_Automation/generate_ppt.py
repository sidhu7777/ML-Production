"""
PPT Generator CLI Tool
Reads configuration and executes PowerPoint presentation updates.
Usage:
    python generate_ppt.py --config sample_config.json
"""

import argparse
import json
import os
import sys
import io

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

from ppt_automation import PPTAutomator


def run_automation_from_config(config_path):
    if not os.path.exists(config_path):
        print(f"Error: Config file not found: {config_path}")
        sys.exit(1)

    with open(config_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    base_dir = os.path.dirname(os.path.abspath(config_path))

    template_path = cfg.get("template_path", "Mobility DT-美濃區(After)-20260729.pptx")
    if not os.path.isabs(template_path):
        template_path = os.path.join(base_dir, template_path)

    output_path = cfg.get("output_path", "Mobility_DT_Output.pptx")
    if not os.path.isabs(output_path):
        output_path = os.path.join(base_dir, output_path)

    print(f"Loading template presentation: {template_path}")
    automator = PPTAutomator(template_path)

    # 1. Update Cover Metadata
    cover_meta = cfg.get("cover_metadata", {})
    if cover_meta:
        print("Updating Cover metadata...")
        automator.update_cover_metadata(
            test_type=cover_meta.get("test_type"),
            region=cover_meta.get("region"),
            area=cover_meta.get("area"),
            completion_date=cover_meta.get("completion_date"),
            total_km=cover_meta.get("total_km"),
        )

    # 2. Update Image Replacements
    image_replacements = cfg.get("image_replacements", [])
    for img_info in image_replacements:
        slide = img_info.get("slide")
        target = img_info.get("target", "main")
        img_path = img_info.get("image_path")
        if img_path:
            if not os.path.isabs(img_path):
                img_path = os.path.join(base_dir, img_path)

            if os.path.exists(img_path):
                print(f"Replacing image on slide {slide} ({target}): {img_path}")
                try:
                    automator.replace_slide_image(slide, img_path, target_type=target)
                except Exception as e:
                    print(f"  Warning: Failed to replace image on slide {slide}: {e}")
            else:
                print(f"  Skipping image replacement on slide {slide} (file not found: {img_path})")

    # 3. Update Table Values
    table_updates = cfg.get("table_updates", [])
    for tbl_info in table_updates:
        slide = tbl_info.get("slide")
        table_idx = tbl_info.get("table_index", 0)
        row_key = tbl_info.get("row_key")
        new_val = tbl_info.get("new_value")
        val_col = tbl_info.get("value_col_idx", 2)
        if slide and row_key and new_val is not None:
            print(f"Updating table on slide {slide} key '{row_key}' -> '{new_val}'")
            automator.update_table_row_by_key(slide, table_idx, row_key, new_val, value_col_idx=val_col)

    # 4. Save presentation
    print(f"Saving output presentation to: {output_path}")
    automator.save(output_path)
    print("PPT Generation complete!")


def main():
    parser = argparse.ArgumentParser(description="Generate Mobility DT PowerPoint Report")
    parser.add_argument(
        "--config",
        default="sample_config.json",
        help="Path to JSON config file (default: sample_config.json)",
    )
    args = parser.parse_args()
    run_automation_from_config(args.config)


if __name__ == "__main__":
    main()
