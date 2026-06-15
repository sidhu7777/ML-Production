import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.report_engine.llm_integration import generate_report_text, parse_and_validate_llm_output
from tools.report_engine.pdf_generator import PDFReportGenerator


def _write_json(path: Path, payload):
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str), encoding="utf-8")


def _read_metadata(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _find_latest_metadata_artifact(base_dir: Path) -> Path:
    candidates = sorted(base_dir.glob("report_engine_project_196_*/processed/report_metadata.json"))
    if not candidates:
        raise FileNotFoundError(f"No report_metadata.json found under {base_dir}")
    return candidates[-1]


def _build_pdf_content_preview(report_text: dict, images_dir: Path) -> list[dict]:
    def _entries(section_title: str, text_key: str | None, image_specs: list[tuple[str, str]]):
        text = report_text.get(text_key, "") if text_key else ""
        images = []
        for subdir, filename in image_specs:
            path = images_dir / subdir / filename
            images.append(
                {
                    "file": str(path),
                    "exists": path.exists(),
                }
            )
        return {
            "section": section_title,
            "text": text,
            "images": images,
        }

    preview = [
        _entries("1. Introduction", "Introduction", []),
        _entries("2. Area Summary", None, [("kpi_maps", "base_route_map.png")]),
        _entries("3. Drive Summary", "Drive Summary", [("kpi_analysis", "drive_summary.png")]),
        _entries(
            "4. KPI Summary",
            "KPI Summary",
            [("kpi_analysis", "kpi_summary.png"), ("kpi_analysis", "session_table.png")],
        ),
        _entries(
            "5.a Band",
            "Map View - Band",
            [
                ("kpi_maps", "band_map.png"),
                ("kpi_analysis", "band_pie.png"),
                ("kpi_analysis", "band_table.png"),
            ],
        ),
        _entries(
            "5.b RSRP",
            "Map View - RSRP",
            [
                ("kpi_maps", "rsrp_map.png"),
                ("kpi_maps", "rsrp_poor_regions.png"),
                ("kpi_analysis", "rsrp_range_table.png"),
                ("kpi_analysis", "cdf_rsrp.png"),
            ],
        ),
        _entries(
            "5.c RSRQ",
            "Map View - RSRQ",
            [
                ("kpi_maps", "rsrq_map.png"),
                ("kpi_maps", "rsrq_poor_regions.png"),
                ("kpi_analysis", "rsrq_range_table.png"),
                ("kpi_analysis", "cdf_rsrq.png"),
            ],
        ),
        _entries(
            "5.d SINR",
            "Map View - SINR",
            [
                ("kpi_maps", "sinr_map.png"),
                ("kpi_analysis", "sinr_range_table.png"),
                ("kpi_analysis", "cdf_sinr.png"),
            ],
        ),
        _entries(
            "5.e DL Throughput",
            "Map View - DL Throughput",
            [
                ("kpi_maps", "dl_map.png"),
                ("kpi_analysis", "dl_range_table.png"),
                ("kpi_analysis", "cdf_dl_tpt.png"),
            ],
        ),
        _entries(
            "5.f UL Throughput",
            "Map View - UL Throughput",
            [
                ("kpi_maps", "ul_map.png"),
                ("kpi_analysis", "ul_range_table.png"),
                ("kpi_analysis", "cdf_ul_tpt.png"),
            ],
        ),
        _entries(
            "5.g MOS",
            "Map View - MOS",
            [
                ("kpi_maps", "mos_map.png"),
                ("kpi_analysis", "mos_range_table.png"),
                ("kpi_analysis", "cdf_mos.png"),
            ],
        ),
        _entries(
            "6. PCI Summary",
            "PCI Summary",
            [
                ("kpi_maps", "pci_map.png"),
                ("kpi_analysis", "pci_distribution.png"),
                ("kpi_analysis", "cdf_pci.png"),
                ("kpi_analysis", "pci_table.png"),
                ("kpi_analysis", "pci_poor_rsrp.png"),
                ("kpi_analysis", "pci_poor_rsrq.png"),
            ],
        ),
    ]
    return preview


def _write_pdf_content_preview_markdown(path: Path, preview: list[dict]):
    lines = ["# PDF Content Preview", ""]
    for section in preview:
        lines.append(f"## {section['section']}")
        lines.append("")
        lines.append("Text:")
        lines.append(section["text"] or "(none)")
        lines.append("")
        lines.append("Artifacts:")
        for item in section["images"]:
            status = "exists" if item["exists"] else "missing"
            lines.append(f"- {status}: {item['file']}")
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="Generate real report-text debug artifacts from saved metadata.")
    parser.add_argument(
        "--metadata-path",
        help="Path to an existing processed/report_metadata.json. Defaults to latest project 196 debug artifact.",
    )
    parser.add_argument(
        "--artifacts-dir",
        default="tests/output/report_engine_content_debug",
        help="Directory to write generated report-text artifacts.",
    )
    parser.add_argument(
        "--skip-llm",
        action="store_true",
        help="Only generate rule-based/parse fallback outputs; skip live LLM call.",
    )
    args = parser.parse_args()

    artifacts_dir = Path(args.artifacts_dir)
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    metadata_path = Path(args.metadata_path) if args.metadata_path else _find_latest_metadata_artifact(
        Path("tests/output/report_engine_real")
    )
    metadata = _read_metadata(metadata_path)
    images_dir = metadata_path.parents[1] / "images"
    _write_json(artifacts_dir / "input_metadata.json", metadata)

    fallback_from_invalid = parse_and_validate_llm_output(
        raw_text="not-json-response",
        metadata=metadata,
        output_path=str(artifacts_dir / "report_text_parse_fallback.json"),
    )
    _write_json(artifacts_dir / "report_text_parse_fallback_copy.json", fallback_from_invalid)

    partial_llm = parse_and_validate_llm_output(
        raw_text=json.dumps(
            {
                "Introduction": "Debug introduction placeholder.",
                "Area Summary": metadata.get("area_summary", {}),
            }
        ),
        metadata=metadata,
        output_path=str(artifacts_dir / "report_text_missing_sections.json"),
    )
    _write_json(artifacts_dir / "report_text_missing_sections_copy.json", partial_llm)

    if not args.skip_llm:
        llm_report = generate_report_text(
            metadata=metadata,
            output_path=str(artifacts_dir / "report_text_hybrid_live.json"),
            verbose=False,
        )
        _write_json(artifacts_dir / "report_text_hybrid_live_copy.json", llm_report)
        preview = _build_pdf_content_preview(llm_report, images_dir)
        _write_json(artifacts_dir / "pdf_content_preview.json", preview)
        _write_pdf_content_preview_markdown(artifacts_dir / "pdf_content_preview.md", preview)

        gen = PDFReportGenerator(
            output_path=str(artifacts_dir / "debug_report_preview.pdf"),
            images_dir=str(images_dir),
        )
        gen.generate_report(llm_report, metadata, verbose=False)

    print(f"Content debug artifacts written to: {artifacts_dir}")
    print(f"Metadata source: {metadata_path}")


if __name__ == "__main__":
    main()
