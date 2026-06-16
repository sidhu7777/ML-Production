import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from tools.report_engine.llm_integration import generate_report_text, parse_and_validate_llm_output
from tools.report_engine.metadata_generator import write_metadata_file
from tools.report_engine.kpi_analysis import (
    generate_drive_summary_images,
    generate_pci_poor_rsrp,
    generate_pci_poor_rsrq,
)
from tools.report_engine.pdf_generator import PDFReportGenerator
from tools.report_engine.map_generator import detect_handover_events, generate_handover_map


def _build_metadata(include_rsrp: bool = True):
    kpi_summary = {
        "SINR": {
            "average": 6.5,
            "min": -3.0,
            "max": 18.0,
            "poor_threshold": 0,
            "poor_count": 12,
            "poor_percentage": 4.2,
            "range_min": 0,
            "range_max": 10,
            "range_percentage": 61.5,
        },
        "DL": {
            "average": 14.2,
            "min": 0.5,
            "max": 58.0,
            "poor_threshold": 5,
            "poor_count": 25,
            "poor_percentage": 8.5,
            "excellent_threshold_value": 19,
            "excellent_percentage": 27.4,
        },
        "UL": {
            "average": 6.4,
            "min": 0.2,
            "max": 15.0,
            "poor_threshold": 5,
            "poor_count": 42,
            "poor_percentage": 13.1,
            "range_min": 4,
            "range_max": 6,
            "range_percentage": 34.7,
        },
        "MOS": {
            "average": 3.1,
            "min": 1.2,
            "max": 4.4,
            "poor_threshold": 2.0,
            "poor_count": 16,
            "poor_percentage": 5.6,
        },
    }
    if include_rsrp:
        kpi_summary["RSRP"] = {
            "average": -91.4,
            "min": -118.0,
            "max": -64.0,
            "poor_threshold": -105,
            "poor_count": 31,
            "poor_percentage": 9.3,
            "distribution": [
                {"Range": "Fair: -100 to -90", "percentage": 35.0, "cdf": 67.0},
            ],
        }

    return {
        "location": {"city": "Chennai", "country": "India"},
        "introduction": "Data includes 1200 samples from 4 sessions.",
        "area_summary": {
            "Overview": "Drive route covers key operational areas.",
            "Hotspots & Marked Locations": "Anna Nagar, T Nagar",
            "Crowded & High-Traffic Locations": "T Nagar",
            "Major Areas Covered": "The drive covered major areas including Anna Nagar and T Nagar.",
        },
        "drive_summary": {
            "distance_covered": 42.7,
            "total_samples": 1200,
            "total_sessions": 4,
            "number_of_days": 2,
            "start_date": "2026-06-01",
            "end_date": "2026-06-02",
        },
        "kpi_summary": kpi_summary,
        "band_summary": [
            {"band": "B3", "sample_count": 500, "sample_percentage": 41.67},
            {"band": "B40", "sample_count": 420, "sample_percentage": 35.0},
        ],
        "pci_summary": {
            "total_unique_pci": 18,
            "top_30_pci_percentage": 96.4,
        },
    }


def _build_report_text():
    return {
        "Introduction": "Intro paragraph.",
        "Area Summary": {
            "Overview": "Area overview.",
            "Hotspots & Marked Locations": "A, B",
            "Crowded & High-Traffic Locations": "B",
            "Major Areas Covered": "Covered areas paragraph.",
        },
        "Drive Summary": "Drive summary paragraph.",
        "KPI Summary": "KPI summary paragraph.",
        "Map View - Band": "",
        "Map View - RSRP": "",
        "Map View - RSRQ": "",
        "Map View - SINR": "",
        "Map View - DL Throughput": "",
        "Map View - UL Throughput": "",
        "Map View - MOS": "",
        "PCI Summary": "PCI summary paragraph.",
    }


def _paragraph_texts(story):
    texts = []
    for flowable in story:
        if hasattr(flowable, "getPlainText"):
            texts.append(flowable.getPlainText())
    return texts


def _artifact_dir():
    out_dir = Path(__file__).resolve().parent / "output" / "report_engine_harness"
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def _write_json(path: Path, payload):
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _write_story_preview(path: Path, story_texts):
    path.write_text("\n".join(story_texts), encoding="utf-8")


def _install_fake_groq(monkeypatch, raw_content):
    from tools.report_engine import llm_integration

    class _Message:
        def __init__(self, content):
            self.content = content

    class _Choice:
        def __init__(self, content):
            self.message = _Message(content)

    class _Response:
        def __init__(self, content):
            self.choices = [_Choice(content)]

    class _Completions:
        @staticmethod
        def create(*args, **kwargs):
            return _Response(raw_content)

    class _Chat:
        completions = _Completions()

    class _GroqClient:
        def __init__(self, api_key=None):
            self.chat = _Chat()

    monkeypatch.setattr(llm_integration, "Groq", _GroqClient)


def _install_failing_groq(monkeypatch, exc_factory):
    from tools.report_engine import llm_integration

    class _Completions:
        @staticmethod
        def create(*args, **kwargs):
            raise exc_factory()

    class _Chat:
        completions = _Completions()

    class _GroqClient:
        def __init__(self, api_key=None):
            self.chat = _Chat()

    monkeypatch.setattr(llm_integration, "Groq", _GroqClient)


def test_parse_and_validate_llm_output_fills_missing_sections(tmp_path):
    metadata = _build_metadata(include_rsrp=False)
    output_path = tmp_path / "report_text.json"

    report = parse_and_validate_llm_output(
        raw_text=json.dumps(
            {
                "Introduction": "LLM intro.",
                "Area Summary": metadata["area_summary"],
            }
        ),
        metadata=metadata,
        output_path=str(output_path),
    )

    assert output_path.exists()
    assert report["Introduction"] == "LLM intro."
    assert report["Area Summary"] == metadata["area_summary"]
    assert report["Map View - RSRP"] == "Not available."
    assert "DL throughput" in report["Map View - DL Throughput"]
    assert report["PCI Summary"].startswith("The drive test observed")


def test_parse_and_validate_llm_output_does_not_invent_missing_kpi_values(tmp_path):
    metadata = _build_metadata(include_rsrp=False)
    output_path = tmp_path / "report_text.json"

    report = parse_and_validate_llm_output(
        raw_text=json.dumps(
            {
                "Introduction": "LLM intro.",
                "Area Summary": metadata["area_summary"],
            }
        ),
        metadata=metadata,
        output_path=str(output_path),
    )

    assert output_path.exists()
    assert report["Map View - RSRP"] == "Not available."
    assert "-91.4" not in report["Map View - RSRP"]
    assert "31" not in report["Map View - RSRP"]


def test_generate_report_text_hybrid_path_merges_llm_and_supported_synthesized_sections(monkeypatch, tmp_path):
    metadata = _build_metadata(include_rsrp=True)
    output_path = tmp_path / "report_text.json"
    _install_fake_groq(
        monkeypatch,
        json.dumps(
            {
                "Introduction": "LLM introduction for Chennai, India.",
                "Drive Summary": "LLM drive summary paragraph.",
                "KPI Summary": "LLM KPI summary paragraph.",
                "Map View - RSRQ": "LLM RSRQ paragraph.",
                "Map View - SINR": "LLM SINR paragraph.",
                "Map View - RSRP": "LLM RSRP paragraph.",
                "Map View - DL Throughput": "LLM DL paragraph.",
                "Map View - UL Throughput": "LLM UL paragraph.",
                "Map View - MOS": "LLM MOS paragraph.",
            }
        ),
    )

    report = generate_report_text(
        metadata=metadata,
        output_path=str(output_path),
        verbose=False,
    )

    assert output_path.exists()
    assert report["Introduction"] == "LLM introduction for Chennai, India."
    assert report["Drive Summary"] == "LLM drive summary paragraph."
    assert report["KPI Summary"] == "LLM KPI summary paragraph."
    assert report["Map View - RSRQ"] == "LLM RSRQ paragraph."
    assert report["Map View - SINR"] == "LLM SINR paragraph."
    assert report["Map View - RSRP"] == "LLM RSRP paragraph."
    assert report["Area Summary"] == metadata["area_summary"]
    assert report["Map View - Band"].startswith("The majority of samples")
    assert report["Map View - DL Throughput"] == "LLM DL paragraph."
    assert report["Map View - UL Throughput"] == "LLM UL paragraph."
    assert report["Map View - MOS"] == "LLM MOS paragraph."
    assert report["PCI Summary"].startswith("The drive test observed")


def test_generate_report_text_should_backfill_kpi_summary_after_partial_llm_success(monkeypatch, tmp_path):
    metadata = _build_metadata(include_rsrp=False)
    output_path = tmp_path / "report_text.json"
    _install_fake_groq(
        monkeypatch,
        json.dumps(
            {
                "Introduction": "LLM introduction for Chennai, India.",
                "Drive Summary": "LLM drive summary paragraph.",
                "Map View - SINR": "LLM SINR paragraph.",
            }
        ),
    )

    report = generate_report_text(
        metadata=metadata,
        output_path=str(output_path),
        verbose=False,
    )

    assert output_path.exists()
    assert report["KPI Summary"]


def test_generate_report_text_should_backfill_missing_absent_kpi_section_after_partial_llm_success(monkeypatch, tmp_path):
    metadata = _build_metadata(include_rsrp=False)
    output_path = tmp_path / "report_text.json"
    _install_fake_groq(
        monkeypatch,
        json.dumps(
            {
                "Introduction": "LLM introduction for Chennai, India.",
                "Drive Summary": "LLM drive summary paragraph.",
                "KPI Summary": "LLM KPI summary paragraph.",
                "Map View - SINR": "LLM SINR paragraph.",
            }
        ),
    )

    report = generate_report_text(
        metadata=metadata,
        output_path=str(output_path),
        verbose=False,
    )

    assert output_path.exists()
    assert report["Map View - RSRP"] == "Not available."


def test_pdf_generator_skips_empty_kpi_headings(tmp_path, monkeypatch):
    output_path = tmp_path / "report.pdf"
    gen = PDFReportGenerator(output_path=str(output_path), images_dir=str(tmp_path))
    monkeypatch.setattr(gen.doc, "multiBuild", lambda story, canvasmaker=None: None)

    gen.generate_report(_build_report_text(), _build_metadata())

    texts = _paragraph_texts(gen.story)

    assert "5. Map View" not in texts
    assert "b) RSRP" not in texts
    assert "g) MOS" not in texts


def test_hybrid_report_text_can_feed_pdf_builder_without_error(monkeypatch, tmp_path):
    metadata = _build_metadata()
    report_text_path = tmp_path / "report_text.json"
    _install_fake_groq(
        monkeypatch,
        json.dumps(
            {
                "Introduction": "LLM introduction.",
                "Drive Summary": "LLM drive summary.",
                "KPI Summary": "LLM KPI summary.",
                "Map View - RSRP": "LLM RSRP paragraph.",
                "Map View - RSRQ": "Not available.",
                "Map View - SINR": "LLM SINR paragraph.",
                "Map View - DL Throughput": "LLM DL paragraph.",
                "Map View - UL Throughput": "LLM UL paragraph.",
                "Map View - MOS": "LLM MOS paragraph.",
                "Map View - Band": "LLM Band paragraph.",
                "PCI Summary": "LLM PCI paragraph.",
            }
        ),
    )
    report = generate_report_text(
        metadata=metadata,
        output_path=str(report_text_path),
        verbose=False,
    )

    output_path = tmp_path / "report.pdf"
    gen = PDFReportGenerator(output_path=str(output_path), images_dir=str(tmp_path))
    monkeypatch.setattr(gen.doc, "multiBuild", lambda story, canvasmaker=None: None)

    gen.generate_report(report, metadata)

    texts = _paragraph_texts(gen.story)
    assert "1. Introduction" in texts
    assert "5. Map View" in texts
    assert "b) RSRP" in texts
    assert "LLM RSRP paragraph." in texts


def test_debug_write_report_engine_output_artifacts(monkeypatch):
    out_dir = _artifact_dir()
    metadata = _build_metadata()

    hybrid_report_path = out_dir / "report_text_hybrid_success.json"
    missing_kpi_report_path = out_dir / "report_text_missing_kpi.json"
    parse_fallback_report_path = out_dir / "report_text_parse_fallback.json"
    metadata_path = out_dir / "report_metadata_sample.json"
    story_preview_path = out_dir / "pdf_story_preview.txt"

    _write_json(metadata_path, metadata)

    _install_fake_groq(
        monkeypatch,
        json.dumps(
            {
                "Introduction": "LLM introduction for Chennai, India.",
                "Drive Summary": "The drive test was conducted across the configured route and collected representative field samples for analysis.",
                "KPI Summary": "Coverage, quality, and throughput KPIs were reviewed across the drive route, with detailed interpretation captured in the map view sections.",
                "Map View - Band": "Most samples were observed on Band B3, followed by Band B40, indicating concentration on the primary serving layers used during the drive.",
                "Map View - RSRP": "RSRP indicates received signal strength. The measured values show stable mid-band coverage across most of the route, with weaker samples concentrated in a smaller subset of locations.",
                "Map View - RSRQ": "RSRQ reflects signal quality and network loading conditions. The observed values suggest moderate quality overall with some local degradation where radio conditions are less favorable.",
                "Map View - SINR": "SINR captures the balance between wanted signal and interference. The route shows generally usable signal quality, with lower values in a limited share of samples.",
                "Map View - DL Throughput": "Downlink throughput results indicate usable user-plane performance on the drive route, with stronger performance in the higher-throughput portions of the sample set.",
                "Map View - UL Throughput": "Uplink throughput remains serviceable for most of the drive, though a subset of samples stays near the lower threshold range.",
                "Map View - MOS": "MOS results indicate voice quality is broadly acceptable across the route, with a smaller set of lower-quality samples needing attention.",
                "PCI Summary": "PCI distribution shows activity across a moderate number of serving identities, with most samples concentrated among the top observed PCIs.",
            }
        ),
    )
    hybrid_report = generate_report_text(
        metadata=metadata,
        output_path=str(hybrid_report_path),
        verbose=False,
    )

    missing_kpi_report = parse_and_validate_llm_output(
        raw_text=json.dumps(
            {
                "Introduction": "LLM intro for missing KPI case.",
                "Area Summary": metadata["area_summary"],
            }
        ),
        metadata=_build_metadata(include_rsrp=False),
        output_path=str(missing_kpi_report_path),
    )

    parse_fallback_report = parse_and_validate_llm_output(
        raw_text="not-json-response",
        metadata=metadata,
        output_path=str(parse_fallback_report_path),
    )

    gen = PDFReportGenerator(output_path=str(out_dir / "debug_preview.pdf"), images_dir=str(out_dir))
    gen.doc.multiBuild = lambda story, canvasmaker=None: None
    gen.generate_report(hybrid_report, metadata)
    story_texts = _paragraph_texts(gen.story)
    _write_story_preview(story_preview_path, story_texts)

    assert metadata_path.exists()
    assert hybrid_report_path.exists()
    assert missing_kpi_report_path.exists()
    assert parse_fallback_report_path.exists()
    assert story_preview_path.exists()
    assert hybrid_report["Introduction"]
    assert missing_kpi_report["Map View - RSRP"] == "Not available."
    assert parse_fallback_report["PCI Summary"]


def test_pdf_generator_should_skip_kpi_heading_when_no_text_and_no_images(tmp_path, monkeypatch):
    output_path = tmp_path / "report.pdf"
    gen = PDFReportGenerator(output_path=str(output_path), images_dir=str(tmp_path))
    monkeypatch.setattr(gen.doc, "multiBuild", lambda story, canvasmaker=None: None)

    gen.generate_report(_build_report_text(), _build_metadata())

    texts = _paragraph_texts(gen.story)

    assert "b) RSRP" not in texts


def test_generate_report_text_should_fallback_after_request_failures(monkeypatch, tmp_path):
    from tools.report_engine import llm_integration

    _install_failing_groq(monkeypatch, lambda: RuntimeError("413 request too large"))

    output_path = tmp_path / "report_text.json"
    report = llm_integration.generate_report_text(
        metadata=_build_metadata(),
        output_path=str(output_path),
        verbose=False,
    )

    assert output_path.exists()
    assert report["Introduction"]
    assert report["Map View - RSRP"]


def test_generate_report_text_should_retry_three_times_before_fallback(monkeypatch, tmp_path):
    from tools.report_engine import llm_integration

    attempts = {"count": 0}

    class _Completions:
        @staticmethod
        def create(*args, **kwargs):
            attempts["count"] += 1
            raise RuntimeError("413 request too large")

    class _Chat:
        completions = _Completions()

    class _GroqClient:
        def __init__(self, api_key=None):
            self.chat = _Chat()

    monkeypatch.setattr(llm_integration, "Groq", _GroqClient)

    output_path = tmp_path / "report_text.json"
    llm_integration.generate_report_text(
        metadata=_build_metadata(),
        output_path=str(output_path),
        verbose=False,
    )

    assert attempts["count"] == 3


def test_request_failure_fallback_output_should_feed_pdf_builder(monkeypatch, tmp_path):
    from tools.report_engine import llm_integration

    _install_failing_groq(monkeypatch, lambda: RuntimeError("413 request too large"))

    report_text_path = tmp_path / "report_text.json"
    metadata = _build_metadata()
    report = llm_integration.generate_report_text(
        metadata=metadata,
        output_path=str(report_text_path),
        verbose=False,
    )

    output_path = tmp_path / "report.pdf"
    gen = PDFReportGenerator(output_path=str(output_path), images_dir=str(tmp_path))
    monkeypatch.setattr(gen.doc, "multiBuild", lambda story, canvasmaker=None: None)
    gen.generate_report(report, metadata)

    texts = _paragraph_texts(gen.story)
    assert report_text_path.exists()
    assert "1. Introduction" in texts


def test_write_metadata_file_serializes_numpy_scalars(tmp_path):
    output_path = tmp_path / "report_metadata.json"
    metadata = {
        "drive_summary": {
            "total_sessions": np.int64(3),
            "number_of_days": np.int64(1),
            "distance_covered": np.float64(0.0),
        },
        "kpi_summary": {
            "RSRP": {
                "poor_count": np.int64(266),
                "poor_percentage": np.float64(26.47),
            }
        },
    }

    write_metadata_file(metadata, str(output_path))

    loaded = json.loads(output_path.read_text(encoding="utf-8"))
    assert loaded["drive_summary"]["total_sessions"] == 3
    assert loaded["drive_summary"]["number_of_days"] == 1
    assert loaded["kpi_summary"]["RSRP"]["poor_count"] == 266


def test_generate_report_text_serializes_numpy_scalars_in_llm_payload(monkeypatch, tmp_path):
    metadata = _build_metadata()
    metadata["drive_summary"]["total_sessions"] = np.int64(4)
    metadata["pci_summary"]["total_unique_pci"] = np.int64(18)

    output_path = tmp_path / "report_text.json"
    _install_failing_groq(monkeypatch, lambda: RuntimeError("forced failure after prompt build"))

    report = generate_report_text(
        metadata=metadata,
        output_path=str(output_path),
        verbose=False,
    )

    assert output_path.exists()
    assert report["Introduction"]


def test_drive_summary_falls_back_to_session_rows_when_network_timestamps_invalid(tmp_path, monkeypatch):
    network_df = pd.DataFrame(
        [
            {"session_id": 4178, "timestamp": None},
            {"session_id": 4180, "timestamp": None},
        ]
    )

    session_df = pd.DataFrame(
        [
            {"id": 4178, "start_time": "2026-03-25 18:29:00", "end_time": "2026-03-25 19:31:00", "distance": 0.0},
            {"id": 4180, "start_time": "2026-03-25 18:30:00", "end_time": "2026-03-25 19:25:00", "distance": 0.0},
        ]
    )

    monkeypatch.setattr("tools.report_engine.kpi_analysis.IMAGE_DIR", str(tmp_path))
    monkeypatch.setattr("tools.report_engine.kpi_analysis.get_session_data_for_drive_summary", lambda session_ids: session_df)

    summary = generate_drive_summary_images([4178, 4180], total_samples=6331, network_df=network_df)

    assert summary is not None
    assert summary["total_sessions"] == 2
    assert summary["start_date"] == "2026-03-25"


def test_pci_poor_tables_ignore_empty_band_values(tmp_path, monkeypatch):
    monkeypatch.setattr("tools.report_engine.kpi_analysis.IMAGE_DIR", str(tmp_path))

    df = pd.DataFrame(
        [
            {"pci": 101, "rsrp": -110, "rsrq": -18, "band": "B3", "cell_id": "A"},
            {"pci": 101, "rsrp": -108, "rsrq": -17, "band": None, "cell_id": None},
            {"pci": 101, "rsrp": -109, "rsrq": -16, "band": "B40", "cell_id": "B"},
        ]
    )

    generate_pci_poor_rsrp(df)
    generate_pci_poor_rsrq(df)

    assert (tmp_path / "pci_poor_rsrp.png").exists()
    assert (tmp_path / "pci_poor_rsrq.png").exists()


def test_handover_handles_bridge_dict_timestamps(tmp_path):
    timestamp_1 = {
        "IsValidDateTime": True,
        "Year": 2026,
        "Month": 6,
        "Day": 8,
        "Hour": 13,
        "Minute": 42,
        "Second": 1,
    }
    timestamp_2 = {
        "IsValidDateTime": True,
        "Year": 2026,
        "Month": 6,
        "Day": 8,
        "Hour": 13,
        "Minute": 42,
        "Second": 2,
    }
    df = pd.DataFrame(
        [
            {
                "session_id": 4763,
                "timestamp": timestamp_1,
                "lat": 7.3876,
                "lon": 3.8787,
                "network": "4G",
                "band": "B3",
                "pci": 101,
                "m_alpha_long": "MTN NIGERIA",
            },
            {
                "session_id": 4763,
                "timestamp": timestamp_2,
                "lat": 7.3877,
                "lon": 3.8788,
                "network": "5G",
                "band": "n78",
                "pci": 102,
                "m_alpha_long": "MTN NIGERIA",
            },
        ]
    )

    events = detect_handover_events(df, use_global_detection=True, min_run_length=10)
    assert events

    output_html = tmp_path / "handover.html"
    generate_handover_map(df, events, str(output_html))
    assert output_html.exists()


def test_report_handover_detects_only_band_transitions():
    df = pd.DataFrame(
        [
            {
                "session_id": 1,
                "timestamp": "2026-06-08T13:42:01",
                "lat": 7.3876,
                "lon": 3.8787,
                "network": "4G",
                "band": "B3",
                "pci": 101,
            },
            {
                "session_id": 1,
                "timestamp": "2026-06-08T13:42:02",
                "lat": 7.3877,
                "lon": 3.8788,
                "network": "5G",
                "band": "B3",
                "pci": 102,
            },
        ]
    )

    events = detect_handover_events(df)

    assert events == []


def test_report_handover_band_transitions_are_not_limited_to_500():
    rows = []
    for idx in range(601):
        rows.append(
            {
                "session_id": 1,
                "timestamp": f"2026-06-08T13:{idx // 60:02d}:{idx % 60:02d}",
                "lat": 7.3 + (idx * 0.00001),
                "lon": 3.8 + (idx * 0.00001),
                "network": "4G",
                "band": "B3" if idx % 2 == 0 else "B40",
                "pci": 100 + idx,
            }
        )

    events = detect_handover_events(pd.DataFrame(rows))

    assert len(events) == 600
    assert {event["type"] for event in events} == {"band"}


def test_report_handover_map_renders_band_events_only(tmp_path):
    df = pd.DataFrame(
        [
            {"session_id": 1, "timestamp": "2026-06-08T13:42:01", "lat": 7.3876, "lon": 3.8787},
            {"session_id": 1, "timestamp": "2026-06-08T13:42:02", "lat": 7.3877, "lon": 3.8788},
            {"session_id": 1, "timestamp": "2026-06-08T13:42:03", "lat": 7.3878, "lon": 3.8789},
        ]
    )
    events = [
        {"type": "technology", "session_id": 1, "lat": 7.3876, "lon": 3.8787, "from_value": "4G", "to_value": "5G"},
        {"type": "band", "session_id": 1, "lat": 7.3877, "lon": 3.8788, "from_value": "B3", "to_value": "B40"},
        {"type": "pci", "session_id": 1, "lat": 7.3878, "lon": 3.8789, "from_value": "101", "to_value": "102"},
    ]

    output_html = tmp_path / "handover.html"
    generate_handover_map(df, events, str(output_html))
    html = output_html.read_text(encoding="utf-8")

    assert "Band Handover Events" in html
    assert "Band: B3 -> B40" in html
    assert "Technology: 4G -> 5G" not in html
    assert "Pci: 101 -> 102" not in html
