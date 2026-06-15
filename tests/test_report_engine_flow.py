import json
from pathlib import Path

import pytest

from tools.report_engine.llm_integration import generate_report_text, parse_and_validate_llm_output
from tools.report_engine.pdf_generator import PDFReportGenerator


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


@pytest.mark.xfail(
    reason="Current generate_report_text() does not backfill KPI Summary when the LLM returns valid JSON but omits that key.",
    strict=True,
)
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


@pytest.mark.xfail(
    reason="Current generate_report_text() does not backfill a missing Map View KPI section when valid LLM JSON omits it and metadata has no matching KPI entry.",
    strict=True,
)
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


def test_pdf_generator_currently_keeps_empty_kpi_headings(tmp_path, monkeypatch):
    output_path = tmp_path / "report.pdf"
    gen = PDFReportGenerator(output_path=str(output_path), images_dir=str(tmp_path))
    monkeypatch.setattr(gen.doc, "multiBuild", lambda story, canvasmaker=None: None)

    gen.generate_report(_build_report_text(), _build_metadata())

    texts = _paragraph_texts(gen.story)

    assert "5. Map View" in texts
    assert "b) RSRP" in texts
    assert "g) MOS" in texts


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


@pytest.mark.xfail(
    reason="Current PDF generator always adds hardcoded KPI subsection headings even when text and images are absent.",
    strict=True,
)
def test_pdf_generator_should_skip_kpi_heading_when_no_text_and_no_images(tmp_path, monkeypatch):
    output_path = tmp_path / "report.pdf"
    gen = PDFReportGenerator(output_path=str(output_path), images_dir=str(tmp_path))
    monkeypatch.setattr(gen.doc, "multiBuild", lambda story, canvasmaker=None: None)

    gen.generate_report(_build_report_text(), _build_metadata())

    texts = _paragraph_texts(gen.story)

    assert "b) RSRP" not in texts


@pytest.mark.xfail(
    reason="Current generate_report_text() raises if the Groq request fails before a response object exists.",
    strict=True,
)
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


@pytest.mark.xfail(
    reason="Current generate_report_text() has no retry loop around the Groq request.",
    strict=True,
)
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


@pytest.mark.xfail(
    reason="Current generate_report_text() cannot produce fallback report_text.json after a request-level Groq failure, so the downstream PDF path cannot be exercised.",
    strict=True,
)
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
