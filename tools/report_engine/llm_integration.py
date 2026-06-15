# src/llm_integration.py

import os
import json
import time
from typing import Dict
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

REQUIRED_KEYS = [
    "Introduction",
    "Area Summary",
    "Drive Summary",
    "KPI Summary",
    "Map View - Band",
    "Map View - RSRP",
    "Map View - RSRQ",
    "Map View - SINR",
    "Map View - DL Throughput",
    "Map View - UL Throughput",
    "Map View - MOS",
    "PCI Summary",
]

SYSTEM_PROMPT = """
You are a telecom RF drive-test reporting engine. Produce professional, concise report
text suitable for inclusion in an executive PDF. Follow these rules STRICTLY.

REQUIRED OUTPUT CHARACTERISTICS:
- Return a single valid JSON object (no surrounding markdown or commentary).
- Keys MUST exactly match the required report sections.
- `Area Summary` MUST be a JSON object with nested keys (do not stringify).
- All other sections MUST be plain text strings.
- Do NOT invent or alter numeric metrics; use only the values present in METADATA.

AREA SUMMARY RULES:
- `Area Summary` is a dict with these keys: Overview, Hotspots & Marked Locations,
  Crowded & High-Traffic Locations, Major Areas Covered.
- Keep "Hotspots & Marked Locations" and "Crowded & High-Traffic Locations" as
  simple comma-separated location strings (DO NOT expand into dict with explanations).
- Keep "Major Areas Covered" as a paragraph string.
- Example: {"Hotspots & Marked Locations": "Location A, Location B, Location C"}

KPI SUMMARY RULES:
- `KPI Summary` must be a short (1–2 sentences) GENERIC neutral paragraph WITHOUT
  any quality judgments (no "acceptable", "good", "excellent", "poor", etc.).
- Simply state that KPI metrics were analyzed and detailed findings are in Map View.
- Example: "Network KPI metrics including coverage, quality, and throughput were analyzed
  across the drive route. Detailed performance analysis is provided in the Map View sections."
- DO NOT include per-KPI numeric values or quality assessments in KPI Summary.
MAP VIEW / PER-KPI RULES:
- For each KPI map view (e.g., `Map View - RSRP`), produce ONE readable, report-level paragraph.
- Write naturally and professionally, not just listing metadata values.
- CRITICAL: Use ONLY the EXACT numeric values from METADATA. DO NOT invent, estimate, or calculate new numbers.
- Each paragraph MUST:
    1) Start with the KPI name and its significance for network performance.
    2) Describe the overall performance using metadata values: `average`, `min`, `max`.
    3) Explain what the numbers mean for user experience (e.g., signal strength, quality, speed).
    4) Highlight areas of concern if `poor_count` and `poor_percentage` are significant.
    5) For throughput KPIs (DL/UL), mention if performance exceeds excellent thresholds.
    6) For quality KPIs (MOS/SINR), describe the impact on voice/data quality.
    7) For Band: Use the EXACT sample_percentage values from band_summary array for each band.
       DO NOT calculate or combine percentages. List top 2-3 bands with their exact percentages.
    8) Use simple hyphens (-) for ranges, format as "X to Y" with spaces.
    9) Make it readable as a report paragraph, not just a metadata dump.
    10) Example: "RSRP (Reference Signal Received Power) is a key indicator of signal strength. "
         "The measured values show an average of -81.23 dBm, ranging from -117.0 to -58.0 dBm. "
         "The network demonstrates good signal coverage with only 2.17% of samples (203 locations) "
         "falling below the acceptable threshold, indicating strong coverage across most of the route."

VALIDATION / FORMAT:
- Do not include lists, tables, or embedded JSON in Map View strings — just
    natural-language paragraphs. Keep paragraphs short and factual.

If any required KPI fields are missing in METADATA, return a concise paragraph
stating which metrics were unavailable, but do not invent values.
"""


def _json_safe_llm(value):
    if value is None:
        return None
    try:
        if value is json.loads("null"):
            return None
    except Exception:
        pass
    if isinstance(value, dict):
        return {k: _json_safe_llm(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_json_safe_llm(v) for v in value]
    if isinstance(value, tuple):
        return [_json_safe_llm(v) for v in value]
    if hasattr(value, "isoformat") and not isinstance(value, str):
        try:
            return value.isoformat()
        except Exception:
            pass
    if hasattr(value, "item"):
        try:
            return _json_safe_llm(value.item())
        except Exception:
            pass
    try:
        if value != value:
            return None
    except Exception:
        pass
    return value

def generate_report_text(
    metadata: Dict,
    output_path: str = "data/processed/report_text.json",
    model: str = "llama-3.1-8b-instant",
    temperature: float = 0.0,
    max_tokens: int = 2500,
    max_retries: int = 3,
    verbose: bool = False,
) -> Dict:

    client = Groq(api_key=os.getenv("GROQ_API_KEY"))

    # Use Area Summary directly from metadata (LLM just repeats it anyway)
    report = {"Area Summary": metadata.get("area_summary", {})}
    
    # Keep the prompt intact but send only the narrative-critical fields.
    drive_summary = metadata.get("drive_summary", {}) if isinstance(metadata, dict) else {}
    if isinstance(drive_summary, dict):
        drive_summary = {
            k: drive_summary.get(k)
            for k in [
                "distance_covered",
                "total_samples",
                "total_sessions",
                "number_of_days",
                "start_date",
                "end_date",
            ]
            if k in drive_summary
        }

    llm_metadata = {
        "location": metadata.get("location", {}),
        "introduction": metadata.get("introduction"),
        "drive_summary": drive_summary,
        "kpi_summary": metadata.get("kpi_summary", {}),
        "band_summary": metadata.get("band_summary", []),
        "pci_summary": metadata.get("pci_summary", {}),
    }
    llm_metadata = _json_safe_llm(llm_metadata)
    
    prompt = f"""
Generate a telecom drive-test report using the metadata below.

METADATA:
{json.dumps(llm_metadata, indent=2)}

OUTPUT FORMAT:
- Introduction: Write a professional introduction for an  drive test report (3-4 sentences).
    Use a tone similar to the provided sample (executive summary style), but keep all specifics dynamic.
    Mention the test location if available in metadata.location.city/country (e.g., "in <city>, <country>").
    Explain the PURPOSE and IMPORTANCE of drive testing in general terms (signal strength, coverage, provider performance).
    Mention that map visualizations highlight strong/weak coverage areas.
    State that the report provides actionable insights for optimization and deployment planning.
    DO NOT include specific drive details (distance, samples, dates) - those belong in Drive Summary.
    DO NOT invent any facts not present in metadata.
  
- Drive Summary: Describe THIS specific drive test using ALL available fields from drive_summary:
  distance_covered, total_samples, total_sessions, number_of_days, start_date, end_date.
  Write as a proper paragraph mentioning the test duration, coverage, and data collection.
  Example: "The drive test was conducted over X days from [start_date] to [end_date], 
  covering Y km with Z samples collected across N sessions."
  
- KPI Summary: A short (1–2 sentences) GENERIC neutral paragraph WITHOUT any quality judgments.
  Simply state that KPI metrics were analyzed and detailed findings are in Map View.
  
- Map View sections: For each KPI, write a readable report-level paragraph that:
  * Explains what the KPI measures and why it matters
  * Describes the measured performance using average, min, max values
  * Interprets what the numbers mean for user experience
  * Use ACTUAL NUMERIC VALUES instead of word 'threshold'
  * Example: Say '436 samples below 3.0' NOT '436 samples below the poor threshold'
  * Example: Say 'exceeding 15 Mbps' NOT 'exceeding the excellent threshold'
  * For Band: Use EXACT percentages from band_summary (e.g., B8: 39.44%, B40: 34.62%)
  * CRITICAL: Use ONLY exact numbers from metadata - DO NOT invent or calculate new values
  * Makes it readable as a professional report, not just metadata listing
  
- PCI Summary: Describe PCI distribution using pci_summary and band_summary metadata.

Return JSON with EXACT keys:
{[k for k in REQUIRED_KEYS if k != "Area Summary"]}
"""

    if verbose:
        print("Calling LLM to generate report text...")

    response = None
    last_error = None
    for attempt in range(1, max_retries + 1):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                temperature=temperature,
                max_tokens=max_tokens,
            )
            break
        except Exception as e:
            last_error = e
            if verbose:
                print(f"LLM request failed on attempt {attempt}/{max_retries}: {e}")
            if attempt < max_retries:
                time.sleep(min(attempt, 2))

    if response is None:
        if verbose:
            print("LLM request failed after retries, synthesizing fallbacks from metadata.")
        report = _fill_missing_sections_from_metadata(report, metadata)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        return report

    try:
        source_map = {"Area Summary": "metadata"}
        raw_content = response.choices[0].message.content

        # Strip markdown code blocks if present
        if raw_content.startswith("```"):
            raw_content = raw_content.split("\n", 1)[1] if "\n" in raw_content else raw_content[3:]
            if raw_content.endswith("```"):
                raw_content = raw_content.rsplit("```", 1)[0]
            raw_content = raw_content.strip()

        # Extract JSON object even if extra text is present
        start = raw_content.find("{")
        end = raw_content.rfind("}")
        if start != -1 and end != -1 and end > start:
            raw_content = raw_content[start:end+1]

        llm_output = json.loads(raw_content)
        if verbose:
            print("✓ LLM response successfully parsed")
            print(f"✓ LLM generated {len(llm_output)} sections")
        # Merge LLM output with pre-set Area Summary
        report.update(llm_output)
        # Ensure metadata Area Summary is preserved (LLM often outputs placeholders)
        report["Area Summary"] = metadata.get("area_summary", report.get("Area Summary", {}))

        # Replace placeholder tokens with real metadata values (if present)
        loc = (metadata or {}).get("location", {}) if isinstance(metadata, dict) else {}
        city = loc.get("city")
        country = loc.get("country")
        if city or country:
            location_text = ", ".join([v for v in [city, country] if v])
            for k, v in report.items():
                if k == "Area Summary" or not isinstance(v, str):
                    continue
                v = v.replace("[location.city]", city or "").replace("[location.country]", country or "")
                v = v.replace("[location]", location_text)
                # Clean up leftover double spaces and dangling commas
                v = " ".join(v.split())
                v = v.replace(", ,", ",").replace(" ,", ",")
                report[k] = v

        # Fill any missing sections from synthesis and report sources
        missing = [k for k in REQUIRED_KEYS if k not in report]
        if missing:
            report = _fill_missing_sections_from_metadata(report, metadata, missing_keys=missing)
            for k in missing:
                if k in report:
                    source_map[k] = "synth"
        for k in REQUIRED_KEYS:
            if k in llm_output:
                source_map[k] = "llm"
        if verbose:
            if missing:
                print(f"⚠️ Missing sections from LLM, synthesized: {missing}")
            print("Section sources:")
            for k in REQUIRED_KEYS:
                print(f"  - {k}: {source_map.get(k, 'unknown')}")

        if verbose:
            # Save raw LLM output for inspection
            debug_path = output_path.replace(".json", "_llm_raw.txt")
            with open(debug_path, "w", encoding="utf-8") as f:
                f.write(response.choices[0].message.content or "")
            print(f"Raw LLM output saved: {debug_path}")
    except Exception as e:
        # LLM returned non-JSON or empty response — synthesize safe fallbacks
        if verbose:
            print(" LLM parse failed, synthesizing fallbacks from metadata:", e)
            print(f" LLM raw response: {response.choices[0].message.content[:500]}...")
        source_map = {k: "synth" for k in REQUIRED_KEYS}
        source_map["Area Summary"] = "metadata"
        report = _fill_missing_sections_from_metadata(report, metadata)
        if verbose:
            print("Section sources:")
            for k in REQUIRED_KEYS:
                print(f"  - {k}: {source_map.get(k, 'unknown')}")
            debug_path = output_path.replace(".json", "_llm_raw.txt")
            with open(debug_path, "w", encoding="utf-8") as f:
                f.write(response.choices[0].message.content or "")
            print(f"Raw LLM output saved: {debug_path}")

    # ---------------- VALIDATION ----------------

    for k in REQUIRED_KEYS:
        if k not in report:
            raise ValueError(f"Missing section: {k}")

    if not isinstance(report["Area Summary"], dict):
        raise ValueError("Area Summary MUST be a structured object")

    for k, v in report.items():
        if k != "Area Summary" and not isinstance(v, str):
            raise ValueError(f"Section '{k}' must be a string")

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    return report


def _fill_missing_sections_from_metadata(report: Dict, metadata: Dict, missing_keys=None) -> Dict:
    if missing_keys is None:
        missing_keys = [k for k in REQUIRED_KEYS if k not in report or report.get(k) in (None, "")]

    synth = _synthesize_missing_sections(metadata, missing_keys)
    for k in missing_keys:
        if k in report and report.get(k) not in (None, ""):
            continue
        if k == "Area Summary":
            report[k] = metadata.get("area_summary") or synth.get(k) or {"Overview": "Area summary not available."}
            continue
        if k in synth and isinstance(synth[k], str) and synth[k].strip():
            report[k] = synth[k]
        elif k == "KPI Summary":
            report[k] = (
                "Network KPI metrics including coverage, quality, and throughput were analyzed across the drive route. "
                "Detailed performance analysis is provided in the Map View sections below."
            )
        else:
            report[k] = "Not available."
    return report


def parse_and_validate_llm_output(
    raw_text: str,
    metadata: Dict,
    output_path: str,
) -> Dict:
    """
    Parse raw LLM output JSON and enforce required sections/types.
    Intended for tests and offline validation.
    """
    try:
        report = json.loads(raw_text)
    except Exception:
        report = {}

    if not isinstance(report, dict):
        report = {}

    # Normalize Area Summary to a structured object if needed
    area = report.get("Area Summary")
    if isinstance(area, str):
        report["Area Summary"] = {"Overview": area}

    synth = _synthesize_missing_sections(metadata, REQUIRED_KEYS)

    for k in REQUIRED_KEYS:
        if k not in report or report.get(k) in (None, ""):
            if k == "KPI Summary":
                report[k] = (
                    "Network KPI metrics including coverage, quality, and throughput were analyzed across the drive route. "
                    "Detailed performance analysis is provided in the Map View sections below."
                )
            else:
                report[k] = synth.get(k) or "Not available."

    # Type enforcement
    if not isinstance(report.get("Area Summary"), dict):
        report["Area Summary"] = {"Overview": "Area summary not available."}

    for k, v in report.items():
        if k != "Area Summary" and not isinstance(v, str):
            report[k] = str(v)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    return report


def _synthesize_missing_sections(metadata: Dict, missing_keys):
    """
    Build simple fallbacks for missing report sections using available metadata.
    Returns a dict of key->text (or structured object for Area Summary).
    """
    out = {}
    if not isinstance(metadata, dict):
        return out

    kpi_summary = metadata.get("kpi_summary", {})
    area = metadata.get("area_summary")
    drive = metadata.get("drive_summary", {})
    pci = metadata.get("pci_summary", {})
    band_summary = metadata.get("band_summary", [])

    for key in missing_keys:
        if key == "Introduction":
            # Calculate number of days from start and end dates
            start = drive.get("start_date")
            end = drive.get("end_date")
            num_days = drive.get("number_of_days", "N/A")
            
            out[key] = metadata.get("introduction") or (
                f"This report presents a comprehensive RF drive test analysis conducted to evaluate network performance and quality. "
                f"RF drive testing is a critical methodology for assessing real-world network coverage, signal strength, and service quality. "
                f"The evaluation encompasses key performance indicators including signal measurements, quality metrics, and throughput analysis. "
                f"This assessment provides valuable insights for network optimization and performance improvement."
            )

        elif key == "Area Summary":
            if isinstance(area, dict):
                out[key] = area
            else:
                out[key] = {
                    "Overview": "Area summary not available from LLM.",
                }

        elif key == "Drive Summary":
            if isinstance(drive, dict):
                dist = drive.get("distance_covered")
                samples = drive.get("total_samples")
                sessions = drive.get("total_sessions")
                num_days = drive.get("number_of_days")
                start = drive.get("start_date")
                end = drive.get("end_date")
                
                if start and end and num_days:
                    out[key] = (
                        f"The drive test was conducted over {num_days} day(s) from {start} to {end}, "
                        f"covering a total distance of {dist} km. A total of {samples:,} samples were collected "
                        f"across {sessions} test sessions, providing comprehensive network performance data."
                    )
                else:
                    out[key] = f"The drive covered {dist} km with {samples} samples across {sessions} sessions."
            else:
                out[key] = "Drive summary not available."

        elif key.startswith("Map View - "):
            sub = key.replace("Map View - ", "").strip()
            data = None
            for kk, vv in (kpi_summary or {}).items():
                if not kk:
                    continue
                if sub.lower() in str(kk).lower() or str(kk).lower() in sub.lower():
                    data = vv
                    break

            # Build a deterministic 1-2 sentence paragraph referencing metadata
            if data:
                parts = []
                avg = data.get("average")
                mn = data.get("min")
                mx = data.get("max")
                poor_count = data.get("poor_count") or data.get("poor_samples")
                poor_pct = data.get("poor_percentage")

                # Use fixed threshold analysis for specific KPIs (uniform reporting)
                if sub.lower() in ["dl throughput", "dl"]:
                    poor_thresh = data.get("poor_threshold")
                    poor_count = data.get("poor_count")
                    poor_pct = data.get("poor_percentage")
                    excellent_thresh = data.get("excellent_threshold_value")
                    excellent_pct = data.get("excellent_percentage")
                    
                    if poor_count is not None and poor_pct is not None and poor_thresh is not None:
                        parts.append(f"{poor_pct}% of the total samples have DL throughput <={poor_thresh}Mbps counting to {poor_count} samples.")
                        if excellent_pct is not None and excellent_pct > 0 and excellent_thresh is not None:
                            parts.append(f"Although we have {excellent_pct}% of samples > {excellent_thresh} Mbps speed.")
                
                elif sub.lower() in ["ul throughput", "ul"]:
                    poor_thresh = data.get("poor_threshold")
                    poor_count = data.get("poor_count")
                    poor_pct = data.get("poor_percentage")
                    range_min = data.get("range_min")
                    range_max = data.get("range_max")
                    range_pct = data.get("range_percentage")
                    
                    if poor_count is not None and poor_pct is not None and poor_thresh is not None:
                        parts.append(f"{poor_pct}% of the total samples have UL throughput <={poor_thresh}Mbps counting to {poor_count} samples.")
                        if range_pct is not None and range_min is not None and range_max is not None:
                            parts.append(f"{range_pct}% of the samples are within the range of {range_min}Mbps~{range_max}Mbps.")
                
                elif sub.lower() == "mos":
                    poor_thresh = data.get("poor_threshold")
                    poor_pct = data.get("poor_percentage")
                    
                    if poor_pct is not None and poor_thresh is not None:
                        parts.append(f"MOS quality to maintain good voice experience requires values above {poor_thresh:.1f}. Currently {poor_pct}% of samples fall below this value.")
                
                elif sub.lower() == "sinr":
                    poor_thresh = data.get("poor_threshold")
                    poor_pct = data.get("poor_percentage")
                    range_min = data.get("range_min")
                    range_max = data.get("range_max")
                    range_pct = data.get("range_percentage")
                    
                    if avg is not None:
                        parts.append(f"SINR shows average {avg}, range {mn} to {mx}.")
                        if poor_pct is not None and poor_pct > 0 and poor_thresh is not None:
                            parts.append(f"{poor_pct}% of samples are below {poor_thresh}.")
                        if range_pct is not None and range_min is not None and range_max is not None:
                            parts.append(f"{range_pct}% of samples are in the range {range_min}-{range_max}.")
                
                # For RSRP/RSRQ, keep existing poor region logic
                elif sub.lower() in ["rsrp", "rsrq"]:
                    facts = []
                    if avg is not None:
                        facts.append(f"average {avg}")
                    if mn is not None and mx is not None:
                        facts.append(f"range {mn} to {mx}")
                    if poor_count is not None:
                        facts.append(f"{poor_count} poor samples")
                    if poor_pct is not None:
                        facts.append(f"{poor_pct}% poor")

                    if facts:
                        parts.append(f"{sub} shows {', '.join(facts)}.")
                    
                    # Determine which quality range the average falls into
                    distribution = data.get("distribution", [])
                    avg_quality = None
                    if avg is not None and distribution and isinstance(distribution, list):
                        for item in distribution:
                            if isinstance(item, dict) and "Range" in item:
                                range_str = item.get("Range", "")
                                # Extract label and min/max from "label: min to max" format
                                if ": " in range_str:
                                    label_part = range_str.split(":")[0].strip()
                                    range_part = range_str.split(":")[1].strip()
                                    # Skip Auto-generated gap ranges
                                    if "Auto" in label_part or "Gap" in label_part or "Min" in label_part or "Max" in label_part:
                                        continue
                                    # Parse min/max from "min to max"
                                    try:
                                        parts_range = range_part.split(" to ")
                                        if len(parts_range) == 2:
                                            range_min = float(parts_range[0].strip())
                                            range_max = float(parts_range[1].strip())
                                            # Check if average falls in this range
                                            if range_min <= avg <= range_max:
                                                avg_quality = label_part
                                                break
                                    except:
                                        continue
                    
                    # Add quality assessment based on average
                    if avg_quality:
                        parts.append(f"The average indicates {avg_quality} performance.")
                out[key] = " ".join(parts)

            elif sub.lower() == "band" and band_summary:
                # Use all bands from band_summary with exact percentages
                band_parts = []
                for idx, band_data in enumerate(band_summary[:4]):  # Top 4 bands
                    band_name = band_data.get('band')
                    band_pct = band_data.get('sample_percentage')
                    if idx == 0:
                        band_parts.append(f"The majority of samples ({band_pct}%) were collected on Band {band_name}")
                    else:
                        band_parts.append(f"Band {band_name} ({band_pct}%)")
                
                if len(band_parts) > 1:
                    out[key] = f"{band_parts[0]}, followed by {', '.join(band_parts[1:])}."
                elif band_parts:
                    out[key] = f"{band_parts[0]}."
                else:
                    out[key] = "Band distribution data not available."

        elif key == "PCI Summary":
            if isinstance(pci, dict):
                total = pci.get("total_unique_pci")
                top30 = pci.get("top_30_pci_percentage")
                if total is not None and top30 is not None:
                    out[key] = f"The drive test observed {int(total)} unique PCI cells; top 30 account for {float(top30):.2f}% of samples."
            else:
                out[key] = "PCI summary not available."

    return out
