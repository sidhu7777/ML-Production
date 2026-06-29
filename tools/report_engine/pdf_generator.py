# src/pdf_generator.py

import os
import json
import re
from datetime import datetime
from typing import Dict, Any, Optional
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Image, PageBreak, CondPageBreak,
    Table, TableStyle, KeepTogether
)
from reportlab.platypus.tableofcontents import TableOfContents
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.pdfgen import canvas
from PIL import Image as PILImage


class PageNumCanvas(canvas.Canvas):
    """Custom canvas to add page numbers to each page"""
    def __init__(self, *args, **kwargs):
        canvas.Canvas.__init__(self, *args, **kwargs)
        self.pages = []
        
    def showPage(self):
        self.pages.append(dict(self.__dict__))
        self._startPage()
        
    def save(self):
        page_count = len(self.pages)
        for page_num, page in enumerate(self.pages, 1):
            self.__dict__.update(page)
            self.draw_page_number(page_num, page_count)
            canvas.Canvas.showPage(self)
        canvas.Canvas.save(self)
        
    def draw_page_number(self, page_num, page_count):
        page_text = f"Page {page_num}"
        self.setFont("Helvetica", 9)
        self.drawRightString(
            A4[0] - 40,  # 40 points from right edge
            30,  # 30 points from bottom
            page_text
        )


class TOCDocTemplate(SimpleDocTemplate):
    """DocTemplate that collects TOC entries via afterFlowable notifications."""

    def afterFlowable(self, flowable):
        if hasattr(flowable, "toc_level"):
            level = flowable.toc_level
            text = flowable.toc_text
            key = getattr(flowable, "toc_key", None)
            if key:
                self.canv.bookmarkPage(key)
            self.notify("TOCEntry", (level, text, self.page, key))


# =====================================================
# NATIVE TABLE + IMAGE HANDLING
# Data tables are rendered as crisp native ReportLab tables (selectable text,
# tiny size); map screenshots are palette-compressed; charts keep full pixels.
# =====================================================

# Filenames that are DATA TABLES (full-width, no 4-inch height squeeze).
TABLE_IMAGE_FILES = {
    "band_table.png", "pci_table.png", "session_table.png",
    "kpi_summary.png", "drive_summary.png", "poor_kpi_stats.png",
    "network_quality_summary.png", "pci_cdf_table.png", "indoor_outdoor_stats.png",
    "app_analytics_part1.png", "app_analytics_part2.png",
    "pci_poor_rsrp.png", "pci_poor_rsrq.png",
    "speed_stats.png", "latency_stats.png", "jitter_stats.png",
    "rsrp_range_table.png", "rsrq_range_table.png", "sinr_range_table.png",
    "dl_range_table.png", "ul_range_table.png", "mos_range_table.png",
}

# Charts: keep original pixels (quantization softens axis/label text).
CHART_IMAGE_FILES = {
    "pci_distribution.png", "band_pie.png",
    "cdf_rsrp.png", "cdf_rsrq.png", "cdf_sinr.png",
    "cdf_dl_tpt.png", "cdf_ul_tpt.png", "cdf_mos.png", "cdf_pci.png",
    "speed_hist.png", "latency_hist.png", "jitter_hist.png",
}

TABLE_MAX_WIDTH = 6.6 * inch    # full usable A4 text width (margins 40+40pt)
TABLE_MAX_HEIGHT = 8.6 * inch   # nearly full page height (no 4-inch squeeze)

_NATIVE_TABLE_STYLES = getSampleStyleSheet()
_NATIVE_CELL_STYLE = ParagraphStyle(
    "NativeTableCell",
    parent=_NATIVE_TABLE_STYLES["BodyText"],
    fontName="Helvetica", fontSize=8, leading=9,
    textColor=colors.HexColor("#1f2937"),
)
_NATIVE_HEADER_STYLE = ParagraphStyle(
    "NativeTableHeader",
    parent=_NATIVE_CELL_STYLE,
    fontName="Helvetica-Bold", textColor=colors.white,
)


def make_native_table(df):
    """Render a compact, crisp native ReportLab table from a DataFrame."""
    import pandas as pd

    table_data = [[Paragraph(str(col), _NATIVE_HEADER_STYLE) for col in df.columns]]
    for row in df.itertuples(index=False):
        table_data.append([
            Paragraph("" if pd.isna(value) else str(value), _NATIVE_CELL_STYLE)
            for value in row
        ])

    col_widths = None
    if len(df.columns) == 5:
        col_widths = [0.55 * inch, 0.9 * inch, 0.95 * inch, 0.95 * inch, 3.25 * inch]
    elif len(df.columns) > 0:
        col_widths = [TABLE_MAX_WIDTH / len(df.columns)] * len(df.columns)

    table = Table(table_data, repeatRows=1, hAlign="LEFT", colWidths=col_widths)
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#4472C4")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 8),
        ("LEADING", (0, 0), (-1, -1), 9),
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#9aa6b2")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.whitesmoke, colors.HexColor("#eef2f7")]),
        ("LEFTPADDING", (0, 0), (-1, -1), 4),
        ("RIGHTPADDING", (0, 0), (-1, -1), 4),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
    ]))
    return table


class PDFReportGenerator:

    def __init__(
        self,
        output_path="data/processed/drive_test_report.pdf",
        images_dir="data/images",
        processed_dir="data/processed",
        native_tables=None,
        scratch_dir=None,
    ):
        self.output_path = output_path
        self.images_dir = images_dir
        self.processed_dir = processed_dir
        # {filename: DataFrame} rendered as native tables instead of PNGs.
        self.native_tables = native_tables or {}
        # Where compressed copies of map PNGs are written.
        self.scratch_dir = scratch_dir or os.path.join(
            os.path.dirname(output_path) or ".", "_img_opt"
        )
        os.makedirs(self.scratch_dir, exist_ok=True)
        self._opt_savings = 0

        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        self.doc = TOCDocTemplate(
            output_path,
            pagesize=A4,
            rightMargin=40,
            leftMargin=40,
            topMargin=50,
            bottomMargin=50
        )

        self.story = []
        self.styles = getSampleStyleSheet()
        self._setup_styles()

    # --------------------------------------------------
    # STYLES
    # --------------------------------------------------
    def _setup_styles(self):
        self.styles.add(ParagraphStyle(
            name="CustomTitle",
            parent=self.styles["Heading1"],
            fontSize=24,
            alignment=TA_CENTER,
            spaceAfter=30
        ))

        self.styles.add(ParagraphStyle(
            name="Section",
            parent=self.styles["Heading2"],
            fontSize=16,
            spaceBefore=20,
            spaceAfter=10,
            textColor=colors.HexColor("#1f4788")
        ))

        self.styles.add(ParagraphStyle(
            name="SubSection",
            parent=self.styles["Heading3"],
            fontSize=13,
            spaceBefore=12,
            spaceAfter=6
        ))

        self.styles.add(ParagraphStyle(
            name="Body",
            parent=self.styles["BodyText"],
            fontSize=11,
            leading=15
        ))
        
        # TOC Styles
        self.styles.add(ParagraphStyle(
            name="TOCHeading",
            parent=self.styles["Heading1"],
            fontSize=18,
            textColor=colors.HexColor("#1f4788"),
            spaceAfter=12
        ))
        
        self.styles.add(ParagraphStyle(
            name="TOCLevel1",
            parent=self.styles["Normal"],
            fontSize=11,
            leftIndent=0,
            firstLineIndent=0,
            spaceBefore=3,
            spaceAfter=3,
            leading=14
        ))
        
        self.styles.add(ParagraphStyle(
            name="TOCLevel2",
            parent=self.styles["Normal"],
            fontSize=10,
            leftIndent=20,
            firstLineIndent=0,
            spaceBefore=2,
            spaceAfter=2,
            leading=12
        ))

    # --------------------------------------------------
    # HELPERS
    # --------------------------------------------------
    def add_text(self, title, text):
        if text and text.strip():
            self.story.append(Paragraph(title, self.styles["Section"]))
            self.story.append(Paragraph(text, self.styles["Body"]))
            self.story.append(Spacer(1, 12))

    def add_sub(self, title):
        self.story.append(Paragraph(title, self.styles["SubSection"]))

    def render_section(self, title: str, content: Any, level: int = 1):
        """
        Recursively render a section. `content` may be a string, dict or list.
        - string: render as a paragraph under `title`.
        - dict: if contains 'Overview' render it first; then iterate other keys
          and render each key as a subsection (dynamic, no hardcoding).
        - list: for list of dicts with 'name' use name as subsection title.
        """
        # choose style for title based on level
        if level == 1:
            title_style = self.styles["Section"]
        else:
            title_style = self.styles["SubSection"]

        # Skip rendering entirely if there's no content to avoid empty headings
        if content is None:
            return
        if isinstance(content, str) and not content.strip():
            return
        if isinstance(content, (dict, list)) and len(content) == 0:
            return

        # Render title
        if title:
            # Add bookmark/anchor for TOC
            if level == 1:
                title_para = Paragraph(f'<a name="{title}"/>{title}', title_style)
            else:
                title_para = Paragraph(f'<a name="{title}"/>{title}', title_style)
            self.story.append(title_para)

        # Render string content
        if isinstance(content, str):
            if content.strip():
                self.story.append(Paragraph(content, self.styles["Body"]))
                self.story.append(Spacer(1, 6))  # Reduced from 8 to 6
            return

        # Render dict content
        if isinstance(content, dict):
            # overview first
            overview = None
            for k in ("Overview", "overview"):
                if k in content and isinstance(content[k], str):
                    overview = content[k]
                    break
            if overview:
                self.story.append(Paragraph(overview, self.styles["Body"]))
                self.story.append(Spacer(1, 8))

            # Fallback: iterate keys dynamically, skip any overview/summary keys already rendered
            for k, v in content.items():
                if k in ("Overview", "overview"):
                    continue
                if k in ("Conclusion", "conclusion"):
                    # render at end
                    continue

                # If the value is string, render as subsection with that title
                if isinstance(v, str):
                    self.story.append(Paragraph(k, self.styles["SubSection"]))
                    self.story.append(Paragraph(v, self.styles["Body"]))
                    self.story.append(Spacer(1, 4))  # Reduced from 6 to 4
                else:
                    # nested structure: recurse
                    self.render_section(k, v, level=level + 1)

            # conclusion if present
            concl = content.get("Conclusion") or content.get("conclusion")
            if concl and isinstance(concl, str):
                self.story.append(Paragraph(concl, self.styles["Body"]))
                self.story.append(Spacer(1, 10))
            return

        # Render list content
        if isinstance(content, list):
            # If it's a list of dicts with 'name', render each as subsection
            rendered = False
            for item in content:
                if isinstance(item, dict) and item.get("name"):
                    name = item.get("name")
                    # derive a body text for the item
                    body = item.get("description") or item.get("text") or item.get("summary") or json.dumps(item)
                    self.story.append(Paragraph(name, self.styles["SubSection"]))
                    if isinstance(body, str) and body.strip():
                        self.story.append(Paragraph(body, self.styles["Body"]))
                    self.story.append(Spacer(1, 6))
                    rendered = True
                elif isinstance(item, str):
                    self.story.append(Paragraph(item, self.styles["Body"]))
                    rendered = True

            if not rendered:
                # fallback: render short joined representation
                joined = ", ".join(str(i) for i in content[:12])
                self.story.append(Paragraph(joined, self.styles["Body"]))
            self.story.append(Spacer(1, 8))
            return

        # fallback
        if content is not None:
            self.story.append(Paragraph(str(content), self.styles["Body"]))
            self.story.append(Spacer(1, 4))  # Reduced from 8

    def _append_visual(self, visual):
        """Bind a heading to the visual that immediately follows it (KeepTogether)."""
        if self.story and isinstance(self.story[-1], Paragraph) and hasattr(self.story[-1], "toc_level"):
            heading = self.story.pop()
            keep = KeepTogether([heading, Spacer(1, 4), visual, Spacer(1, 6)])
            keep.toc_level = heading.toc_level
            keep.toc_text = heading.toc_text
            keep.toc_key = heading.toc_key
            self.story.append(keep)
            return
        self.story.append(visual)
        self.story.append(Spacer(1, 6))

    def _compress_png(self, path):
        """Palette-quantize + optimize a map PNG to shrink file size; keep .png."""
        try:
            out = os.path.join(self.scratch_dir, os.path.basename(path))
            with PILImage.open(path) as im:
                q = im.convert("RGB").quantize(colors=256, method=PILImage.FASTOCTREE)
                q.save(out, format="PNG", optimize=True)
            before, after = os.path.getsize(path), os.path.getsize(out)
            if after < before:
                self._opt_savings += (before - after)
                return out
            return path
        except Exception as exc:
            print(f" Warning: image compress failed for {path}: {exc}")
            return path

    def _place_image(self, path, max_width, max_height):
        img = Image(path)
        iw, ih = img.imageWidth, img.imageHeight
        scale = min(max_width / iw, max_height / ih, 1.0)
        img.drawWidth = iw * scale
        img.drawHeight = ih * scale
        self._append_visual(img)

    def add_image(self, filename, subdir="kpi_maps", max_height=4*inch):
        # Native table takes priority — crisp, selectable, tiny.
        if filename in self.native_tables:
            table_df = self.native_tables[filename]
            if table_df is not None and not table_df.empty:
                self._append_visual(make_native_table(table_df))
            return

        path = os.path.join(self.images_dir, subdir, filename)
        if not os.path.exists(path):
            print(f" Warning: missing {path}")
            return

        if filename in TABLE_IMAGE_FILES:
            # Table image fallback: full width, no 4-inch squeeze.
            self._place_image(path, TABLE_MAX_WIDTH, TABLE_MAX_HEIGHT)
        elif filename in CHART_IMAGE_FILES:
            # Charts: full quality (no quantization), full width.
            self._place_image(path, TABLE_MAX_WIDTH, max_height)
        else:
            # Maps: palette-compress to shrink the PDF, then place.
            self._place_image(self._compress_png(path), 5.8 * inch, max_height)

    def has_image(self, filename, subdir="kpi_maps"):
        if filename in self.native_tables:
            return True
        return os.path.exists(os.path.join(self.images_dir, subdir, filename))

    def has_text(self, text):
        return isinstance(text, str) and text.strip() and text.strip().lower() != "not available."

    def has_any_image(self, images):
        return any(self.has_image(**img) for img in images)

    def add_image_section(self, title, key, images, page_break=False, cond_break=False):
        if not self.has_any_image(images):
            return False
        if page_break:
            self.story.append(PageBreak())
        elif cond_break:
            self.story.append(CondPageBreak(3 * inch))
        self.add_toc_heading(title, self.styles["Section"], 0, key)
        for img in images:
            self.add_image(**img)
        return True

    def add_image_subsection(self, title, key, images, toc_text=None):
        if not self.has_any_image(images):
            return False

        # Native poor-PCI tables: show "Top N of M PCIs" in the heading.
        if len(images) == 1 and images[0]["filename"] in self.native_tables:
            filename = images[0]["filename"]
            table_df = self.native_tables[filename]
            if table_df is None or table_df.empty:
                return False
            display_title = title
            if filename in {"pci_poor_rsrp.png", "pci_poor_rsrq.png"}:
                shown = len(table_df)
                total = int(table_df.attrs.get("total_poor_pci", shown))
                suffix = f"Top {shown} of {total} PCIs" if total > shown else f"{shown} PCIs"
                display_title = f"{title} ({suffix})"
            self.add_toc_heading(display_title, self.styles["SubSection"], 1, key, toc_text or display_title)
            self.add_image(**images[0])
            return True

        self.add_toc_heading(title, self.styles["SubSection"], 1, key, toc_text or title)
        for img in images:
            self.add_image(**img)
        return True

    # --------------------------------------------------
    # COVER
    # --------------------------------------------------
    def add_cover(self, metadata):
        self.story.append(Spacer(1, 2*inch))
        self.story.append(Paragraph("Drive Test Report", self.styles["CustomTitle"]))

        loc = metadata.get("location", {})
        self.story.append(Paragraph(
            f"{loc.get('city','')} , {loc.get('country','')}",
            self.styles["Body"]
        ))

        self.story.append(Spacer(1, 12))
        self.story.append(Paragraph(
            f"Generated on {datetime.now().strftime('%B %d, %Y')}",
            self.styles["Body"]
        ))

        self.story.append(PageBreak())
        
    def add_table_of_contents(self):
        """Add professional TOC with dotted lines and dynamic page numbers"""
        # Title
        self.story.append(Paragraph("<b>Table of Contents</b>", self.styles["TOCHeading"]))
        self.story.append(Spacer(1, 0.2 * inch))
        
        # Create TOC object with proper formatting
        from reportlab.platypus.tableofcontents import TableOfContents
        toc = TableOfContents()
        
        # Use existing TOC styles from _setup_styles
        toc.levelStyles = [
            self.styles['TOCLevel1'],
            self.styles['TOCLevel2'],
        ]
        
        self.story.append(toc)
        self.story.append(PageBreak())

    def add_toc_heading(self, text, style, level, key, toc_text=None):
        """Add a heading paragraph and register it for TOC via afterFlowable."""
        para = Paragraph(text, style)
        para.toc_level = level
        para.toc_text = toc_text or text
        para.toc_key = key
        self.story.append(para)

    # --------------------------------------------------
    # REPORT
    # --------------------------------------------------
    def generate_report(self, report_text, metadata, verbose=False):

        self.add_cover(metadata)
        self.add_table_of_contents()

        # 1. Introduction
        self.add_toc_heading('1. Introduction', self.styles["Section"], 0, "sec1")
        if report_text.get("Introduction"):
            self.story.append(Paragraph(report_text["Introduction"], self.styles["Body"]))
        
        # 2. Area Summary
        area_summary = report_text.get("Area Summary", "")
        if area_summary or self.has_image("base_route_map.png"):
            self.story.append(CondPageBreak(3 * inch))
            self.add_toc_heading('2. Area Summary', self.styles["Section"], 0, "sec2")
            self.render_section(None, area_summary)
            self.add_image("base_route_map.png")
        
        # 3. Drive Summary
        drive_text = report_text.get("Drive Summary")
        if self.has_text(drive_text) or self.has_image("drive_summary.png", subdir="kpi_analysis"):
            self.story.append(PageBreak())
            self.add_toc_heading('3. Drive Summary', self.styles["Section"], 0, "sec3")
            if self.has_text(drive_text):
                self.story.append(Paragraph(drive_text, self.styles["Body"]))
            self.add_image("drive_summary.png", subdir="kpi_analysis")

        # 4. KPI Summary
        kpi_summary_images = [
            {"filename": "kpi_summary.png", "subdir": "kpi_analysis"},
            {"filename": "session_table.png", "subdir": "kpi_analysis"},
        ]
        kpi_text = report_text.get("KPI Summary")
        if not kpi_text or not isinstance(kpi_text, str) or not kpi_text.strip():
            kpi_text = (
                "A concise executive summary of overall KPI performance across the drive, "
                "highlighting overall network health and any major issues observed."
            )
        if self.has_text(kpi_text) or self.has_any_image(kpi_summary_images):
            self.story.append(CondPageBreak(3 * inch))
            self.add_toc_heading('4. KPI Summary', self.styles["Section"], 0, "sec4")
            if self.has_text(kpi_text):
                self.story.append(Paragraph(kpi_text, self.styles["Body"]))
            for img in kpi_summary_images:
                self.add_image(**img)

        # 5. MAP VIEW
        map_view_started = False

        def ensure_map_view():
            nonlocal map_view_started
            if not map_view_started:
                self.story.append(PageBreak())
                self.add_toc_heading('5. Map View', self.styles["Section"], 0, "sec5")
                map_view_started = True

        def kpi_sub(label, anchor, toc_label, text, imgs):
            has_text = self.has_text(text)
            has_any_image = any(self.has_image(**img) for img in imgs)
            if not has_text and not has_any_image:
                return
            ensure_map_view()
            self.add_toc_heading(label, self.styles["SubSection"], 1, anchor, toc_label)
            if has_text:
                self.story.append(Paragraph(text, self.styles["Body"]))
                self.story.append(Spacer(1, 6))
            for img in imgs:
                self.add_image(**img)

        # a) Band
        kpi_sub("a) Band", "sec5a", "a) Band", report_text.get("Map View - Band", ""), [
            {"filename": "band_map.png"},
            {"filename": "band_pie.png", "subdir": "kpi_analysis"},
            {"filename": "band_table.png", "subdir": "kpi_analysis"},
        ])

        # b) RSRP
        kpi_sub("b) RSRP", "sec5b", "b) RSRP", report_text.get("Map View - RSRP", ""), [
            {"filename": "rsrp_map.png"},
            {"filename": "rsrp_poor_regions.png"},
            {"filename": "rsrp_range_table.png", "subdir": "kpi_analysis"},
            {"filename": "cdf_rsrp.png", "subdir": "kpi_analysis"},
        ])

        # c) RSRQ
        kpi_sub("c) RSRQ", "sec5c", "c) RSRQ", report_text.get("Map View - RSRQ", ""), [
            {"filename": "rsrq_map.png"},
            {"filename": "rsrq_poor_regions.png"},
            {"filename": "rsrq_range_table.png", "subdir": "kpi_analysis"},
            {"filename": "cdf_rsrq.png", "subdir": "kpi_analysis"},
        ])

        # d) SINR
        kpi_sub("d) SINR", "sec5d", "d) SINR", report_text.get("Map View - SINR", ""), [
            {"filename": "sinr_map.png"},
            {"filename": "sinr_range_table.png", "subdir": "kpi_analysis"},
            {"filename": "cdf_sinr.png", "subdir": "kpi_analysis"},
        ])

        # e) DL Throughput
        kpi_sub("e) DL Throughput", "sec5e", "e) DL Throughput", report_text.get("Map View - DL Throughput", ""), [
            {"filename": "dl_map.png"},
            {"filename": "dl_range_table.png", "subdir": "kpi_analysis"},
            {"filename": "cdf_dl_tpt.png", "subdir": "kpi_analysis"},
        ])

        # f) UL Throughput
        kpi_sub("f) UL Throughput", "sec5f", "f) UL Throughput", report_text.get("Map View - UL Throughput", ""), [
            {"filename": "ul_map.png"},
            {"filename": "ul_range_table.png", "subdir": "kpi_analysis"},
            {"filename": "cdf_ul_tpt.png", "subdir": "kpi_analysis"},
        ])

        # g) MOS
        kpi_sub("g) MOS", "sec5g", "g) MOS", report_text.get("Map View - MOS", ""), [
            {"filename": "mos_map.png"},
            {"filename": "mos_range_table.png", "subdir": "kpi_analysis"},
            {"filename": "cdf_mos.png", "subdir": "kpi_analysis"},
        ])

        # 6. PCI Summary
        pci_main_images = [
            {"filename": "pci_map.png"},
            {"filename": "pci_distribution.png", "subdir": "kpi_analysis"},
            {"filename": "cdf_pci.png", "subdir": "kpi_analysis"},
        ]
        pci_subsections = [
            ('a) Top 30 PCI Values', "sec6a", [{"filename": "pci_table.png", "subdir": "kpi_analysis"}]),
            ('b) PCI with Poor RSRP', "sec6b", [{"filename": "pci_poor_rsrp.png", "subdir": "kpi_analysis"}]),
            ('c) PCI with Poor RSRQ', "sec6c", [{"filename": "pci_poor_rsrq.png", "subdir": "kpi_analysis"}]),
        ]
        pci_text = report_text.get("PCI Summary")
        has_pci_content = (
            self.has_text(pci_text)
            or self.has_any_image(pci_main_images)
            or any(self.has_any_image(images) for _, _, images in pci_subsections)
        )
        if has_pci_content:
            self.story.append(PageBreak())
            self.add_toc_heading('6. PCI Summary', self.styles["Section"], 0, "sec6")
            if self.has_text(pci_text):
                self.story.append(Paragraph(pci_text, self.styles["Body"]))
            for img in pci_main_images:
                self.add_image(**img)
            for title, key, images in pci_subsections:
                self.add_image_subsection(title, key, images)

        # 7. App Analytics
        self.add_image_section('7. App Analytics', "sec7", [
            {"filename": "app_analytics_part1.png", "subdir": "kpi_analysis"},
            {"filename": "app_analytics_part2.png", "subdir": "kpi_analysis"},
        ], cond_break=True)

        # 8. Indoor/Outdoor Summary
        self.add_image_section('8. Indoor/Outdoor Summary', "sec8", [
            {"filename": "indoor_outdoor_stats.png", "subdir": "kpi_analysis"},
        ], cond_break=True)

        # 9. Performance Summary
        performance_subsections = [
            ('a) Network Quality Metrics', "sec9a", [{"filename": "network_quality_summary.png", "subdir": "kpi_analysis"}]),
            ('b) Speed Metrics', "sec9b", [{"filename": "speed_hist.png", "subdir": "kpi_analysis"}]),
            ('c) Latency Distribution', "sec9c", [{"filename": "latency_hist.png", "subdir": "kpi_analysis"}]),
            ('d) Jitter Distribution', "sec9d", [{"filename": "jitter_hist.png", "subdir": "kpi_analysis"}]),
        ]
        if any(self.has_any_image(images) for _, _, images in performance_subsections):
            self.story.append(CondPageBreak(3 * inch))
            self.add_toc_heading('9. Performance Summary', self.styles["Section"], 0, "sec9")
            for title, key, images in performance_subsections:
                self.add_image_subsection(title, key, images)

        # 10. Handover Analysis
        if self.has_image("handover_map.png"):
            self.story.append(PageBreak())
            self.add_toc_heading('10. Handover Analysis', self.styles["Section"], 0, "sec10")
            self.add_image("handover_map.png")

        # Build with multiple passes for TOC and page numbers
        self.doc.multiBuild(self.story, canvasmaker=PageNumCanvas)


def generate_pdf_report(
    metadata_path="data/processed/report_metadata.json",
    report_text_path="data/processed/report_text.json",
    output_path="data/processed/drive_test_report.pdf",
    images_dir="data/images",
    verbose=False,
    native_tables=None,
    scratch_dir=None,
):
    with open(metadata_path) as f:
        metadata = json.load(f)
    with open(report_text_path) as f:
        report_text = json.load(f)

    gen = PDFReportGenerator(
        output_path, images_dir,
        native_tables=native_tables, scratch_dir=scratch_dir,
    )
    gen.generate_report(report_text, metadata, verbose)
    if native_tables:
        print(f" [PDF] image optimization saved ~{gen._opt_savings / (1024 * 1024):.1f} MB")
    return output_path


def main():
    generate_pdf_report(verbose=True)


if __name__ == "__main__":
    main()

