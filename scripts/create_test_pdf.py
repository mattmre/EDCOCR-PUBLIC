#!/usr/bin/env python3
"""Generate a small 2-page test PDF for OCR pipeline canary testing.

Output: tests/fixtures/test-canary-2page.pdf

Requires: pip install fpdf2
"""

import os

from fpdf import FPDF


def create_canary_pdf(output_path: str) -> None:
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=25)

    # --- Page 1: Title + paragraph ---
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 24)
    pdf.cell(0, 15, "OCR Canary Test Document", new_x="LMARGIN", new_y="NEXT", align="C")
    pdf.ln(10)

    pdf.set_font("Helvetica", "", 12)
    paragraph = (
        "This document serves as a canary test artifact for the Industrial Adaptive "
        "OCR pipeline. It contains two pages of structured content designed to validate "
        "end-to-end processing through extraction, optical character recognition, "
        "assembly, and compression stages. The first page provides a block of plain "
        "English text to verify that the PaddleOCR engine can accurately detect and "
        "transcribe standard typeset characters at three hundred dots per inch. "
        "Successful recognition of this paragraph confirms that the GPU worker, "
        "language detection, and text-layer embedding stages are functioning correctly. "
        "The second page contains a tabular data sample that exercises the layout "
        "analysis and structured extraction components of the pipeline."
    )
    pdf.multi_cell(0, 7, paragraph)

    # --- Page 2: Table ---
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 18)
    pdf.cell(0, 12, "Sample Data Table", new_x="LMARGIN", new_y="NEXT", align="C")
    pdf.ln(8)

    # Table header
    col_widths = [60, 50, 50]
    headers = ["Name", "Date", "Amount"]
    pdf.set_font("Helvetica", "B", 12)
    for i, header in enumerate(headers):
        pdf.cell(col_widths[i], 10, header, border=1, align="C")
    pdf.ln()

    # Table rows
    rows = [
        ("Alice Johnson", "2026-01-15", "$1,250.00"),
        ("Robert Chen", "2026-02-03", "$3,780.50"),
        ("Maria Garcia", "2026-02-28", "$925.75"),
    ]
    pdf.set_font("Helvetica", "", 12)
    for row in rows:
        for i, cell in enumerate(row):
            pdf.cell(col_widths[i], 10, cell, border=1, align="C")
        pdf.ln()

    dirname = os.path.dirname(output_path)
    if dirname:
        os.makedirs(dirname, exist_ok=True)
    pdf.output(output_path)
    print(f"Created: {output_path} ({os.path.getsize(output_path)} bytes)")


if __name__ == "__main__":
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    out = os.path.join(root, "tests", "fixtures", "test-canary-2page.pdf")
    create_canary_pdf(out)
