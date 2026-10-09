"""Render results/final2/FINAL2_REPORT.md to a single PDF.

Uses markdown -> HTML -> reportlab (no system LaTeX needed). Tables and code
blocks are rendered; long lines wrap. Reads only the generated files.
"""

import os
import re
import html

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
R = os.path.join(ROOT, "results", "final2")
SRC = os.path.join(R, "FINAL2_REPORT.md")
DST = os.path.join(R, "FINAL2_REPORT.pdf")


def parse(md_text):
    """Return a list of flowable specs from a small subset of markdown."""
    import markdown as mdlib
    # split into blocks; render tables with the markdown lib, then convert to
    # reportlab Tables, everything else to Paragraphs.
    lines = md_text.splitlines()
    blocks = []
    i = 0
    while i < len(lines):
        ln = lines[i]
        if ln.startswith("```"):
            buf = []
            i += 1
            while i < len(lines) and not lines[i].startswith("```"):
                buf.append(lines[i]); i += 1
            i += 1
            blocks.append(("code", "\n".join(buf)))
        elif ln.startswith("|") and i + 1 < len(lines) and set(lines[i + 1].replace("|", "").strip()) <= set("-: "):
            buf = [ln]; i += 1
            while i < len(lines) and lines[i].startswith("|"):
                buf.append(lines[i]); i += 1
            blocks.append(("table", buf))
        elif ln.strip() == "":
            i += 1
        else:
            buf = [ln]; i += 1
            while i < len(lines) and lines[i].strip() != "" and not lines[i].startswith(("|", "```")):
                buf.append(lines[i]); i += 1
            blocks.append(("text", "\n".join(buf)))
    return blocks


def main():
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib import colors
    from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer, Table,
                                    TableStyle, Preformatted)
    from reportlab.lib.units import mm

    ss = getSampleStyleSheet()
    h1 = ParagraphStyle("h1", parent=ss["Heading1"], fontSize=16, spaceAfter=8)
    h2 = ParagraphStyle("h2", parent=ss["Heading2"], fontSize=12.5, spaceAfter=6)
    h3 = ParagraphStyle("h3", parent=ss["Heading3"], fontSize=10.5)
    body = ParagraphStyle("body", parent=ss["BodyText"], fontSize=8.4, leading=11)
    cell = ParagraphStyle("cell", parent=ss["BodyText"], fontSize=6.2, leading=7.4)
    code = ParagraphStyle("code", parent=ss["Code"], fontSize=7, leading=8.5)

    def inline(t):
        t = html.escape(t)
        t = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", t)
        t = re.sub(r"`(.+?)`", r"<font face='Courier'>\1</font>", t)
        return t

    story = []
    for kind, payload in parse(open(SRC).read()):
        if kind == "code":
            story.append(Preformatted(payload, code)); story.append(Spacer(1, 4))
        elif kind == "table":
            rows = []
            for r_i, row in enumerate(payload):
                if r_i == 1:
                    continue
                cells = [c.strip() for c in row.strip().strip("|").split("|")]
                rows.append([Paragraph(inline(c), cell) for c in cells])
            t = Table(rows, repeatRows=1)
            t.setStyle(TableStyle([
                ("GRID", (0, 0), (-1, -1), 0.3, colors.grey),
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#eef1f5")),
                ("FONTSIZE", (0, 0), (-1, -1), 6.2),
                ("VALIGN", (0, 0), (-1, -1), "TOP")]))
            story.append(t); story.append(Spacer(1, 6))
        else:
            for para in payload.split("\n"):
                para = para.rstrip()
                if not para:
                    continue
                if para.startswith("# "):
                    story.append(Paragraph(inline(para[2:]), h1))
                elif para.startswith("## "):
                    story.append(Paragraph(inline(para[3:]), h2))
                elif para.startswith("### "):
                    story.append(Paragraph(inline(para[4:]), h3))
                elif re.match(r"^\d+\.\s", para):
                    story.append(Paragraph(inline(para), body))
                else:
                    story.append(Paragraph(inline(para), body))
            story.append(Spacer(1, 3))

    doc = SimpleDocTemplate(DST, pagesize=A4,
                            leftMargin=14 * mm, rightMargin=14 * mm,
                            topMargin=14 * mm, bottomMargin=14 * mm,
                            title="RAPT Final-2 Report", author="Anonymous")
    doc.build(story)
    print("wrote", DST)


if __name__ == "__main__":
    main()
