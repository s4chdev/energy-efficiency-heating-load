"""Render report.md into report.pdf.

Usage: python build_report.py
Requires: markdown, xhtml2pdf (pip install markdown xhtml2pdf)
"""

from __future__ import annotations

import re
from pathlib import Path

import markdown
from xhtml2pdf import pisa

ROOT = Path(__file__).parent
SOURCE = ROOT / "report.md"
OUTPUT = ROOT / "report.pdf"

CSS = """
@page { size: A4; margin: 2cm 1.9cm 2cm 1.9cm;
        @bottom-center { content: counter(page); font-size: 8pt; color: #666; } }
body { font-family: Helvetica, Arial, sans-serif; font-size: 9.6pt;
       line-height: 1.45; color: #1c1c1c; }
h1 { font-size: 15pt; color: #1F3864; border-bottom: 1.5pt solid #1F3864;
     padding-bottom: 3pt; margin: 16pt 0 8pt 0; page-break-after: avoid; }
h2 { font-size: 11.5pt; color: #1F3864; margin: 12pt 0 5pt 0;
     page-break-after: avoid; }
h3 { font-size: 10.2pt; color: #2E5395; margin: 10pt 0 4pt 0;
     page-break-after: avoid; }
p { margin: 0 0 6pt 0; text-align: justify; }
ul, ol { margin: 0 0 7pt 0; padding-left: 16pt; }
li p { text-align: left; }
li { margin-bottom: 2pt; text-align: left; }
table { border-collapse: collapse; width: 100%; margin: 6pt 0 10pt 0;
        font-size: 8.5pt; page-break-inside: avoid; }
th { background: #1F3864; color: white; text-align: left; padding: 4pt 5pt;
     border: 0.5pt solid #1F3864; font-size: 8.5pt; }
td { padding: 3.5pt 5pt; border: 0.5pt solid #a8b4c8; vertical-align: top; }
tr:nth-child(even) td { background: #f2f5fa; }
code { font-family: Courier, monospace; font-size: 8.6pt; background: #eef1f6;
       padding: 0 1.5pt; }
pre { background: #f5f6f8; border-left: 2.5pt solid #4a7ebb; padding: 6pt 8pt;
      font-family: Courier, monospace; font-size: 8.2pt; line-height: 1.3;
      page-break-inside: avoid; }
pre code { background: none; }
img { max-width: 100%; }
em { color: #444; }
.titleblock { text-align: center; margin-bottom: 18pt; }
.titleblock .doctitle { font-size: 19pt; font-weight: bold; color: #1F3864;
                        margin-bottom: 5pt; line-height: 1.25; }
.titleblock .subtitle { font-size: 10.5pt; color: #444; font-style: italic;
                        margin-bottom: 12pt; }
.titleblock .authors { font-size: 10pt; margin-bottom: 6pt; }
.titleblock .date { font-size: 9pt; color: #666; }
hr { border: none; border-top: 0.5pt solid #bbb; }
"""


def parse_front_matter(text: str) -> tuple[dict, str]:
    if not text.startswith("---\n"):
        return {}, text
    end = text.index("\n---\n", 4)
    block, body = text[4:end], text[end + 5 :]
    meta: dict[str, str] = {}
    key = None
    for line in block.splitlines():
        if re.match(r"^\s+- ", line) and key:
            meta.setdefault(key, "")
            meta[key] += ("|" if meta[key] else "") + line.strip()[2:].strip().strip('"')
        elif ":" in line:
            key, value = line.split(":", 1)
            key = key.strip()
            meta[key] = value.strip().strip('"')
    return meta, body


def build_html(text: str) -> str:
    meta, body = parse_front_matter(text)
    body = body.split("\n# 1. Title Page", 1)[1]
    body = "# 1. Title Page" + body
    # The cover block below already carries the title, so drop the duplicated
    # heading body of section 1 and keep only the course/date line.
    body = body.replace(
        "**Predicting Building Heating Load: An End-to-End Machine Learning Mini Project**\n\n"
        "Energy Efficiency Dataset (UCI ENB2012)\n\n",
        "",
    )
    body = body.replace("(2–4 members as permitted)", "(2-4 members as permitted)")
    html = markdown.markdown(
        body, extensions=["tables", "fenced_code", "sane_lists", "toc"]
    )
    # xhtml2pdf collapses the space that follows an inline </code>; put it back.
    html = re.sub(r"(</code>)(?![ \n.,;:)</])", r"\1 ", html)
    title_block = ""
    if meta.get("title"):
        authors = meta.get("author", "").split("|")
        authors_html = "<br/>".join(a.strip().strip('"') for a in authors if a.strip())
        title_block = f"""
        <div class="titleblock">
          <div class="doctitle">{meta['title']}</div>
          <div class="subtitle">{meta.get('subtitle', '')}</div>
          <div class="authors">{authors_html}</div>
          <div class="date">{meta.get('date', '')}</div>
        </div>
        <hr/>
        """
    return f"<html><head><style>{CSS}</style></head><body>{title_block}{html}</body></html>"


def main() -> None:
    html = build_html(SOURCE.read_text())
    (ROOT / "report.html").write_text(html)
    with OUTPUT.open("wb") as fh:
        result = pisa.CreatePDF(html, dest=fh, base_url=str(ROOT))
    if result.err:
        raise SystemExit(f"PDF generation reported {result.err} error(s)")
    print(f"Wrote {OUTPUT} ({OUTPUT.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
