"""PDFVisionLoader showcase.

Demonstrates the major features of the vision-based PDF loader:
- Page rendering + extraction via a multimodal chat model
- Page selection (`pages=[...]`) and performance controls (`dpi`, `model_timeout_s`)
- Optional extraction of headers/footers/printed page number (structured fields)
- PDF metadata extraction
- TOC extraction (embedded TOC or best-effort vision scan across multiple pages)
- Multiple output formats: Markdown, HTML, JSON

Prereqs:
- Copy `examples/.env.example` -> `examples/.env` and set your provider + API key.
- Install PyMuPDF: `uv add pymupdf`

Run:
    uv run python examples/retrieval/pdf_vision_showcase.py

Notes:
- This example makes multiple model calls (Markdown + HTML + JSON). If you're rate-limited,
  reduce the number of pages, raise backoff, or switch providers.
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

# Add examples directory to path for config import
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import get_model

from agenticflow.document.loaders import PDFVisionLoader
from agenticflow.document.loaders.handlers.pdf_vision import OutputFormat


def _print_doc_header(title: str) -> None:
    print("\n" + "=" * 90)
    print(title)
    print("=" * 90)


def _safe_write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _wrap_html_fragment(fragment: str, *, title: str) -> str:
    # Minimal wrapper for browser viewing. No external assets.
    return (
        "<!doctype html>\n"
        "<html lang=\"en\">\n"
        "<head>\n"
        "  <meta charset=\"utf-8\" />\n"
        "  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\" />\n"
        f"  <title>{title}</title>\n"
        "  <style>body{font-family:system-ui,-apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif;line-height:1.45;margin:24px;max-width:980px}</style>\n"
        "</head>\n"
        "<body>\n"
        f"{fragment}\n"
        "</body>\n"
        "</html>\n"
    )


async def main() -> None:
    data_dir = Path(__file__).parent.parent / "data"
    pdf_path = data_dir / "financial_report.pdf"

    out_dir = data_dir / "pdf_output" / "pdf_vision_showcase" / pdf_path.stem
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[output] Writing extracted files to: {out_dir}")

    if not pdf_path.exists():
        raise FileNotFoundError(f"Missing example PDF: {pdf_path}")

    model = get_model()
    print(
        f"Using model: {getattr(model, 'model', None) or getattr(model, 'deployment', None) or type(model).__name__}"
    )

    # Best-effort: keep provider calls bounded.
    try:
        model.timeout = 90.0  # type: ignore[attr-defined]
    except Exception:
        pass

    # This example is intentionally heavyweight: multiple extractions + optional TOC vision scan.
    # If the PDF has no embedded TOC, TOC extraction can add up to `toc_max_pages` extra model calls.
    md_pages = 2
    html_pages = 1
    json_pages = 1
    toc_max_pages = 5
    print(
        "\n[perf] Expected model calls (worst-case): "
        f"markdown={md_pages}, html={html_pages}, json={json_pages}, toc_scan<= {toc_max_pages} "
        f"(total<= {md_pages + html_pages + json_pages + toc_max_pages})"
    )

    # 1) Markdown extraction + metadata + TOC (embedded or vision) + timing
    _print_doc_header("1) MARKDOWN + METADATA + TOC + TIMING (pages 1-2)")
    loader_md = PDFVisionLoader(model=model, output_format=OutputFormat.MARKDOWN, dpi=110)

    docs_md = await loader_md.load(
        pdf_path,
        pages=[1, 2],
        extract_metadata=True,
        extract_toc=True,
        toc_max_pages=toc_max_pages,
        timing=True,
        model_timeout_s=90,
        include_printed_page_number=True,
    )

    # Save Markdown outputs for visual verification (VS Code Markdown Preview).
    combined_md = "\n\n---\n\n".join(
        [f"# Page {d.metadata.get('page')}\n\n{d.text}" for d in docs_md]
    )
    _safe_write_text(out_dir / "extracted_markdown_pages_1_2.md", combined_md)

    first = docs_md[0]
    print(f"source: {first.metadata.get('source')}")
    print(f"output_format: {first.metadata.get('output_format')}")

    if "pdf_metadata" in first.metadata:
        print("--- PDF metadata (subset) ---")
        pdf_meta = first.metadata["pdf_metadata"]
        for k in ("title", "author", "subject", "keywords", "producer", "creator"):
            if pdf_meta.get(k):
                print(f"{k}: {pdf_meta.get(k)}")

    toc = first.metadata.get("toc")
    if isinstance(toc, list):
        print(f"--- TOC entries: {len(toc)} ---")
        for entry in toc[:20]:
            print(f"- L{entry.get('level')}: {entry.get('title')} (p{entry.get('page')})")
        _safe_write_text(out_dir / "toc.json", json.dumps(toc, ensure_ascii=False, indent=2))

    timing = first.metadata.get("timing")
    if isinstance(timing, dict):
        print("--- Timing ---")
        if timing.get("pdf_open_ms") is not None:
            print(f"pdf_open_ms: {timing.get('pdf_open_ms'):.1f}")
        if timing.get("pdf_metadata_ms") is not None:
            print(f"pdf_metadata_ms: {timing.get('pdf_metadata_ms'):.1f}")
        toc_info = timing.get("toc")
        if isinstance(toc_info, dict) and toc_info.get("ms") is not None:
            print(
                f"toc_ms: {toc_info.get('ms'):.1f} (source={toc_info.get('source')}, entries={toc_info.get('entries')})"
            )
            attempts = toc_info.get("attempts")
            if isinstance(attempts, list) and attempts:
                for a in attempts:
                    print(
                        f"  toc_page={a.get('page')} render_ms={a.get('render_ms'):.1f} model_ms={a.get('model_ms'):.1f} "
                        f"is_toc={a.get('is_toc')} entries={a.get('entries')} timeout={a.get('timeout')}"
                    )

        pages_timing = timing.get("pages")
        if isinstance(pages_timing, list) and pages_timing:
            print("--- Per-page timing (render + model) ---")
            for p in pages_timing:
                print(
                    f"page={p.get('page')} render_ms={p.get('render_ms'):.1f} model_ms={p.get('model_ms'):.1f} "
                    f"png_bytes={p.get('bytes')}"
                )
        if timing.get("total_ms") is not None:
            print(f"total_ms: {timing.get('total_ms'):.1f}")

    for d in docs_md:
        print("\n--- PAGE (markdown) ---")
        print(f"pdf page: {d.metadata.get('page')} / {d.metadata.get('total_pages')}")
        print(f"printed page number: {d.metadata.get('printed_page_number')}")
        print(d.text[:1200])

    # 2) HTML extraction (fragment) + structured fields
    _print_doc_header("2) HTML (fragment) + HEADER/FOOTER + PRINTED PAGE NUMBER (page 1)")
    loader_html = PDFVisionLoader(model=model, output_format=OutputFormat.HTML, dpi=160)
    docs_html = await loader_html.load(
        pdf_path,
        pages=[1],
        include_headers=True,
        include_footers=True,
        include_printed_page_number=True,
        extract_metadata=False,
        extract_toc=False,
        model_timeout_s=90,
    )
    doc_html = docs_html[0]
    print(f"header: {doc_html.metadata.get('header')}")
    print(f"footer: {doc_html.metadata.get('footer')}")
    print(f"printed page number: {doc_html.metadata.get('printed_page_number')}")
    print("--- HTML excerpt ---")
    print(doc_html.text[:1200])

    # Save HTML fragment and a wrapped HTML document for browser viewing.
    _safe_write_text(out_dir / "extracted_page_1_fragment.html", doc_html.text)
    _safe_write_text(
        out_dir / "extracted_page_1_view.html",
        _wrap_html_fragment(doc_html.text, title=f"{pdf_path.name} - page 1 (pdf_vision HTML)"),
    )

    # 3) JSON output format (full JSON payload in Document.text)
    _print_doc_header("3) JSON OUTPUT (full payload) (page 1)")
    loader_json = PDFVisionLoader(model=model, output_format=OutputFormat.JSON, dpi=160)
    docs_json = await loader_json.load(
        pdf_path,
        pages=[1],
        include_headers=True,
        include_footers=True,
        include_printed_page_number=True,
        extract_metadata=False,
        extract_toc=False,
        model_timeout_s=90,
    )
    doc_json = docs_json[0]
    print("--- Raw JSON excerpt ---")
    print(doc_json.text[:1200])

    # Save JSON output for inspection.
    _safe_write_text(out_dir / "extracted_page_1.json", doc_json.text)

    try:
        parsed = json.loads(doc_json.text)
        print("--- Parsed JSON keys ---")
        print(
            ", ".join(k for k in ("content", "header", "footer", "printed_page_number", "error") if k in parsed)
        )

        # Also save the inner content field if present (makes quick eyeballing easier).
        content = parsed.get("content")
        if isinstance(content, str) and content.strip():
            _safe_write_text(out_dir / "extracted_page_1_from_json_content.txt", content)
    except Exception:
        pass


if __name__ == "__main__":
    asyncio.run(main())
