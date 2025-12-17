"""Vision-based PDF loader.

This loader renders PDF pages to images and uses a vision-capable chat model
(e.g., OpenAI GPT-4o) to extract content into structured outputs.

Why:
- Works with scanned/image-only PDFs (OCR + layout reasoning via model)
- Can extract headers/footers, printed page numbers, and rich metadata
- Can produce multiple formats (markdown/text/json)

Notes:
- Requires `pymupdf` (import name: `fitz`) for PDF rendering.
- Model must support multimodal inputs (image + text). AgenticFlow's
  `convert_messages` preserves OpenAI-style multimodal message content.

Example:
    >>> from agenticflow.document.loaders.handlers.pdf_vision import PDFVisionLoader
    >>> from agenticflow.models import ChatModel
    >>> loader = PDFVisionLoader(model=ChatModel(model="gpt-4o"), output_format="markdown")
    >>> docs = await loader.load("document.pdf", include_headers=True, include_footers=True)
"""

from __future__ import annotations

import asyncio
import base64
import json
import time
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any

from agenticflow.document.loaders.base import BaseLoader
from agenticflow.document.types import Document
from agenticflow.models.base import BaseChatModel
from agenticflow.observability import ObservabilityLogger


class OutputFormat(StrEnum):
    MARKDOWN = "markdown"
    HTML = "html"
    TEXT = "text"
    JSON = "json"


def _parse_output_format(value: OutputFormat | str) -> OutputFormat:
    if isinstance(value, OutputFormat):
        return value
    try:
        return OutputFormat(str(value).lower())
    except ValueError as e:
        allowed = ", ".join(repr(v.value) for v in OutputFormat)
        raise ValueError(f"Invalid output_format={value!r}. Allowed: {allowed}") from e


@dataclass(frozen=True, slots=True)
class PDFVisionOptions:
    """Options for vision extraction.

    Attributes:
        dpi: Render resolution. Higher improves OCR at cost of latency.
        pages: Optional 1-based page numbers to process (e.g., [4]). If provided,
            this takes precedence over max_pages.
        max_pages: Limit number of pages processed (None = all).
        output_format: Output format.
        include_headers: Ask for header extraction per page.
        include_footers: Ask for footer extraction per page.
        include_pdf_page_number: Include the PDF page index (1-based) in output.
        include_printed_page_number: Ask model to extract printed page number if present.
        extract_metadata: Include PDF metadata (title/author/etc) from the file.
        extract_toc: Include table of contents.
        toc_max_pages: If the PDF has no embedded TOC, scan the first N pages with vision to
            detect a TOC page and extract entries.
        model_timeout_s: Optional timeout hint forwarded to model providers (best-effort).
        timing: If True, captures timing metrics in the first returned Document's metadata.
    """

    dpi: int = 200
    pages: list[int] | None = None
    max_pages: int | None = None
    output_format: OutputFormat = OutputFormat.MARKDOWN
    include_headers: bool = False
    include_footers: bool = False
    include_pdf_page_number: bool = True
    include_printed_page_number: bool = False
    extract_metadata: bool = True
    extract_toc: bool = True
    toc_max_pages: int = 5
    model_timeout_s: float | None = None
    timing: bool = False


class PDFVisionLoader(BaseLoader):
    """Render PDF pages to images and extract via a vision model."""

    supported_extensions = [".pdf"]

    def __init__(
        self,
        *,
        model: BaseChatModel,
        encoding: str = "utf-8",
        output_format: OutputFormat | str = OutputFormat.MARKDOWN,
        dpi: int = 200,
    ) -> None:
        super().__init__(encoding=encoding)
        self.model = model
        self.default_output_format: OutputFormat = _parse_output_format(output_format)
        self.default_dpi = dpi
        self._logger = ObservabilityLogger("pdf_vision_loader")

    async def load(self, path: str | Path, **kwargs: Any) -> list[Document]:
        """Load a PDF and extract content using vision.

        Keyword Args:
            output_format: "markdown" | "html" | "text" | "json".
            dpi: Render dpi.
            pages: Optional list of 1-based page numbers to process.
            max_pages: Max pages to process.
            include_headers: Whether to extract headers.
            include_footers: Whether to extract footers.
            include_pdf_page_number: Include PDF page index.
            include_printed_page_number: Extract printed page number.
            extract_metadata: Include PDF metadata.
            extract_toc: Include embedded or vision-extracted TOC.
            toc_max_pages: Vision TOC scan limit.
            model_timeout_s: Best-effort timeout hint.
            timing: If True, include timing details in metadata.

        Returns:
            One Document per processed page.
        """

        pdf_path = Path(path)
        if not pdf_path.exists():
            raise FileNotFoundError(str(pdf_path))

        t0 = time.monotonic()

        options = PDFVisionOptions(
            dpi=int(kwargs.pop("dpi", self.default_dpi)),
            pages=kwargs.pop("pages", None),
            max_pages=kwargs.pop("max_pages", None),
            output_format=_parse_output_format(kwargs.pop("output_format", self.default_output_format)),
            include_headers=bool(kwargs.pop("include_headers", False)),
            include_footers=bool(kwargs.pop("include_footers", False)),
            include_pdf_page_number=bool(kwargs.pop("include_pdf_page_number", True)),
            include_printed_page_number=bool(kwargs.pop("include_printed_page_number", False)),
            extract_metadata=bool(kwargs.pop("extract_metadata", True)),
            extract_toc=bool(kwargs.pop("extract_toc", True)),
            toc_max_pages=int(kwargs.pop("toc_max_pages", 5)),
            model_timeout_s=kwargs.pop("model_timeout_s", None),
            timing=bool(kwargs.pop("timing", False)),
        )

        if kwargs:
            # Surface unknown kwargs early (helps callers and avoids silent no-ops)
            unknown = ", ".join(sorted(kwargs.keys()))
            raise TypeError(f"Unknown PDFVisionLoader options: {unknown}")

        try:
            import fitz  # type: ignore
        except ImportError as e:
            raise ImportError(
                "PDFVisionLoader requires 'pymupdf' (import name 'fitz'). Install with: uv add pymupdf"
            ) from e

        doc = fitz.open(str(pdf_path))
        total_pages = doc.page_count

        timing: dict[str, Any] = {
            "total_pages": total_pages,
            "selected_pages": None,
            "pdf_open_ms": (time.monotonic() - t0) * 1000.0,
            "toc": None,
            "pages": [],
            "total_ms": None,
        }

        if options.pages is not None:
            # Validate and preserve order while removing duplicates.
            if not isinstance(options.pages, list) or not options.pages:
                raise ValueError("pages must be a non-empty list of 1-based page numbers")
            normalized: list[int] = []
            seen: set[int] = set()
            for p in options.pages:
                try:
                    page_num = int(p)
                except Exception as e:
                    raise ValueError(f"Invalid page number in pages: {p!r}") from e
                if page_num < 1 or page_num > total_pages:
                    raise ValueError(f"Page number out of range: {page_num} (valid: 1..{total_pages})")
                if page_num not in seen:
                    normalized.append(page_num)
                    seen.add(page_num)
            selected_pages = normalized
        else:
            max_pages = options.max_pages or total_pages
            max_pages = min(max_pages, total_pages)
            selected_pages = list(range(1, max_pages + 1))

        timing["selected_pages"] = selected_pages

        pdf_metadata: dict[str, Any] | None = None
        toc: list[dict[str, Any]] | None = None

        if options.extract_metadata:
            t_meta = time.monotonic()
            pdf_metadata = dict(doc.metadata or {})
            pdf_metadata["page_count"] = total_pages
            timing["pdf_metadata_ms"] = (time.monotonic() - t_meta) * 1000.0

        if options.extract_toc:
            t_toc = time.monotonic()
            toc = _extract_embedded_toc(doc)
            toc_info: dict[str, Any] = {
                "source": "embedded" if toc else None,
                "entries": len(toc or []),
                "scan_pages": 0,
                "attempts": [],
                "ms": None,
            }
            if (not toc) and options.toc_max_pages > 0:
                # If `pages` is specified (e.g., [4]) we still scan the *first* pages
                # for a TOC, since TOCs are usually front-matter.
                toc_scan_pages = (
                    min(options.toc_max_pages, total_pages)
                    if options.pages is not None
                    else min(options.toc_max_pages, max_pages)
                )
                toc_info["scan_pages"] = toc_scan_pages
                toc, attempts = await _extract_toc_with_vision(
                    model=self.model,
                    pdf_path=pdf_path,
                    doc=doc,
                    dpi=options.dpi,
                    scan_pages=toc_scan_pages,
                    timeout_s=options.model_timeout_s,
                    timing=options.timing,
                )
                toc_info["attempts"] = attempts
                toc_info["source"] = "vision" if toc else toc_info["source"]
                toc_info["entries"] = len(toc or [])
            toc_info["ms"] = (time.monotonic() - t_toc) * 1000.0
            timing["toc"] = toc_info

        documents: list[Document] = []

        for page_number in selected_pages:
            page_index = page_number - 1
            page = doc.load_page(page_index)
            t_render = time.monotonic()
            png_bytes = _render_page_png(page, dpi=options.dpi)
            render_ms = (time.monotonic() - t_render) * 1000.0

            t_model = time.monotonic()
            extracted = await _extract_page_with_vision(
                model=self.model,
                png_bytes=png_bytes,
                page_number=page_number,
                total_pages=total_pages,
                options=options,
            )
            model_ms = (time.monotonic() - t_model) * 1000.0

            if options.timing:
                timing["pages"].append(
                    {
                        "page": page_number,
                        "render_ms": render_ms,
                        "model_ms": model_ms,
                        "bytes": len(png_bytes),
                    }
                )

            page_metadata: dict[str, Any] = {
                "page": page_number,
                "total_pages": total_pages,
                "loader": "PDFVisionLoader",
                "output_format": options.output_format.value,
            }
            if options.include_printed_page_number and extracted.get("printed_page_number") is not None:
                page_metadata["printed_page_number"] = extracted.get("printed_page_number")
            if options.include_headers and extracted.get("header") is not None:
                page_metadata["header"] = extracted.get("header")
            if options.include_footers and extracted.get("footer") is not None:
                page_metadata["footer"] = extracted.get("footer")

            # Attach doc-level metadata to the first returned page only (keeps per-page metadata smaller)
            if page_number == selected_pages[0]:
                if pdf_metadata is not None:
                    page_metadata["pdf_metadata"] = pdf_metadata
                if toc is not None:
                    page_metadata["toc"] = toc
                if options.timing:
                    timing["total_ms"] = (time.monotonic() - t0) * 1000.0
                    page_metadata["timing"] = timing

            documents.append(self._create_document(extracted["content"], pdf_path, **page_metadata))

        try:
            doc.close()
        except Exception:
            pass

        return documents


def _render_page_png(page: Any, *, dpi: int) -> bytes:
    """Render a PDF page to PNG bytes via PyMuPDF."""
    # PDF points are 72 dpi.
    zoom = max(dpi, 72) / 72.0
    import fitz  # type: ignore

    mat = fitz.Matrix(zoom, zoom)
    pix = page.get_pixmap(matrix=mat, alpha=False)
    return pix.tobytes("png")


def _data_url_for_png(png_bytes: bytes) -> str:
    b64 = base64.b64encode(png_bytes).decode("ascii")
    return f"data:image/png;base64,{b64}"


def _build_page_prompt(*, page_number: int, total_pages: int, options: PDFVisionOptions) -> str:
    parts: list[str] = []
    parts.append(
        "You are a document extraction engine. Read the provided PDF page image and extract content faithfully."
    )

    needs_structured_fields = (
        options.output_format == OutputFormat.JSON
        or options.include_headers
        or options.include_footers
        or options.include_printed_page_number
    )

    if needs_structured_fields:
        if options.output_format == OutputFormat.MARKDOWN:
            content_kind = "clean GitHub-flavored Markdown"
        elif options.output_format == OutputFormat.HTML:
            content_kind = "clean semantic HTML (a fragment; no <html>/<head>/<body>)"
        else:
            content_kind = "plain text"
        parts.append(
            "Return STRICT JSON only (no markdown fences) with keys: "
            f"content (string; {content_kind}), header (string|null), footer (string|null), "
            "printed_page_number (string|null)."
        )
    else:
        if options.output_format == OutputFormat.MARKDOWN:
            parts.append("Return the page content as clean GitHub-flavored Markdown.")
        elif options.output_format == OutputFormat.HTML:
            parts.append(
                "Return the page content as clean semantic HTML (a fragment only; do not include <html>, <head>, or <body>)."
            )
        elif options.output_format == OutputFormat.TEXT:
            parts.append("Return the page content as plain text.")

    if options.include_headers:
        parts.append("If there is a repeated header, extract it into header.")
    if options.include_footers:
        parts.append("If there is a repeated footer, extract it into footer.")
    if options.include_printed_page_number:
        parts.append("If a printed page number is visible on the page, extract it into printed_page_number.")

    if options.include_pdf_page_number:
        parts.append(f"This is PDF page {page_number} of {total_pages}.")

    return "\n".join(parts)


async def _extract_page_with_vision(
    *,
    model: BaseChatModel,
    png_bytes: bytes,
    page_number: int,
    total_pages: int,
    options: PDFVisionOptions,
) -> dict[str, Any]:
    """Call a vision model to extract a single page."""

    prompt = _build_page_prompt(page_number=page_number, total_pages=total_pages, options=options)

    messages: list[dict[str, Any]] = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": _data_url_for_png(png_bytes)}},
            ],
        }
    ]

    try:
        if options.model_timeout_s is not None:
            resp = await asyncio.wait_for(model.ainvoke(messages), timeout=float(options.model_timeout_s))
        else:
            resp = await model.ainvoke(messages)
    except asyncio.TimeoutError:
        if options.output_format == OutputFormat.JSON:
            return {
                "content": json.dumps(
                    {
                        "content": "",
                        "header": None,
                        "footer": None,
                        "printed_page_number": None,
                        "error": f"vision model timed out after {options.model_timeout_s}s",
                    },
                    ensure_ascii=False,
                ),
                "header": None,
                "footer": None,
                "printed_page_number": None,
            }
        return {
            "content": f"[PDFVisionLoader] vision model timed out after {options.model_timeout_s}s",
            "header": None,
            "footer": None,
            "printed_page_number": None,
        }
    raw = (resp.content or "").strip()

    needs_structured_fields = (
        options.output_format == OutputFormat.JSON
        or options.include_headers
        or options.include_footers
        or options.include_printed_page_number
    )

    if needs_structured_fields:
        try:
            parsed = json.loads(raw)
        except Exception:
            # Best-effort fallback: wrap raw in JSON envelope
            parsed = {"content": raw, "header": None, "footer": None, "printed_page_number": None}
        # Ensure required keys exist
        parsed.setdefault("content", "")
        parsed.setdefault("header", None)
        parsed.setdefault("footer", None)
        parsed.setdefault("printed_page_number", None)

        content_value: str
        if options.output_format == OutputFormat.JSON:
            content_value = json.dumps(parsed, ensure_ascii=False)
        else:
            content_value = str(parsed.get("content") or "")

        return {
            "content": content_value,
            "header": parsed.get("header"),
            "footer": parsed.get("footer"),
            "printed_page_number": parsed.get("printed_page_number"),
        }

    return {"content": raw, "header": None, "footer": None, "printed_page_number": None}


def _extract_embedded_toc(doc: Any) -> list[dict[str, Any]]:
    """Extract embedded table-of-contents if present."""
    try:
        toc_raw = doc.get_toc(simple=True)
    except Exception:
        return []

    toc: list[dict[str, Any]] = []
    for item in toc_raw or []:
        # item: [level, title, page]
        if not isinstance(item, (list, tuple)) or len(item) < 3:
            continue
        level, title, page = item[0], item[1], item[2]
        toc.append({"level": int(level), "title": str(title), "page": int(page)})
    return toc


async def _extract_toc_with_vision(
    *,
    model: BaseChatModel,
    pdf_path: Path,
    doc: Any,
    dpi: int,
    scan_pages: int,
    timeout_s: float | None,
    timing: bool,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Best-effort vision TOC extraction by scanning the first pages.

    Behavior:
    - Scans up to `scan_pages` from the front.
    - If it detects a TOC page, it continues scanning subsequent pages to
      capture multi-page TOCs until it hits a non-TOC page (or runs out of budget).
    """
    _ = pdf_path  # reserved for future debug output

    attempts: list[dict[str, Any]] = []

    in_toc = False
    merged_entries: list[dict[str, Any]] = []

    for page_index in range(scan_pages):
        page = doc.load_page(page_index)
        t_render = time.monotonic()
        png_bytes = _render_page_png(page, dpi=dpi)
        render_ms = (time.monotonic() - t_render) * 1000.0

        prompt = (
            "You are a document understanding engine. Determine if this page is a Table of Contents (TOC) page, "
            "or a continuation of a TOC from a previous page.\n"
            "If it is a TOC/continuation page, extract as many entries as are clearly visible.\n"
            "TOC entries often look like: 'Section Title .......... 12' or '1.2 Subsection 7'.\n"
            "Rules:\n"
            "- Output STRICT JSON only (no markdown fences, no prose).\n"
            "- Schema: {\"is_toc_page\": true|false, \"entries\": [{\"level\": number, \"title\": string, \"page\": number}]}\n"
            "- 'level' should be 1 for top-level, 2 for indented/subsections, etc (best effort).\n"
            "- 'page' must be a number (omit entries you can't parse a page number for).\n"
            "If it is NOT a TOC page, output: {\"is_toc_page\": false, \"entries\": []}."
        )

        messages: list[dict[str, Any]] = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": _data_url_for_png(png_bytes)}},
                ],
            }
        ]

        t_model = time.monotonic()
        try:
            if timeout_s is not None:
                resp = await asyncio.wait_for(model.ainvoke(messages), timeout=float(timeout_s))
            else:
                resp = await model.ainvoke(messages)
        except asyncio.TimeoutError:
            if timing:
                attempts.append(
                    {
                        "page": page_index + 1,
                        "is_toc": False,
                        "entries": 0,
                        "render_ms": render_ms,
                        "model_ms": (time.monotonic() - t_model) * 1000.0,
                        "timeout": True,
                    }
                )
            continue
        raw = (resp.content or "").strip()
        model_ms = (time.monotonic() - t_model) * 1000.0
        try:
            parsed = json.loads(raw)
        except Exception:
            continue

        is_toc_page = bool(parsed.get("is_toc_page") is True)
        parsed_entries = parsed.get("entries")

        if is_toc_page and isinstance(parsed_entries, list):
            page_entries: list[dict[str, Any]] = []
            for e in parsed_entries:
                if not isinstance(e, dict):
                    continue
                if "title" not in e or "page" not in e:
                    continue
                try:
                    page_entries.append(
                        {
                            "level": max(1, int(e.get("level", 1))),
                            "title": str(e.get("title")),
                            "page": int(e.get("page")),
                        }
                    )
                except Exception:
                    continue

            if timing:
                attempts.append(
                    {
                        "page": page_index + 1,
                        "is_toc": True,
                        "entries": len(page_entries),
                        "render_ms": render_ms,
                        "model_ms": model_ms,
                        "timeout": False,
                    }
                )

            # Start or continue a multi-page TOC if we got anything useful.
            if page_entries:
                in_toc = True
                merged_entries.extend(page_entries)
                continue

            # It's a TOC page but produced no entries; if we've already started
            # collecting, treat this as a weak continuation and keep scanning.
            if in_toc:
                continue

        # If we were in a TOC block and now hit a non-TOC page, stop early.
        if in_toc and not is_toc_page:
            break

        if timing:
            attempts.append(
                {
                    "page": page_index + 1,
                    "is_toc": is_toc_page,
                    "entries": 0,
                    "render_ms": render_ms,
                    "model_ms": model_ms,
                    "timeout": False,
                }
            )

    if not merged_entries:
        return [], attempts

    # Best-effort normalize + de-dupe (TOCs sometimes repeat headers across pages).
    seen: set[tuple[int, str, int]] = set()
    normalized: list[dict[str, Any]] = []
    for entry in merged_entries:
        key = (int(entry.get("level", 1)), str(entry.get("title", "")), int(entry.get("page", 0)))
        if key in seen:
            continue
        seen.add(key)
        normalized.append({"level": key[0], "title": key[1], "page": key[2]})

    return normalized, attempts


__all__ = ["PDFVisionLoader", "PDFVisionOptions"]
