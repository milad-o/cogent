"""High-performance PDF loader optimized for HTML output with complex table support.

This module provides a modern PDF-to-HTML loader using pdfplumber for intelligent
table extraction, with true CPU parallelization for processing multiple pages concurrently.

Features:
- Converts PDF to semantic HTML with proper table structures
- Intelligent table detection and extraction using pdfplumber
- True CPU parallelization using ProcessPoolExecutor
- Batch processing with configurable chunk sizes
- Superior table handling with actual HTML <table> elements
- No OCR (assumes text-based PDFs) for speed
- Comprehensive observability with structured logging
- Progress tracking and metrics

Example:
    >>> from agenticflow.document.loaders.handlers.pdf_html import PDFHTMLLoader
    >>> 
    >>> loader = PDFHTMLLoader(max_workers=4, batch_size=10)
    >>> docs = await loader.load(Path("large_document.pdf"))
    >>> print(f"Loaded {len(docs)} pages as HTML")

Note:
    Requires: uv add pdfplumber
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from enum import Enum
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING, Any

from agenticflow.document.loaders.base import BaseLoader
from agenticflow.document.types import Document
from agenticflow.observability import ObservabilityLogger, LogLevel

if TYPE_CHECKING:
    pass


# ==============================================================================
# Enums for PDF Processing
# ==============================================================================


class PDFProcessingStatus(Enum):
    """Status of PDF processing operations."""

    PENDING = "pending"
    LOADING = "loading"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    PARTIAL = "partial"  # Some pages failed


class PageStatus(Enum):
    """Status of individual page processing."""

    PENDING = "pending"
    PROCESSING = "processing"
    SUCCESS = "success"
    FAILED = "failed"
    EMPTY = "empty"  # Page has no extractable text


class OutputFormat(Enum):
    """Output format for PDF extraction."""

    HTML = "html"
    TEXT = "text"
    JSON = "json"


# ==============================================================================
# Dataclasses for Tracking
# ==============================================================================


@dataclass(frozen=True, slots=True, kw_only=True)
class PDFConfig:
    """Configuration for PDF processing.

    Attributes:
        max_workers: Maximum parallel workers for CPU-bound processing.
        batch_size: Number of pages to process per batch.
        dpi: Image resolution if extracting images.
        output_format: Output format (html, text, json).
        ignore_images: Whether to skip image extraction.
        ignore_graphics: Whether to skip graphics extraction.
        force_text: Force text extraction even for image-based PDFs.
        fontsize_limit: Minimum font size to consider as text.
        header: Whether to include page headers in output.
        footer: Whether to include page footers in output.
    """

    max_workers: int = 4
    batch_size: int = 10
    dpi: int = 150
    output_format: OutputFormat = OutputFormat.HTML
    ignore_images: bool = False
    ignore_graphics: bool = False
    force_text: bool = True
    fontsize_limit: float = 3.0
    header: bool = True
    footer: bool = True


@dataclass(slots=True, kw_only=True)
class PageResult:
    """Result of processing a single page.

    Attributes:
        page_number: 1-based page number.
        status: Processing status.
        content: Extracted content (HTML/text).
        tables_count: Number of tables detected.
        images_count: Number of images on page.
        processing_time_ms: Time to process in milliseconds.
        error: Error message if failed.
    """

    page_number: int
    status: PageStatus
    content: str = ""
    tables_count: int = 0
    images_count: int = 0
    processing_time_ms: float = 0.0
    error: str | None = None


@dataclass(slots=True, kw_only=True)
class PDFProcessingResult:
    """Result of processing an entire PDF.

    Attributes:
        file_path: Path to the PDF file.
        status: Overall processing status.
        total_pages: Total pages in document.
        successful_pages: Number of successfully processed pages.
        failed_pages: Number of failed pages.
        empty_pages: Number of pages with no text.
        page_results: Individual page results.
        total_time_ms: Total processing time in milliseconds.
        metadata: Additional document metadata.
    """

    file_path: Path
    status: PDFProcessingStatus
    total_pages: int
    successful_pages: int = 0
    failed_pages: int = 0
    empty_pages: int = 0
    page_results: list[PageResult] = field(default_factory=list)
    total_time_ms: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def success_rate(self) -> float:
        """Calculate success rate as percentage."""
        if self.total_pages == 0:
            return 0.0
        return (self.successful_pages / self.total_pages) * 100

    @property
    def documents(self) -> list[Document]:
        """Convert page results to Document objects."""
        docs: list[Document] = []
        for page_result in self.page_results:
            if page_result.status != PageStatus.SUCCESS:
                continue
            if not page_result.content.strip():
                continue

            docs.append(
                Document(
                    text=page_result.content,
                    metadata={
                        "source": str(self.file_path),
                        "page": page_result.page_number,
                        "total_pages": self.total_pages,
                        "tables_count": page_result.tables_count,
                        "images_count": page_result.images_count,
                        "processing_time_ms": page_result.processing_time_ms,
                        "format": "html",
                        **self.metadata,
                    },
                )
            )
        return docs

    def to_html(
        self,
        *,
        include_page_breaks: bool = True,
        include_page_numbers: bool = False,
        page_break_style: str = "hr",
    ) -> str:
        """Convert all pages to a single HTML string.
        
        Args:
            include_page_breaks: Add separators between pages (default: True).
            include_page_numbers: Add page number headers (default: False).
            page_break_style: Separator style ("hr", "div", or custom HTML).
            
        Returns:
            Complete HTML content as a single string.
            
        Example:
            >>> result = await loader.load("doc.pdf", tracking=True)
            >>> html = result.to_html(include_page_numbers=True)
        """
        parts: list[str] = []

        def _page_block(page_number: int, content: str) -> str:
            """Format a single page block with optional dividers and page number."""
            if include_page_numbers and include_page_breaks:
                if page_break_style == "hr":
                    return f'<hr>\n<div class="page-number">Page {page_number}</div>\n{content}\n<hr>'
                elif page_break_style == "div":
                    return f'<div class="page-separator"></div>\n<div class="page-number">Page {page_number}</div>\n{content}\n<div class="page-separator"></div>'
                else:
                    return f'{page_break_style}\n<div class="page-number">Page {page_number}</div>\n{content}\n{page_break_style}'
            
            if include_page_numbers:
                return f'<div class="page-number">Page {page_number}</div>\n\n{content}'
            
            if include_page_breaks:
                if page_break_style == "hr":
                    return f"<hr>\n\n{content}\n\n<hr>"
                elif page_break_style == "div":
                    return f'<div class="page-separator"></div>\n\n{content}\n\n<div class="page-separator"></div>'
                else:
                    return f"{page_break_style}\n\n{content}\n\n{page_break_style}"
            
            return content

        for page_result in self.page_results:
            if page_result.status != PageStatus.SUCCESS:
                continue
            if not page_result.content.strip():
                continue

            page_content = page_result.content.strip()
            parts.append(_page_block(page_result.page_number, page_content))

        return "\n\n".join(parts)

    def save(
        self,
        output_path: str | Path,
        *,
        mode: str = "single",
        include_page_breaks: bool = True,
        include_page_numbers: bool = False,
        page_break_style: str = "hr",
        encoding: str = "utf-8",
    ) -> Path | list[Path]:
        """Save extracted content to file(s).
        
        Args:
            output_path: Output file path (for single mode) or directory (for pages mode).
            mode: Save mode:
                - "single": One combined HTML file (default).
                - "pages": Separate file per page.
                - "json": Export as JSON with metadata.
            include_page_breaks: Add separators between pages (single mode only).
            include_page_numbers: Add page number divs (single mode only).
            page_break_style: Separator style ("hr", "div", or custom HTML).
            encoding: Text encoding for output files.
            
        Returns:
            Path to saved file (single/json mode) or list of paths (pages mode).
            
        Example:
            >>> result = await loader.load("doc.pdf", tracking=True)
            >>> 
            >>> # Save as single HTML file
            >>> result.save("output.html")
            >>> 
            >>> # Save each page separately
            >>> result.save("output_dir/", mode="pages")
            >>> 
            >>> # Save with page numbers
            >>> result.save("output.html", include_page_numbers=True)
            >>> 
            >>> # Export as JSON
            >>> result.save("output.json", mode="json")
        """
        import json
        
        output_path = Path(output_path)
        
        if mode == "single":
            # Combine all pages into one file
            content = self.to_html(
                include_page_breaks=include_page_breaks,
                include_page_numbers=include_page_numbers,
                page_break_style=page_break_style,
            )
            
            # Wrap in basic HTML structure
            html_doc = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{self.file_path.name}</title>
    <style>
        body {{ font-family: Arial, sans-serif; max-width: 900px; margin: 40px auto; padding: 20px; }}
        table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
        th {{ background-color: #f2f2f2; }}
        .page-number {{ font-weight: bold; color: #666; margin: 20px 0; }}
        .page-separator {{ margin: 40px 0; border-top: 2px solid #ddd; }}
        hr {{ margin: 40px 0; }}
    </style>
</head>
<body>
{content}
</body>
</html>"""
            
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(html_doc, encoding=encoding)
            return output_path
        
        elif mode == "pages":
            # Save each page as separate file
            output_path.mkdir(parents=True, exist_ok=True)
            stem = self.file_path.stem
            saved_paths: list[Path] = []
            
            for page_result in self.page_results:
                if page_result.status != PageStatus.SUCCESS:
                    continue
                if not page_result.content.strip():
                    continue

                page_content = page_result.content.strip()
                if include_page_numbers:
                    page_content = f'<div class="page-number">Page {page_result.page_number}</div>\n\n{page_content}'
                
                # Wrap each page in HTML structure
                html_doc = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{self.file_path.name} - Page {page_result.page_number}</title>
    <style>
        body {{ font-family: Arial, sans-serif; max-width: 900px; margin: 40px auto; padding: 20px; }}
        table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
        th {{ background-color: #f2f2f2; }}
        .page-number {{ font-weight: bold; color: #666; margin: 20px 0; }}
    </style>
</head>
<body>
{page_content}
</body>
</html>"""

                page_path = output_path / f"{stem}_page_{page_result.page_number:04d}.html"
                page_path.write_text(html_doc, encoding=encoding)
                saved_paths.append(page_path)
            
            return saved_paths
        
        elif mode == "json":
            # Export as JSON with full metadata
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            export_data = {
                "source": str(self.file_path),
                "filename": self.file_path.name,
                "total_pages": self.total_pages,
                "successful_pages": self.successful_pages,
                "failed_pages": self.failed_pages,
                "empty_pages": self.empty_pages,
                "success_rate": self.success_rate,
                "processing_time_ms": self.total_time_ms,
                "metadata": self.metadata,
                "pages": [
                    {
                        "page_number": pr.page_number,
                        "status": pr.status.value,
                        "content": pr.content,
                        "tables_count": pr.tables_count,
                        "images_count": pr.images_count,
                        "processing_time_ms": pr.processing_time_ms,
                        "error": pr.error,
                    }
                    for pr in self.page_results
                ],
            }
            
            output_path.write_text(
                json.dumps(export_data, indent=2, ensure_ascii=False),
                encoding=encoding,
            )
            return output_path
        
        else:
            msg = f"Invalid mode: {mode}. Use 'single', 'pages', or 'json'."
            raise ValueError(msg)


@dataclass(frozen=True, slots=True, kw_only=True)
class ProcessingMetrics:
    """Metrics for PDF processing performance.

    Attributes:
        total_pages: Total pages processed.
        total_time_ms: Total processing time.
        avg_page_time_ms: Average time per page.
        pages_per_second: Processing throughput.
        batch_count: Number of batches processed.
    """

    total_pages: int
    total_time_ms: float
    avg_page_time_ms: float
    pages_per_second: float
    batch_count: int


# ==============================================================================
# Worker Functions (for ProcessPoolExecutor)
# ==============================================================================


def _process_page_batch(
    pdf_path: str,
    page_numbers: Sequence[int],
    config_dict: dict[str, Any],
) -> list[dict[str, Any]]:
    """Process a batch of pages in a separate process.

    This function runs in a ProcessPoolExecutor worker process.
    It uses pdfplumber to extract text and tables as semantic HTML.

    Args:
        pdf_path: Path to the PDF file.
        page_numbers: 0-based page numbers to process.
        config_dict: Configuration as dict (must be picklable).

    Returns:
        List of page result dictionaries.
    """
    try:
        import pdfplumber
    except ImportError as e:
        return [
            {
                "page_number": pn + 1,
                "status": "failed",
                "content": "",
                "error": f"pdfplumber not installed: {e}",
                "processing_time_ms": 0.0,
            }
            for pn in page_numbers
        ]

    results: list[dict[str, Any]] = []

    try:
        pdf = pdfplumber.open(pdf_path)
    except Exception as e:
        pdf = None
        return [
            {
                "page_number": pn + 1,
                "status": "failed",
                "content": "",
                "error": f"Failed to open PDF: {e}",
                "processing_time_ms": 0.0,
            }
            for pn in page_numbers
        ]

    for page_num in page_numbers:
        start_time = time.perf_counter()

        try:
            page = pdf.pages[page_num]
            
            # Extract tables with their positions
            table_objects = page.find_tables()
            tables = [t.extract() for t in table_objects]
            table_info = [(t.bbox[1], idx, 'table') for idx, t in enumerate(table_objects)]  # (top_y, index, type)
            
            # Extract text with positions
            text_blocks = []
            if table_objects:
                # Get table bounding boxes
                table_bboxes = [t.bbox for t in table_objects]
                
                # Filter page to exclude table areas
                def not_within_tables(obj):
                    """Check if object is not within any table bbox."""
                    obj_bbox = (obj['x0'], obj['top'], obj['x1'], obj['bottom'])
                    for table_bbox in table_bboxes:
                        # Check if object overlaps with table
                        if (obj_bbox[0] < table_bbox[2] and obj_bbox[2] > table_bbox[0] and
                            obj_bbox[1] < table_bbox[3] and obj_bbox[3] > table_bbox[1]):
                            return False
                    return True
                
                # Extract text with positions from non-table regions
                filtered_page = page.filter(not_within_tables)
                chars = filtered_page.chars
                
                # Group chars into lines based on y-position
                if chars:
                    lines = []
                    current_line = []
                    current_y = None
                    
                    for char in sorted(chars, key=lambda c: (c['top'], c['x0'])):
                        if current_y is None or abs(char['top'] - current_y) < 3:  # Same line (tolerance of 3 pixels)
                            current_line.append(char)
                            current_y = char['top']
                        else:
                            if current_line:
                                line_text = ''.join(c['text'] for c in current_line)
                                line_y = current_line[0]['top']
                                lines.append((line_y, line_text))
                            current_line = [char]
                            current_y = char['top']
                    
                    # Add last line
                    if current_line:
                        line_text = ''.join(c['text'] for c in current_line)
                        line_y = current_line[0]['top']
                        lines.append((line_y, line_text))
                    
                    # Group lines into paragraphs
                    text_blocks = _group_positioned_lines(lines)
            else:
                # No tables - extract all text
                chars = page.chars
                if chars:
                    lines = []
                    current_line = []
                    current_y = None
                    
                    for char in sorted(chars, key=lambda c: (c['top'], c['x0'])):
                        if current_y is None or abs(char['top'] - current_y) < 3:
                            current_line.append(char)
                            current_y = char['top']
                        else:
                            if current_line:
                                line_text = ''.join(c['text'] for c in current_line)
                                line_y = current_line[0]['top']
                                lines.append((line_y, line_text))
                            current_line = [char]
                            current_y = char['top']
                    
                    if current_line:
                        line_text = ''.join(c['text'] for c in current_line)
                        line_y = current_line[0]['top']
                        lines.append((line_y, line_text))
                    
                    text_blocks = _group_positioned_lines(lines)
            
            # Convert to HTML with position information
            html_content = _convert_to_html_positioned(text_blocks, tables, table_info, page_num + 1)
            
            elapsed_ms = (time.perf_counter() - start_time) * 1000

            tables_count = len(tables)
            # Count images (if available)
            images_count = len(page.images) if hasattr(page, 'images') else 0

            if html_content.strip():
                results.append(
                    {
                        "page_number": page_num + 1,
                        "status": "success",
                        "content": html_content,
                        "tables_count": tables_count,
                        "images_count": images_count,
                        "processing_time_ms": elapsed_ms,
                        "error": None,
                    }
                )
            else:
                results.append(
                    {
                        "page_number": page_num + 1,
                        "status": "empty",
                        "content": "",
                        "tables_count": 0,
                        "images_count": 0,
                        "processing_time_ms": elapsed_ms,
                        "error": None,
                    }
                )

        except Exception as e:
            elapsed_ms = (time.perf_counter() - start_time) * 1000
            results.append(
                {
                    "page_number": page_num + 1,
                    "status": "failed",
                    "content": "",
                    "error": str(e),
                    "processing_time_ms": elapsed_ms,
                }
            )

    if pdf:
        pdf.close()
    return results


def _group_positioned_lines(lines: list[tuple[float, str]]) -> list[tuple[float, str, str]]:
    """Group positioned lines into paragraphs.
    
    Args:
        lines: List of (y_position, text) tuples.
        
    Returns:
        List of (y_position, paragraph_text, paragraph_type) tuples.
        Type can be: 'heading', 'table_caption', 'paragraph'.
    """
    if not lines:
        return []
    
    paragraphs = []
    current_para_lines = []
    last_y = None
    
    for y, text in lines:
        text = text.strip()
        if not text:
            continue
            
        # Skip PDF metadata
        if _is_pdf_metadata(text):
            continue
            
        # Skip table-like text
        if _looks_like_table_text(text):
            continue
        
        # Check for paragraph break (large y-gap)
        if last_y is not None and (y - last_y) > 15:  # Large gap = new paragraph
            if current_para_lines:
                # Process accumulated lines
                para_y = current_para_lines[0][0]
                para_text = ' '.join(line[1] for line in current_para_lines)
                para_type = _classify_paragraph(para_text)
                paragraphs.append((para_y, para_text, para_type))
                current_para_lines = []
        
        current_para_lines.append((y, text))
        last_y = y
    
    # Add last paragraph
    if current_para_lines:
        para_y = current_para_lines[0][0]
        para_text = ' '.join(line[1] for line in current_para_lines)
        para_type = _classify_paragraph(para_text)
        paragraphs.append((para_y, para_text, para_type))
    
    return paragraphs


def _classify_paragraph(text: str) -> str:
    """Classify a paragraph as heading, table_caption, or paragraph.
    
    Args:
        text: Paragraph text.
        
    Returns:
        Classification: 'heading', 'table_caption', or 'paragraph'.
    """
    # Check if it's all uppercase and short (likely a heading)
    if text.isupper() and len(text) < 100:
        return 'heading'
    
    # Check if it starts with "Table N" (table caption)
    if text.startswith('Table ') and (':' in text or len(text) < 20):
        return 'table_caption'
    
    return 'paragraph'


def _convert_to_html_positioned(
    text_blocks: list[tuple[float, str, str]],
    tables: list[list[list[str | None]]],
    table_info: list[tuple[float, int, str]],
    page_number: int
) -> str:
    """Convert positioned text blocks and tables to HTML, preserving document order.
    
    Args:
        text_blocks: List of (y_position, text, type) tuples.
        tables: List of extracted tables.
        table_info: List of (y_position, table_index, 'table') tuples.
        page_number: 1-based page number.
        
    Returns:
        HTML content with preserved document order.
    """
    # Combine text blocks and tables into a single list
    elements: list[tuple[float, str, str, int | None]] = []
    
    # Add text blocks (y, text, type, None)
    for y, text, para_type in text_blocks:
        elements.append((y, text, para_type, None))
    
    # Add table placeholders (y, '', 'table', table_index)
    for y, idx, element_type in table_info:
        elements.append((y, '', element_type, idx))
    
    # Sort by y-position to preserve document order
    elements.sort(key=lambda x: x[0])
    
    # Generate HTML
    html_parts = []
    for y, content, element_type, table_idx in elements:
        if element_type == 'table' and table_idx is not None:
            # Render table
            if table_idx < len(tables) and tables[table_idx]:
                html_parts.append(_table_to_html(tables[table_idx], table_idx))
        elif element_type == 'heading':
            # Escape HTML
            content = content.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
            html_parts.append(f'<h3>{content}</h3>')
        elif element_type == 'table_caption':
            # Escape HTML
            content = content.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
            html_parts.append(f'<h4>{content}</h4>')
        elif element_type == 'paragraph':
            # Escape HTML
            content = content.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
            html_parts.append(f'<p>{content}</p>')
    
    return '\n'.join(html_parts)


def _convert_to_html(text: str, tables: list[list[list[str | None]]], table_positions: list[tuple[float, int]], page_number: int) -> str:
    """Convert extracted text and tables to semantic HTML, preserving document order.
    
    Args:
        text: Extracted text from the page (with table regions excluded).
        tables: List of extracted tables (each table is a list of rows, each row is a list of cells).
        table_positions: List of (y_position, table_index) tuples for each table.
        page_number: 1-based page number.
        
    Returns:
        HTML content with proper structure and preserved document order.
    """
    # Process text content - split into paragraphs
    paragraphs = text.split('\n\n')
    
    # If no double newlines found, split by single newlines and group intelligently
    if len(paragraphs) == 1 and '\n' in text:
        lines = text.split('\n')
        paragraphs = _group_lines_into_paragraphs(lines)
    
    # Process each paragraph and estimate its position
    # Since we don't have exact positions for text after filtering, we'll use a heuristic:
    # - Text at the beginning of the extracted text likely appears before tables
    # - Text mentioning "Table N" likely appears just before that table
    
    html_parts: list[tuple[float, str]] = []  # (position, html_content)
    
    # Process text paragraphs
    para_position = 0.0  # Estimate position
    for para_idx, para in enumerate(paragraphs):
        para = para.strip()
        if not para:
            continue
            
        # Skip if this looks like it's part of a table (heuristic)
        if _looks_like_table_text(para):
            continue
        
        # Skip common PDF metadata/footer patterns
        if _is_pdf_metadata(para):
            continue
            
        # Escape HTML special characters
        para = para.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
        
        # Determine HTML format
        if para.isupper() and len(para) < 100:
            html_content = f'<h3>{para}</h3>'
        elif para.startswith('Table ') and ':' in para:
            html_content = f'<h4>{para}</h4>'
            # If this is a table caption, position it just before the corresponding table
            # Extract table number
            import re
            match = re.match(r'Table (\d+)', para)
            if match and table_positions:
                table_num = int(match.group(1))
                # Find if we have a table at a similar position
                # Use the next table's position minus a small offset
                for pos, idx in table_positions:
                    if idx < len(tables):
                        # Position this caption just before the table
                        para_position = pos - 1.0
                        break
        elif para.startswith('Table ') and len(para) < 20:
            html_content = f'<h4>{para}</h4>'
            import re
            match = re.match(r'Table (\d+)', para)
            if match and table_positions:
                table_num = int(match.group(1))
                for pos, idx in table_positions:
                    if idx < len(tables):
                        para_position = pos - 1.0
                        break
        else:
            para_html = para.replace('\n', '<br>\n')
            html_content = f'<p>{para_html}</p>'
        
        html_parts.append((para_position, html_content))
        para_position += 1.0  # Increment for next paragraph
    
    # Add tables with their actual positions
    for pos, idx in table_positions:
        if idx < len(tables) and tables[idx]:
            table_html = _table_to_html(tables[idx], idx)
            html_parts.append((pos, table_html))
    
    # Sort by position to preserve document order
    html_parts.sort(key=lambda x: x[0])
    
    # Extract just the HTML content
    return '\n'.join(html for _, html in html_parts)


def _group_lines_into_paragraphs(lines: list[str]) -> list[str]:
    """Group lines into logical paragraphs.
    
    Headings, table captions, and list items are kept separate.
    Regular text lines are grouped together.
    
    Args:
        lines: List of text lines.
        
    Returns:
        List of paragraph strings.
    """
    paragraphs: list[str] = []
    current_group: list[str] = []
    
    for line in lines:
        line = line.strip()
        if not line:
            if current_group:
                paragraphs.append('\n'.join(current_group))
                current_group = []
            continue
        
        # Check if this is a heading or special line
        is_heading = (
            line.isupper() or 
            line.startswith('Table ') or
            line.startswith('Figure ') or
            line.startswith('(') and ')' in line and len(line) < 100  # Footnotes
        )
        
        if is_heading:
            # Finish current group
            if current_group:
                paragraphs.append('\n'.join(current_group))
                current_group = []
            # Add heading as separate paragraph
            paragraphs.append(line)
        else:
            # Add to current group
            current_group.append(line)
    
    # Add final group
    if current_group:
        paragraphs.append('\n'.join(current_group))
    
    return paragraphs


def _table_to_html(table: list[list[str | None]], table_idx: int) -> str:
    """Convert a table to HTML with proper rowspan/colspan for merged cells.
    
    Args:
        table: List of rows, each row is a list of cells (None indicates merged cell).
        table_idx: Index of this table on the page.
        
    Returns:
        HTML table string with rowspan/colspan attributes.
    """
    if not table or len(table) == 0:
        return ""
    
    # Calculate rowspan and colspan for merged cells
    merge_info = _calculate_cell_spans(table)
    
    html_parts = [f'<table class="table-{table_idx}">']
    
    # First row is usually header
    if len(table) > 0:
        html_parts.append('  <thead>')
        html_parts.append('    <tr>')
        for col_idx, cell in enumerate(table[0]):
            # Skip if this cell is part of a previous colspan
            if (0, col_idx) in merge_info['skip']:
                continue
                
            cell_text = str(cell).strip() if cell else ''
            cell_text = cell_text.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
            
            # Add colspan/rowspan if needed
            attrs = []
            if (0, col_idx) in merge_info['colspan']:
                attrs.append(f'colspan="{merge_info["colspan"][(0, col_idx)]}"')
            if (0, col_idx) in merge_info['rowspan']:
                attrs.append(f'rowspan="{merge_info["rowspan"][(0, col_idx)]}"')
            
            attr_str = ' ' + ' '.join(attrs) if attrs else ''
            html_parts.append(f'      <th{attr_str}>{cell_text}</th>')
        html_parts.append('    </tr>')
        html_parts.append('  </thead>')
    
    # Rest are body rows
    if len(table) > 1:
        html_parts.append('  <tbody>')
        for row_idx, row in enumerate(table[1:], start=1):
            html_parts.append('    <tr>')
            for col_idx, cell in enumerate(row):
                # Skip if this cell is part of a previous colspan or rowspan
                if (row_idx, col_idx) in merge_info['skip']:
                    continue
                    
                cell_text = str(cell).strip() if cell else ''
                cell_text = cell_text.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
                
                # Add colspan/rowspan if needed
                attrs = []
                if (row_idx, col_idx) in merge_info['colspan']:
                    attrs.append(f'colspan="{merge_info["colspan"][(row_idx, col_idx)]}"')
                if (row_idx, col_idx) in merge_info['rowspan']:
                    attrs.append(f'rowspan="{merge_info["rowspan"][(row_idx, col_idx)]}"')
                
                attr_str = ' ' + ' '.join(attrs) if attrs else ''
                html_parts.append(f'      <td{attr_str}>{cell_text}</td>')
            html_parts.append('    </tr>')
        html_parts.append('  </tbody>')
    
    html_parts.append('</table>')
    
    return '\n'.join(html_parts)


def _calculate_cell_spans(table: list[list[str | None]]) -> dict[str, dict[tuple[int, int], int]]:
    """Calculate rowspan and colspan for merged cells in a table.
    
    pdfplumber uses None to indicate cells that are part of a merge.
    This function detects the merge patterns and calculates the appropriate spans.
    
    Args:
        table: List of rows, each row is a list of cells.
        
    Returns:
        Dictionary with 'rowspan', 'colspan', and 'skip' mappings.
    """
    if not table:
        return {'rowspan': {}, 'colspan': {}, 'skip': set()}
    
    rowspan: dict[tuple[int, int], int] = {}
    colspan: dict[tuple[int, int], int] = {}
    skip: set[tuple[int, int]] = set()
    
    num_rows = len(table)
    num_cols = len(table[0]) if table else 0
    
    # Calculate colspan (horizontal merges within same row)
    for row_idx, row in enumerate(table):
        col_idx = 0
        while col_idx < len(row):
            if row[col_idx] is not None:
                # Count consecutive None values to the right
                span = 1
                for next_col in range(col_idx + 1, len(row)):
                    if row[next_col] is None:
                        span += 1
                        skip.add((row_idx, next_col))
                    else:
                        break
                
                if span > 1:
                    colspan[(row_idx, col_idx)] = span
                col_idx += span
            else:
                col_idx += 1
    
    # Calculate rowspan (vertical merges across rows)
    for col_idx in range(num_cols):
        row_idx = 0
        while row_idx < num_rows:
            # Skip if this cell is already part of a colspan
            if (row_idx, col_idx) in skip:
                row_idx += 1
                continue
                
            if row_idx < len(table) and col_idx < len(table[row_idx]):
                cell_value = table[row_idx][col_idx]
                
                if cell_value is not None:
                    # Count consecutive None values below
                    span = 1
                    for next_row in range(row_idx + 1, num_rows):
                        if (col_idx < len(table[next_row]) and 
                            table[next_row][col_idx] is None and
                            (next_row, col_idx) not in skip):  # Not part of colspan
                            span += 1
                            skip.add((next_row, col_idx))
                        else:
                            break
                    
                    if span > 1:
                        rowspan[(row_idx, col_idx)] = span
            
            row_idx += 1
    
    return {'rowspan': rowspan, 'colspan': colspan, 'skip': skip}


def _looks_like_table_text(text: str) -> bool:
    """Heuristic to detect if text is likely part of a table.
    
    Args:
        text: Text to check.
        
    Returns:
        True if text looks like table content.
    """
    # Multiple spaces often indicate columnar data
    if '    ' in text or '\t' in text:
        return True
    
    # Multiple numbers separated by spaces
    import re
    numbers = re.findall(r'\d+', text)
    if len(numbers) > 3 and len(text) < 200:
        return True
    
    return False


def _is_pdf_metadata(text: str) -> bool:
    """Detect common PDF metadata, headers, footers, and page numbers.
    
    Args:
        text: Text to check.
        
    Returns:
        True if text looks like PDF metadata that should be filtered out.
    """
    import re
    
    # Common patterns to filter
    patterns = [
        r'^Page\s+\d+\s+of\s+\d+',  # "Page X of Y"
        r'Fileid:.*source',  # PDF generation metadata
        r'^\d+:\d+\s*-\s*\d+-[A-Za-z]+-\d{4}$',  # Timestamps like "11:23 - 17-Dec-2024"
        r'^The type and rule above prints on all proofs',  # Common PDF proof text
        r'Userid:\s*\w+\s+Schema:',  # User/schema metadata
        r'Draft\s+Ok\s+to\s+Print',  # Draft markers
        r'Leadpct:\s*\d+%',  # PDF layout metadata
    ]
    
    for pattern in patterns:
        if re.search(pattern, text, re.IGNORECASE):
            return True
    
    # Very short lines that are just page numbers
    if re.match(r'^\d+$', text.strip()) and len(text.strip()) <= 3:
        return True
    
    return False


def _clean_html_content(html: str) -> str:
    """Deprecated - kept for compatibility.
    
    Previously cleaned PyMuPDF HTML output.
    Now we generate semantic HTML directly.
    """
    return html


def _markdown_to_html_tables(markdown: str) -> str:
    """Deprecated - no longer used. Kept for compatibility."""
    return markdown


def _convert_markdown_table_to_html(table_lines: list[str]) -> str:
    """Deprecated - no longer used. Kept for compatibility."""
    return "\n".join(table_lines)


def _get_pdf_page_count(pdf_path: str) -> tuple[int, dict[str, Any]]:
    """Get the page count and metadata of a PDF.

    Args:
        pdf_path: Path to the PDF file.

    Returns:
        Tuple of (page_count, metadata).
    """
    try:
        import pdfplumber
        
        with pdfplumber.open(pdf_path) as pdf:
            page_count = len(pdf.pages)
            metadata = pdf.metadata or {}
            return page_count, metadata
    except Exception:
        # Fallback to pymupdf if pdfplumber fails
        try:
            import pymupdf
            
            doc = pymupdf.open(pdf_path)
            page_count = doc.page_count
            metadata = dict(doc.metadata) if doc.metadata else {}
            doc.close()
            return page_count, metadata
        except Exception:
            return 0, {}


# ==============================================================================
# Main Loader Class
# ==============================================================================


class PDFHTMLLoader(BaseLoader):
    """High-performance PDF loader with semantic HTML output for complex tables.

    Uses pdfplumber for intelligent table extraction and semantic HTML generation.
    Supports true CPU parallelization for processing large documents.

    Features:
        - Converts PDF to semantic HTML with proper <table> structures
        - Intelligent table detection using pdfplumber
        - True parallel processing with ProcessPoolExecutor
        - Batch processing for memory efficiency
        - Superior table handling with actual HTML elements
        - Comprehensive observability and tracking

    Example:
        >>> loader = PDFHTMLLoader(max_workers=4, batch_size=10)
        >>> result = await loader.load(Path("document.pdf"))
        >>> print(f"Success rate: {result.success_rate:.1f}%")
        >>> print(f"Total time: {result.total_time_ms:.0f}ms")
        >>> docs = result.documents
    """

    supported_extensions = [".pdf"]

    def __init__(
        self,
        *,
        max_workers: int = 4,
        batch_size: int = 10,
        dpi: int = 150,
        ignore_images: bool = False,
        ignore_graphics: bool = False,
        force_text: bool = True,
        fontsize_limit: float = 3.0,
        header: bool = True,
        footer: bool = True,
        show_progress: bool = True,
        verbose: bool = False,
        encoding: str = "utf-8",
        parallel: bool = True,
    ) -> None:
        """Initialize the PDF HTML loader.

        Args:
            max_workers: Maximum parallel workers for CPU-bound processing.
            batch_size: Number of pages to process per batch.
            dpi: Image resolution for extraction.
            ignore_images: Whether to ignore images.
            ignore_graphics: Whether to ignore vector graphics.
            force_text: Extract text even over images/graphics.
            fontsize_limit: Minimum font size to consider.
            header: Include page headers in output (default: True).
            footer: Include page footers in output (default: True).
            show_progress: Print human-friendly progress messages.
            verbose: Enable structured logging for observability (default: False).
            encoding: Text encoding (not used for PDFs).
            parallel: Whether to process pages in parallel (default: True).
        """
        super().__init__(encoding)

        self.config = PDFConfig(
            max_workers=max_workers,
            batch_size=batch_size,
            dpi=dpi,
            ignore_images=ignore_images,
            ignore_graphics=ignore_graphics,
            force_text=force_text,
            fontsize_limit=fontsize_limit,
            header=header,
            footer=footer,
        )
        self._show_progress = show_progress
        self._verbose = verbose
        self._last_result: PDFProcessingResult | None = None
        self._parallel = parallel

        # Only create logger if verbose mode is enabled
        self._log: ObservabilityLogger | None = None
        if verbose:
            self._log = ObservabilityLogger(
                name="agenticflow.document.loaders.pdf_html",
                level=LogLevel.DEBUG,
            )
            self._log.set_context(
                loader="PDFHTMLLoader",
                max_workers=max_workers,
                batch_size=batch_size,
            )

    async def load(
        self,
        path: str | Path,
        *,
        tracking: bool = False,
        **kwargs: Any,
    ) -> list[Document] | PDFProcessingResult:
        """Load a PDF file and convert to HTML documents.

        Args:
            path: Path to the PDF file (str or Path).
            tracking: If True, return PDFProcessingResult with metrics.
                If False (default), return list of Documents.
            **kwargs: Additional options (overrides config), e.g. parallel=False to force
                sequential processing.

        Returns:
            list[Document] by default, or PDFProcessingResult if tracking=True.

        Raises:
            ImportError: If pdfplumber is not installed.
            FileNotFoundError: If PDF file doesn't exist.
        
        Example:
            >>> # Simple usage - load and save
            >>> await loader.load("doc.pdf")
            >>> loader.save("output.html")
            >>> 
            >>> # With tracking - get detailed metrics
            >>> result = await loader.load("doc.pdf", tracking=True)
            >>> print(f"Success: {result.success_rate:.0%}")
            >>> loader.save("output.html")
        """
        result = await self._load_with_tracking(path, **kwargs)
        self._last_result = result
        if tracking:
            return result
        return result.documents

    def save(
        self,
        output_path: str | Path,
        *,
        mode: str = "single",
        include_page_breaks: bool = True,
        include_page_numbers: bool = False,
        page_break_style: str = "hr",
        encoding: str = "utf-8",
    ) -> Path | list[Path]:
        """Save the last loaded documents to file(s).

        Args:
            output_path: Output file path (single/json) or directory (pages).
            mode: Save mode:
                - "single": One combined HTML file (default).
                - "pages": Separate file per document.
                - "json": Export as JSON with metadata.
            include_page_breaks: Add separators between documents (single mode).
            include_page_numbers: Add page/document number divs.
            page_break_style: Separator style ("hr", "div", or custom HTML).
            encoding: Text encoding for output files.

        Returns:
            Path to saved file (single/json) or list of paths (pages).

        Raises:
            RuntimeError: If no documents have been loaded yet.

        Example:
            >>> await loader.load("doc.pdf")
            >>> loader.save("output.html")
            >>> 
            >>> # Save with page numbers
            >>> loader.save("output.html", include_page_numbers=True)
            >>> 
            >>> # Save each page separately
            >>> loader.save("pages/", mode="pages")
        """
        if self._last_result is None:
            msg = "No documents loaded. Call load() first."
            raise RuntimeError(msg)

        return self._last_result.save(
            output_path,
            mode=mode,
            include_page_breaks=include_page_breaks,
            include_page_numbers=include_page_numbers,
            page_break_style=page_break_style,
            encoding=encoding,
        )

    async def _load_with_tracking(
        self,
        path: str | Path,
        **kwargs: Any,
    ) -> PDFProcessingResult:
        """Internal method to load PDF with full tracking."""
        path = Path(path)
        if not path.exists():
            msg = f"PDF file not found: {path}"
            raise FileNotFoundError(msg)

        start_time = time.perf_counter()

        if self._log:
            self._log.info("pdf_loading_started", file_path=str(path))

        # Get page count and metadata
        try:
            total_pages, metadata = await asyncio.to_thread(_get_pdf_page_count, str(path))
        except Exception as e:
            if self._log:
                self._log.error("pdf_metadata_failed", error=str(e))
            return PDFProcessingResult(
                file_path=path,
                status=PDFProcessingStatus.FAILED,
                total_pages=0,
                total_time_ms=(time.perf_counter() - start_time) * 1000,
            )

        if self._show_progress:
            print(f"📄 Loading PDF: {path.name} ({total_pages} pages)")

        # Override parallel setting if specified in kwargs
        parallel = kwargs.get("parallel", self._parallel)

        # Process pages
        if parallel and total_pages > 1:
            page_results = await self._process_parallel(path, total_pages)
        else:
            page_results = await self._process_sequential(path, total_pages)

        # Calculate metrics
        successful_pages = sum(1 for pr in page_results if pr.status == PageStatus.SUCCESS)
        failed_pages = sum(1 for pr in page_results if pr.status == PageStatus.FAILED)
        empty_pages = sum(1 for pr in page_results if pr.status == PageStatus.EMPTY)

        total_time_ms = (time.perf_counter() - start_time) * 1000

        if successful_pages == total_pages:
            status = PDFProcessingStatus.COMPLETED
        elif successful_pages > 0:
            status = PDFProcessingStatus.PARTIAL
        else:
            status = PDFProcessingStatus.FAILED

        result = PDFProcessingResult(
            file_path=path,
            status=status,
            total_pages=total_pages,
            successful_pages=successful_pages,
            failed_pages=failed_pages,
            empty_pages=empty_pages,
            page_results=page_results,
            total_time_ms=total_time_ms,
            metadata=metadata,
        )

        if self._show_progress:
            print(f"✅ Loaded {successful_pages}/{total_pages} pages in {total_time_ms:.0f}ms")

        if self._log:
            self._log.info(
                "pdf_loading_completed",
                total_pages=total_pages,
                successful_pages=successful_pages,
                failed_pages=failed_pages,
                total_time_ms=total_time_ms,
            )

        return result

    async def _process_parallel(
        self,
        path: Path,
        total_pages: int,
    ) -> list[PageResult]:
        """Process pages in parallel using ProcessPoolExecutor."""
        # Create batches of page numbers (0-based)
        page_numbers = list(range(total_pages))
        batches = [
            page_numbers[i : i + self.config.batch_size]
            for i in range(0, len(page_numbers), self.config.batch_size)
        ]

        if self._log:
            self._log.debug(
                "parallel_processing_started",
                total_pages=total_pages,
                batch_count=len(batches),
                batch_size=self.config.batch_size,
            )

        config_dict = {
            "force_text": self.config.force_text,
            "ignore_images": self.config.ignore_images,
            "ignore_graphics": self.config.ignore_graphics,
            "fontsize_limit": self.config.fontsize_limit,
            "dpi": self.config.dpi,
            "header": self.config.header,
            "footer": self.config.footer,
        }

        all_results: list[dict[str, Any]] = []

        # Process batches in parallel
        with ProcessPoolExecutor(max_workers=self.config.max_workers) as executor:
            worker_fn = partial(_process_page_batch, str(path), config_dict=config_dict)

            loop = asyncio.get_running_loop()
            tasks = [loop.run_in_executor(executor, worker_fn, batch) for batch in batches]

            batch_results = await asyncio.gather(*tasks)
            for batch_result in batch_results:
                all_results.extend(batch_result)

        # Convert to PageResult objects
        page_results = [
            PageResult(
                page_number=r["page_number"],
                status=PageStatus(r["status"]),
                content=r.get("content", ""),
                tables_count=r.get("tables_count", 0),
                images_count=r.get("images_count", 0),
                processing_time_ms=r.get("processing_time_ms", 0.0),
                error=r.get("error"),
            )
            for r in all_results
        ]

        # Sort by page number to maintain order
        page_results.sort(key=lambda pr: pr.page_number)

        return page_results

    async def _process_sequential(
        self,
        path: Path,
        total_pages: int,
    ) -> list[PageResult]:
        """Process pages sequentially (for debugging or single-page PDFs)."""
        if self._log:
            self._log.debug("sequential_processing_started", total_pages=total_pages)

        page_numbers = list(range(total_pages))

        config_dict = {
            "force_text": self.config.force_text,
            "ignore_images": self.config.ignore_images,
            "ignore_graphics": self.config.ignore_graphics,
            "fontsize_limit": self.config.fontsize_limit,
            "dpi": self.config.dpi,
            "header": self.config.header,
            "footer": self.config.footer,
        }

        result_dicts = await asyncio.to_thread(
            _process_page_batch,
            str(path),
            page_numbers,
            config_dict,
        )

        page_results = [
            PageResult(
                page_number=r["page_number"],
                status=PageStatus(r["status"]),
                content=r.get("content", ""),
                tables_count=r.get("tables_count", 0),
                images_count=r.get("images_count", 0),
                processing_time_ms=r.get("processing_time_ms", 0.0),
                error=r.get("error"),
            )
            for r in result_dicts
        ]

        return page_results


__all__ = [
    "PDFHTMLLoader",
    "PDFProcessingResult",
    "PDFConfig",
    "PageResult",
    "PageStatus",
    "PDFProcessingStatus",
    "OutputFormat",
    "ProcessingMetrics",
]
