"""High-performance PDF loader optimized for LLM consumption.

This module provides a modern PDF-to-Markdown loader using pymupdf4llm,
with true CPU parallelization for processing multiple pages concurrently.

Features:
- Converts PDF to clean Markdown format (best for LLM/RAG)
- True CPU parallelization using ProcessPoolExecutor
- Batch processing with configurable chunk sizes
- No OCR (assumes text-based PDFs) for speed
- Comprehensive observability with structured logging
- Progress tracking and metrics

Example:
    >>> from agenticflow.document.loaders.handlers.pdf_llm import PDFMarkdownLoader
    >>> 
    >>> loader = PDFMarkdownLoader(max_workers=4, batch_size=10)
    >>> docs = await loader.load(Path("large_document.pdf"))
    >>> print(f"Loaded {len(docs)} pages as Markdown")

Note:
    Requires: uv add pymupdf4llm pymupdf
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

    MARKDOWN = "markdown"
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
        output_format: Output format (markdown, text, json).
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
    output_format: OutputFormat = OutputFormat.MARKDOWN
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
        content: Extracted content (Markdown/text).
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
    total_pages: int = 0
    successful_pages: int = 0
    failed_pages: int = 0
    empty_pages: int = 0
    page_results: list[PageResult] = field(default_factory=list)
    total_time_ms: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def success_rate(self) -> float:
        """Calculate the success rate as a ratio (0.0-1.0)."""
        if self.total_pages == 0:
            return 0.0
        return self.successful_pages / self.total_pages

    @property
    def documents(self) -> list[Document]:
        """Get successful page results as Documents."""
        return self.to_documents()

    def to_documents(self) -> list[Document]:
        """Convert successful page results to Documents."""
        documents: list[Document] = []
        for result in self.page_results:
            if result.status == PageStatus.SUCCESS and result.content.strip():
                doc = Document(
                    text=result.content,
                    metadata={
                        "source": str(self.file_path),
                        "filename": self.file_path.name,
                        "file_type": ".pdf",
                        "page": result.page_number,
                        "total_pages": self.total_pages,
                        "tables_count": result.tables_count,
                        "images_count": result.images_count,
                        "format": "markdown",
                        **self.metadata,
                    },
                )
                documents.append(doc)
        return documents

    def to_markdown(
        self,
        *,
        include_page_breaks: bool = True,
        include_page_numbers: bool = False,
        page_break_style: str = "---",
    ) -> str:
        """Convert all pages to a single Markdown string.
        
        Args:
            include_page_breaks: Add separators between pages (default: True).
            include_page_numbers: Add page number headers (default: False).
            page_break_style: Separator style ("---", "***", "===", or custom).
            
        Returns:
            Complete Markdown content as a single string.
            
        Example:
            >>> result = await loader.load_with_tracking("doc.pdf")
            >>> markdown = result.to_markdown(include_page_numbers=True)
        """
        parts: list[str] = []
        
        for page_result in self.page_results:
            if page_result.status != PageStatus.SUCCESS:
                continue
            if not page_result.content.strip():
                continue
            
            if include_page_numbers:
                parts.append(f"<!-- Page {page_result.page_number} -->")
            
            parts.append(page_result.content.strip())
        
        separator = f"\n\n{page_break_style}\n\n" if include_page_breaks else "\n\n"
        return separator.join(parts)

    def save(
        self,
        output_path: str | Path,
        *,
        mode: str = "single",
        include_page_breaks: bool = True,
        include_page_numbers: bool = False,
        page_break_style: str = "---",
        encoding: str = "utf-8",
    ) -> Path | list[Path]:
        """Save extracted content to file(s).
        
        Args:
            output_path: Output file path (for single mode) or directory (for pages mode).
            mode: Save mode:
                - "single": One combined Markdown file (default).
                - "pages": Separate file per page.
                - "json": Export as JSON with metadata.
            include_page_breaks: Add separators between pages (single mode only).
            include_page_numbers: Add page number comments (single mode only).
            page_break_style: Separator style ("---", "***", "===", or custom).
            encoding: Text encoding for output files.
            
        Returns:
            Path to saved file (single/json mode) or list of paths (pages mode).
            
        Example:
            >>> result = await loader.load_with_tracking("doc.pdf")
            >>> 
            >>> # Save as single Markdown file
            >>> result.save("output.md")
            >>> 
            >>> # Save each page separately
            >>> result.save("output_dir/", mode="pages")
            >>> 
            >>> # Save with page numbers in comments
            >>> result.save("output.md", include_page_numbers=True)
            >>> 
            >>> # Export as JSON
            >>> result.save("output.json", mode="json")
        """
        import json
        
        output_path = Path(output_path)
        
        if mode == "single":
            # Combine all pages into one file
            content = self.to_markdown(
                include_page_breaks=include_page_breaks,
                include_page_numbers=include_page_numbers,
                page_break_style=page_break_style,
            )
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(content, encoding=encoding)
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
                
                page_path = output_path / f"{stem}_page_{page_result.page_number:04d}.md"
                page_path.write_text(page_result.content, encoding=encoding)
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
    It uses pymupdf4llm to convert pages to Markdown.

    Args:
        pdf_path: Path to the PDF file.
        page_numbers: 0-based page numbers to process.
        config_dict: Configuration as dict (must be picklable).

    Returns:
        List of page result dictionaries.
    """
    try:
        # Import pymupdf.layout BEFORE pymupdf4llm to enable advanced layout features
        # This activates header/footer extraction and disables legacy mode warnings
        try:
            import pymupdf.layout  # noqa: F401
        except ImportError:
            pass  # pymupdf_layout not installed, will use legacy mode

        import pymupdf4llm
    except ImportError as e:
        return [
            {
                "page_number": pn + 1,
                "status": "failed",
                "content": "",
                "error": f"pymupdf4llm not installed: {e}",
                "processing_time_ms": 0.0,
            }
            for pn in page_numbers
        ]

    results: list[dict[str, Any]] = []

    for page_num in page_numbers:
        start_time = time.perf_counter()

        try:
            # Use page_chunks=True for per-page output (works in legacy mode)
            # In layout mode, result is a string regardless of page_chunks
            result = pymupdf4llm.to_markdown(
                pdf_path,
                pages=[page_num],
                page_chunks=True,
                force_text=config_dict.get("force_text", True),
                ignore_images=config_dict.get("ignore_images", False),
                ignore_graphics=config_dict.get("ignore_graphics", False),
                fontsize_limit=config_dict.get("fontsize_limit", 3.0),
                dpi=config_dict.get("dpi", 150),
                header=config_dict.get("header", True),
                footer=config_dict.get("footer", True),
                use_ocr=False,
            )

            elapsed_ms = (time.perf_counter() - start_time) * 1000

            # Handle both legacy mode (list[dict]) and layout mode (str)
            if isinstance(result, str):
                # Layout mode returns string directly
                content = result
                tables_count = 0
                images_count = 0
            elif result and len(result) > 0:
                # Legacy mode returns list of dicts
                chunk = result[0]
                content = chunk.get("text", "")
                tables_count = len(chunk.get("tables", []))
                images_count = len(chunk.get("images", []))
            else:
                content = ""
                tables_count = 0
                images_count = 0

            if content.strip():
                results.append(
                    {
                        "page_number": page_num + 1,
                        "status": "success",
                        "content": content,
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

    return results


def _get_pdf_page_count(pdf_path: str) -> tuple[int, dict[str, Any]]:
    """Get the page count and metadata of a PDF.

    Args:
        pdf_path: Path to the PDF file.

    Returns:
        Tuple of (page_count, metadata).
    """
    import pymupdf

    doc = pymupdf.open(pdf_path)
    page_count = doc.page_count
    metadata = dict(doc.metadata) if doc.metadata else {}
    doc.close()
    return page_count, metadata


# ==============================================================================
# Main Loader Class
# ==============================================================================


class PDFMarkdownLoader(BaseLoader):
    """High-performance PDF loader with Markdown output for LLM/RAG.

    Uses pymupdf4llm for optimal text extraction and conversion to Markdown.
    Supports true CPU parallelization for processing large documents.

    Features:
        - Converts PDF to clean Markdown (best for LLMs)
        - True parallel processing with ProcessPoolExecutor
        - Batch processing for memory efficiency
        - Comprehensive observability and tracking
        - Auto-detects pymupdf_layout for enhanced extraction

    Example:
        >>> loader = PDFMarkdownLoader(max_workers=4, batch_size=10)
        >>> result = await loader.load(Path("document.pdf"))
        >>> print(f"Success rate: {result.success_rate:.1f}%")
        >>> print(f"Total time: {result.total_time_ms:.0f}ms")
        >>> docs = result.to_documents()
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
    ) -> None:
        """Initialize the PDF Markdown loader.

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

        # Only create logger if verbose mode is enabled
        self._log: ObservabilityLogger | None = None
        if verbose:
            self._log = ObservabilityLogger(
                name="agenticflow.document.loaders.pdf",
                level=LogLevel.DEBUG,
            )
            self._log.set_context(
                loader="PDFMarkdownLoader",
                max_workers=max_workers,
                batch_size=batch_size,
            )

    async def load(
        self,
        path: str | Path,
        *,
        save_to: str | Path | None = None,
        save_mode: str = "single",
        include_page_breaks: bool = True,
        include_page_numbers: bool = False,
        page_break_style: str = "---",
        **kwargs: Any,
    ) -> list[Document]:
        """Load a PDF file and convert to Markdown documents.

        Args:
            path: Path to the PDF file (str or Path).
            save_to: Optional path to save output. If provided, saves the result.
            save_mode: Save mode when save_to is set:
                - "single": One combined Markdown file (default).
                - "pages": Separate file per page (save_to should be a directory).
                - "json": Export as JSON with metadata.
            include_page_breaks: Add separators between pages (single mode).
            include_page_numbers: Add page number comments.
            page_break_style: Separator style ("---", "***", "===").
            **kwargs: Additional options (overrides config).

        Returns:
            List of Document objects (one per page with content).

        Raises:
            ImportError: If pymupdf4llm is not installed.
            FileNotFoundError: If PDF file doesn't exist.
        
        Example:
            >>> # Just load
            >>> docs = await loader.load("doc.pdf")
            >>> 
            >>> # Load and save to file
            >>> docs = await loader.load("doc.pdf", save_to="output.md")
            >>> 
            >>> # Load and save each page separately
            >>> docs = await loader.load("doc.pdf", save_to="pages/", save_mode="pages")
            >>> 
            >>> # Load and save as JSON
            >>> docs = await loader.load("doc.pdf", save_to="output.json", save_mode="json")
        """
        result = await self._load_with_tracking(path, **kwargs)
        
        if save_to is not None:
            result.save(
                save_to,
                mode=save_mode,
                include_page_breaks=include_page_breaks,
                include_page_numbers=include_page_numbers,
                page_break_style=page_break_style,
            )
        
        return result.documents

    async def load_with_tracking(
        self,
        path: str | Path,
        **kwargs: Any,
    ) -> PDFProcessingResult:
        """Load a PDF with full processing tracking and metrics.
        
        Use this method when you need detailed information about the
        PDF processing, such as success rate, timing, or failed pages.

        Args:
            path: Path to the PDF file (str or Path).
            **kwargs: Additional options (overrides config).

        Returns:
            PDFProcessingResult with documents and metrics.
            
        Example:
            >>> result = await loader.load_with_tracking("doc.pdf")
            >>> print(f"Success rate: {result.success_rate:.0%}")
            >>> print(f"Time: {result.total_time_ms:.0f}ms")
            >>> docs = result.documents
        """
        return await self._load_with_tracking(path, **kwargs)

    async def _load_with_tracking(
        self,
        path: str | Path,
        **kwargs: Any,
    ) -> PDFProcessingResult:
        """Load a PDF with full processing tracking.

        This method provides detailed metrics and status tracking
        for each page processed.

        Args:
            path: Path to the PDF file (str or Path).
            **kwargs: Override configuration options.

        Returns:
            PDFProcessingResult with detailed tracking information.
        """
        # Convert str to Path if needed
        if isinstance(path, str):
            path = Path(path)
        
        if not path.exists():
            raise FileNotFoundError(f"PDF file not found: {path}")

        start_time = time.perf_counter()
        show_progress = kwargs.pop("show_progress", self._show_progress)

        if self._log:
            self._log.info("PDF processing started", file=str(path))

        if show_progress:
            print(f"      📖 Loading {path.name}...")

        try:
            # Get page count and metadata
            loop = asyncio.get_running_loop()
            page_count, doc_metadata = await loop.run_in_executor(
                None,
                _get_pdf_page_count,
                str(path),
            )

            if self._log:
                self._log.info(
                    "PDF metadata loaded",
                    file=str(path),
                    page_count=page_count,
                    title=doc_metadata.get("title", ""),
                )

            if show_progress:
                title = doc_metadata.get("title", "")
                title_str = f" - {title}" if title else ""
                print(f"      📄 {page_count} pages{title_str}")

            if page_count == 0:
                return PDFProcessingResult(
                    file_path=path,
                    status=PDFProcessingStatus.COMPLETED,
                    total_pages=0,
                    metadata=doc_metadata,
                    total_time_ms=(time.perf_counter() - start_time) * 1000,
                )

            # Process pages in parallel batches
            page_results = await self._process_pages_parallel(
                path,
                page_count,
                **kwargs,
            )

            # Aggregate results
            total_time_ms = (time.perf_counter() - start_time) * 1000
            successful = sum(1 for r in page_results if r.status == PageStatus.SUCCESS)
            failed = sum(1 for r in page_results if r.status == PageStatus.FAILED)
            empty = sum(1 for r in page_results if r.status == PageStatus.EMPTY)

            # Determine overall status
            if failed == 0:
                status = PDFProcessingStatus.COMPLETED
            elif successful > 0:
                status = PDFProcessingStatus.PARTIAL
            else:
                status = PDFProcessingStatus.FAILED

            result = PDFProcessingResult(
                file_path=path,
                status=status,
                total_pages=page_count,
                successful_pages=successful,
                failed_pages=failed,
                empty_pages=empty,
                page_results=page_results,
                total_time_ms=total_time_ms,
                metadata=doc_metadata,
            )

            if self._log:
                self._log.info(
                    "PDF processing completed",
                    file=str(path),
                    status=status.value,
                    total_pages=page_count,
                    successful_pages=successful,
                    failed_pages=failed,
                    empty_pages=empty,
                    total_time_ms=total_time_ms,
                    success_rate=result.success_rate,
                )

            if show_progress:
                pages_per_sec = page_count / (total_time_ms / 1000) if total_time_ms > 0 else 0
                print(f"      ✓ {successful}/{page_count} pages in {total_time_ms/1000:.1f}s ({pages_per_sec:.1f} pages/sec)")

            return result

        except Exception as e:
            if self._log:
                self._log.error("PDF processing failed", file=str(path), error=str(e))
            if show_progress:
                print(f"      ✗ Error: {e}")
            return PDFProcessingResult(
                file_path=path,
                status=PDFProcessingStatus.FAILED,
                total_time_ms=(time.perf_counter() - start_time) * 1000,
                metadata={"error": str(e)},
            )

    async def _process_pages_parallel(
        self,
        path: Path,
        page_count: int,
        **kwargs: Any,
    ) -> list[PageResult]:
        """Process all pages in parallel batches.

        Uses ProcessPoolExecutor for true CPU parallelization.

        Args:
            path: Path to PDF file.
            page_count: Total number of pages.
            **kwargs: Override configuration.

        Returns:
            List of PageResult for each page.
        """
        # Build config dict for worker processes
        config_dict = {
            "force_text": kwargs.get("force_text", self.config.force_text),
            "ignore_images": kwargs.get("ignore_images", self.config.ignore_images),
            "ignore_graphics": kwargs.get(
                "ignore_graphics", self.config.ignore_graphics
            ),
            "fontsize_limit": kwargs.get("fontsize_limit", self.config.fontsize_limit),
            "dpi": kwargs.get("dpi", self.config.dpi),
            "header": kwargs.get("header", self.config.header),
            "footer": kwargs.get("footer", self.config.footer),
        }

        batch_size = kwargs.get("batch_size", self.config.batch_size)
        max_workers = kwargs.get("max_workers", self.config.max_workers)

        # Create batches of page numbers
        all_pages = list(range(page_count))
        batches: list[list[int]] = [
            all_pages[i : i + batch_size] for i in range(0, len(all_pages), batch_size)
        ]

        if self._log:
            self._log.debug(
                "Processing batches",
                batch_count=len(batches),
            )

        loop = asyncio.get_running_loop()
        all_results: list[PageResult] = []

        # Process batches in parallel using ProcessPoolExecutor
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            # Create tasks for all batches
            process_func = partial(
                _process_page_batch,
                str(path),
                config_dict=config_dict,
            )

            futures = [
                loop.run_in_executor(executor, process_func, batch) for batch in batches
            ]

            # Gather all results
            batch_results = await asyncio.gather(*futures, return_exceptions=True)

            for batch_idx, result in enumerate(batch_results):
                if isinstance(result, Exception):
                    # Handle batch-level failure
                    if self._log:
                        self._log.error(
                            "Batch processing failed",
                            batch_index=batch_idx,
                            error=str(result),
                        )
                    # Mark all pages in batch as failed
                    for page_num in batches[batch_idx]:
                        all_results.append(
                            PageResult(
                                page_number=page_num + 1,
                                status=PageStatus.FAILED,
                                error=str(result),
                            )
                        )
                else:
                    # Convert dict results to PageResult objects
                    for page_dict in result:
                        status_str = page_dict.get("status", "failed")
                        status = PageStatus[status_str.upper()]

                        all_results.append(
                            PageResult(
                                page_number=page_dict["page_number"],
                                status=status,
                                content=page_dict.get("content", ""),
                                tables_count=page_dict.get("tables_count", 0),
                                images_count=page_dict.get("images_count", 0),
                                processing_time_ms=page_dict.get(
                                    "processing_time_ms", 0.0
                                ),
                                error=page_dict.get("error"),
                            )
                        )

        # Sort by page number
        all_results.sort(key=lambda r: r.page_number)
        return all_results

    def get_metrics(self, result: PDFProcessingResult) -> ProcessingMetrics:
        """Calculate processing metrics from a result.

        Args:
            result: PDFProcessingResult from load_with_tracking.

        Returns:
            ProcessingMetrics with performance data.
        """
        total_pages = result.total_pages
        total_time_ms = result.total_time_ms

        avg_page_time = total_time_ms / total_pages if total_pages > 0 else 0.0
        pages_per_second = (
            (total_pages / (total_time_ms / 1000)) if total_time_ms > 0 else 0.0
        )
        batch_count = (
            total_pages + self.config.batch_size - 1
        ) // self.config.batch_size

        return ProcessingMetrics(
            total_pages=total_pages,
            total_time_ms=total_time_ms,
            avg_page_time_ms=avg_page_time,
            pages_per_second=pages_per_second,
            batch_count=batch_count,
        )


__all__ = [
    "PDFMarkdownLoader",
    "PDFConfig",
    "PDFProcessingResult",
    "PageResult",
    "ProcessingMetrics",
    "PDFProcessingStatus",
    "PageStatus",
    "OutputFormat",
]
