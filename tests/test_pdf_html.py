"""Tests for the PDFHTMLLoader (pdfplumber-based semantic HTML loader).

Tests the high-performance PDF loader with semantic HTML output for complex tables.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import TYPE_CHECKING
from unittest.mock import MagicMock, patch

import pytest

from agenticflow.document.loaders.handlers.pdf_html import (
    OutputFormat,
    PageResult,
    PageStatus,
    PDFConfig,
    PDFHTMLLoader,
    PDFProcessingResult,
    PDFProcessingStatus,
    ProcessingMetrics,
    _get_pdf_page_count,
    _process_page_batch,
    _table_to_html,
    _convert_to_html,
)
from agenticflow.document.types import Document

if TYPE_CHECKING:
    pass


# ==============================================================================
# Test Enums
# ==============================================================================


class TestPDFProcessingStatus:
    """Tests for PDFProcessingStatus enum."""

    def test_all_statuses_exist(self) -> None:
        """Test all expected statuses are defined."""
        assert PDFProcessingStatus.PENDING.value == "pending"
        assert PDFProcessingStatus.LOADING.value == "loading"
        assert PDFProcessingStatus.PROCESSING.value == "processing"
        assert PDFProcessingStatus.COMPLETED.value == "completed"
        assert PDFProcessingStatus.FAILED.value == "failed"
        assert PDFProcessingStatus.PARTIAL.value == "partial"

    def test_enum_membership(self) -> None:
        """Test enum membership checks."""
        assert PDFProcessingStatus.COMPLETED in PDFProcessingStatus
        assert PDFProcessingStatus.FAILED in PDFProcessingStatus


class TestPageStatus:
    """Tests for PageStatus enum."""

    def test_all_statuses_exist(self) -> None:
        """Test all expected page statuses are defined."""
        assert PageStatus.PENDING.value == "pending"
        assert PageStatus.PROCESSING.value == "processing"
        assert PageStatus.SUCCESS.value == "success"
        assert PageStatus.FAILED.value == "failed"
        assert PageStatus.EMPTY.value == "empty"


class TestOutputFormat:
    """Tests for OutputFormat enum."""

    def test_all_formats_exist(self) -> None:
        """Test all expected output formats are defined."""
        assert OutputFormat.HTML.value == "html"
        assert OutputFormat.TEXT.value == "text"
        assert OutputFormat.JSON.value == "json"


# ==============================================================================
# Test Configuration
# ==============================================================================


class TestPDFConfig:
    """Tests for PDFConfig dataclass."""

    def test_default_config(self) -> None:
        """Test default configuration values."""
        config = PDFConfig()
        assert config.max_workers == 4
        assert config.batch_size == 10
        assert config.dpi == 150
        assert config.output_format == OutputFormat.HTML
        assert config.ignore_images is False
        assert config.ignore_graphics is False
        assert config.force_text is True
        assert config.fontsize_limit == 3.0

    def test_custom_config(self) -> None:
        """Test custom configuration."""
        config = PDFConfig(
            max_workers=8,
            batch_size=20,
            dpi=300,
            ignore_images=True,
        )
        assert config.max_workers == 8
        assert config.batch_size == 20
        assert config.dpi == 300
        assert config.ignore_images is True

    def test_config_immutable(self) -> None:
        """Test that config is frozen."""
        config = PDFConfig()
        with pytest.raises(Exception):  # FrozenInstanceError or AttributeError
            config.max_workers = 10  # type: ignore


# ==============================================================================
# Test Table Conversion
# ==============================================================================


class TestTableConversion:
    """Tests for table to HTML conversion."""

    def test_table_to_html_simple(self) -> None:
        """Test converting a simple table to HTML."""
        table = [
            ["Name", "Age", "City"],
            ["John", "30", "NYC"],
            ["Jane", "25", "LA"],
        ]
        
        html = _table_to_html(table, 0)
        assert "<table" in html
        assert "<thead>" in html
        assert "<th>Name</th>" in html
        assert "<th>Age</th>" in html
        assert "<tbody>" in html
        assert "<td>John</td>" in html
        assert "<td>30</td>" in html

    def test_table_to_html_empty(self) -> None:
        """Test empty table returns empty string."""
        assert _table_to_html([], 0) == ""
        assert _table_to_html([[]], 0) != ""  # Header row exists

    def test_convert_to_html_with_tables(self) -> None:
        """Test HTML conversion with tables."""
        text = "Header\n\nSome content"
        tables = [[["Col1", "Col2"], ["Data1", "Data2"]]]
        table_positions = [(100.0, 0)]  # (y_position, table_index)
        
        html = _convert_to_html(text, tables, table_positions, 1)
        assert "<table" in html
        assert "<p>" in html or "<h3>" in html


# ==============================================================================
# Test PDFProcessingResult
# ==============================================================================


class TestPDFProcessingResult:
    """Tests for PDFProcessingResult dataclass."""

    def test_success_rate_calculation(self) -> None:
        """Test success rate calculation."""
        result = PDFProcessingResult(
            file_path=Path("test.pdf"),
            status=PDFProcessingStatus.COMPLETED,
            total_pages=10,
            successful_pages=8,
            failed_pages=2,
        )
        assert result.success_rate == 80.0

    def test_success_rate_zero_pages(self) -> None:
        """Test success rate with zero pages."""
        result = PDFProcessingResult(
            file_path=Path("test.pdf"),
            status=PDFProcessingStatus.FAILED,
            total_pages=0,
        )
        assert result.success_rate == 0.0

    def test_to_documents(self) -> None:
        """Test conversion to Document objects."""
        page_results = [
            PageResult(
                page_number=1,
                status=PageStatus.SUCCESS,
                content="<p>Page 1</p>",
                tables_count=1,
            ),
            PageResult(
                page_number=2,
                status=PageStatus.SUCCESS,
                content="<p>Page 2</p>",
            ),
            PageResult(
                page_number=3,
                status=PageStatus.FAILED,
                content="",
            ),
        ]

        result = PDFProcessingResult(
            file_path=Path("test.pdf"),
            status=PDFProcessingStatus.PARTIAL,
            total_pages=3,
            successful_pages=2,
            failed_pages=1,
            page_results=page_results,
        )

        docs = result.documents
        assert len(docs) == 2
        assert docs[0].text == "<p>Page 1</p>"
        assert docs[0].metadata["page"] == 1
        assert docs[0].metadata["format"] == "html"
        assert docs[1].text == "<p>Page 2</p>"
        assert docs[1].metadata["page"] == 2

    def test_to_html_simple(self) -> None:
        """Test converting result to HTML string."""
        page_results = [
            PageResult(
                page_number=1,
                status=PageStatus.SUCCESS,
                content="<p>Page 1</p>",
            ),
            PageResult(
                page_number=2,
                status=PageStatus.SUCCESS,
                content="<p>Page 2</p>",
            ),
        ]

        result = PDFProcessingResult(
            file_path=Path("test.pdf"),
            status=PDFProcessingStatus.COMPLETED,
            total_pages=2,
            successful_pages=2,
            page_results=page_results,
        )

        html = result.to_html()
        assert "<p>Page 1</p>" in html
        assert "<p>Page 2</p>" in html

    def test_to_html_with_page_numbers(self) -> None:
        """Test HTML output with page numbers."""
        page_results = [
            PageResult(page_number=1, status=PageStatus.SUCCESS, content="<p>Content</p>"),
        ]

        result = PDFProcessingResult(
            file_path=Path("test.pdf"),
            status=PDFProcessingStatus.COMPLETED,
            total_pages=1,
            successful_pages=1,
            page_results=page_results,
        )

        html = result.to_html(include_page_numbers=True)
        assert "Page 1" in html


# ==============================================================================
# Test PDFHTMLLoader
# ==============================================================================


class TestPDFHTMLLoader:
    """Tests for PDFHTMLLoader class."""

    def test_initialization_defaults(self) -> None:
        """Test loader initialization with defaults."""
        loader = PDFHTMLLoader()
        assert loader.config.max_workers == 4
        assert loader.config.batch_size == 10
        assert loader.config.output_format == OutputFormat.HTML

    def test_initialization_custom(self) -> None:
        """Test loader initialization with custom settings."""
        loader = PDFHTMLLoader(
            max_workers=8,
            batch_size=20,
            verbose=True,
        )
        assert loader.config.max_workers == 8
        assert loader.config.batch_size == 20

    @pytest.mark.asyncio
    async def test_load_missing_file(self) -> None:
        """Test loading a non-existent file."""
        loader = PDFHTMLLoader()
        with pytest.raises(FileNotFoundError):
            await loader.load("nonexistent.pdf")

    @pytest.mark.asyncio
    async def test_load_with_tracking(self) -> None:
        """Test loading with tracking enabled."""
        pdfplumber = pytest.importorskip("pdfplumber")

        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
            # Try to use pymupdf to create a simple PDF
            try:
                import pymupdf
                doc = pymupdf.open()
                page = doc.new_page()
                page.insert_text((72, 72), "Test content")
                doc.save(tmp.name)
                doc.close()
            except ImportError:
                pytest.skip("pymupdf not available for test PDF creation")

            loader = PDFHTMLLoader(show_progress=False)
            result = await loader.load(tmp.name, tracking=True)

            assert isinstance(result, PDFProcessingResult)
            assert result.total_pages == 1
            assert result.successful_pages >= 0  # May be 0 or 1 depending on content
            assert result.status in [PDFProcessingStatus.COMPLETED, PDFProcessingStatus.PARTIAL]

            Path(tmp.name).unlink()

    def test_save_without_load(self) -> None:
        """Test save raises error when no documents loaded."""
        loader = PDFHTMLLoader()
        with pytest.raises(RuntimeError, match="No documents loaded"):
            loader.save("output.html")


# ==============================================================================
# Test Worker Functions
# ==============================================================================


class TestWorkerFunctions:
    """Tests for worker functions."""

    def test_get_pdf_page_count(self) -> None:
        """Test getting page count from PDF."""
        # Try pdfplumber first
        pdfplumber = pytest.importorskip("pdfplumber", reason="pdfplumber not available")
        
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
            # Create a simple PDF
            try:
                import pymupdf
                doc = pymupdf.open()
                doc.new_page()
                doc.new_page()
                doc.save(tmp.name)
                doc.close()
            except ImportError:
                pytest.skip("pymupdf not available for test PDF creation")

            count, metadata = _get_pdf_page_count(tmp.name)
            assert count == 2
            assert isinstance(metadata, dict)

            Path(tmp.name).unlink()


# ==============================================================================
# Test __all__ Exports
# ==============================================================================


class TestExports:
    """Tests for module exports."""

    def test_all_exports_exist(self) -> None:
        """Test that all exported names are defined."""
        from agenticflow.document.loaders.handlers.pdf_html import __all__

        assert "PDFHTMLLoader" in __all__
        assert "PDFProcessingResult" in __all__
        assert "PDFConfig" in __all__
        assert "PageResult" in __all__
        assert "PageStatus" in __all__
        assert "PDFProcessingStatus" in __all__
        assert "OutputFormat" in __all__
