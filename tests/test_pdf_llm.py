"""Tests for the PDFMarkdownLoader (pymupdf4llm-based loader).

Tests the high-performance PDF loader with true CPU parallelization.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import TYPE_CHECKING
from unittest.mock import MagicMock, patch

import pytest

from agenticflow.document.loaders.handlers.pdf_llm import (
    OutputFormat,
    PageResult,
    PageStatus,
    PDFConfig,
    PDFMarkdownLoader,
    PDFProcessingResult,
    PDFProcessingStatus,
    ProcessingMetrics,
    _get_pdf_page_count,
    _process_page_batch,
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
        """Test enum values can be accessed by name."""
        assert PDFProcessingStatus["COMPLETED"] == PDFProcessingStatus.COMPLETED
        assert PDFProcessingStatus["PARTIAL"] == PDFProcessingStatus.PARTIAL


class TestPageStatus:
    """Tests for PageStatus enum."""

    def test_all_statuses_exist(self) -> None:
        """Test all expected page statuses are defined."""
        assert PageStatus.PENDING.value == "pending"
        assert PageStatus.PROCESSING.value == "processing"
        assert PageStatus.SUCCESS.value == "success"
        assert PageStatus.FAILED.value == "failed"
        assert PageStatus.EMPTY.value == "empty"

    def test_enum_from_string(self) -> None:
        """Test converting string to enum."""
        assert PageStatus["SUCCESS"] == PageStatus.SUCCESS
        assert PageStatus["FAILED"] == PageStatus.FAILED
        assert PageStatus["EMPTY"] == PageStatus.EMPTY


class TestOutputFormat:
    """Tests for OutputFormat enum."""

    def test_all_formats_exist(self) -> None:
        """Test all expected output formats are defined."""
        assert OutputFormat.MARKDOWN.value == "markdown"
        assert OutputFormat.TEXT.value == "text"
        assert OutputFormat.JSON.value == "json"


# ==============================================================================
# Test Dataclasses
# ==============================================================================


class TestPDFConfig:
    """Tests for PDFConfig dataclass."""

    def test_default_values(self) -> None:
        """Test default configuration values."""
        config = PDFConfig()

        assert config.max_workers == 4
        assert config.batch_size == 10
        assert config.dpi == 150
        assert config.output_format == OutputFormat.MARKDOWN
        assert config.ignore_images is False
        assert config.ignore_graphics is False
        assert config.force_text is True
        assert config.fontsize_limit == 3.0

    def test_custom_values(self) -> None:
        """Test custom configuration values."""
        config = PDFConfig(
            max_workers=8,
            batch_size=20,
            dpi=300,
            output_format=OutputFormat.TEXT,
            ignore_images=True,
            force_text=False,
        )

        assert config.max_workers == 8
        assert config.batch_size == 20
        assert config.dpi == 300
        assert config.output_format == OutputFormat.TEXT
        assert config.ignore_images is True
        assert config.force_text is False

    def test_frozen_immutability(self) -> None:
        """Test that config is immutable (frozen=True)."""
        config = PDFConfig()

        with pytest.raises(AttributeError):
            config.max_workers = 8  # type: ignore[misc]


class TestPageResult:
    """Tests for PageResult dataclass."""

    def test_create_success_result(self) -> None:
        """Test creating a successful page result."""
        result = PageResult(
            page_number=1,
            status=PageStatus.SUCCESS,
            content="# Page Title\n\nSome content here.",
            tables_count=2,
            images_count=1,
            processing_time_ms=45.5,
        )

        assert result.page_number == 1
        assert result.status == PageStatus.SUCCESS
        assert result.content == "# Page Title\n\nSome content here."
        assert result.tables_count == 2
        assert result.images_count == 1
        assert result.processing_time_ms == 45.5
        assert result.error is None

    def test_create_failed_result(self) -> None:
        """Test creating a failed page result."""
        result = PageResult(
            page_number=5,
            status=PageStatus.FAILED,
            error="Failed to extract text",
            processing_time_ms=10.0,
        )

        assert result.page_number == 5
        assert result.status == PageStatus.FAILED
        assert result.content == ""
        assert result.error == "Failed to extract text"

    def test_create_empty_result(self) -> None:
        """Test creating an empty page result."""
        result = PageResult(
            page_number=3,
            status=PageStatus.EMPTY,
            processing_time_ms=5.0,
        )

        assert result.status == PageStatus.EMPTY
        assert result.content == ""
        assert result.error is None


class TestPDFProcessingResult:
    """Tests for PDFProcessingResult dataclass."""

    def test_create_completed_result(self) -> None:
        """Test creating a completed processing result."""
        page_results = [
            PageResult(page_number=1, status=PageStatus.SUCCESS, content="Page 1"),
            PageResult(page_number=2, status=PageStatus.SUCCESS, content="Page 2"),
        ]

        result = PDFProcessingResult(
            file_path=Path("/test/doc.pdf"),
            status=PDFProcessingStatus.COMPLETED,
            total_pages=2,
            successful_pages=2,
            failed_pages=0,
            empty_pages=0,
            page_results=page_results,
            total_time_ms=100.0,
        )

        assert result.status == PDFProcessingStatus.COMPLETED
        assert result.total_pages == 2
        assert result.successful_pages == 2
        assert result.success_rate == 1.0

    def test_success_rate_calculation(self) -> None:
        """Test success rate calculation."""
        result = PDFProcessingResult(
            file_path=Path("/test/doc.pdf"),
            status=PDFProcessingStatus.PARTIAL,
            total_pages=10,
            successful_pages=7,
            failed_pages=2,
            empty_pages=1,
        )

        assert result.success_rate == 0.7

    def test_success_rate_zero_pages(self) -> None:
        """Test success rate with zero pages."""
        result = PDFProcessingResult(
            file_path=Path("/test/doc.pdf"),
            status=PDFProcessingStatus.COMPLETED,
            total_pages=0,
        )

        assert result.success_rate == 0.0

    def test_to_documents(self) -> None:
        """Test converting page results to Documents."""
        page_results = [
            PageResult(
                page_number=1,
                status=PageStatus.SUCCESS,
                content="# Title\n\nContent",
                tables_count=1,
            ),
            PageResult(
                page_number=2,
                status=PageStatus.EMPTY,
                content="",
            ),
            PageResult(
                page_number=3,
                status=PageStatus.SUCCESS,
                content="More content",
            ),
        ]

        result = PDFProcessingResult(
            file_path=Path("/test/document.pdf"),
            status=PDFProcessingStatus.COMPLETED,
            total_pages=3,
            successful_pages=2,
            empty_pages=1,
            page_results=page_results,
            metadata={"title": "Test Document"},
        )

        docs = result.to_documents()

        assert len(docs) == 2  # Only successful pages with content
        assert all(isinstance(d, Document) for d in docs)
        assert docs[0].text == "# Title\n\nContent"
        assert docs[0].metadata["page"] == 1
        assert docs[0].metadata["total_pages"] == 3
        assert docs[0].metadata["tables_count"] == 1
        assert docs[0].metadata["format"] == "markdown"
        assert docs[0].metadata["title"] == "Test Document"


class TestProcessingMetrics:
    """Tests for ProcessingMetrics dataclass."""

    def test_create_metrics(self) -> None:
        """Test creating processing metrics."""
        metrics = ProcessingMetrics(
            total_pages=100,
            total_time_ms=5000.0,
            avg_page_time_ms=50.0,
            pages_per_second=20.0,
            batch_count=10,
        )

        assert metrics.total_pages == 100
        assert metrics.total_time_ms == 5000.0
        assert metrics.avg_page_time_ms == 50.0
        assert metrics.pages_per_second == 20.0
        assert metrics.batch_count == 10

    def test_frozen_immutability(self) -> None:
        """Test that metrics is immutable (frozen=True)."""
        metrics = ProcessingMetrics(
            total_pages=100,
            total_time_ms=5000.0,
            avg_page_time_ms=50.0,
            pages_per_second=20.0,
            batch_count=10,
        )

        with pytest.raises(AttributeError):
            metrics.total_pages = 200  # type: ignore[misc]


# ==============================================================================
# Test Worker Functions
# ==============================================================================


class TestProcessPageBatch:
    """Tests for _process_page_batch worker function."""

    def test_returns_error_when_pymupdf4llm_not_available(self) -> None:
        """Test graceful handling when pymupdf4llm is not installed."""
        with patch.dict("sys.modules", {"pymupdf4llm": None}):
            # Force reimport to trigger ImportError
            import importlib
            import sys

            # This test verifies the error handling structure
            # The actual import happens inside the function

    def test_batch_processing_structure(self) -> None:
        """Test that batch results have correct structure."""
        # Mock pymupdf4llm.to_markdown
        mock_chunks = [
            {
                "text": "# Page Content\n\nSome text here.",
                "tables": [],
                "images": [],
            }
        ]

        with patch("pymupdf4llm.to_markdown", return_value=mock_chunks):
            config_dict = {
                "include_header": False,
                "include_footer": False,
                "use_ocr": False,
                "force_text": True,
                "ignore_images": False,
                "ignore_graphics": False,
                "fontsize_limit": 3.0,
                "dpi": 150,
            }

            results = _process_page_batch(
                "/fake/path.pdf",
                [0, 1],  # 0-based page numbers
                config_dict,
            )

            assert len(results) == 2
            for result in results:
                assert "page_number" in result
                assert "status" in result
                assert "content" in result
                assert "processing_time_ms" in result


class TestGetPdfPageCount:
    """Tests for _get_pdf_page_count helper function."""

    def test_returns_page_count_and_metadata(self) -> None:
        """Test that function returns page count and metadata."""
        # Create a mock document
        mock_doc = MagicMock()
        mock_doc.page_count = 10
        mock_doc.metadata = {"title": "Test PDF", "author": "Test Author"}

        with patch("pymupdf.open", return_value=mock_doc):
            page_count, metadata = _get_pdf_page_count("/fake/path.pdf")

            assert page_count == 10
            assert metadata["title"] == "Test PDF"
            assert metadata["author"] == "Test Author"
            mock_doc.close.assert_called_once()


# ==============================================================================
# Test PDFMarkdownLoader
# ==============================================================================


class TestPDFMarkdownLoader:
    """Tests for PDFMarkdownLoader class."""

    def test_init_default_config(self) -> None:
        """Test loader initialization with default config."""
        loader = PDFMarkdownLoader()

        assert loader.config.max_workers == 4
        assert loader.config.batch_size == 10
        assert loader.config.dpi == 150

    def test_init_custom_config(self) -> None:
        """Test loader initialization with custom config."""
        loader = PDFMarkdownLoader(
            max_workers=8,
            batch_size=20,
            dpi=300,
            ignore_images=True,
        )

        assert loader.config.max_workers == 8
        assert loader.config.batch_size == 20
        assert loader.config.dpi == 300
        assert loader.config.ignore_images is True

    def test_supported_extensions(self) -> None:
        """Test that loader supports .pdf extension."""
        assert ".pdf" in PDFMarkdownLoader.supported_extensions

    @pytest.mark.asyncio
    async def test_load_file_not_found(self) -> None:
        """Test that FileNotFoundError is raised for missing files."""
        loader = PDFMarkdownLoader()

        with pytest.raises(FileNotFoundError, match="PDF file not found"):
            await loader.load(Path("/nonexistent/file.pdf"))

    @pytest.mark.asyncio
    async def test_load_with_tracking_file_not_found(self) -> None:
        """Test load_with_tracking raises FileNotFoundError."""
        loader = PDFMarkdownLoader()

        with pytest.raises(FileNotFoundError):
            await loader.load_with_tracking(Path("/nonexistent/file.pdf"))

    def test_get_metrics(self) -> None:
        """Test metrics calculation from processing result."""
        loader = PDFMarkdownLoader(batch_size=10)

        result = PDFProcessingResult(
            file_path=Path("/test/doc.pdf"),
            status=PDFProcessingStatus.COMPLETED,
            total_pages=100,
            successful_pages=100,
            total_time_ms=5000.0,
        )

        metrics = loader.get_metrics(result)

        assert metrics.total_pages == 100
        assert metrics.total_time_ms == 5000.0
        assert metrics.avg_page_time_ms == 50.0
        assert metrics.pages_per_second == 20.0
        assert metrics.batch_count == 10

    def test_get_metrics_zero_pages(self) -> None:
        """Test metrics calculation with zero pages."""
        loader = PDFMarkdownLoader()

        result = PDFProcessingResult(
            file_path=Path("/test/doc.pdf"),
            status=PDFProcessingStatus.COMPLETED,
            total_pages=0,
            total_time_ms=100.0,
        )

        metrics = loader.get_metrics(result)

        assert metrics.avg_page_time_ms == 0.0
        assert metrics.pages_per_second == 0.0

    def test_get_metrics_zero_time(self) -> None:
        """Test metrics calculation with zero processing time."""
        loader = PDFMarkdownLoader()

        result = PDFProcessingResult(
            file_path=Path("/test/doc.pdf"),
            status=PDFProcessingStatus.COMPLETED,
            total_pages=10,
            total_time_ms=0.0,
        )

        metrics = loader.get_metrics(result)

        assert metrics.pages_per_second == 0.0


# ==============================================================================
# Integration Tests (require actual PDF)
# ==============================================================================


class TestPDFMarkdownLoaderIntegration:
    """Integration tests for PDFMarkdownLoader.

    These tests require pymupdf and pymupdf4llm to be installed,
    and will create/use actual PDF files.
    """

    @pytest.fixture
    def sample_pdf(self, tmp_path: Path) -> Path:
        """Create a simple PDF for testing."""
        try:
            import pymupdf
        except ImportError:
            pytest.skip("pymupdf not installed")

        pdf_path = tmp_path / "sample.pdf"
        doc = pymupdf.open()  # New empty document

        # Add a page with text
        page = doc.new_page()
        text_point = pymupdf.Point(50, 50)
        page.insert_text(text_point, "Sample PDF Content for Testing")

        # Add another page
        page2 = doc.new_page()
        page2.insert_text(text_point, "Second page content")

        doc.save(str(pdf_path))
        doc.close()

        return pdf_path

    @pytest.mark.asyncio
    async def test_load_real_pdf(self, sample_pdf: Path) -> None:
        """Test loading an actual PDF file."""
        loader = PDFMarkdownLoader(max_workers=2, batch_size=1)
        docs = await loader.load(sample_pdf)

        assert len(docs) >= 1
        assert all(isinstance(d, Document) for d in docs)

    @pytest.mark.asyncio
    async def test_load_with_tracking_real_pdf(self, sample_pdf: Path) -> None:
        """Test load_with_tracking on actual PDF."""
        loader = PDFMarkdownLoader(max_workers=2, batch_size=1)
        result = await loader.load_with_tracking(sample_pdf)

        assert result.status in (
            PDFProcessingStatus.COMPLETED,
            PDFProcessingStatus.PARTIAL,
        )
        assert result.total_pages == 2
        assert result.total_time_ms > 0
        assert len(result.page_results) == 2

    @pytest.mark.asyncio
    async def test_parallel_processing(self, sample_pdf: Path) -> None:
        """Test that parallel processing works correctly."""
        loader = PDFMarkdownLoader(max_workers=4, batch_size=1)
        result = await loader.load_with_tracking(sample_pdf)

        # All pages should be processed
        assert len(result.page_results) == result.total_pages

        # Pages should be sorted by page number
        page_numbers = [r.page_number for r in result.page_results]
        assert page_numbers == sorted(page_numbers)


# ==============================================================================
# Test Module Exports
# ==============================================================================


class TestModuleExports:
    """Tests for module-level exports."""

    def test_all_exports_available(self) -> None:
        """Test that all expected symbols are exported."""
        from agenticflow.document.loaders.handlers.pdf_llm import __all__

        expected = [
            "PDFMarkdownLoader",
            "PDFConfig",
            "PDFProcessingResult",
            "PageResult",
            "ProcessingMetrics",
            "PDFProcessingStatus",
            "PageStatus",
            "OutputFormat",
        ]

        for name in expected:
            assert name in __all__

    def test_imports_from_loaders_package(self) -> None:
        """Test that exports are available from loaders package."""
        from agenticflow.document.loaders import (
            OutputFormat,
            PageResult,
            PageStatus,
            PDFConfig,
            PDFMarkdownLoader,
            PDFProcessingResult,
            PDFProcessingStatus,
            ProcessingMetrics,
        )

        assert PDFMarkdownLoader is not None
        assert PDFConfig is not None
        assert PDFProcessingResult is not None
        assert PageResult is not None
        assert ProcessingMetrics is not None
        assert PDFProcessingStatus is not None
        assert PageStatus is not None
        assert OutputFormat is not None
