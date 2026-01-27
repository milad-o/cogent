"""PDF loaders with multiple processing strategies."""

from cogent.documents.loaders.pdf.pdf import PDFLoader
from cogent.documents.loaders.pdf.pdf_markdown import (
    OutputFormat,
    PageResult,
    PageStatus,
    PDFConfig,
    PDFMarkdownLoader,
    PDFProcessingResult,
    PDFProcessingStatus,
    ProcessingMetrics,
)
from cogent.documents.loaders.pdf.pdf_vision import (
    PDFVisionLoader,
    PDFVisionOptions,
)

__all__ = [
    "PDFLoader",
    "PDFMarkdownLoader",
    "PDFVisionLoader",
    # PDF Markdown types
    "PDFConfig",
    "PDFProcessingResult",
    "PageResult",
    "ProcessingMetrics",
    "PDFProcessingStatus",
    "PageStatus",
    "OutputFormat",
    "PDFVisionOptions",
]
