"""Document loaders submodule.

This module provides document loading capabilities for various file formats.

**Supported Formats:**
- Text: .txt, .md, .rst, .log
- Documents: .pdf, .docx
- Data: .csv, .json, .jsonl, .xlsx
- Web: .html, .htm
- Code: .py, .js, .ts, .java, .cpp, .go, .rs, and more

**Usage:**
    >>> from cogent.documents.loaders import DocumentLoader, load_documents
    >>>
    >>> # Using DocumentLoader class
    >>> loader = DocumentLoader()
    >>> docs = await loader.load("document.pdf")
    >>> docs = await loader.load_directory("./docs", glob="**/*.md")
    >>>
    >>> # Using convenience function
    >>> docs = await load_documents("./data")
    >>>
    >>> # High-performance PDF loading with Markdown output (for LLM/RAG)
    >>> from cogent.documents.loaders import PDFMarkdownLoader
    >>> loader = PDFMarkdownLoader(max_workers=4, batch_size=10)
    >>> docs = await loader.load("large_document.pdf")
"""

from cogent.documents.loaders.base import BaseLoader
from cogent.documents.loaders.code import CodeLoader
from cogent.documents.loaders.data import CSVLoader, JSONLoader, XLSXLoader
from cogent.documents.loaders.loader import DocumentLoader
from cogent.documents.loaders.markup import HTMLLoader, MarkdownLoader
from cogent.documents.loaders.pdf import (
    OutputFormat,
    PageResult,
    PageStatus,
    PDFConfig,
    PDFLoader,
    PDFMarkdownLoader,
    PDFProcessingResult,
    PDFProcessingStatus,
    PDFVisionLoader,
    PDFVisionOptions,
    ProcessingMetrics,
)
from cogent.documents.loaders.plaintext import TextLoader
from cogent.documents.loaders.registry import LOADERS, get_loader, register_loader
from cogent.documents.loaders.utils import (
    load_documents,
    load_documents_sync,
)
from cogent.documents.loaders.word import WordLoader

__all__ = [
    # Core
    "BaseLoader",
    "DocumentLoader",
    # Handlers
    "TextLoader",
    "MarkdownLoader",
    "HTMLLoader",
    "PDFLoader",
    "PDFMarkdownLoader",
    "PDFVisionLoader",
    "WordLoader",
    "CSVLoader",
    "JSONLoader",
    "XLSXLoader",
    "CodeLoader",
    # PDF LLM types
    "PDFConfig",
    "PDFProcessingResult",
    "PageResult",
    "ProcessingMetrics",
    "PDFProcessingStatus",
    "PageStatus",
    "OutputFormat",
    "PDFVisionOptions",
    # Registry
    "LOADERS",
    "get_loader",
    "register_loader",
    # Convenience
    "load_documents",
    "load_documents_sync",
]
