"""Document loaders submodule.

This module provides document loading capabilities for various file formats.

**Supported Formats:**
- Text: .txt, .md, .rst, .log
- Documents: .pdf, .docx
- Data: .csv, .json, .jsonl, .xlsx
- Web: .html, .htm
- Code: .py, .js, .ts, .java, .cpp, .go, .rs, and more

**Usage:**
    >>> from agenticflow.document.loaders import DocumentLoader, load_documents
    >>> 
    >>> # Using DocumentLoader class
    >>> loader = DocumentLoader()
    >>> docs = await loader.load("document.pdf")
    >>> docs = await loader.load_directory("./docs", glob="**/*.md")
    >>> 
    >>> # Using convenience function
    >>> docs = await load_documents("./data")
"""

from agenticflow.document.loaders.base import BaseLoader
from agenticflow.document.loaders.handlers import (
    CodeLoader,
    CSVLoader,
    HTMLLoader,
    JSONLoader,
    MarkdownLoader,
    PDFLoader,
    TextLoader,
    WordLoader,
    XLSXLoader,
)
from agenticflow.document.loaders.loader import DocumentLoader
from agenticflow.document.loaders.registry import LOADERS, get_loader, register_loader
from agenticflow.document.loaders.utils import (
    load_documents,
    load_documents_sync,
)

__all__ = [
    # Core
    "BaseLoader",
    "DocumentLoader",
    # Handlers
    "TextLoader",
    "MarkdownLoader",
    "HTMLLoader",
    "PDFLoader",
    "WordLoader",
    "CSVLoader",
    "JSONLoader",
    "XLSXLoader",
    "CodeLoader",
    # Registry
    "LOADERS",
    "get_loader",
    "register_loader",
    # Convenience
    "load_documents",
    "load_documents_sync",
]
