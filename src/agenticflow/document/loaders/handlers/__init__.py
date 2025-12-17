"""File type handlers for document loading.

This submodule contains specialized loaders for different file types.
Each loader inherits from BaseLoader and implements format-specific loading logic.
"""

from agenticflow.document.loaders.handlers.code import CodeLoader
from agenticflow.document.loaders.handlers.csv import CSVLoader
from agenticflow.document.loaders.handlers.html import HTMLLoader
from agenticflow.document.loaders.handlers.json import JSONLoader
from agenticflow.document.loaders.handlers.markdown import MarkdownLoader
from agenticflow.document.loaders.handlers.pdf import PDFLoader
from agenticflow.document.loaders.handlers.pdf_llm import (
    OutputFormat,
    PageResult,
    PageStatus,
    PDFConfig,
    PDFMarkdownLoader,
    PDFProcessingResult,
    PDFProcessingStatus,
    ProcessingMetrics,
)
from agenticflow.document.loaders.handlers.pdf_vision import PDFVisionLoader, PDFVisionOptions
from agenticflow.document.loaders.handlers.text import TextLoader
from agenticflow.document.loaders.handlers.word import WordLoader
from agenticflow.document.loaders.handlers.xlsx import XLSXLoader

__all__ = [
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
]

