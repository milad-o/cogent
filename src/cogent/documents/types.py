"""Types for document processing.

This module defines data structures used in the document processing pipeline.
For core Document and DocumentMetadata types, see cogent.core.document.
"""

from __future__ import annotations

from enum import StrEnum


class FileType(StrEnum):
    """Supported file types for document loading.

    Each value maps to the file extension (without dot).
    """

    # Text
    TEXT = "txt"
    MARKDOWN = "md"
    RST = "rst"
    LOG = "log"

    # Documents
    PDF = "pdf"
    DOCX = "docx"
    DOC = "doc"

    # Data
    CSV = "csv"
    TSV = "tsv"
    JSON = "json"
    JSONL = "jsonl"
    XLSX = "xlsx"
    XLS = "xls"

    # Web
    HTML = "html"
    HTM = "htm"
    XML = "xml"

    # Code
    PYTHON = "py"
    JAVASCRIPT = "js"
    TYPESCRIPT = "ts"
    JAVA = "java"
    CPP = "cpp"
    C = "c"
    GO = "go"
    RUST = "rs"
    RUBY = "rb"
    PHP = "php"
    SWIFT = "swift"
    KOTLIN = "kt"
    SCALA = "scala"
    SQL = "sql"
    SHELL = "sh"
    YAML = "yaml"
    TOML = "toml"
    CSS = "css"

    @classmethod
    def from_extension(cls, ext: str) -> FileType | None:
        """Get FileType from file extension.

        Args:
            ext: File extension (with or without leading dot).

        Returns:
            Matching FileType or None if not found.
        """
        ext = ext.lstrip(".").lower()

        # Handle aliases
        aliases = {
            "markdown": "md",
            "text": "txt",
            "ndjson": "jsonl",
            "jsx": "js",
            "tsx": "ts",
            "h": "c",
            "hpp": "cpp",
            "bash": "sh",
            "zsh": "sh",
            "yml": "yaml",
            "scss": "css",
            "less": "css",
        }
        ext = aliases.get(ext, ext)

        for file_type in cls:
            if file_type.value == ext:
                return file_type
        return None


class SplitterType(StrEnum):
    """Types of text splitters available."""

    RECURSIVE = "recursive"
    CHARACTER = "character"
    SENTENCE = "sentence"
    MARKDOWN = "markdown"
    HTML = "html"
    CODE = "code"
    SEMANTIC = "semantic"
    TOKEN = "token"


__all__ = [
    "FileType",
    "SplitterType",
]
