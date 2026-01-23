"""Types for document processing.

This module defines data structures used in the document processing pipeline.
For core Document and DocumentMetadata types, see agenticflow.core.document.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from agenticflow.core import Document


class FileType(Enum):
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


class SplitterType(Enum):
    """Types of text splitters available."""

    RECURSIVE = "recursive"
    CHARACTER = "character"
    SENTENCE = "sentence"
    MARKDOWN = "markdown"
    HTML = "html"
    CODE = "code"
    SEMANTIC = "semantic"
    TOKEN = "token"


@dataclass
class TextChunk:
    """A chunk of text with metadata and position information.

    This is the data structure for representing split text after
    document processing. Use `to_document()` to convert to Document
    for vector store indexing.

    Attributes:
        text: The text content of the chunk (primary field).
        metadata: Metadata inherited from source document plus chunk info.
        start_index: Character position in original text (optional).
        end_index: End character position in original text (optional).

    Example:
        >>> chunk = TextChunk(
        ...     text="First paragraph.",
        ...     metadata={"chunk_index": 0, "source": "doc.txt"},
        ...     start_index=0,
        ...     end_index=16,
        ... )
        >>> doc = chunk.to_document()  # Convert for indexing
    """

    text: str
    metadata: dict[str, Any] = field(default_factory=dict)
    start_index: int | None = None
    end_index: int | None = None

    # Alias for backward compatibility
    @property
    def content(self) -> str:
        """Alias for text (backward compatibility)."""
        return self.text

    @content.setter
    def content(self, value: str) -> None:
        """Set text via content alias."""
        self.text = value

    def __len__(self) -> int:
        """Return the length of the text."""
        return len(self.text)

    def __repr__(self) -> str:
        """Return a readable representation."""
        preview = self.text[:50] + "..." if len(self.text) > 50 else self.text
        return f"TextChunk(len={len(self.text)}, text='{preview}')"

    def to_dict(self) -> dict[str, Any]:
        """Convert chunk to dictionary.

        Returns:
            Dictionary with all chunk data.
        """
        return {
            "text": self.text,
            "metadata": self.metadata,
            "start_index": self.start_index,
            "end_index": self.end_index,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TextChunk:
        """Create chunk from dictionary.

        Args:
            data: Dictionary with chunk data.

        Returns:
            New TextChunk instance.
        """
        text = data.get("text") or data.get("content", "")
        return cls(
            text=text,
            metadata=data.get("metadata", {}),
            start_index=data.get("start_index"),
            end_index=data.get("end_index"),
        )

    def to_document(self) -> Document:
        """Convert chunk to a Document for vector store indexing.

        Returns:
            Document with chunk text and metadata.
        """
        from agenticflow.core import DocumentMetadata
        
        # Handle both dict and DocumentMetadata
        if isinstance(self.metadata, DocumentMetadata):
            metadata = self.metadata
        else:
            metadata = DocumentMetadata.from_dict(self.metadata)
        
        return Document(text=self.text, metadata=metadata)


__all__ = [
    "TextChunk",
    "FileType",
    "SplitterType",
]
