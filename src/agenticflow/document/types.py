"""Core types for document processing.

This module defines the fundamental data structures used throughout
the document processing pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


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
class Document:
    """A loaded document with content and metadata.
    
    This is the primary data structure for representing loaded documents
    before they are split into chunks.
    
    Attributes:
        content: The text content of the document.
        metadata: Metadata about the document including source, type, etc.
        
    Example:
        >>> doc = Document(
        ...     content="Hello, World!",
        ...     metadata={"source": "example.txt", "file_type": ".txt"}
        ... )
        >>> print(doc.content)
        Hello, World!
    """
    
    content: str
    metadata: dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self) -> None:
        """Ensure required metadata fields exist."""
        if "source" not in self.metadata:
            self.metadata["source"] = "unknown"
    
    def __len__(self) -> int:
        """Return the length of the content."""
        return len(self.content)
    
    def __repr__(self) -> str:
        """Return a readable representation."""
        preview = self.content[:50] + "..." if len(self.content) > 50 else self.content
        source = self.metadata.get("source", "unknown")
        return f"Document(content='{preview}', source='{source}')"
    
    def to_dict(self) -> dict[str, Any]:
        """Convert document to dictionary.
        
        Returns:
            Dictionary with content and metadata.
        """
        return {
            "content": self.content,
            "metadata": self.metadata,
        }
    
    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Document:
        """Create document from dictionary.
        
        Args:
            data: Dictionary with content and optional metadata.
            
        Returns:
            New Document instance.
        """
        return cls(
            content=data.get("content", ""),
            metadata=data.get("metadata", {}),
        )


@dataclass
class TextChunk:
    """A chunk of text with metadata and position information.
    
    This is the primary data structure for representing split text
    after document processing.
    
    Attributes:
        content: The text content of the chunk.
        metadata: Metadata inherited from source document plus chunk info.
        start_index: Character position in original text (optional).
        end_index: End character position in original text (optional).
        
    Example:
        >>> chunk = TextChunk(
        ...     content="First paragraph.",
        ...     metadata={"chunk_index": 0, "source": "doc.txt"},
        ...     start_index=0,
        ...     end_index=16
        ... )
    """
    
    content: str
    metadata: dict[str, Any] = field(default_factory=dict)
    start_index: int | None = None
    end_index: int | None = None
    
    def __len__(self) -> int:
        """Return the length of the content."""
        return len(self.content)
    
    def __repr__(self) -> str:
        """Return a readable representation."""
        preview = self.content[:50] + "..." if len(self.content) > 50 else self.content
        return f"TextChunk(len={len(self.content)}, content='{preview}')"
    
    def to_dict(self) -> dict[str, Any]:
        """Convert chunk to dictionary.
        
        Returns:
            Dictionary with all chunk data.
        """
        return {
            "content": self.content,
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
        return cls(
            content=data.get("content", ""),
            metadata=data.get("metadata", {}),
            start_index=data.get("start_index"),
            end_index=data.get("end_index"),
        )
    
    def to_document(self) -> Document:
        """Convert chunk to a Document.
        
        Useful for further processing or when APIs expect Document type.
        
        Returns:
            Document with chunk content and metadata.
        """
        return Document(content=self.content, metadata=self.metadata.copy())


__all__ = [
    "Document",
    "TextChunk",
    "FileType",
    "SplitterType",
]
