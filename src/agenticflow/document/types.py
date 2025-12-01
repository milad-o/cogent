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


def _generate_doc_id(text: str) -> str:
    """Generate a unique ID based on content hash."""
    import hashlib
    import uuid
    content_hash = hashlib.sha256(text.encode()).hexdigest()[:8]
    unique_suffix = uuid.uuid4().hex[:8]
    return f"doc_{content_hash}_{unique_suffix}"


@dataclass
class Document:
    """A document with text content, metadata, and optional embedding.
    
    This is THE unified document type used throughout agenticflow for:
    - Loading documents from files
    - Storing in vector stores
    - Retrieval results
    
    Attributes:
        text: The text content of the document (primary field).
        metadata: Metadata about the document (source, type, etc.).
        embedding: Optional pre-computed embedding vector.
        id: Unique identifier (auto-generated if not provided).
        
    Example:
        >>> doc = Document(text="Hello, World!")
        >>> doc.id
        'doc_a1b2c3...'
        
        >>> # With metadata
        >>> doc = Document(
        ...     text="Python is great",
        ...     metadata={"source": "tutorial.md", "language": "en"},
        ... )
    """
    
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)
    embedding: list[float] | None = None
    id: str = ""
    
    def __post_init__(self) -> None:
        """Generate ID if not provided, ensure source in metadata."""
        if not self.id:
            self.id = _generate_doc_id(self.text)
        if "source" not in self.metadata:
            self.metadata["source"] = "unknown"
    
    # Alias for backward compatibility with document loaders
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
        source = self.metadata.get("source", "unknown")
        return f"Document(id={self.id!r}, text={preview!r}, source={source!r})"
    
    def to_dict(self) -> dict[str, Any]:
        """Convert document to dictionary.
        
        Returns:
            Dictionary with id, text, metadata, and embedding.
        """
        return {
            "id": self.id,
            "text": self.text,
            "metadata": self.metadata,
            "embedding": self.embedding,
        }
    
    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Document:
        """Create document from dictionary.
        
        Args:
            data: Dictionary with text/content and optional metadata.
            
        Returns:
            New Document instance.
        """
        # Support both 'text' and 'content' keys
        text = data.get("text") or data.get("content", "")
        return cls(
            text=text,
            metadata=data.get("metadata", {}),
            embedding=data.get("embedding"),
            id=data.get("id", ""),
        )


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
        return Document(text=self.text, metadata=self.metadata.copy())


__all__ = [
    "Document",
    "TextChunk",
    "FileType",
    "SplitterType",
]
