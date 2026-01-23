"""Core document types for AgenticFlow.

This module defines the fundamental document structures used throughout
the framework for RAG, retrieval, and document processing.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import Any


@dataclass
class DocumentMetadata:
    """Structured metadata for documents with provenance and tracking.

    This replaces the old dict[str, Any] metadata with a typed, validated
    structure that provides:
    - Type safety with IDE autocomplete
    - Consistent field names across the codebase
    - Provenance tracking (who created, from what source)
    - Chunking awareness (parent/child relationships)
    - Content metrics (token counts, character counts)

    Attributes:
        id: Unique document identifier (auto-generated)
        timestamp: Unix timestamp when document was created
        source: Source identifier (file path, URL, etc.)
        source_type: Type of source ("pdf", "markdown", "web", "api", etc.)
        page: Page number (for paginated documents like PDFs)
        chunk_index: Index of this chunk in sequence (0-based)
        chunk_total: Total number of chunks from parent document
        start_char: Starting character position in parent document
        end_char: Ending character position in parent document
        token_count: Number of tokens in document text
        char_count: Number of characters in document text
        loader: Name of loader that created this document
        created_by: Agent or tool that created this document
        parent_id: ID of parent document (for chunks)
        custom: Additional user-defined metadata fields

    Example:
        >>> metadata = DocumentMetadata(
        ...     source="report.pdf",
        ...     source_type="pdf",
        ...     page=5,
        ...     loader="PDFMarkdownLoader"
        ... )
        >>> metadata.id
        'doc_a1b2c3d4e5f6g7h8'
    """

    # Identification & timing
    id: str = field(default_factory=lambda: f"doc_{uuid.uuid4().hex[:16]}")
    timestamp: float = field(default_factory=time.time)

    # Source information
    source: str = "unknown"
    source_type: str | None = None

    # Content positioning (for chunked documents)
    page: int | None = None
    chunk_index: int | None = None
    chunk_total: int | None = None
    start_char: int | None = None
    end_char: int | None = None

    # Content metrics
    token_count: int | None = None
    char_count: int | None = None

    # Provenance & relationships
    loader: str | None = None
    created_by: str | None = None
    parent_id: str | None = None

    # Custom fields (extensibility)
    custom: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert metadata to dictionary for serialization.

        Returns:
            Dictionary with all metadata fields, excluding None values.
        """
        data = {
            "id": self.id,
            "timestamp": self.timestamp,
            "source": self.source,
        }

        # Optional fields (only include if not None)
        if self.source_type is not None:
            data["source_type"] = self.source_type
        if self.page is not None:
            data["page"] = self.page
        if self.chunk_index is not None:
            data["chunk_index"] = self.chunk_index
        if self.chunk_total is not None:
            data["chunk_total"] = self.chunk_total
        if self.start_char is not None:
            data["start_char"] = self.start_char
        if self.end_char is not None:
            data["end_char"] = self.end_char
        if self.token_count is not None:
            data["token_count"] = self.token_count
        if self.char_count is not None:
            data["char_count"] = self.char_count
        if self.loader is not None:
            data["loader"] = self.loader
        if self.created_by is not None:
            data["created_by"] = self.created_by
        if self.parent_id is not None:
            data["parent_id"] = self.parent_id
        if self.custom:
            data["custom"] = self.custom

        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> DocumentMetadata:
        """Create DocumentMetadata from dictionary.

        For backward compatibility, any fields not recognized as standard
        metadata fields will be placed in the custom dict.

        Args:
            data: Dictionary with metadata fields.

        Returns:
            New DocumentMetadata instance.
        """
        # Standard fields that DocumentMetadata recognizes
        standard_fields = {
            "id", "timestamp", "source", "source_type", "page",
            "chunk_index", "chunk_total", "start_char", "end_char",
            "token_count", "char_count", "loader", "created_by",
            "parent_id", "custom"
        }
        
        # Collect unknown fields into custom
        custom = data.get("custom", {}).copy() if data.get("custom") else {}
        for key, value in data.items():
            if key not in standard_fields:
                custom[key] = value
        
        return cls(
            id=data.get("id", f"doc_{uuid.uuid4().hex[:16]}"),
            timestamp=data.get("timestamp", time.time()),
            source=data.get("source", "unknown"),
            source_type=data.get("source_type"),
            page=data.get("page"),
            chunk_index=data.get("chunk_index"),
            chunk_total=data.get("chunk_total"),
            start_char=data.get("start_char"),
            end_char=data.get("end_char"),
            token_count=data.get("token_count"),
            char_count=data.get("char_count"),
            loader=data.get("loader"),
            created_by=data.get("created_by"),
            parent_id=data.get("parent_id"),
            custom=custom,
        )


@dataclass
class Document:
    """A document with text content, structured metadata, and optional embedding.

    This is THE unified document type used throughout agenticflow for:
    - Loading documents from files
    - Storing in vector stores
    - Retrieval results

    Attributes:
        text: The text content of the document (primary field).
        metadata: Structured metadata with type-safe fields.
        embedding: Optional pre-computed embedding vector.

    Example:
        >>> doc = Document(text="Hello, World!")
        >>> doc.id  # Auto-generated via metadata
        'doc_a1b2c3d4e5f6g7h8'

        >>> # With structured metadata
        >>> doc = Document(
        ...     text="Python is great",
        ...     metadata=DocumentMetadata(
        ...         source="tutorial.md",
        ...         source_type="markdown",
        ...         custom={"language": "en"}
        ...     )
        ... )
        >>> doc.source
        'tutorial.md'
        >>> doc.metadata.source_type
        'markdown'
    """

    text: str
    metadata: DocumentMetadata = field(default_factory=DocumentMetadata)
    embedding: list[float] | None = None

    def __post_init__(self) -> None:
        """Auto-populate char_count in metadata."""
        if self.metadata.char_count is None:
            self.metadata.char_count = len(self.text)

    # Convenience properties for common metadata access
    @property
    def id(self) -> str:
        """Get document ID from metadata."""
        return self.metadata.id

    @property
    def source(self) -> str:
        """Get source from metadata."""
        return self.metadata.source

    @property
    def page(self) -> int | None:
        """Get page number from metadata."""
        return self.metadata.page

    @property
    def chunk_index(self) -> int | None:
        """Get chunk index from metadata."""
        return self.metadata.chunk_index

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
        return f"Document(id={self.id!r}, text={preview!r}, source={self.source!r})"

    def to_dict(self) -> dict[str, Any]:
        """Convert document to dictionary.

        Returns:
            Dictionary with text, metadata, and embedding.
        """
        return {
            "text": self.text,
            "metadata": self.metadata.to_dict(),
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
        
        # Handle metadata (dict or DocumentMetadata)
        metadata_data = data.get("metadata", {})
        if isinstance(metadata_data, DocumentMetadata):
            metadata = metadata_data
        elif isinstance(metadata_data, dict):
            metadata = DocumentMetadata.from_dict(metadata_data)
        else:
            metadata = DocumentMetadata()
        
        return cls(
            text=text,
            metadata=metadata,
            embedding=data.get("embedding"),
        )


__all__ = [
    "Document",
    "DocumentMetadata",
]
