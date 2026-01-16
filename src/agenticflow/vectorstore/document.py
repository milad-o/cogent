"""Document class for vector store.

Re-exports the unified Document class from agenticflow.document.
Also provides utility functions for document creation and splitting.
"""

from __future__ import annotations

from typing import Any

# Re-export the unified Document class
from agenticflow.document.types import Document

__all__ = [
    "Document",
    "create_documents",
    "split_text",
    "split_documents",
]


def create_documents(
    texts: list[str],
    metadatas: list[dict[str, Any]] | None = None,
    ids: list[str] | None = None,
) -> list[Document]:
    """Create multiple documents from texts.

    Args:
        texts: List of text contents.
        metadatas: Optional list of metadata dicts (one per text).
        ids: Optional list of IDs (one per text).

    Returns:
        List of Document objects.

    Raises:
        ValueError: If metadatas or ids length doesn't match texts.
    """
    if metadatas and len(metadatas) != len(texts):
        msg = f"metadatas length ({len(metadatas)}) must match texts length ({len(texts)})"
        raise ValueError(msg)

    if ids and len(ids) != len(texts):
        msg = f"ids length ({len(ids)}) must match texts length ({len(texts)})"
        raise ValueError(msg)

    documents = []
    for i, text in enumerate(texts):
        doc = Document(
            text=text,
            metadata=metadatas[i] if metadatas else {},
            id=ids[i] if ids else "",
        )
        documents.append(doc)

    return documents


# ============================================================
# Text Splitting Utilities
# ============================================================

def split_text(
    text: str,
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
    separator: str = "\n\n",
) -> list[str]:
    """Split text into chunks with overlap.

    Simple character-based splitting with configurable separator.

    Args:
        text: Text to split.
        chunk_size: Maximum size of each chunk.
        chunk_overlap: Number of characters to overlap between chunks.
        separator: Preferred split point (falls back to space).

    Returns:
        List of text chunks.

    Example:
        >>> chunks = split_text("Long document...", chunk_size=500)
        >>> len(chunks)
        3
    """
    if len(text) <= chunk_size:
        return [text]

    chunks: list[str] = []
    start = 0

    while start < len(text):
        end = start + chunk_size

        if end >= len(text):
            chunk = text[start:].strip()
            if chunk:
                chunks.append(chunk)
            break

        # Try to find a good split point
        chunk = text[start:end]

        # Look for separator
        split_pos = chunk.rfind(separator)
        if split_pos == -1:
            # Fall back to space
            split_pos = chunk.rfind(" ")
        if split_pos == -1 or split_pos == 0:
            # No good split point, just use chunk_size
            split_pos = chunk_size
        else:
            # Include the separator position
            split_pos += 1

        chunk_text = text[start:start + split_pos].strip()
        if chunk_text:
            chunks.append(chunk_text)

        # Move start forward, accounting for overlap
        new_start = start + split_pos - chunk_overlap

        # Ensure we make progress (prevent infinite loop)
        if new_start <= start:
            new_start = start + max(1, split_pos)

        start = new_start

    return chunks


def split_documents(
    documents: list[Document],
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
) -> list[Document]:
    """Split documents into smaller chunks.

    Preserves metadata from parent document.

    Args:
        documents: Documents to split.
        chunk_size: Maximum size of each chunk.
        chunk_overlap: Number of characters to overlap.

    Returns:
        List of chunked documents.
    """
    chunked: list[Document] = []

    for doc in documents:
        chunks = split_text(doc.text, chunk_size, chunk_overlap)

        for i, chunk_text in enumerate(chunks):
            chunk_doc = Document(
                text=chunk_text,
                metadata={
                    **doc.metadata,
                    "_chunk_index": i,
                    "_parent_id": doc.id,
                },
            )
            chunked.append(chunk_doc)

    return chunked
