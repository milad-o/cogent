"""Base class for text splitters."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Callable, Sequence

from agenticflow.document.types import Document

if TYPE_CHECKING:
    pass


class BaseSplitter(ABC):
    """Abstract base class for text splitters.
    
    All splitters inherit from this class and implement the split_text method.
    Splitters return Document objects directly for seamless vectorstore integration.
    
    Args:
        chunk_size: Target size for each chunk.
        chunk_overlap: Number of characters to overlap between chunks.
        length_function: Function to measure text length (default: len).
        keep_separator: Whether to keep separators in output.
        strip_whitespace: Whether to strip whitespace from chunks.
    """
    
    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        length_function: Callable[[str], int] | None = None,
        keep_separator: bool = False,
        strip_whitespace: bool = True,
    ):
        if chunk_overlap >= chunk_size:
            raise ValueError(
                f"chunk_overlap ({chunk_overlap}) must be less than "
                f"chunk_size ({chunk_size})"
            )
        
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.length_function = length_function or len
        self.keep_separator = keep_separator
        self.strip_whitespace = strip_whitespace
    
    @abstractmethod
    def split_text(self, text: str) -> list[Document]:
        """Split text into document chunks.
        
        Args:
            text: The text to split.
            
        Returns:
            List of Document objects (chunks).
        """
        ...
    
    def split_documents(
        self,
        documents: Sequence[Document],
    ) -> list[Document]:
        """Split multiple documents into chunks.
        
        Args:
            documents: List of Document objects to split.
            
        Returns:
            List of Document chunks with inherited metadata.
        """
        chunks = []
        for doc in documents:
            doc_chunks = self.split_text(doc.text)
            for chunk in doc_chunks:
                # Inherit document metadata, but preserve chunk-specific fields
                # (like chunk_index, start_index, end_index)
                chunk_specific = {
                    k: v for k, v in chunk.metadata.items()
                    if k in ("chunk_index", "start_index", "end_index")
                }
                chunk.metadata = {**doc.metadata, **chunk_specific}
            chunks.extend(doc_chunks)
        return chunks
    
    def _merge_splits(
        self,
        splits: list[str],
        separator: str = "",
    ) -> list[Document]:
        """Merge splits into chunks respecting size limits.
        
        Args:
            splits: List of text pieces to merge.
            separator: Separator to use when joining.
            
        Returns:
            List of merged Document chunks.
        """
        chunks: list[Document] = []
        current_chunk: list[str] = []
        current_length = 0
        current_start = 0
        position = 0
        
        for split in splits:
            split_length = self.length_function(split)
            
            # Check if adding this split exceeds chunk size
            total = current_length + split_length
            if current_chunk:
                total += self.length_function(separator)
            
            if total > self.chunk_size and current_chunk:
                # Save current chunk
                content = separator.join(current_chunk)
                if self.strip_whitespace:
                    content = content.strip()
                if content:
                    chunks.append(Document(
                        text=content,
                        metadata={
                            "chunk_index": len(chunks),
                            "start_index": current_start,
                            "end_index": position,
                        },
                    ))
                
                # Handle overlap
                overlap_start = current_start
                while current_chunk and current_length > self.chunk_overlap:
                    popped = current_chunk.pop(0)
                    current_length -= self.length_function(popped)
                    if current_chunk:
                        current_length -= self.length_function(separator)
                    overlap_start += self.length_function(popped) + self.length_function(separator)
                
                current_start = overlap_start
            
            current_chunk.append(split)
            current_length += split_length
            if len(current_chunk) > 1:
                current_length += self.length_function(separator)
            position += split_length + self.length_function(separator)
        
        # Add final chunk
        if current_chunk:
            content = separator.join(current_chunk)
            if self.strip_whitespace:
                content = content.strip()
            if content:
                chunks.append(Document(
                    text=content,
                    metadata={
                        "chunk_index": len(chunks),
                        "start_index": current_start,
                        "end_index": position,
                    },
                ))
        
        return chunks


# Alias for backward compatibility
TextSplitter = BaseSplitter


__all__ = ["BaseSplitter", "TextSplitter"]
