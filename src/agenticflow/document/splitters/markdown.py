"""Markdown-aware text splitter."""

from __future__ import annotations

import re
from typing import Any

from agenticflow.document.splitters.base import BaseSplitter
from agenticflow.document.splitters.character import RecursiveCharacterSplitter
from agenticflow.document.types import Document


class MarkdownSplitter(BaseSplitter):
    """Split markdown text respecting document structure.
    
    Preserves heading hierarchy and splits at section boundaries.
    Each chunk includes its heading context.
    
    Args:
        chunk_size: Target chunk size.
        chunk_overlap: Overlap between chunks.
        headers_to_split_on: List of header levels to split on.
        return_each_section: Return each section as separate chunk.
        
    Example:
        >>> splitter = MarkdownSplitter(chunk_size=1000)
        >>> chunks = splitter.split_text(markdown_text)
    """
    
    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        headers_to_split_on: list[str] | None = None,
        return_each_section: bool = False,
        **kwargs: Any,
    ):
        super().__init__(chunk_size=chunk_size, chunk_overlap=chunk_overlap, **kwargs)
        self.headers_to_split_on = headers_to_split_on or ["#", "##", "###"]
        self.return_each_section = return_each_section
    
    def split_text(self, text: str) -> list[Document]:
        """Split markdown by headers and content."""
        sections = self._split_by_headers(text)
        
        if self.return_each_section:
            chunks = []
            for section in sections:
                chunks.append(Document(
                    text=section["content"],
                    metadata={
                        "headers": section["headers"],
                        "chunk_index": len(chunks),
                    }
                ))
            return chunks
        
        # Merge sections into appropriately sized chunks
        chunks: list[Document] = []
        current_content: list[str] = []
        current_length = 0
        current_headers: dict[str, str] = {}
        
        for section in sections:
            section_text = section["content"]
            section_length = self.length_function(section_text)
            
            # If section is too large, split it further
            if section_length > self.chunk_size:
                # Save current chunk first
                if current_content:
                    chunks.append(Document(
                        text="\n\n".join(current_content),
                        metadata={
                            "headers": dict(current_headers),
                            "chunk_index": len(chunks),
                        }
                    ))
                    current_content = []
                    current_length = 0
                
                # Split large section with recursive splitter
                sub_splitter = RecursiveCharacterSplitter(
                    chunk_size=self.chunk_size,
                    chunk_overlap=self.chunk_overlap,
                )
                sub_chunks = sub_splitter.split_text(section_text)
                for sub_chunk in sub_chunks:
                    sub_chunk.metadata["headers"] = section["headers"]
                    sub_chunk.metadata["chunk_index"] = len(chunks)
                    chunks.append(sub_chunk)
                
                current_headers = section["headers"].copy()
                continue
            
            # Check if we need to start new chunk
            if current_length + section_length > self.chunk_size and current_content:
                chunks.append(Document(
                    text="\n\n".join(current_content),
                    metadata={
                        "headers": dict(current_headers),
                        "chunk_index": len(chunks),
                    }
                ))
                current_content = []
                current_length = 0
            
            current_content.append(section_text)
            current_length += section_length
            current_headers.update(section["headers"])
        
        # Add final chunk
        if current_content:
            chunks.append(Document(
                text="\n\n".join(current_content),
                metadata={
                    "headers": dict(current_headers),
                    "chunk_index": len(chunks),
                }
            ))
        
        return chunks
    
    def _split_by_headers(self, text: str) -> list[dict[str, Any]]:
        """Split markdown into sections by headers."""
        header_pattern = r'^(#{1,6})\s+(.+)$'
        
        lines = text.split("\n")
        sections: list[dict[str, Any]] = []
        current_headers: dict[str, str] = {}
        current_content: list[str] = []
        
        for line in lines:
            header_match = re.match(header_pattern, line)
            
            if header_match:
                # Save previous section
                if current_content:
                    content_text = "\n".join(current_content).strip()
                    if content_text:
                        sections.append({
                            "headers": dict(current_headers),
                            "content": content_text,
                        })
                    current_content = []
                
                # Update header hierarchy
                level = header_match.group(1)
                title = header_match.group(2).strip()
                
                # Clear lower-level headers
                levels_to_clear = [h for h in current_headers if len(h) >= len(level)]
                for h in levels_to_clear:
                    del current_headers[h]
                
                current_headers[level] = title
                
                # Add header to content if desired
                if level in self.headers_to_split_on:
                    current_content.append(line)
            else:
                current_content.append(line)
        
        # Add final section
        if current_content:
            content_text = "\n".join(current_content).strip()
            if content_text:
                sections.append({
                    "headers": dict(current_headers),
                    "content": content_text,
                })
        
        return sections


__all__ = ["MarkdownSplitter"]
