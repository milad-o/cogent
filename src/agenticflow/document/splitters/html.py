"""HTML-aware text splitter."""

from __future__ import annotations

import re
from typing import Any

from agenticflow.document.splitters.base import BaseSplitter
from agenticflow.document.types import Document


class HTMLSplitter(BaseSplitter):
    """Split HTML content by semantic tags.
    
    Respects HTML structure and splits at tag boundaries.
    
    Args:
        chunk_size: Target chunk size.
        chunk_overlap: Overlap between chunks.
        tags_to_split_on: HTML tags to use as split points.
        
    Example:
        >>> splitter = HTMLSplitter(chunk_size=1000)
        >>> chunks = splitter.split_text(html_content)
    """
    
    DEFAULT_SPLIT_TAGS = [
        "article", "section", "div", "p", "h1", "h2", "h3", "h4", "h5", "h6",
        "ul", "ol", "li", "table", "tr", "blockquote", "pre", "code",
    ]
    
    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        tags_to_split_on: list[str] | None = None,
        **kwargs: Any,
    ):
        super().__init__(chunk_size=chunk_size, chunk_overlap=chunk_overlap, **kwargs)
        self.tags_to_split_on = tags_to_split_on or self.DEFAULT_SPLIT_TAGS
    
    def split_text(self, text: str) -> list[Document]:
        """Split HTML by tags."""
        # Try to use BeautifulSoup if available
        import importlib.util
        if importlib.util.find_spec("bs4") is not None:
            return self._split_with_bs4(text)
        else:
            return self._split_with_regex(text)
    
    def _split_with_bs4(self, text: str) -> list[Document]:
        """Split using BeautifulSoup."""
        from bs4 import BeautifulSoup
        
        soup = BeautifulSoup(text, "html.parser")
        
        # Remove script and style
        for element in soup(["script", "style"]):
            element.decompose()
        
        sections: list[str] = []
        
        # Find all split-worthy elements
        for tag in self.tags_to_split_on:
            for element in soup.find_all(tag):
                text_content = element.get_text(separator=" ", strip=True)
                if text_content:
                    sections.append(text_content)
        
        # If no sections found, get all text
        if not sections:
            sections = [soup.get_text(separator=" ", strip=True)]
        
        return self._merge_splits(sections, "\n\n")
    
    def _split_with_regex(self, text: str) -> list[Document]:
        """Fallback regex-based splitting."""
        # Build pattern for tags
        tag_pattern = "|".join(self.tags_to_split_on)
        pattern = rf'<({tag_pattern})[^>]*>(.*?)</\1>'
        
        sections = []
        for match in re.finditer(pattern, text, re.IGNORECASE | re.DOTALL):
            content = re.sub(r'<[^>]+>', ' ', match.group(2))
            content = re.sub(r'\s+', ' ', content).strip()
            if content:
                sections.append(content)
        
        if not sections:
            # Just strip all tags and split
            plain_text = re.sub(r'<[^>]+>', ' ', text)
            plain_text = re.sub(r'\s+', ' ', plain_text).strip()
            sections = [plain_text]
        
        return self._merge_splits(sections, "\n\n")


__all__ = ["HTMLSplitter"]
