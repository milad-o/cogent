"""Text splitters submodule.

This module provides text splitting/chunking capabilities for document processing.

**Available Splitters:**
- RecursiveCharacterSplitter: Hierarchical separator-based splitting
- CharacterSplitter: Simple single-separator splitting
- SentenceSplitter: Sentence boundary detection
- MarkdownSplitter: Markdown structure-aware splitting
- HTMLSplitter: HTML tag-based splitting
- CodeSplitter: Language-aware code splitting
- SemanticSplitter: Embedding-based semantic chunking
- TokenSplitter: Token count-based splitting

**Usage:**
    >>> from cogent.documents.splitters import (
    ...     RecursiveCharacterSplitter,
    ...     split_text,
    ... )
    >>>
    >>> # Using splitter class
    >>> splitter = RecursiveCharacterSplitter(chunk_size=1000, chunk_overlap=200)
    >>> chunks = splitter.split_text(text)
    >>>
    >>> # Using convenience function
    >>> chunks = split_text(text, chunk_size=1000, splitter_type="recursive")
"""

from cogent.documents.splitters.base import BaseSplitter
from cogent.documents.splitters.character import (
    CharacterSplitter,
    RecursiveCharacterSplitter,
)
from cogent.documents.splitters.code import CodeSplitter
from cogent.documents.splitters.html import HTMLSplitter
from cogent.documents.splitters.markdown import MarkdownSplitter
from cogent.documents.splitters.semantic import SemanticSplitter
from cogent.documents.splitters.sentence import SentenceSplitter
from cogent.documents.splitters.table_aware import TableAwareSplitter
from cogent.documents.splitters.token import TokenSplitter
from cogent.documents.splitters.utils import split_text

__all__ = [
    # Base
    "BaseSplitter",
    # Character-based
    "RecursiveCharacterSplitter",
    "CharacterSplitter",
    # Sentence-based
    "SentenceSplitter",
    # Structure-aware
    "MarkdownSplitter",
    "HTMLSplitter",
    "CodeSplitter",
    "TableAwareSplitter",
    # Semantic
    "SemanticSplitter",
    # Token-based
    "TokenSplitter",
    # Convenience
    "split_text",
]
