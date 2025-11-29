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
    >>> from agenticflow.document.splitters import (
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

from agenticflow.document.splitters.base import BaseSplitter
from agenticflow.document.splitters.character import (
    CharacterSplitter,
    RecursiveCharacterSplitter,
)
from agenticflow.document.splitters.code import CodeSplitter
from agenticflow.document.splitters.html import HTMLSplitter
from agenticflow.document.splitters.markdown import MarkdownSplitter
from agenticflow.document.splitters.semantic import SemanticSplitter
from agenticflow.document.splitters.sentence import SentenceSplitter
from agenticflow.document.splitters.token import TokenSplitter
from agenticflow.document.splitters.utils import split_text

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
    # Semantic
    "SemanticSplitter",
    # Token-based
    "TokenSplitter",
    # Convenience
    "split_text",
]
