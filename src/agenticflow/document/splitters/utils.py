"""Utility functions for text splitting."""

from __future__ import annotations

from typing import Any

from agenticflow.document.types import TextChunk


def split_text(
    text: str,
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
    splitter_type: str = "recursive",
    **kwargs: Any,
) -> list[TextChunk]:
    """Split text using specified strategy.
    
    Args:
        text: Text to split.
        chunk_size: Target chunk size.
        chunk_overlap: Overlap between chunks.
        splitter_type: Type of splitter (recursive, sentence, markdown, code, token).
        **kwargs: Additional splitter arguments.
        
    Returns:
        List of TextChunk objects.
        
    Example:
        >>> chunks = split_text(text, chunk_size=500, splitter_type="recursive")
        >>> chunks = split_text(markdown, splitter_type="markdown")
        >>> chunks = split_text(code, splitter_type="code", language="python")
    """
    from agenticflow.document.splitters.character import (
        CharacterSplitter,
        RecursiveCharacterSplitter,
    )
    from agenticflow.document.splitters.code import CodeSplitter
    from agenticflow.document.splitters.html import HTMLSplitter
    from agenticflow.document.splitters.markdown import MarkdownSplitter
    from agenticflow.document.splitters.sentence import SentenceSplitter
    from agenticflow.document.splitters.token import TokenSplitter
    
    splitters = {
        "recursive": RecursiveCharacterSplitter,
        "character": CharacterSplitter,
        "sentence": SentenceSplitter,
        "markdown": MarkdownSplitter,
        "html": HTMLSplitter,
        "code": CodeSplitter,
        "token": TokenSplitter,
    }
    
    splitter_class = splitters.get(splitter_type.lower())
    if not splitter_class:
        raise ValueError(
            f"Unknown splitter type: {splitter_type}. "
            f"Supported: {', '.join(splitters.keys())}"
        )
    
    splitter = splitter_class(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        **kwargs,
    )
    
    return splitter.split_text(text)


__all__ = ["split_text"]
