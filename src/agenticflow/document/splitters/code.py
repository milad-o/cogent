"""Language-aware code splitter."""

from __future__ import annotations

import re
from typing import Any

from agenticflow.document.splitters.base import BaseSplitter
from agenticflow.document.types import TextChunk


class CodeSplitter(BaseSplitter):
    """Split source code respecting language syntax.
    
    Splits at function/class boundaries and preserves code blocks.
    
    Args:
        language: Programming language (python, javascript, java, etc.)
        chunk_size: Target chunk size.
        chunk_overlap: Overlap between chunks.
        
    Example:
        >>> splitter = CodeSplitter(language="python", chunk_size=1500)
        >>> chunks = splitter.split_text(python_code)
    """
    
    # Language-specific split patterns
    LANGUAGE_PATTERNS: dict[str, list[str]] = {
        "python": [
            r'\nclass\s+\w+',           # Class definitions
            r'\ndef\s+\w+',             # Function definitions
            r'\nasync\s+def\s+\w+',     # Async functions
            r'\n@\w+',                  # Decorators
            r'\n\n',                    # Double newlines
        ],
        "javascript": [
            r'\nclass\s+\w+',
            r'\nfunction\s+\w+',
            r'\nconst\s+\w+\s*=\s*(?:async\s+)?\(',
            r'\nlet\s+\w+\s*=\s*(?:async\s+)?\(',
            r'\nexport\s+',
            r'\n\n',
        ],
        "typescript": [
            r'\nclass\s+\w+',
            r'\ninterface\s+\w+',
            r'\ntype\s+\w+',
            r'\nfunction\s+\w+',
            r'\nconst\s+\w+\s*=\s*(?:async\s+)?\(',
            r'\nexport\s+',
            r'\n\n',
        ],
        "java": [
            r'\npublic\s+class\s+\w+',
            r'\nprivate\s+class\s+\w+',
            r'\nprotected\s+class\s+\w+',
            r'\npublic\s+\w+\s+\w+\s*\(',
            r'\nprivate\s+\w+\s+\w+\s*\(',
            r'\n\n',
        ],
        "go": [
            r'\nfunc\s+',
            r'\ntype\s+\w+\s+struct',
            r'\ntype\s+\w+\s+interface',
            r'\n\n',
        ],
        "rust": [
            r'\nfn\s+\w+',
            r'\nimpl\s+',
            r'\nstruct\s+\w+',
            r'\nenum\s+\w+',
            r'\ntrait\s+\w+',
            r'\n\n',
        ],
        "cpp": [
            r'\nclass\s+\w+',
            r'\nstruct\s+\w+',
            r'\n\w+\s+\w+\s*\([^)]*\)\s*{',  # Function definitions
            r'\n\n',
        ],
        "c": [
            r'\nstruct\s+\w+',
            r'\n\w+\s+\w+\s*\([^)]*\)\s*{',
            r'\n\n',
        ],
    }
    
    def __init__(
        self,
        language: str = "python",
        chunk_size: int = 1500,
        chunk_overlap: int = 200,
        **kwargs: Any,
    ):
        super().__init__(chunk_size=chunk_size, chunk_overlap=chunk_overlap, **kwargs)
        self.language = language.lower()
        self.patterns = self.LANGUAGE_PATTERNS.get(
            self.language,
            [r'\n\n', r'\n']  # Default fallback
        )
    
    def split_text(self, text: str) -> list[TextChunk]:
        """Split code by language-specific patterns."""
        # Try each pattern in order
        for pattern in self.patterns:
            sections = self._split_by_pattern(text, pattern)
            if len(sections) > 1:
                chunks = self._merge_splits(sections, "\n")
                if chunks:
                    for chunk in chunks:
                        chunk.metadata["language"] = self.language
                    return chunks
        
        # Fallback to simple line splitting
        lines = text.split("\n")
        chunks = self._merge_splits(lines, "\n")
        for chunk in chunks:
            chunk.metadata["language"] = self.language
        return chunks
    
    def _split_by_pattern(self, text: str, pattern: str) -> list[str]:
        """Split text by regex pattern, keeping the matched delimiter."""
        splits = re.split(f'({pattern})', text)
        
        # Recombine: each split starts with its delimiter
        sections = []
        current = ""
        
        for i, part in enumerate(splits):
            if re.match(pattern, part):
                if current.strip():
                    sections.append(current)
                current = part
            else:
                current += part
        
        if current.strip():
            sections.append(current)
        
        return sections


__all__ = ["CodeSplitter"]
