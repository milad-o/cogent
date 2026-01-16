"""Character-based text splitters."""

from __future__ import annotations

import re
from typing import Any

from agenticflow.document.splitters.base import BaseSplitter
from agenticflow.document.types import Document


class RecursiveCharacterSplitter(BaseSplitter):
    """Split text recursively using a hierarchy of separators.

    Tries to split by larger semantic units first (paragraphs),
    then falls back to smaller units (sentences, words, characters).

    Args:
        chunk_size: Target chunk size.
        chunk_overlap: Overlap between chunks.
        separators: List of separators to try in order.

    Example:
        >>> splitter = RecursiveCharacterSplitter(chunk_size=500, chunk_overlap=50)
        >>> chunks = splitter.split_text(long_text)
    """

    DEFAULT_SEPARATORS = [
        "\n\n",      # Paragraphs
        "\n",        # Lines
        ". ",        # Sentences (with space)
        "! ",        # Exclamations
        "? ",        # Questions
        "; ",        # Semicolons
        ", ",        # Clauses
        " ",         # Words
        "",          # Characters
    ]

    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        separators: list[str] | None = None,
        **kwargs: Any,
    ):
        super().__init__(chunk_size=chunk_size, chunk_overlap=chunk_overlap, **kwargs)
        self.separators = separators or self.DEFAULT_SEPARATORS.copy()

    def split_text(self, text: str) -> list[Document]:
        """Split text using recursive separator strategy."""
        return self._split_text(text, self.separators)

    def _split_text(
        self,
        text: str,
        separators: list[str],
    ) -> list[Document]:
        """Recursively split text."""
        # Base case: no separators left
        if not separators:
            return [Document(text=text, metadata={"chunk_index": 0})]

        separator = separators[0]
        remaining_separators = separators[1:]

        # Split by current separator
        if separator:
            if self.keep_separator:
                # Keep separator at end of each piece
                splits = re.split(f"({re.escape(separator)})", text)
                # Recombine: pair each content with its separator
                merged = []
                for i in range(0, len(splits) - 1, 2):
                    merged.append(splits[i] + (splits[i + 1] if i + 1 < len(splits) else ""))
                if len(splits) % 2 == 1:
                    merged.append(splits[-1])
                splits = merged
            else:
                splits = text.split(separator)
        else:
            # Character-level split
            splits = list(text)

        # Process splits
        chunks: list[Document] = []
        current_parts: list[str] = []
        current_length = 0

        for split in splits:
            split_length = self.length_function(split)

            # If single split is too large, recursively split it
            if split_length > self.chunk_size:
                # First, save current accumulated parts
                if current_parts:
                    chunks.extend(
                        self._merge_splits(current_parts, separator if self.keep_separator else "")
                    )
                    current_parts = []
                    current_length = 0

                # Recursively split the large piece
                if remaining_separators:
                    sub_chunks = self._split_text(split, remaining_separators)
                    chunks.extend(sub_chunks)
                else:
                    # Can't split further, just add as is
                    chunks.append(Document(
                        text=split,
                        metadata={"chunk_index": len(chunks)},
                    ))
                continue

            # Check if we need to start a new chunk
            total_length = current_length + split_length
            if current_parts:
                total_length += self.length_function(separator)

            if total_length > self.chunk_size and current_parts:
                # Create chunk from accumulated parts
                chunks.extend(
                    self._merge_splits(current_parts, separator if self.keep_separator else "")
                )

                # Start new chunk with overlap
                overlap_parts = []
                overlap_length = 0
                for part in reversed(current_parts):
                    part_len = self.length_function(part)
                    if overlap_length + part_len <= self.chunk_overlap:
                        overlap_parts.insert(0, part)
                        overlap_length += part_len
                    else:
                        break

                current_parts = overlap_parts
                current_length = overlap_length

            current_parts.append(split)
            current_length += split_length

        # Handle remaining parts
        if current_parts:
            chunks.extend(
                self._merge_splits(current_parts, separator if self.keep_separator else "")
            )

        # Renumber chunk indices
        for i, chunk in enumerate(chunks):
            chunk.metadata["chunk_index"] = i

        return chunks


class CharacterSplitter(BaseSplitter):
    """Simple character-based splitter using a single separator.

    Args:
        separator: The separator to split on.
        chunk_size: Target chunk size.
        chunk_overlap: Overlap between chunks.

    Example:
        >>> splitter = CharacterSplitter(separator="\\n\\n", chunk_size=1000)
        >>> chunks = splitter.split_text(text)
    """

    def __init__(
        self,
        separator: str = "\n\n",
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        **kwargs: Any,
    ):
        super().__init__(chunk_size=chunk_size, chunk_overlap=chunk_overlap, **kwargs)
        self.separator = separator

    def split_text(self, text: str) -> list[Document]:
        """Split text by separator."""
        splits = text.split(self.separator) if self.separator else list(text)

        return self._merge_splits(splits, self.separator if self.keep_separator else " ")


__all__ = ["RecursiveCharacterSplitter", "CharacterSplitter"]
