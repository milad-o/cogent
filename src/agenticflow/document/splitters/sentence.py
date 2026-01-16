"""Sentence-based text splitter."""

from __future__ import annotations

import re
from typing import Any

from agenticflow.document.splitters.base import BaseSplitter
from agenticflow.document.types import Document


class SentenceSplitter(BaseSplitter):
    """Split text by sentence boundaries.

    Uses regex to identify sentence endings and respects
    common abbreviations to avoid false splits.

    Args:
        chunk_size: Target chunk size.
        chunk_overlap: Overlap between chunks.
        min_sentence_length: Minimum characters for a valid sentence.

    Example:
        >>> splitter = SentenceSplitter(chunk_size=500)
        >>> chunks = splitter.split_text(text)
    """

    # Common abbreviations that don't end sentences
    ABBREVIATIONS = {
        "mr", "mrs", "ms", "dr", "prof", "sr", "jr",
        "vs", "etc", "fig", "eg", "ie", "al", "vol",
        "inc", "ltd", "corp", "co",
    }

    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        min_sentence_length: int = 10,
        **kwargs: Any,
    ):
        super().__init__(chunk_size=chunk_size, chunk_overlap=chunk_overlap, **kwargs)
        self.min_sentence_length = min_sentence_length

    def split_text(self, text: str) -> list[Document]:
        """Split text into sentences then merge into chunks."""
        sentences = self._split_sentences(text)
        return self._merge_splits(sentences, " ")

    def _split_sentences(self, text: str) -> list[str]:
        """Split text into individual sentences."""
        # Pattern matches sentence-ending punctuation followed by space and capital
        pattern = r'(?<=[.!?])\s+(?=[A-Z])'

        # Split on the pattern
        raw_sentences = re.split(pattern, text)

        sentences = []
        current = ""

        for sent in raw_sentences:
            sent = sent.strip()
            if not sent:
                continue

            # Check if previous sentence ended with abbreviation
            if current:
                last_word = current.split()[-1].rstrip(".").lower() if current.split() else ""
                if last_word in self.ABBREVIATIONS:
                    current = current + " " + sent
                    continue

            if current and len(current) >= self.min_sentence_length:
                sentences.append(current)

            current = sent

        # Add final sentence
        if current and len(current) >= self.min_sentence_length:
            sentences.append(current)
        elif current and sentences:
            sentences[-1] = sentences[-1] + " " + current
        elif current:
            sentences.append(current)

        return sentences


__all__ = ["SentenceSplitter"]
