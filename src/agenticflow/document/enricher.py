"""Document metadata enrichment utilities.

Provides tools to automatically extract and add metadata to documents
for improved retrieval with HybridRetriever and filtering.
"""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from agenticflow.document.types import Document

if TYPE_CHECKING:
    pass


@dataclass
class EnricherConfig:
    """Configuration for metadata enrichment.

    Attributes:
        extract_keywords: Extract top keywords from text.
        keyword_count: Number of keywords to extract.
        count_words: Add word count to metadata.
        count_chars: Add character count to metadata.
        detect_language: Detect document language (requires langdetect).
        extract_dates: Extract dates mentioned in text.
        extract_emails: Extract email addresses.
        extract_urls: Extract URLs.
        compute_reading_time: Estimate reading time in minutes.
        words_per_minute: Reading speed for time estimation.
    """

    extract_keywords: bool = True
    keyword_count: int = 10
    count_words: bool = True
    count_chars: bool = False
    detect_language: bool = False
    extract_dates: bool = False
    extract_emails: bool = False
    extract_urls: bool = False
    compute_reading_time: bool = True
    words_per_minute: int = 200


# Common English stop words for keyword extraction
STOP_WORDS = frozenset({
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from",
    "has", "he", "in", "is", "it", "its", "of", "on", "that", "the",
    "to", "was", "were", "will", "with", "this", "but", "they",
    "have", "had", "what", "when", "where", "who", "which", "why", "how",
    "all", "each", "every", "both", "few", "more", "most", "other",
    "some", "such", "no", "nor", "not", "only", "own", "same", "so",
    "than", "too", "very", "can", "just", "should", "now", "or", "if",
    "about", "into", "through", "during", "before", "after", "above",
    "below", "between", "under", "again", "further", "then", "once",
    "here", "there", "any", "because", "been", "being", "could", "did",
    "do", "does", "doing", "down", "get", "got", "him", "his",
    "her", "hers", "herself", "himself", "i", "me", "my", "myself",
    "we", "our", "ours", "ourselves", "you", "your", "yours", "yourself",
})


class MetadataEnricher:
    """Enrich documents with automatically extracted metadata.

    Extracts keywords, counts, dates, and other useful metadata
    from document text to enable better filtering and hybrid retrieval.

    Example:
        >>> from agenticflow.document import MetadataEnricher, Document
        >>>
        >>> enricher = MetadataEnricher(
        ...     extract_keywords=True,
        ...     count_words=True,
        ...     compute_reading_time=True,
        ... )
        >>>
        >>> docs = [Document(text="Python is a programming language...")]
        >>> enriched = enricher.enrich(docs)
        >>>
        >>> print(enriched[0].metadata)
        >>> # {'keywords': ['python', 'programming', 'language'],
        >>> #  'word_count': 42, 'reading_time_minutes': 1}

    With custom extractors:
        >>> def extract_version(doc: Document) -> str | None:
        ...     match = re.search(r'v?(\\d+\\.\\d+\\.\\d+)', doc.text)
        ...     return match.group(1) if match else None
        >>>
        >>> enricher = MetadataEnricher(
        ...     custom_extractors={"version": extract_version}
        ... )
    """

    def __init__(
        self,
        config: EnricherConfig | None = None,
        *,
        extract_keywords: bool = True,
        keyword_count: int = 10,
        count_words: bool = True,
        count_chars: bool = False,
        detect_language: bool = False,
        extract_dates: bool = False,
        extract_emails: bool = False,
        extract_urls: bool = False,
        compute_reading_time: bool = True,
        words_per_minute: int = 200,
        custom_extractors: dict[str, Callable[[Document], Any]] | None = None,
        stop_words: frozenset[str] | None = None,
    ) -> None:
        """Initialize the enricher.

        Args:
            config: Configuration object (alternative to individual params).
            extract_keywords: Extract top keywords from text.
            keyword_count: Number of keywords to extract.
            count_words: Add word count to metadata.
            count_chars: Add character count to metadata.
            detect_language: Detect document language.
            extract_dates: Extract dates mentioned in text.
            extract_emails: Extract email addresses.
            extract_urls: Extract URLs.
            compute_reading_time: Estimate reading time.
            words_per_minute: Reading speed for time estimation.
            custom_extractors: Dict of field_name -> extractor function.
            stop_words: Custom stop words for keyword extraction.
        """
        if config:
            self._config = config
        else:
            self._config = EnricherConfig(
                extract_keywords=extract_keywords,
                keyword_count=keyword_count,
                count_words=count_words,
                count_chars=count_chars,
                detect_language=detect_language,
                extract_dates=extract_dates,
                extract_emails=extract_emails,
                extract_urls=extract_urls,
                compute_reading_time=compute_reading_time,
                words_per_minute=words_per_minute,
            )

        self._custom_extractors = custom_extractors or {}
        self._stop_words = stop_words or STOP_WORDS

    def enrich(
        self,
        documents: list[Document],
        *,
        in_place: bool = True,
    ) -> list[Document]:
        """Enrich documents with extracted metadata.

        Args:
            documents: Documents to enrich.
            in_place: If True, modify documents in place. If False, return copies.

        Returns:
            Enriched documents.
        """
        if not in_place:
            documents = [
                Document(
                    text=doc.text,
                    metadata=dict(doc.metadata),
                    id=doc.id,
                )
                for doc in documents
            ]

        for doc in documents:
            self._enrich_document(doc)

        return documents

    def enrich_one(self, document: Document, *, in_place: bool = True) -> Document:
        """Enrich a single document.

        Args:
            document: Document to enrich.
            in_place: If True, modify in place.

        Returns:
            Enriched document.
        """
        return self.enrich([document], in_place=in_place)[0]

    def _enrich_document(self, doc: Document) -> None:
        """Enrich a single document in place."""
        text = doc.text
        config = self._config

        # Word count (used by multiple features)
        words = text.split()
        word_count = len(words)

        if config.count_words:
            doc.metadata["word_count"] = word_count

        if config.count_chars:
            doc.metadata["char_count"] = len(text)

        if config.compute_reading_time:
            doc.metadata["reading_time_minutes"] = max(
                1, round(word_count / config.words_per_minute)
            )

        if config.extract_keywords:
            doc.metadata["keywords"] = self._extract_keywords(
                text, config.keyword_count
            )

        if config.detect_language:
            doc.metadata["language"] = self._detect_language(text)

        if config.extract_dates:
            doc.metadata["dates"] = self._extract_dates(text)

        if config.extract_emails:
            doc.metadata["emails"] = self._extract_emails(text)

        if config.extract_urls:
            doc.metadata["urls"] = self._extract_urls(text)

        # Run custom extractors
        for field_name, extractor in self._custom_extractors.items():
            try:
                value = extractor(doc)
                if value is not None:
                    doc.metadata[field_name] = value
            except Exception:
                # Skip failed extractors
                pass

    def _extract_keywords(self, text: str, count: int) -> list[str]:
        """Extract top keywords from text.

        Uses simple word frequency after filtering stop words.
        """
        # Tokenize: lowercase, keep only alphanumeric
        words = re.findall(r'\b[a-z]{3,}\b', text.lower())

        # Filter stop words and count
        word_counts = Counter(
            word for word in words
            if word not in self._stop_words
        )

        # Return top keywords
        return [word for word, _ in word_counts.most_common(count)]

    def _detect_language(self, text: str) -> str | None:
        """Detect document language using langdetect."""
        try:
            from langdetect import detect
            return detect(text[:1000])  # Use first 1000 chars for speed
        except ImportError:
            return None
        except Exception:
            return None

    def _extract_dates(self, text: str) -> list[str]:
        """Extract date patterns from text."""
        patterns = [
            # ISO format: 2024-01-15
            r'\b\d{4}-\d{2}-\d{2}\b',
            # US format: 01/15/2024 or 1/15/2024
            r'\b\d{1,2}/\d{1,2}/\d{4}\b',
            # Written: January 15, 2024
            r'\b(?:January|February|March|April|May|June|July|August|'
            r'September|October|November|December)\s+\d{1,2},?\s+\d{4}\b',
            # Short written: Jan 15, 2024
            r'\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)'
            r'\s+\d{1,2},?\s+\d{4}\b',
        ]

        dates = []
        for pattern in patterns:
            dates.extend(re.findall(pattern, text, re.IGNORECASE))

        return list(set(dates))[:10]  # Dedupe and limit

    def _extract_emails(self, text: str) -> list[str]:
        """Extract email addresses from text."""
        pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        emails = re.findall(pattern, text)
        return list(set(emails))[:10]

    def _extract_urls(self, text: str) -> list[str]:
        """Extract URLs from text."""
        pattern = r'https?://[^\s<>"{}|\\^`\[\]]+'
        urls = re.findall(pattern, text)
        return list(set(urls))[:10]


def enrich_documents(
    documents: list[Document],
    *,
    extract_keywords: bool = True,
    count_words: bool = True,
    compute_reading_time: bool = True,
    custom_extractors: dict[str, Callable[[Document], Any]] | None = None,
    **kwargs: Any,
) -> list[Document]:
    """Convenience function to enrich documents.

    Args:
        documents: Documents to enrich.
        extract_keywords: Extract keywords.
        count_words: Count words.
        compute_reading_time: Estimate reading time.
        custom_extractors: Custom extraction functions.
        **kwargs: Additional config options.

    Returns:
        Enriched documents (modified in place).

    Example:
        >>> from agenticflow.document import enrich_documents
        >>>
        >>> docs = loader.load_directory("./docs")
        >>> enrich_documents(docs)
        >>> # Now all docs have keywords, word_count, reading_time_minutes
    """
    enricher = MetadataEnricher(
        extract_keywords=extract_keywords,
        count_words=count_words,
        compute_reading_time=compute_reading_time,
        custom_extractors=custom_extractors,
        **kwargs,
    )
    return enricher.enrich(documents)


__all__ = [
    "EnricherConfig",
    "MetadataEnricher",
    "enrich_documents",
    "STOP_WORDS",
]
