"""Result processing utilities for retrieval results.

Simple, composable functions for common result transformations:
- filter_by_score: Remove results below a threshold
- top_k: Keep only top K results
- add_citations: Add «1», «2» markers to results
- format_context: Format results into a context string for LLM prompts
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from agenticflow.retriever.base import RetrievalResult


def filter_by_score(
    results: list[RetrievalResult],
    min_score: float,
) -> list[RetrievalResult]:
    """Filter results by minimum score.

    Args:
        results: Retrieval results to filter.
        min_score: Minimum score threshold (results below are removed).

    Returns:
        Filtered list of results.

    Example:
        results = await retriever.retrieve(query, k=10, include_scores=True)
        results = filter_by_score(results, min_score=0.7)
    """
    return [r for r in results if r.score >= min_score]


def top_k(
    results: list[RetrievalResult],
    k: int,
) -> list[RetrievalResult]:
    """Keep only top K results.

    Args:
        results: Retrieval results.
        k: Maximum number of results to keep.

    Returns:
        Top K results (assumes already sorted by score).

    Example:
        results = await retriever.retrieve(query, k=20, include_scores=True)
        results = top_k(results, k=5)  # Keep only best 5
    """
    return results[:k]


def add_citations(
    results: list[RetrievalResult],
    *,
    marker_format: str = "«{n}»",
    start_index: int = 1,
) -> list[RetrievalResult]:
    """Add citation markers to retrieval results.

    Adds a `citation` field to each result's metadata with the marker
    (e.g., «1», «2»). Use this to reference sources in generated text.

    Args:
        results: Retrieval results.
        marker_format: Format string for markers. {n} is replaced with index.
        start_index: Starting citation number (default: 1).

    Returns:
        Results with citation markers in metadata.

    Example:
        results = add_citations(results)
        # results[0].metadata["citation"] == "«1»"
        # results[1].metadata["citation"] == "«2»"

        # In prompt:
        # "According to «1», Python is..."
        #
        # Sources:
        # «1» {results[0].document.text[:100]}
    """
    from agenticflow.retriever.base import RetrievalResult

    cited: list[RetrievalResult] = []

    for i, r in enumerate(results):
        marker = marker_format.format(n=i + start_index)
        new_metadata = {**r.metadata, "citation": marker}

        cited.append(
            RetrievalResult(
                document=r.document,
                score=r.score,
                retriever_name=r.retriever_name,
                metadata=new_metadata,
            )
        )

    return cited


def format_context(
    results: list[RetrievalResult],
    *,
    include_citations: bool = True,
    include_source: bool = True,
    separator: str = "\n\n---\n\n",
) -> str:
    """Format retrieval results into a context string for LLM prompts.

    Args:
        results: Retrieval results (optionally with citations added).
        include_citations: Prefix each chunk with its citation marker.
        include_source: Include source metadata if available.
        separator: Separator between chunks.

    Returns:
        Formatted context string.

    Example:
        results = add_citations(results)
        context = format_context(results)

        # Output:
        # «1» [Source: doc.pdf]
        # This is the first chunk of text...
        #
        # ---
        #
        # «2» [Source: other.pdf]
        # This is the second chunk...
    """
    chunks: list[str] = []

    for r in results:
        parts: list[str] = []

        # Citation marker
        if include_citations and "citation" in r.metadata:
            parts.append(r.metadata["citation"])

        # Source info
        if include_source:
            source = r.document.metadata.get("source") or r.document.metadata.get("filename")
            if source:
                parts.append(f"[Source: {source}]")

        # Build header
        header = " ".join(parts)

        # Full chunk
        if header:
            chunks.append(f"{header}\n{r.document.text}")
        else:
            chunks.append(r.document.text)

    return separator.join(chunks)


def format_citations_reference(
    results: list[RetrievalResult],
    *,
    max_preview_length: int = 100,
) -> str:
    """Format a citations reference section for LLM responses.

    Creates a numbered list of sources that can be appended to LLM output.

    Args:
        results: Retrieval results with citations added.
        max_preview_length: Max characters to show from each source.

    Returns:
        Formatted citations reference.

    Example:
        results = add_citations(results)
        reference = format_citations_reference(results)

        # Output:
        # Sources:
        # «1» doc.pdf: This is a preview of the first document...
        # «2» other.pdf: This is a preview of the second...
    """
    if not results:
        return ""

    lines = ["Sources:"]

    for r in results:
        marker = r.metadata.get("citation", "")
        source = (
            r.document.metadata.get("source")
            or r.document.metadata.get("filename")
            or "Unknown"
        )

        preview = r.document.text[:max_preview_length]
        if len(r.document.text) > max_preview_length:
            preview += "..."

        # Clean up preview (remove newlines)
        preview = preview.replace("\n", " ").strip()

        lines.append(f"{marker} {source}: {preview}")

    return "\n".join(lines)


__all__ = [
    "filter_by_score",
    "top_k",
    "add_citations",
    "format_context",
    "format_citations_reference",
]
