"""Document summarization strategies for large documents.

Provides multiple approaches for summarizing documents that exceed LLM context limits:

- **MapReduceSummarizer**: Parallel chunk summarization, then combine summaries
- **RefineSummarizer**: Sequential refinement with each chunk
- **HierarchicalSummarizer**: Tree-based recursive summarization

All strategies handle documents of arbitrary length by chunking and aggregating.

Example:
    ```python
    from agenticflow.document import MapReduceSummarizer, RefineSummarizer
    from agenticflow.models import ChatModel

    # Map-reduce: Fast, parallel-friendly
    summarizer = MapReduceSummarizer(
        model=ChatModel(model="gpt-4o-mini"),
        chunk_size=4000,
    )
    summary = await summarizer.summarize(long_document)

    # Refine: Better coherence, sequential
    summarizer = RefineSummarizer(
        model=ChatModel(model="gpt-4o-mini"),
        chunk_size=4000,
    )
    summary = await summarizer.summarize(long_document, context="financial report")
    ```
"""

from __future__ import annotations

import asyncio
import math
import sys
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import StrEnum
from typing import TYPE_CHECKING

from agenticflow.observability import LogLevel, ObservabilityLogger

if TYPE_CHECKING:
    from agenticflow.models import Model


class SummarizationStrategy(StrEnum):
    """Available summarization strategies."""

    MAP_REDUCE = "map_reduce"
    REFINE = "refine"
    HIERARCHICAL = "hierarchical"


@dataclass(frozen=True, slots=True, kw_only=True)
class SummaryResult:
    """Result of document summarization."""

    summary: str
    """The final summary text."""

    strategy: SummarizationStrategy
    """Strategy used for summarization."""

    chunks_processed: int
    """Number of chunks processed."""

    reduction_ratio: float
    """Original length / summary length."""

    intermediate_summaries: list[str] = field(default_factory=list)
    """Intermediate summaries (for debugging/analysis)."""

    metadata: dict[str, object] = field(default_factory=dict)
    """Additional metadata about the summarization process."""


class BaseSummarizer(ABC):
    """Base class for document summarization strategies."""

    # Default prompts - can be overridden
    CHUNK_SUMMARY_PROMPT = """Summarize the following text concisely, capturing the main points and key information.

Text:
{text}

Summary:"""

    COMBINE_SUMMARIES_PROMPT = """Combine the following summaries into a single coherent summary.
Eliminate redundancy while preserving all important information.

Summaries:
{summaries}

Combined Summary:"""

    REFINE_PROMPT = """You have an existing summary and new content to incorporate.
Refine the summary to include the new information while maintaining coherence.

Existing Summary:
{existing_summary}

New Content:
{new_content}

Refined Summary:"""

    FINAL_SUMMARY_PROMPT = """Create a final, polished summary from the following content.
{context_instruction}

Content:
{content}

Final Summary:"""

    def __init__(
        self,
        model: Model,
        *,
        chunk_size: int = 4000,
        chunk_overlap: int = 200,
        max_summary_length: int | None = None,
        show_progress: bool = False,
        log_level: LogLevel = LogLevel.WARNING,
    ) -> None:
        """Initialize the summarizer.

        Args:
            model: Language model for generating summaries.
            chunk_size: Maximum characters per chunk.
            chunk_overlap: Overlap between chunks for context continuity.
            max_summary_length: Target maximum summary length (soft limit).
            show_progress: Show progress during summarization.
            log_level: Logging level for the summarizer.
        """
        self._model = model
        self._chunk_size = chunk_size
        self._chunk_overlap = chunk_overlap
        self._max_summary_length = max_summary_length
        self._show_progress = show_progress
        self._log = ObservabilityLogger("document.summarizer", level=log_level)

    async def _call_model(self, prompt: str) -> str:
        """Call the model with a prompt and return the response text.

        Handles the model's message-based interface.
        """
        messages = [{"role": "user", "content": prompt}]
        response = await self._model.ainvoke(messages)
        return response.content

    def _chunk_text(self, text: str) -> list[str]:
        """Split text into overlapping chunks.

        Attempts to break at natural boundaries (paragraphs, sentences).
        """
        chunks: list[str] = []
        start = 0
        text_len = len(text)

        while start < text_len:
            end = min(start + self._chunk_size, text_len)
            chunk = text[start:end]

            # Try to break at natural boundaries if not at end
            if end < text_len:
                # Priority: double newline > sentence end > single newline
                for separator in ["\n\n", ". ", ".\n", "\n", " "]:
                    last_sep = chunk.rfind(separator)
                    if last_sep > self._chunk_size // 2:  # At least half the chunk
                        chunk = chunk[:last_sep + len(separator)]
                        break

            stripped = chunk.strip()
            if stripped:
                chunks.append(stripped)

            # Move start, accounting for overlap
            new_start = start + len(chunk) - self._chunk_overlap
            if new_start <= start:
                # Prevent infinite loop - ensure we always make progress
                start = end
            else:
                start = new_start

        return chunks

    @abstractmethod
    async def summarize(
        self,
        text: str,
        *,
        context: str | None = None,
        max_length: int | None = None,
    ) -> SummaryResult:
        """Summarize the given text.

        Args:
            text: The text to summarize.
            context: Optional context about the document (e.g., "financial report").
            max_length: Override the default max summary length.

        Returns:
            SummaryResult with the summary and metadata.
        """
        ...


class MapReduceSummarizer(BaseSummarizer):
    """Map-Reduce summarization strategy.

    1. **Map**: Summarize each chunk independently (parallelizable)
    2. **Reduce**: Combine chunk summaries into final summary

    Best for:
    - Large documents where parallelism helps
    - When chunks are relatively independent
    - Speed over perfect coherence

    Example:
        ```python
        summarizer = MapReduceSummarizer(
            model=ChatModel(model="gpt-4o-mini"),
            chunk_size=4000,
            max_concurrent=5,  # Parallel summarization
        )
        result = await summarizer.summarize(long_document)
        print(result.summary)
        ```
    """

    def __init__(
        self,
        model: Model,
        *,
        chunk_size: int = 4000,
        chunk_overlap: int = 200,
        max_summary_length: int | None = None,
        max_concurrent: int = 5,
        show_progress: bool = False,
        log_level: LogLevel = LogLevel.WARNING,
    ) -> None:
        """Initialize MapReduce summarizer.

        Args:
            model: Language model for generating summaries.
            chunk_size: Maximum characters per chunk.
            chunk_overlap: Overlap between chunks.
            max_summary_length: Target maximum summary length.
            max_concurrent: Maximum concurrent summarization tasks.
            show_progress: Show progress during summarization.
            log_level: Logging level.
        """
        super().__init__(
            model,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            max_summary_length=max_summary_length,
            show_progress=show_progress,
            log_level=log_level,
        )
        self._max_concurrent = max_concurrent

    async def _summarize_chunk(self, chunk: str, index: int) -> str:
        """Summarize a single chunk."""
        prompt = self.CHUNK_SUMMARY_PROMPT.format(text=chunk)
        summary = await self._call_model(prompt)

        self._log.debug(
            "Chunk summarized",
            chunk_index=index,
            input_len=len(chunk),
            output_len=len(summary),
        )

        return summary

    async def _combine_summaries(
        self,
        summaries: list[str],
        context: str | None = None,
    ) -> str:
        """Combine multiple summaries into one."""
        combined_text = "\n\n---\n\n".join(
            f"[Section {i+1}]\n{s}" for i, s in enumerate(summaries)
        )

        # If combined summaries are still too long, recurse
        if len(combined_text) > self._chunk_size * 2:
            self._log.debug(
                "Recursive combination needed",
                summaries_count=len(summaries),
                combined_len=len(combined_text),
            )
            # Split summaries into groups and combine recursively
            mid = len(summaries) // 2
            left = await self._combine_summaries(summaries[:mid], context)
            right = await self._combine_summaries(summaries[mid:], context)
            summaries = [left, right]
            combined_text = "\n\n---\n\n".join(summaries)

        context_instruction = ""
        if context:
            context_instruction = f"This is a {context}. "

        prompt = self.FINAL_SUMMARY_PROMPT.format(
            context_instruction=context_instruction,
            content=combined_text,
        )

        return await self._call_model(prompt)

    async def summarize(
        self,
        text: str,
        *,
        context: str | None = None,
        max_length: int | None = None,
    ) -> SummaryResult:
        """Summarize using map-reduce strategy."""
        start_time = time.perf_counter()
        original_length = len(text)

        # Check if text fits in a single chunk
        if original_length <= self._chunk_size:
            self._log.info("Document fits in single chunk, direct summarization")
            context_instruction = f"This is a {context}. " if context else ""
            prompt = self.FINAL_SUMMARY_PROMPT.format(
                context_instruction=context_instruction,
                content=text,
            )
            summary = await self._call_model(prompt)

            return SummaryResult(
                summary=summary,
                strategy=SummarizationStrategy.MAP_REDUCE,
                chunks_processed=1,
                reduction_ratio=original_length / len(summary) if summary else 1.0,
                intermediate_summaries=[],
            )

        # Chunk the document
        chunks = self._chunk_text(text)
        total_chunks = len(chunks)
        completed = 0

        self._log.info(
            "Starting map-reduce summarization",
            total_chunks=total_chunks,
            original_length=original_length,
            max_concurrent=self._max_concurrent,
        )

        if self._show_progress:
            print(f"📄 Summarizing document ({original_length:,} chars) in {total_chunks} chunks...")
            print(f"   Strategy: map-reduce | Concurrency: {self._max_concurrent}")
            sys.stdout.flush()

        # Map phase: summarize chunks with concurrency limit
        semaphore = asyncio.Semaphore(self._max_concurrent)
        map_start = time.perf_counter()

        async def bounded_summarize(chunk: str, index: int) -> str:
            nonlocal completed
            async with semaphore:
                chunk_start = time.perf_counter()
                result = await self._summarize_chunk(chunk, index)
                chunk_time = time.perf_counter() - chunk_start
                completed += 1

                self._log.debug(
                    "Chunk summarized",
                    chunk=index + 1,
                    total=total_chunks,
                    chunk_time_s=round(chunk_time, 2),
                    input_chars=len(chunk),
                    output_chars=len(result),
                )

                if self._show_progress:
                    elapsed = time.perf_counter() - map_start
                    rate = completed / elapsed if elapsed > 0 else 0
                    eta = (total_chunks - completed) / rate if rate > 0 else 0
                    print(f"  ✓ Chunk {completed}/{total_chunks} ({chunk_time:.1f}s) | Rate: {rate:.1f}/s | ETA: {eta:.0f}s")
                    sys.stdout.flush()
                return result

        tasks = [
            bounded_summarize(chunk, i)
            for i, chunk in enumerate(chunks)
        ]
        chunk_summaries = await asyncio.gather(*tasks)
        map_time = time.perf_counter() - map_start

        self._log.info(
            "Map phase complete",
            chunks_processed=total_chunks,
            map_time_s=round(map_time, 2),
            chunks_per_sec=round(total_chunks / map_time, 2) if map_time > 0 else 0,
        )

        if self._show_progress:
            print(f"🔄 Combining {total_chunks} summaries... (map phase: {map_time:.1f}s)")
            sys.stdout.flush()

        # Reduce phase: combine summaries
        reduce_start = time.perf_counter()
        final_summary = await self._combine_summaries(list(chunk_summaries), context)
        reduce_time = time.perf_counter() - reduce_start
        total_time = time.perf_counter() - start_time

        reduction = original_length / len(final_summary) if final_summary else 1.0

        self._log.info(
            "Summarization complete",
            total_time_s=round(total_time, 2),
            map_time_s=round(map_time, 2),
            reduce_time_s=round(reduce_time, 2),
            reduction_ratio=round(reduction, 1),
            chars_per_sec=round(original_length / total_time, 0) if total_time > 0 else 0,
        )

        if self._show_progress:
            print(f"✅ Summary complete in {total_time:.1f}s ({len(final_summary):,} chars, {reduction:.1f}x reduction)")
            print(f"   Throughput: {original_length / total_time:,.0f} chars/sec")
            sys.stdout.flush()

        return SummaryResult(
            summary=final_summary,
            strategy=SummarizationStrategy.MAP_REDUCE,
            chunks_processed=total_chunks,
            reduction_ratio=reduction,
            intermediate_summaries=list(chunk_summaries),
            metadata={
                "original_length": original_length,
                "final_length": len(final_summary),
                "max_concurrent": self._max_concurrent,
                "total_time_s": round(total_time, 2),
                "map_time_s": round(map_time, 2),
                "reduce_time_s": round(reduce_time, 2),
                "chunks_per_sec": round(total_chunks / map_time, 2) if map_time > 0 else 0,
            },
        )


class RefineSummarizer(BaseSummarizer):
    """Iterative refinement summarization strategy.

    1. Summarize first chunk
    2. For each subsequent chunk, refine the summary to include new info
    3. Final polish pass

    Best for:
    - Documents with sequential narrative
    - When coherence is more important than speed
    - Smaller documents (sequential processing)

    Example:
        ```python
        summarizer = RefineSummarizer(
            model=ChatModel(model="gpt-4o-mini"),
            chunk_size=4000,
        )
        result = await summarizer.summarize(
            long_document,
            context="quarterly earnings report",
        )
        print(result.summary)
        ```
    """

    async def summarize(
        self,
        text: str,
        *,
        context: str | None = None,
        max_length: int | None = None,
    ) -> SummaryResult:
        """Summarize using iterative refinement."""
        original_length = len(text)

        # Check if text fits in a single chunk
        if original_length <= self._chunk_size:
            self._log.info("Document fits in single chunk, direct summarization")
            context_instruction = f"This is a {context}. " if context else ""
            prompt = self.FINAL_SUMMARY_PROMPT.format(
                context_instruction=context_instruction,
                content=text,
            )
            summary = await self._call_model(prompt)

            return SummaryResult(
                summary=summary,
                strategy=SummarizationStrategy.REFINE,
                chunks_processed=1,
                reduction_ratio=original_length / len(summary) if summary else 1.0,
                intermediate_summaries=[],
            )

        # Chunk the document
        chunks = self._chunk_text(text)
        total_chunks = len(chunks)
        intermediate_summaries: list[str] = []

        self._log.info(
            "Starting refine summarization",
            total_chunks=total_chunks,
            original_length=original_length,
        )

        if self._show_progress:
            print(f"📄 Summarizing document ({original_length:,} chars) with refinement...")

        # Initial summary from first chunk
        current_summary = await self._call_model(
            self.CHUNK_SUMMARY_PROMPT.format(text=chunks[0])
        )
        intermediate_summaries.append(current_summary)

        if self._show_progress:
            print(f"  ✓ Initial summary from chunk 1/{total_chunks}")

        # Refine with each subsequent chunk
        for i, chunk in enumerate(chunks[1:], start=2):
            prompt = self.REFINE_PROMPT.format(
                existing_summary=current_summary,
                new_content=chunk,
            )
            current_summary = await self._call_model(prompt)
            intermediate_summaries.append(current_summary)

            self._log.debug(
                "Refined summary",
                chunk_index=i,
                summary_len=len(current_summary),
            )

            if self._show_progress:
                print(f"  ✓ Refined with chunk {i}/{total_chunks}")

        # Final polish
        if self._show_progress:
            print("🔄 Final polish...")

        context_instruction = f"This is a {context}. " if context else ""
        final_prompt = self.FINAL_SUMMARY_PROMPT.format(
            context_instruction=context_instruction,
            content=current_summary,
        )
        final_summary = await self._call_model(final_prompt)

        if self._show_progress:
            reduction = original_length / len(final_summary) if final_summary else 1.0
            print(f"✅ Summary complete ({len(final_summary):,} chars, {reduction:.1f}x reduction)")

        return SummaryResult(
            summary=final_summary,
            strategy=SummarizationStrategy.REFINE,
            chunks_processed=total_chunks,
            reduction_ratio=original_length / len(final_summary) if final_summary else 1.0,
            intermediate_summaries=intermediate_summaries,
            metadata={
                "original_length": original_length,
                "final_length": len(final_summary),
            },
        )


class HierarchicalSummarizer(BaseSummarizer):
    """Hierarchical tree-based summarization strategy.

    1. Split document into chunks (leaves)
    2. Summarize pairs/groups of chunks (internal nodes)
    3. Recursively summarize until single root summary

    Best for:
    - Very large documents (books, long reports)
    - When you need multi-level abstraction
    - Balanced parallel + coherence tradeoff

    Example:
        ```python
        summarizer = HierarchicalSummarizer(
            model=ChatModel(model="gpt-4o-mini"),
            chunk_size=4000,
            branching_factor=4,  # Combine 4 chunks at a time
        )
        result = await summarizer.summarize(book_text)

        # Access intermediate levels
        for i, level in enumerate(result.metadata.get("levels", [])):
            print(f"Level {i}: {len(level)} summaries")
        ```
    """

    def __init__(
        self,
        model: Model,
        *,
        chunk_size: int = 4000,
        chunk_overlap: int = 200,
        max_summary_length: int | None = None,
        branching_factor: int = 4,
        max_concurrent: int = 5,
        show_progress: bool = False,
        log_level: LogLevel = LogLevel.WARNING,
    ) -> None:
        """Initialize hierarchical summarizer.

        Args:
            model: Language model for generating summaries.
            chunk_size: Maximum characters per chunk.
            chunk_overlap: Overlap between chunks.
            max_summary_length: Target maximum summary length.
            branching_factor: Number of summaries to combine at each level.
            max_concurrent: Maximum concurrent summarization tasks.
            show_progress: Show progress during summarization.
            log_level: Logging level.
        """
        super().__init__(
            model,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            max_summary_length=max_summary_length,
            show_progress=show_progress,
            log_level=log_level,
        )
        self._branching_factor = branching_factor
        self._max_concurrent = max_concurrent

    async def _summarize_group(self, texts: list[str], level: int) -> str:
        """Summarize a group of texts into one."""
        combined = "\n\n---\n\n".join(texts)
        prompt = self.COMBINE_SUMMARIES_PROMPT.format(summaries=combined)
        return await self._call_model(prompt)

    async def summarize(
        self,
        text: str,
        *,
        context: str | None = None,
        max_length: int | None = None,
    ) -> SummaryResult:
        """Summarize using hierarchical tree strategy."""
        original_length = len(text)

        # Check if text fits in a single chunk
        if original_length <= self._chunk_size:
            self._log.info("Document fits in single chunk, direct summarization")
            context_instruction = f"This is a {context}. " if context else ""
            prompt = self.FINAL_SUMMARY_PROMPT.format(
                context_instruction=context_instruction,
                content=text,
            )
            summary = await self._call_model(prompt)

            return SummaryResult(
                summary=summary,
                strategy=SummarizationStrategy.HIERARCHICAL,
                chunks_processed=1,
                reduction_ratio=original_length / len(summary) if summary else 1.0,
                intermediate_summaries=[],
            )

        # Chunk the document
        chunks = self._chunk_text(text)
        total_chunks = len(chunks)

        # Calculate tree depth
        tree_depth = math.ceil(math.log(total_chunks, self._branching_factor))

        self._log.info(
            "Starting hierarchical summarization",
            total_chunks=total_chunks,
            tree_depth=tree_depth,
            branching_factor=self._branching_factor,
        )

        if self._show_progress:
            print(f"📄 Summarizing document ({original_length:,} chars)")
            print(f"   {total_chunks} chunks, {tree_depth} levels, branching factor {self._branching_factor}")

        levels: list[list[str]] = []
        current_level = chunks
        level_num = 0

        semaphore = asyncio.Semaphore(self._max_concurrent)

        while len(current_level) > 1:
            level_num += 1
            next_level: list[str] = []

            # Group current level into batches
            groups: list[list[str]] = []
            for i in range(0, len(current_level), self._branching_factor):
                groups.append(current_level[i:i + self._branching_factor])

            if self._show_progress:
                print(f"  Level {level_num}: Combining {len(current_level)} → {len(groups)} summaries")

            async def bounded_summarize(group: list[str], idx: int) -> str:
                async with semaphore:
                    return await self._summarize_group(group, level_num)

            tasks = [
                bounded_summarize(group, i)
                for i, group in enumerate(groups)
            ]
            next_level = await asyncio.gather(*tasks)

            levels.append(list(next_level))
            current_level = list(next_level)

            self._log.debug(
                "Level complete",
                level=level_num,
                summaries_count=len(current_level),
            )

        # Final polish
        if self._show_progress:
            print("🔄 Final polish...")

        context_instruction = f"This is a {context}. " if context else ""
        final_prompt = self.FINAL_SUMMARY_PROMPT.format(
            context_instruction=context_instruction,
            content=current_level[0] if current_level else "",
        )
        final_summary = await self._call_model(final_prompt)

        if self._show_progress:
            reduction = original_length / len(final_summary) if final_summary else 1.0
            print(f"✅ Summary complete ({len(final_summary):,} chars, {reduction:.1f}x reduction)")

        # Flatten all intermediate summaries
        all_intermediates = [s for level in levels for s in level]

        return SummaryResult(
            summary=final_summary,
            strategy=SummarizationStrategy.HIERARCHICAL,
            chunks_processed=total_chunks,
            reduction_ratio=original_length / len(final_summary) if final_summary else 1.0,
            intermediate_summaries=all_intermediates,
            metadata={
                "original_length": original_length,
                "final_length": len(final_summary),
                "tree_depth": level_num,
                "branching_factor": self._branching_factor,
                "levels": levels,  # Structured by level
            },
        )


# Convenience factory function
def create_summarizer(
    strategy: SummarizationStrategy | str,
    model: Model,
    **kwargs,
) -> BaseSummarizer:
    """Create a summarizer with the specified strategy.

    Args:
        strategy: Summarization strategy to use.
        model: Language model for summarization.
        **kwargs: Additional arguments passed to the summarizer.

    Returns:
        Configured summarizer instance.

    Example:
        ```python
        summarizer = create_summarizer(
            "map_reduce",
            model=ChatModel(model="gpt-4o-mini"),
            chunk_size=4000,
            max_concurrent=5,
        )
        ```
    """
    strategy = SummarizationStrategy(strategy)

    if strategy == SummarizationStrategy.MAP_REDUCE:
        return MapReduceSummarizer(model, **kwargs)
    elif strategy == SummarizationStrategy.REFINE:
        return RefineSummarizer(model, **kwargs)
    elif strategy == SummarizationStrategy.HIERARCHICAL:
        return HierarchicalSummarizer(model, **kwargs)
    else:
        msg = f"Unknown strategy: {strategy}"
        raise ValueError(msg)
