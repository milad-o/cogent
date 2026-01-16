"""
Summarizer capability - Document summarization for agents.

Provides tools for summarizing large documents using various strategies
(map-reduce, refine, hierarchical), enabling agents to process documents
that exceed LLM context limits.

Example:
    ```python
    from agenticflow import Agent
    from agenticflow.capabilities import Summarizer
    from agenticflow.models import ChatModel

    agent = Agent(
        name="DocumentAnalyzer",
        model=model,
        capabilities=[Summarizer(model=ChatModel(model="gpt-4o-mini"))],
    )

    # Agent can now summarize large documents
    await agent.run("Summarize the annual report in data/report.pdf")
    await agent.run("Give me a brief summary of all the markdown files in docs/")
    ```
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from agenticflow.capabilities.base import BaseCapability
from agenticflow.document.summarizer import (
    BaseSummarizer,
    HierarchicalSummarizer,
    MapReduceSummarizer,
    RefineSummarizer,
    SummarizationStrategy,
    SummaryResult,
)
from agenticflow.tools.base import tool

if TYPE_CHECKING:
    from agenticflow.models import Model


@dataclass(frozen=True, slots=True, kw_only=True)
class SummarizerConfig:
    """Configuration for the Summarizer capability."""

    chunk_size: int = 4000
    """Maximum characters per chunk for processing."""

    chunk_overlap: int = 200
    """Overlap between chunks for context continuity."""

    max_concurrent: int = 5
    """Maximum concurrent summarization tasks (for map-reduce/hierarchical)."""

    branching_factor: int = 4
    """Branching factor for hierarchical summarization."""

    default_strategy: str = "map_reduce"
    """Default summarization strategy to use."""

    max_file_size_mb: int = 50
    """Maximum file size to process in MB."""

    show_progress: bool = True
    """Show human-friendly progress output during summarization."""

    supported_extensions: frozenset[str] = frozenset({
        ".txt", ".md", ".markdown", ".rst", ".text",
        ".py", ".js", ".ts", ".java", ".c", ".cpp", ".h", ".hpp",
        ".json", ".yaml", ".yml", ".xml", ".html", ".css",
    })
    """File extensions that can be read as text for summarization."""


class Summarizer(BaseCapability):
    """
    Summarizer capability for processing large documents.

    Provides tools for summarizing text and files using various strategies:
    - **map_reduce**: Fast, parallel-friendly (best for large docs)
    - **refine**: Sequential refinement (best for coherence)
    - **hierarchical**: Tree-based (best for very large docs like books)

    Args:
        model: Language model for generating summaries. If not provided,
            uses the agent's model when attached.
        config: Summarizer configuration options.
        allowed_paths: List of paths the agent can access. If empty, allows
            current working directory only.

    Tools provided:
        - summarize_text: Summarize text content directly
        - summarize_file: Summarize a text file
        - summarize_files: Summarize multiple files together
        - get_summary_strategies: List available strategies with descriptions

    Example:
        ```python
        from agenticflow import Agent
        from agenticflow.capabilities import Summarizer
        from agenticflow.models import ChatModel

        # With explicit model
        agent = Agent(
            name="Summarizer",
            model=main_model,
            capabilities=[
                Summarizer(
                    model=ChatModel(model="gpt-4o-mini"),  # Cheaper model for summaries
                    config=SummarizerConfig(chunk_size=8000),
                )
            ],
        )

        # Or use agent's model (set model=None or omit)
        agent = Agent(
            name="Summarizer",
            model=model,
            capabilities=[Summarizer()],  # Uses agent's model
        )
        ```
    """

    def __init__(
        self,
        model: Model | None = None,
        config: SummarizerConfig | None = None,
        allowed_paths: list[str | Path] | None = None,
    ) -> None:
        self._model = model
        self._config = config or SummarizerConfig()
        self._allowed_paths = [Path(p).resolve() for p in (allowed_paths or ["."])]
        self._summarizers: dict[SummarizationStrategy, BaseSummarizer] = {}

    @property
    def name(self) -> str:
        """Unique name for this capability."""
        return "summarizer"

    @property
    def description(self) -> str:
        """Human-readable description."""
        return "Summarize large documents using map-reduce, refine, or hierarchical strategies"

    @property
    def tools(self) -> list:
        """Tools this capability provides to the agent."""
        return self._get_tools()

    def _get_model(self) -> Model:
        """Get the model to use for summarization."""
        if self._model is not None:
            return self._model

        if self._agent is not None and hasattr(self._agent, "model"):
            return self._agent.model

        msg = "No model available. Provide a model to Summarizer or attach to an agent."
        raise RuntimeError(msg)

    def _get_summarizer(self, strategy: SummarizationStrategy) -> BaseSummarizer:
        """Get or create a summarizer for the given strategy."""
        if strategy not in self._summarizers:
            model = self._get_model()
            config = self._config

            if strategy == SummarizationStrategy.MAP_REDUCE:
                self._summarizers[strategy] = MapReduceSummarizer(
                    model=model,
                    chunk_size=config.chunk_size,
                    chunk_overlap=config.chunk_overlap,
                    max_concurrent=config.max_concurrent,
                    show_progress=config.show_progress,
                )
            elif strategy == SummarizationStrategy.REFINE:
                self._summarizers[strategy] = RefineSummarizer(
                    model=model,
                    chunk_size=config.chunk_size,
                    chunk_overlap=config.chunk_overlap,
                    show_progress=config.show_progress,
                )
            elif strategy == SummarizationStrategy.HIERARCHICAL:
                self._summarizers[strategy] = HierarchicalSummarizer(
                    model=model,
                    chunk_size=config.chunk_size,
                    chunk_overlap=config.chunk_overlap,
                    max_concurrent=config.max_concurrent,
                    branching_factor=config.branching_factor,
                    show_progress=config.show_progress,
                )
            else:
                msg = f"Unknown strategy: {strategy}"
                raise ValueError(msg)

        return self._summarizers[strategy]

    def _validate_path(self, path: str | Path) -> Path:
        """Validate that path is within allowed directories."""
        resolved = Path(path).resolve()

        for allowed in self._allowed_paths:
            try:
                resolved.relative_to(allowed)
                return resolved
            except ValueError:
                continue

        msg = f"Path {resolved} is not within allowed directories"
        raise PermissionError(msg)

    def _check_file_size(self, path: Path) -> None:
        """Check file size is within limits."""
        if path.exists():
            size_mb = path.stat().st_size / (1024 * 1024)
            if size_mb > self._config.max_file_size_mb:
                msg = f"File size ({size_mb:.1f}MB) exceeds limit ({self._config.max_file_size_mb}MB)"
                raise ValueError(msg)

    def _can_read_file(self, path: Path) -> bool:
        """Check if file extension is supported for text reading."""
        return path.suffix.lower() in self._config.supported_extensions

    def _read_file_content(self, path: Path) -> str:
        """Read text content from a file."""
        if not path.exists():
            msg = f"File not found: {path}"
            raise FileNotFoundError(msg)

        if not self._can_read_file(path):
            msg = f"Unsupported file extension: {path.suffix}. Supported: {', '.join(sorted(self._config.supported_extensions))}"
            raise ValueError(msg)

        self._check_file_size(path)
        return path.read_text(encoding="utf-8")

    def _result_to_dict(self, result: SummaryResult, source: str | None = None) -> dict[str, Any]:
        """Convert SummaryResult to a dictionary for tool output."""
        output = {
            "summary": result.summary,
            "strategy": result.strategy.value,
            "chunks_processed": result.chunks_processed,
            "reduction_ratio": round(result.reduction_ratio, 2),
            "original_length": result.metadata.get("original_length"),
            "summary_length": len(result.summary),
        }

        if source:
            output["source"] = source

        return output

    def _get_tools(self) -> list:
        """Return the tools provided by this capability."""

        @tool
        async def summarize_text(
            text: str,
            strategy: str = "map_reduce",
            context: str | None = None,
        ) -> dict[str, Any]:
            """Summarize text content using the specified strategy.

            Use this for summarizing large text that exceeds context limits.
            The text is automatically chunked and processed based on the strategy.

            Args:
                text: The text content to summarize
                strategy: Summarization strategy - "map_reduce" (fast, parallel),
                    "refine" (coherent, sequential), or "hierarchical" (very large docs)
                context: Optional context about the content (e.g., "financial report",
                    "technical documentation") to improve summary quality

            Returns:
                Dictionary with summary, strategy used, chunks processed, and reduction ratio
            """
            strat = SummarizationStrategy(strategy)
            summarizer = self._get_summarizer(strat)

            result = await summarizer.summarize(text, context=context)
            return self._result_to_dict(result)

        @tool
        async def summarize_file(
            path: str,
            strategy: str = "map_reduce",
            context: str | None = None,
        ) -> dict[str, Any]:
            """Summarize a text file using the specified strategy.

            Reads the file content and summarizes it. Works with text files
            like .txt, .md, .py, .json, etc.

            Args:
                path: Path to the file to summarize
                strategy: Summarization strategy - "map_reduce" (fast, parallel),
                    "refine" (coherent, sequential), or "hierarchical" (very large docs)
                context: Optional context about the file (e.g., "Python source code",
                    "meeting notes") to improve summary quality

            Returns:
                Dictionary with summary, source file, strategy used, and metrics
            """
            file_path = self._validate_path(path)
            content = self._read_file_content(file_path)

            strat = SummarizationStrategy(strategy)
            summarizer = self._get_summarizer(strat)

            # Auto-detect context from file extension if not provided
            if context is None:
                ext = file_path.suffix.lower()
                context_map = {
                    ".py": "Python source code",
                    ".js": "JavaScript source code",
                    ".ts": "TypeScript source code",
                    ".md": "Markdown documentation",
                    ".json": "JSON data",
                    ".yaml": "YAML configuration",
                    ".yml": "YAML configuration",
                }
                context = context_map.get(ext)

            result = await summarizer.summarize(content, context=context)
            return self._result_to_dict(result, source=str(file_path))

        @tool
        async def summarize_files(
            paths: list[str],
            strategy: str = "map_reduce",
            context: str | None = None,
            combine: bool = True,
        ) -> dict[str, Any]:
            """Summarize multiple files together or separately.

            Can either combine all files into one summary (combine=True) or
            summarize each file individually (combine=False).

            Args:
                paths: List of file paths to summarize
                strategy: Summarization strategy - "map_reduce", "refine", or "hierarchical"
                context: Optional context about the files
                combine: If True, combines all files into one summary.
                    If False, returns individual summaries for each file.

            Returns:
                Dictionary with combined summary or list of individual summaries
            """
            file_paths = [self._validate_path(p) for p in paths]

            if combine:
                # Combine all content with file markers
                combined_content = []
                for fp in file_paths:
                    content = self._read_file_content(fp)
                    combined_content.append(f"=== {fp.name} ===\n{content}")

                full_content = "\n\n".join(combined_content)

                strat = SummarizationStrategy(strategy)
                summarizer = self._get_summarizer(strat)

                result = await summarizer.summarize(full_content, context=context)
                return {
                    **self._result_to_dict(result),
                    "sources": [str(p) for p in file_paths],
                    "files_count": len(file_paths),
                }
            else:
                # Summarize each file individually
                summaries = []
                for fp in file_paths:
                    content = self._read_file_content(fp)

                    strat = SummarizationStrategy(strategy)
                    summarizer = self._get_summarizer(strat)

                    result = await summarizer.summarize(content, context=context)
                    summaries.append({
                        "file": str(fp),
                        "summary": result.summary,
                        "reduction_ratio": round(result.reduction_ratio, 2),
                    })

                return {
                    "summaries": summaries,
                    "files_count": len(summaries),
                    "strategy": strategy,
                }

        @tool
        def get_summary_strategies() -> dict[str, Any]:
            """Get available summarization strategies and their characteristics.

            Use this to understand which strategy is best for your use case.

            Returns:
                Dictionary describing each available strategy
            """
            return {
                "strategies": {
                    "map_reduce": {
                        "name": "Map-Reduce",
                        "description": "Parallel chunk summarization, then combine. Fast and scalable.",
                        "best_for": ["Large documents", "When speed matters", "Independent sections"],
                        "tradeoffs": "May lose some cross-section context",
                    },
                    "refine": {
                        "name": "Iterative Refinement",
                        "description": "Sequential processing, refining summary with each chunk.",
                        "best_for": ["Narrative documents", "When coherence is critical", "Smaller docs"],
                        "tradeoffs": "Slower (sequential), may drift from early content",
                    },
                    "hierarchical": {
                        "name": "Hierarchical Tree",
                        "description": "Tree-based recursive summarization. Balances parallel and coherence.",
                        "best_for": ["Very large documents (books)", "Multi-level abstraction needed"],
                        "tradeoffs": "More complex, good for 100+ page documents",
                    },
                },
                "default": self._config.default_strategy,
                "config": {
                    "chunk_size": self._config.chunk_size,
                    "max_file_size_mb": self._config.max_file_size_mb,
                },
            }

        return [
            summarize_text,
            summarize_file,
            summarize_files,
            get_summary_strategies,
        ]

    async def shutdown(self) -> None:
        """Clean up resources."""
        self._summarizers.clear()
        await super().shutdown()
