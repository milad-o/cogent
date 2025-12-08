"""
RAG (Retrieval-Augmented Generation) capability.

A thin capability that provides document search tools to agents.
Document loading and indexing happens OUTSIDE this capability.

Example (single retriever):
    ```python
    from agenticflow import Agent
    from agenticflow.capabilities import RAG
    from agenticflow.retriever import DenseRetriever
    from agenticflow.vectorstore import VectorStore
    from agenticflow.document import DocumentLoader, RecursiveCharacterSplitter

    # 1. Load and index documents (outside RAG)
    loader = DocumentLoader()
    docs = await loader.load_directory("docs/")
    chunks = RecursiveCharacterSplitter(chunk_size=1000).split_documents(docs)

    store = VectorStore(embeddings=embeddings)
    await store.add_documents(chunks)

    # 2. Create retriever and RAG
    rag = RAG(DenseRetriever(store))

    # 3. Add to agent
    agent = Agent(model=model, capabilities=[rag])
    answer = await agent.run("What are the key findings?")
    ```

Example (multiple retrievers with fusion):
    ```python
    from agenticflow.retriever import DenseRetriever, BM25Retriever

    dense = DenseRetriever(store)
    sparse = BM25Retriever(chunks)

    # RAG creates ensemble internally
    rag = RAG(
        retrievers=[dense, sparse],
        weights=[0.6, 0.4],
        fusion="rrf",  # or "linear", "max", "voting"
    )
    ```
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Any, Mapping

from agenticflow.capabilities.base import BaseCapability
from agenticflow.tools.base import tool

if TYPE_CHECKING:
    from agenticflow.retriever.base import FusionStrategy, Retriever
    from agenticflow.retriever.rerankers.base import Reranker


# ============================================================
# Citation Types
# ============================================================


class CitationStyle(Enum):
    """Citation formatting styles."""

    NUMERIC = "numeric"  # [1], [2], [3]
    AUTHOR_YEAR = "author_year"  # (Smith, 2023)
    FOOTNOTE = "footnote"  # ¹, ², ³
    INLINE = "inline"  # [source.pdf], [doc.md]


@dataclass(frozen=True, slots=True, kw_only=True)
class CitedPassage:
    """A cited passage from a retrieved document.

    Stores both minimal info for LLM output and full metadata for bibliography.

    Attributes:
        citation_id: Citation reference number (1, 2, 3...).
        source: Source document name.
        page: Page number if available.
        chunk_index: Chunk index within document.
        score: Relevance score (0.0-1.0).
        text: The passage text.
        metadata: Full document metadata (for bibliography, not sent to LLM).
    """

    citation_id: int
    source: str
    page: int | None = None
    chunk_index: int | None = None
    score: float = 0.0
    text: str = ""
    metadata: Mapping[str, Any] | None = None  # Full metadata for bibliography

    def format_reference(
        self,
        style: CitationStyle = CitationStyle.NUMERIC,
        include_page: bool = True,
    ) -> str:
        """Format as citation reference.

        Args:
            style: Citation style to use.
            include_page: Whether to include page number.

        Returns:
            Formatted citation string.
        """
        if style == CitationStyle.NUMERIC:
            if include_page and self.page is not None:
                return f"[{self.citation_id}, p.{self.page}]"
            return f"[{self.citation_id}]"
        elif style == CitationStyle.FOOTNOTE:
            superscripts = "⁰¹²³⁴⁵⁶⁷⁸⁹"
            num_str = "".join(superscripts[int(d)] for d in str(self.citation_id))
            return num_str
        elif style == CitationStyle.INLINE:
            return f"[{self.source}]"
        elif style == CitationStyle.AUTHOR_YEAR:
            return f"({self.source})"
        return f"[{self.citation_id}]"

    def format_full(
        self,
        include_score: bool = False,
        include_metadata_keys: list[str] | None = None,
        markdown: bool = True,
    ) -> str:
        """Format as full bibliography entry.

        Uses stored metadata for rich bibliography if available.

        Args:
            include_score: Whether to include relevance score.
            include_metadata_keys: Additional metadata keys to include (e.g., ["author", "date", "url"]).
            markdown: Use markdown formatting (bold source, italic metadata).

        Returns:
            Full citation with source, page, and optionally score and metadata.

        Example output (markdown=True):
            [1] **report.pdf** p.5 - *Author: Smith, Date: 2024*
            [2] **docs/guide.md** - *Section: Introduction*
        """
        # Build source part
        source_str = f"**{self.source}**" if markdown else self.source
        parts = [f"[{self.citation_id}]", source_str]
        
        # Add page if available
        if self.page is not None:
            parts.append(f"p.{self.page}")
        
        # Build metadata part
        meta_parts = []
        if include_metadata_keys and self.metadata:
            for key in include_metadata_keys:
                if key in self.metadata:
                    value = self.metadata[key]
                    # Handle URLs specially
                    if key.lower() == "url" and markdown:
                        meta_parts.append(f"[link]({value})")
                    else:
                        meta_parts.append(f"{key}: {value}")
        
        # Add score if requested
        if include_score and self.score > 0:
            meta_parts.append(f"relevance: {self.score:.0%}")
        
        # Combine
        result = " ".join(parts)
        if meta_parts:
            meta_str = ", ".join(meta_parts)
            if markdown:
                result += f" — *{meta_str}*"
            else:
                result += f" ({meta_str})"
        
        return result


@dataclass(frozen=True, slots=True, kw_only=True)
class RAGConfig:
    """Configuration for RAG capability.

    Most users only need to set `top_k`. Other options have sensible defaults.

    Example:
        ```python
        # Simple - just use defaults
        rag = RAG(retriever)

        # With bibliography
        rag = RAG(retriever, config=RAGConfig(bibliography=True))

        # Customize top_k
        rag = RAG(retriever, config=RAGConfig(top_k=5))
        ```

    Attributes:
        top_k: Number of passages to retrieve (default: 4).
        bibliography: Enable bibliography in format_response_with_bibliography().
        bibliography_fields: Metadata fields to show in bibliography (e.g., ["author", "date"]).
        citation_style: Citation format - numeric [1], footnote ¹, inline [source.md].
        max_chars: Truncate passages to this length (None = full text).
    """

    # Primary options
    top_k: int = 4
    bibliography: bool = False
    bibliography_fields: tuple[str, ...] = ()
    citation_style: CitationStyle = CitationStyle.NUMERIC
    max_chars: int | None = None

    # Advanced options (rarely needed)
    include_source: bool = True  # Show source in tool output
    include_page: bool = True  # Show page in tool output
    include_score: bool = False  # Show score in tool output (wastes tokens)


# ============================================================
# RAG Capability
# ============================================================

class RAG(BaseCapability):
    """RAG (Retrieval-Augmented Generation) capability.

    A thin capability that provides document search tools to agents.
    Document loading and indexing happens OUTSIDE this capability.

    Example (single retriever):
        ```python
        from agenticflow import Agent
        from agenticflow.capabilities import RAG
        from agenticflow.retriever import DenseRetriever
        from agenticflow.vectorstore import VectorStore

        # Prepare retriever
        store = VectorStore(embeddings=embeddings)
        await store.add_documents(chunks)
        retriever = DenseRetriever(store)

        # Create RAG and add to agent
        rag = RAG(retriever)
        agent = Agent(model=model, capabilities=[rag])
        ```

    Example (multiple retrievers with fusion):
        ```python
        from agenticflow.retriever import DenseRetriever, BM25Retriever

        # Create multiple retrievers
        dense = DenseRetriever(store)
        sparse = BM25Retriever(chunks)

        # RAG fuses them automatically (default: RRF)
        rag = RAG(
            retrievers=[dense, sparse],
            weights=[0.6, 0.4],
            fusion="rrf",  # or "linear", "max", "voting"
        )
        ```
    """

    DEFAULT_INSTRUCTIONS = """When answering questions:
1. Use search_documents to find relevant information first
2. Base answers ONLY on retrieved passages - do not make up information
3. Cite sources using [1], [2], etc. matching the search results
4. Include page numbers when available: [1, p.5]
5. If information isn't in documents, say so explicitly"""

    def __init__(
        self,
        retriever: Retriever | None = None,
        *,
        retrievers: list[Retriever] | None = None,
        weights: list[float] | None = None,
        fusion: FusionStrategy | str = "rrf",
        reranker: Reranker | None = None,
        config: RAGConfig | None = None,
    ) -> None:
        """Create RAG capability.

        Args:
            retriever: Single retriever for document search.
            retrievers: Multiple retrievers to ensemble (alternative to retriever).
            weights: Weights for each retriever (default: equal weights).
            fusion: Fusion strategy: "rrf", "linear", "max", "voting".
            reranker: Optional reranker for two-stage retrieval.
            config: RAG configuration options.

        Example:
            ```python
            # Single retriever
            rag = RAG(DenseRetriever(store))

            # Multiple retrievers with fusion
            rag = RAG(
                retrievers=[dense, sparse],
                weights=[0.6, 0.4],
                fusion="rrf",
            )

            # With reranking
            rag = RAG(
                retriever=dense,
                reranker=CrossEncoderReranker(),
            )
            ```

        Raises:
            ValueError: If neither retriever nor retrievers is provided,
                or if both are provided.
        """
        # Validate inputs
        if retriever is None and not retrievers:
            raise ValueError("Must provide either 'retriever' or 'retrievers'")
        if retriever is not None and retrievers:
            raise ValueError("Provide either 'retriever' or 'retrievers', not both")

        # Build the retriever
        if retrievers:
            from agenticflow.retriever import EnsembleRetriever
            from agenticflow.retriever.base import FusionStrategy as FS

            fusion_strategy = FS(fusion) if isinstance(fusion, str) else fusion
            self._retriever = EnsembleRetriever(
                retrievers=retrievers,
                weights=weights,
                fusion=fusion_strategy,
            )
        else:
            self._retriever = retriever  # type: ignore

        self._reranker = reranker
        self._config = config or RAGConfig()
        self._last_citations: list[CitedPassage] = []

    # ================================================================
    # Properties
    # ================================================================

    @property
    def name(self) -> str:
        return "rag"

    @property
    def description(self) -> str:
        return "Document retrieval and search"

    @property
    def tools(self) -> list:
        return self._create_tools()

    @property
    def retriever(self) -> Retriever:
        """The underlying retriever."""
        return self._retriever

    @property
    def citations(self) -> list[CitedPassage]:
        """Citations from last search."""
        return list(self._last_citations)

    @property
    def top_k(self) -> int:
        """Default number of results to retrieve."""
        return self._config.top_k

    # ================================================================
    # Search
    # ================================================================

    async def search(self, query: str, k: int | None = None) -> list[CitedPassage]:
        """Search for relevant passages.

        Args:
            query: Search query.
            k: Number of results (default: config.top_k).

        Returns:
            List of cited passages with source and score.

        Example:
            ```python
            passages = await rag.search("machine learning best practices", k=5)
            for p in passages:
                print(f"{p.format_reference()}: {p.text[:100]}...")
            ```
        """
        k = k or self._config.top_k

        # Retrieve with scores
        results = await self._retriever.retrieve(query, k=k, include_scores=True)

        # Rerank if configured
        if self._reranker and results:
            results = await self._reranker.rerank(query, results, k=k)

        # Convert to cited passages
        passages = []
        for i, result in enumerate(results, 1):
            doc = result.document
            # Store full metadata for bibliography generation (not sent to LLM)
            full_metadata = dict(doc.metadata) if self._config.bibliography else None
            passage = CitedPassage(
                citation_id=i,
                source=doc.metadata.get("source", "unknown"),
                page=doc.metadata.get("page"),
                chunk_index=doc.metadata.get("chunk_index"),
                score=result.score,
                text=doc.text,
                metadata=full_metadata,
            )
            passages.append(passage)

        self._last_citations = passages
        return passages

    def format_bibliography(
        self,
        passages: list[CitedPassage] | None = None,
        title: str | None = None,
        include_metadata_keys: list[str] | None = None,
    ) -> str:
        """Format passages as bibliography.

        Args:
            passages: Passages to format (default: last search results).
            title: Bibliography section title (default: from config).
            include_metadata_keys: Additional metadata keys to include.

        Returns:
            Formatted bibliography string.
        """
        passages = passages or self._last_citations
        title = title or "References"
        include_metadata_keys = include_metadata_keys or list(self._config.bibliography_fields)

        if not passages:
            return ""

        # Deduplicate by source
        seen_sources: dict[str, CitedPassage] = {}
        for p in passages:
            if p.source not in seen_sources:
                seen_sources[p.source] = p

        lines = [f"\n---\n**{title}:**"]
        for p in seen_sources.values():
            lines.append(
                p.format_full(
                    include_score=self._config.include_score,
                    include_metadata_keys=include_metadata_keys or None,
                )
            )

        return "\n".join(lines)

    def format_response_with_bibliography(
        self,
        response: str,
        passages: list[CitedPassage] | None = None,
    ) -> str:
        """Format agent response with appended bibliography.

        Call this after agent.run() to add bibliography to the response.

        Args:
            response: The agent's response text.
            passages: Passages to cite (default: last search results).

        Returns:
            Response with bibliography appended.

        Example:
            ```python
            rag = RAG(retriever, config=RAGConfig(auto_bibliography=True))
            agent = Agent(model=model, capabilities=[rag])
            response = await agent.run("What are the key findings?")
            
            # Add bibliography
            formatted = rag.format_response_with_bibliography(response)
            print(formatted)
            ```
        """
        bibliography = self.format_bibliography(passages)
        if not bibliography:
            return response
        return f"{response}\n{bibliography}"

    # ================================================================
    # Tools
    # ================================================================

    def _create_tools(self) -> list:
        """Create RAG tools for the agent."""
        cap = self
        cfg = self._config

        @tool
        async def search_documents(query: str, num_results: int = 4) -> str:
            """Search documents for relevant passages.

            Args:
                query: Search query.
                num_results: Number of passages (default: 4).

            Returns:
                Formatted passages with citations [1], [2], etc.
            """
            k = min(num_results, 10)
            passages = await cap.search(query, k=k)

            if not passages:
                return "No relevant passages found."

            # Format output based on config (minimize what LLM sees)
            formatted = []
            for p in passages:
                # Build header with only configured fields
                header_parts = [f"[{p.citation_id}]"]
                if cfg.include_source:
                    header_parts.append(f"Source: {p.source}")
                if cfg.include_page and p.page is not None:
                    header_parts.append(f"Page: {p.page}")
                if cfg.include_score:
                    header_parts.append(f"Score: {p.score:.3f}")
                
                # Optionally truncate long passages
                text = p.text
                if cfg.max_chars and len(text) > cfg.max_chars:
                    text = text[:cfg.max_chars] + "..."
                
                formatted.append(f"{' | '.join(header_parts)}\n{text}")

            return "\n\n---\n\n".join(formatted)

        return [search_documents]


__all__ = [
    "RAG",
    "RAGConfig",
    "CitedPassage",
    "CitationStyle",
]
