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

from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any

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

    Attributes:
        citation_id: Citation reference number (1, 2, 3...).
        source: Source document name.
        page: Page number if available.
        chunk_index: Chunk index within document.
        score: Relevance score (0.0-1.0).
        text: The passage text.
    """

    citation_id: int
    source: str
    page: int | None = None
    chunk_index: int | None = None
    score: float = 0.0
    text: str = ""

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

    def format_full(self, include_score: bool = True) -> str:
        """Format as full bibliography entry.

        Args:
            include_score: Whether to include relevance score.

        Returns:
            Full citation with source, page, and optionally score.
        """
        parts = [f"[{self.citation_id}]", self.source]
        if self.page is not None:
            parts.append(f"p.{self.page}")
        if include_score:
            parts.append(f"(score: {self.score:.2f})")
        return " ".join(parts)


@dataclass(frozen=True, slots=True, kw_only=True)
class RAGConfig:
    """Configuration for RAG capability.

    Attributes:
        top_k: Default number of results to retrieve.
        citation_style: How to format citations.
        include_page_in_citation: Include page numbers in citations.
        include_score_in_bibliography: Include scores in bibliography.
    """

    top_k: int = 4
    citation_style: CitationStyle = CitationStyle.NUMERIC
    include_page_in_citation: bool = True
    include_score_in_bibliography: bool = True


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
            passage = CitedPassage(
                citation_id=i,
                source=doc.metadata.get("source", "unknown"),
                page=doc.metadata.get("page"),
                chunk_index=doc.metadata.get("chunk_index"),
                score=result.score,
                text=doc.text,
            )
            passages.append(passage)

        self._last_citations = passages
        return passages

    def format_bibliography(
        self,
        passages: list[CitedPassage] | None = None,
        title: str = "References",
    ) -> str:
        """Format passages as bibliography.

        Args:
            passages: Passages to format (default: last search results).
            title: Bibliography section title.

        Returns:
            Formatted bibliography string.
        """
        passages = passages or self._last_citations

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
                    include_score=self._config.include_score_in_bibliography,
                )
            )

        return "\n".join(lines)

    # ================================================================
    # Tools
    # ================================================================

    def _create_tools(self) -> list:
        """Create RAG tools for the agent."""
        cap = self

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

            formatted = []
            for p in passages:
                header_parts = [f"[{p.citation_id}]", f"Source: {p.source}"]
                if p.page is not None:
                    header_parts.append(f"Page: {p.page}")
                header_parts.append(f"Score: {p.score:.3f}")
                formatted.append(f"{' | '.join(header_parts)}\n{p.text}")

            return "\n\n---\n\n".join(formatted)

        return [search_documents]


__all__ = [
    "RAG",
    "RAGConfig",
    "CitedPassage",
    "CitationStyle",
]
