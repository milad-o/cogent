"""Base classes and protocols for retrievers.

Defines the core abstractions for the retriever module:
- Retriever: Protocol for all retrievers
- RetrievalResult: Result from retrieval with document and score
- FusionStrategy: Enum for combining results from multiple retrievers
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any, Literal, Protocol, overload, runtime_checkable

from agenticflow.tools.base import BaseTool

if TYPE_CHECKING:
    from agenticflow.observability.bus import EventBus
    from agenticflow.vectorstore import Document


class FusionStrategy(Enum):
    """Strategies for combining results from multiple retrievers.
    
    RRF: Reciprocal Rank Fusion - robust, parameter-free fusion based on ranks
    LINEAR: Weighted linear combination of normalized scores
    MAX: Take maximum score per document across retrievers
    VOTING: Count appearances across retrievers (good for diverse sources)
    """
    
    RRF = "rrf"
    LINEAR = "linear"
    MAX = "max"
    VOTING = "voting"


@dataclass
class RetrievalResult:
    """Result from a retrieval operation.
    
    Attributes:
        document: The retrieved document.
        score: Relevance score (higher is better, normalized 0-1 when possible).
        retriever_name: Name of the retriever that produced this result.
        metadata: Additional retrieval metadata (e.g., rank, raw_score).
    """
    
    document: Document
    score: float
    retriever_name: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self) -> None:
        """Ensure document is properly typed."""
        # Import here to avoid circular imports
        from agenticflow.vectorstore import Document
        if not isinstance(self.document, Document):
            raise TypeError(f"document must be a Document, got {type(self.document)}")


@runtime_checkable
class Retriever(Protocol):
    """Protocol for all retrievers.
    
    Retrievers are responsible for finding relevant documents given a query.
    They can use various strategies: dense (vector), sparse (BM25), or hybrid.
    
    The unified `retrieve()` method supports both document-only and scored results:
    - `retrieve(query)` → list of Documents
    - `retrieve(query, include_scores=True)` → list of RetrievalResult
    
    Example:
        >>> retriever = DenseRetriever(vectorstore)
        >>> 
        >>> # Get documents only
        >>> docs = await retriever.retrieve("What is Python?", k=5)
        >>> 
        >>> # Get documents with scores
        >>> results = await retriever.retrieve("Python", k=5, include_scores=True)
        >>> for result in results:
        ...     print(f"{result.score:.3f}: {result.document.text[:50]}")
    """
    
    @property
    def name(self) -> str:
        """Name of this retriever for identification."""
        ...
    
    @overload
    async def retrieve(
        self,
        query: str,
        k: int = 4,
        filter: dict[str, Any] | None = None,
        *,
        include_scores: Literal[False] = False,
        **kwargs: Any,
    ) -> list[Document]: ...
    
    @overload
    async def retrieve(
        self,
        query: str,
        k: int = 4,
        filter: dict[str, Any] | None = None,
        *,
        include_scores: Literal[True],
        **kwargs: Any,
    ) -> list[RetrievalResult]: ...
    
    async def retrieve(
        self,
        query: str,
        k: int = 4,
        filter: dict[str, Any] | None = None,
        *,
        include_scores: bool = False,
        **kwargs: Any,
    ) -> list[Document] | list[RetrievalResult]:
        """Retrieve documents matching the query.
        
        Args:
            query: The search query.
            k: Number of documents to retrieve.
            filter: Optional metadata filter.
            include_scores: If True, return RetrievalResult with scores.
                           If False (default), return just Documents.
            **kwargs: Additional retriever-specific arguments.
            
        Returns:
            List of Documents or RetrievalResults, ordered by relevance.
        """
        ...
    
    # Keep for backward compatibility
    async def retrieve_with_scores(
        self,
        query: str,
        k: int = 4,
        filter: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> list[RetrievalResult]:
        """Retrieve documents with relevance scores.
        
        Deprecated: Use `retrieve(query, include_scores=True)` instead.
        """
        ...


@runtime_checkable
class Reranker(Protocol):
    """Protocol for rerankers.
    
    Rerankers take an initial set of retrieved documents and re-score them
    using a more expensive but accurate model (e.g., cross-encoder).
    
    Example:
        >>> reranker = CrossEncoderReranker(model="cross-encoder/ms-marco-MiniLM-L-6-v2")
        >>> reranked = await reranker.rerank(query, documents, top_n=5)
    """
    
    @property
    def name(self) -> str:
        """Name of this reranker for identification."""
        ...
    
    async def rerank(
        self,
        query: str,
        documents: list[Document],
        top_n: int | None = None,
    ) -> list[RetrievalResult]:
        """Rerank documents for the given query.
        
        Args:
            query: The search query.
            documents: Documents to rerank.
            top_n: Number of top documents to return (None = all).
            
        Returns:
            Reranked results with new scores.
        """
        ...


class BaseRetriever:
    """Base class for retrievers with common functionality.
    
    Provides default implementations and utility methods.
    Subclasses should override `retrieve_with_scores` for the core logic.
    
    The unified `retrieve()` API:
    - `retrieve(query)` → list of Documents
    - `retrieve(query, include_scores=True)` → list of RetrievalResult
    
    Observability:
        Set `event_bus` to emit RETRIEVAL_START and RETRIEVAL_COMPLETE events.
    """
    
    _name: str = "base"
    _event_bus: EventBus | None = None
    
    @property
    def name(self) -> str:
        """Name of this retriever."""
        return self._name
    
    @property
    def event_bus(self) -> EventBus | None:
        """Get the EventBus (if any)."""
        return self._event_bus
    
    @event_bus.setter
    def event_bus(self, value: EventBus | None) -> None:
        """Set the EventBus for observability."""
        self._event_bus = value
    
    async def _emit(self, event_type: str, data: dict[str, Any]) -> None:
        """Emit an event if event_bus is configured."""
        if self._event_bus:
            from agenticflow.observability.event import EventType
            await self._event_bus.publish(EventType(event_type), {
                "retriever": self.name,
                **data,
            })
    
    @overload
    async def retrieve(
        self,
        query: str,
        k: int = 4,
        filter: dict[str, Any] | None = None,
        *,
        include_scores: Literal[False] = False,
        **kwargs: Any,
    ) -> list[Document]: ...
    
    @overload
    async def retrieve(
        self,
        query: str,
        k: int = 4,
        filter: dict[str, Any] | None = None,
        *,
        include_scores: Literal[True],
        **kwargs: Any,
    ) -> list[RetrievalResult]: ...
    
    async def retrieve(
        self,
        query: str,
        k: int = 4,
        filter: dict[str, Any] | None = None,
        *,
        include_scores: bool = False,
        **kwargs: Any,
    ) -> list[Document] | list[RetrievalResult]:
        """Retrieve documents matching the query.
        
        Args:
            query: The search query.
            k: Number of documents to retrieve.
            filter: Optional metadata filter.
            include_scores: If True, return RetrievalResult with scores.
                           If False (default), return just Documents.
            **kwargs: Additional retriever-specific arguments.
            
        Returns:
            List of Documents or RetrievalResults, ordered by relevance.
        """
        start = time.perf_counter()
        
        await self._emit("retrieval.start", {
            "query": query[:100],
            "k": k,
            "filter": filter,
        })
        
        try:
            results = await self.retrieve_with_scores(query, k=k, filter=filter, **kwargs)
            
            duration_ms = (time.perf_counter() - start) * 1000
            top_scores = [r.score for r in results[:3]] if results else []
            
            await self._emit("retrieval.complete", {
                "query": query[:100],
                "k": k,
                "results_count": len(results),
                "top_scores": top_scores,
                "duration_ms": duration_ms,
            })
            
            if include_scores:
                return results
            return [r.document for r in results]
            
        except Exception as e:
            await self._emit("retrieval.error", {
                "query": query[:100],
                "error": str(e),
                "duration_ms": (time.perf_counter() - start) * 1000,
            })
            raise
    
    async def retrieve_with_scores(
        self,
        query: str,
        k: int = 4,
        filter: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> list[RetrievalResult]:
        """Retrieve documents with scores.
        
        This is the method subclasses should override.
        For external use, prefer `retrieve(query, include_scores=True)`.
        
        Args:
            query: The search query.
            k: Number of documents to retrieve.
            filter: Optional metadata filter.
            **kwargs: Additional retriever-specific arguments.
        """
        raise NotImplementedError("Subclasses must implement retrieve_with_scores")
    
    async def abatch_retrieve(
        self,
        queries: list[str],
        k: int = 4,
        filter: dict[str, Any] | None = None,
        *,
        include_scores: bool = False,
    ) -> list[list[Document]] | list[list[RetrievalResult]]:
        """Batch retrieve for multiple queries.
        
        Default implementation calls retrieve for each query.
        Subclasses can override for more efficient batch processing.
        """
        import asyncio
        tasks = [self.retrieve(q, k=k, filter=filter, include_scores=include_scores) for q in queries]
        return await asyncio.gather(*tasks)
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name!r})"

    def as_tool(
        self,
        *,
        name: str | None = None,
        description: str | None = None,
        k_default: int = 4,
        include_scores: bool = False,
        include_metadata: bool = True,
    ) -> BaseTool:
        """Expose this retriever as a native tool for agents.

        The generated tool wraps ``retrieve`` and returns structured results
        ready for tool-calling models. Scores and metadata can be toggled.

        Args:
            name: Optional tool name. Defaults to ``{retriever_name}_retrieve``.
            description: Optional tool description. Defaults to a generic description.
            k_default: Default number of results to return when ``k`` is not provided.
            include_scores: Include retrieval scores in the tool output.
            include_metadata: Include document metadata in the tool output.

        Returns:
            BaseTool configured for this retriever.
        """

        tool_name = name or f"{self.name}_retrieve"
        tool_description = description or (
            f"Retrieve relevant documents using the {self.name} retriever."
        )

        args_schema = {
            "query": {
                "type": "string",
                "description": "Natural language search query.",
            },
            "k": {
                "type": "integer",
                "description": "Number of results to return.",
                "default": k_default,
                "minimum": 1,
            },
            "filter": {
                "type": "object",
                "description": "Optional metadata filter (field -> value).",
                "additionalProperties": True,
            },
        }

        async def _tool(
            query: str,
            k: int = k_default,
            filter: dict[str, Any] | None = None,
        ) -> list[dict[str, Any]]:
            results = await self.retrieve(
                query,
                k=k,
                filter=filter,
                include_scores=True,
            )

            payload: list[dict[str, Any]] = []
            for r in results:
                entry: dict[str, Any] = {"text": r.document.text}
                if include_metadata:
                    entry["metadata"] = r.document.metadata
                if include_scores:
                    entry["score"] = r.score
                payload.append(entry)

            return payload

        return BaseTool(
            name=tool_name,
            description=tool_description,
            func=_tool,
            args_schema=args_schema,
        )
