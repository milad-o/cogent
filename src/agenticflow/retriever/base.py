"""Base classes and protocols for retrievers.

Defines the core abstractions for the retriever module:
- Retriever: Protocol for all retrievers
- RetrievalResult: Result from retrieval with document and score
- FusionStrategy: Enum for combining results from multiple retrievers
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

if TYPE_CHECKING:
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
    
    All retrievers must implement:
    - retrieve: Get documents matching a query
    - retrieve_with_scores: Get documents with relevance scores
    
    Example:
        >>> retriever = DenseRetriever(vectorstore)
        >>> docs = await retriever.retrieve("What is Python?", k=5)
        >>> results = await retriever.retrieve_with_scores("Python", k=5)
        >>> for result in results:
        ...     print(f"{result.score:.3f}: {result.document.text[:50]}")
    """
    
    @property
    def name(self) -> str:
        """Name of this retriever for identification."""
        ...
    
    async def retrieve(
        self,
        query: str,
        k: int = 4,
        filter: dict[str, Any] | None = None,
    ) -> list[Document]:
        """Retrieve documents matching the query.
        
        Args:
            query: The search query.
            k: Number of documents to retrieve.
            filter: Optional metadata filter.
            
        Returns:
            List of matching documents, ordered by relevance.
        """
        ...
    
    async def retrieve_with_scores(
        self,
        query: str,
        k: int = 4,
        filter: dict[str, Any] | None = None,
    ) -> list[RetrievalResult]:
        """Retrieve documents with relevance scores.
        
        Args:
            query: The search query.
            k: Number of documents to retrieve.
            filter: Optional metadata filter.
            
        Returns:
            List of RetrievalResult with document and score.
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
    Subclasses should override retrieve_with_scores.
    """
    
    _name: str = "base"
    
    @property
    def name(self) -> str:
        """Name of this retriever."""
        return self._name
    
    async def retrieve(
        self,
        query: str,
        k: int = 4,
        filter: dict[str, Any] | None = None,
    ) -> list[Document]:
        """Retrieve documents (convenience method).
        
        Calls retrieve_with_scores and extracts just the documents.
        """
        results = await self.retrieve_with_scores(query, k=k, filter=filter)
        return [r.document for r in results]
    
    async def retrieve_with_scores(
        self,
        query: str,
        k: int = 4,
        filter: dict[str, Any] | None = None,
    ) -> list[RetrievalResult]:
        """Retrieve documents with scores. Must be implemented by subclasses."""
        raise NotImplementedError("Subclasses must implement retrieve_with_scores")
    
    async def abatch_retrieve(
        self,
        queries: list[str],
        k: int = 4,
        filter: dict[str, Any] | None = None,
    ) -> list[list[Document]]:
        """Batch retrieve for multiple queries.
        
        Default implementation calls retrieve for each query.
        Subclasses can override for more efficient batch processing.
        """
        import asyncio
        tasks = [self.retrieve(q, k=k, filter=filter) for q in queries]
        return await asyncio.gather(*tasks)
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name!r})"
