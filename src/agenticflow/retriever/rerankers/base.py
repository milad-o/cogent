"""Base classes and protocols for rerankers.

Rerankers take an initial set of retrieved documents and re-score them
using a more expensive but accurate model.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

from agenticflow.retriever.base import RetrievalResult

if TYPE_CHECKING:
    from agenticflow.vectorstore import Document


@runtime_checkable
class Reranker(Protocol):
    """Protocol for rerankers.
    
    Rerankers re-score documents for better relevance ranking.
    They typically use cross-encoders or other models that
    jointly encode query and document.
    """
    
    @property
    def name(self) -> str:
        """Name of this reranker."""
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
            Reranked results with new scores, sorted by relevance.
        """
        ...


class BaseReranker:
    """Base class for rerankers with common functionality."""
    
    _name: str = "base_reranker"
    
    @property
    def name(self) -> str:
        """Name of this reranker."""
        return self._name
    
    async def rerank(
        self,
        query: str,
        documents: list[Document],
        top_n: int | None = None,
    ) -> list[RetrievalResult]:
        """Rerank documents. Must be implemented by subclasses."""
        raise NotImplementedError("Subclasses must implement rerank")
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name!r})"
