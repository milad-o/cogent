"""Ensemble retriever combining multiple retrievers.

Ensemble retrieval allows combining results from any number of retrievers
using various fusion strategies.
"""

from __future__ import annotations

from typing import Any

from agenticflow.retriever.base import BaseRetriever, FusionStrategy, Retriever, RetrievalResult
from agenticflow.retriever.utils.fusion import fuse_results, normalize_scores
from agenticflow.vectorstore import Document


class EnsembleRetriever(BaseRetriever):
    """Ensemble retriever combining multiple retrievers.
    
    Takes results from multiple retrievers and fuses them using
    a configurable strategy (RRF, linear, max, voting).
    
    This is useful when you want to:
    - Combine different embedding models
    - Use multiple vector stores
    - Mix dense and sparse retrievers
    - Ensemble diverse retrieval strategies
    
    Example:
        >>> from agenticflow.retriever import EnsembleRetriever, DenseRetriever, BM25Retriever
        >>> from agenticflow.vectorstore import VectorStore
        >>> 
        >>> # 1. Create and populate retrievers
        >>> store = VectorStore()
        >>> await store.add_documents(documents)
        >>> dense = DenseRetriever(store)
        >>> 
        >>> bm25 = BM25Retriever(documents)  # Pass docs directly
        >>> 
        >>> # 2. Combine into ensemble
        >>> ensemble = EnsembleRetriever(
        ...     retrievers=[dense, bm25],
        ...     weights=[0.6, 0.4],
        ...     fusion="rrf",
        ... )
        >>> docs = await ensemble.retrieve("query")
    """
    
    _name: str = "ensemble"
    
    def __init__(
        self,
        retrievers: list[Retriever],
        *,
        weights: list[float] | None = None,
        fusion: FusionStrategy | str = FusionStrategy.RRF,
        normalize: bool = True,
        name: str | None = None,
    ) -> None:
        """Create an ensemble retriever.
        
        Args:
            retrievers: List of retrievers to combine.
            weights: Optional weights for each retriever (default: equal).
            fusion: Fusion strategy ("rrf", "linear", "max", "voting").
            normalize: Normalize scores before linear fusion.
            name: Optional custom name.
        """
        if not retrievers:
            raise ValueError("At least one retriever is required")
        
        self._retrievers = retrievers
        self._weights = weights or [1.0] * len(retrievers)
        
        if len(self._weights) != len(retrievers):
            raise ValueError("Number of weights must match number of retrievers")
        
        if isinstance(fusion, str):
            fusion = FusionStrategy(fusion)
        self._fusion = fusion
        self._normalize = normalize
        
        if name:
            self._name = name
    
    async def retrieve_with_scores(
        self,
        query: str,
        k: int = 4,
        filter: dict[str, Any] | None = None,
    ) -> list[RetrievalResult]:
        """Retrieve from all retrievers and fuse results.
        
        Args:
            query: The search query.
            k: Number of documents to retrieve.
            filter: Optional metadata filter.
            
        Returns:
            Fused results from all retrievers.
        """
        import asyncio
        
        # Fetch more from each to ensure enough results after fusion
        fetch_k = k * 2
        
        # Retrieve from all retrievers in parallel
        tasks = [
            retriever.retrieve_with_scores(query, k=fetch_k, filter=filter)
            for retriever in self._retrievers
        ]
        all_results = await asyncio.gather(*tasks)
        
        # Optionally normalize scores for linear fusion
        if self._normalize and self._fusion == FusionStrategy.LINEAR:
            all_results = [normalize_scores(results) for results in all_results]
        
        # Fuse results
        fused = fuse_results(
            all_results,
            strategy=self._fusion,
            weights=self._weights,
            k=k,
        )
        
        # Update retriever name
        for result in fused:
            result.retriever_name = self.name
        
        return fused
    
    def add_retriever(
        self,
        retriever: Retriever,
        weight: float = 1.0,
    ) -> None:
        """Add a retriever to the ensemble.
        
        Args:
            retriever: Retriever to add.
            weight: Weight for this retriever.
        """
        self._retrievers.append(retriever)
        self._weights.append(weight)
    
    def remove_retriever(self, index: int) -> None:
        """Remove a retriever by index.
        
        Args:
            index: Index of retriever to remove.
        """
        del self._retrievers[index]
        del self._weights[index]
    
    @property
    def retrievers(self) -> list[Retriever]:
        """Access the list of retrievers."""
        return self._retrievers
    
    @property
    def weights(self) -> list[float]:
        """Access the weights."""
        return self._weights
    
    def set_weights(self, weights: list[float]) -> None:
        """Update retriever weights.
        
        Args:
            weights: New weights (must match number of retrievers).
        """
        if len(weights) != len(self._retrievers):
            raise ValueError("Number of weights must match number of retrievers")
        self._weights = weights

    async def add_documents(self, documents: list[Document]) -> None:
        """Add documents to all child retrievers that support it.
        
        This is a convenience method for adding the same documents to all
        retrievers at once. For different documents per retriever, add
        directly to each retriever before creating the ensemble.
        
        Args:
            documents: Documents to add to all child retrievers.
            
        Example:
            ```python
            # Same docs for all retrievers (common case):
            ensemble = EnsembleRetriever([dense, bm25])
            await ensemble.add_documents(docs)
            
            # Different docs per retriever:
            await dense_retriever.add_documents(technical_docs)
            bm25_retriever.add_documents(legal_docs)
            ensemble = EnsembleRetriever([dense_retriever, bm25_retriever])
            
            # Mix: shared base + retriever-specific additions
            await ensemble.add_documents(shared_docs)
            await dense_retriever.add_documents(extra_embeddings_only)
            ```
        """
        import asyncio
        
        tasks = []
        for retriever in self._retrievers:
            if hasattr(retriever, "add_documents"):
                add_fn = getattr(retriever, "add_documents")
                if asyncio.iscoroutinefunction(add_fn):
                    tasks.append(add_fn(documents))
                else:
                    add_fn(documents)
        
        if tasks:
            await asyncio.gather(*tasks)


class WeightedRetriever(BaseRetriever):
    """A single retriever with a configurable weight.
    
    Wrapper that applies a weight multiplier to scores.
    Useful when building custom ensembles.
    """
    
    def __init__(
        self,
        retriever: Retriever,
        weight: float = 1.0,
    ) -> None:
        """Create a weighted retriever.
        
        Args:
            retriever: Base retriever.
            weight: Score multiplier.
        """
        self._retriever = retriever
        self._weight = weight
        self._name = f"weighted_{retriever.name}"
    
    async def retrieve_with_scores(
        self,
        query: str,
        k: int = 4,
        filter: dict[str, Any] | None = None,
    ) -> list[RetrievalResult]:
        """Retrieve with weighted scores."""
        results = await self._retriever.retrieve_with_scores(query, k=k, filter=filter)
        
        return [
            RetrievalResult(
                document=r.document,
                score=r.score * self._weight,
                retriever_name=self.name,
                metadata={**r.metadata, "original_score": r.score, "weight": self._weight},
            )
            for r in results
        ]
