"""Hybrid retriever combining dense and sparse retrieval.

Hybrid retrieval combines the strengths of:
- Dense (semantic): Good for meaning and paraphrases
- Sparse (BM25): Good for exact terms and keywords

The combination often outperforms either approach alone.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from agenticflow.retriever.base import BaseRetriever, FusionStrategy, RetrievalResult
from agenticflow.retriever.dense import DenseRetriever
from agenticflow.retriever.sparse import BM25Retriever
from agenticflow.retriever.utils.fusion import fuse_results
from agenticflow.vectorstore import Document

if TYPE_CHECKING:
    from agenticflow.vectorstore import VectorStore


class HybridRetriever(BaseRetriever):
    """Hybrid retriever combining dense and sparse search.
    
    Uses both vector similarity (dense) and BM25 (sparse) to find
    documents, then fuses results using a configurable strategy.
    
    This is often the best approach for RAG applications as it
    combines semantic understanding with keyword matching.
    
    Example:
        >>> from agenticflow.vectorstore import VectorStore
        >>> from agenticflow.retriever import HybridRetriever
        >>> 
        >>> store = VectorStore()
        >>> await store.add_texts(["Python is great", "JavaScript rocks"])
        >>> 
        >>> retriever = HybridRetriever(vectorstore=store)
        >>> docs = await retriever.retrieve("Python programming language")
    """
    
    _name: str = "hybrid"
    
    def __init__(
        self,
        vectorstore: VectorStore | None = None,
        *,
        dense_retriever: DenseRetriever | None = None,
        sparse_retriever: BM25Retriever | None = None,
        dense_weight: float = 0.7,
        sparse_weight: float = 0.3,
        fusion: FusionStrategy | str = FusionStrategy.RRF,
        name: str | None = None,
    ) -> None:
        """Create a hybrid retriever.
        
        Args:
            vectorstore: VectorStore for dense retrieval (creates DenseRetriever).
            dense_retriever: Existing dense retriever (alternative to vectorstore).
            sparse_retriever: Existing sparse retriever (or creates new one).
            dense_weight: Weight for dense results (default: 0.7).
            sparse_weight: Weight for sparse results (default: 0.3).
            fusion: Fusion strategy ("rrf", "linear", "max", "voting").
            name: Optional custom name.
        """
        if vectorstore is None and dense_retriever is None:
            raise ValueError("Either vectorstore or dense_retriever must be provided")
        
        self._dense = dense_retriever or DenseRetriever(vectorstore)  # type: ignore
        self._sparse = sparse_retriever or BM25Retriever()
        self._dense_weight = dense_weight
        self._sparse_weight = sparse_weight
        
        if isinstance(fusion, str):
            fusion = FusionStrategy(fusion)
        self._fusion = fusion
        
        if name:
            self._name = name
        
        # Track if we need to sync sparse index
        self._vectorstore = vectorstore
        self._sparse_synced = sparse_retriever is not None
    
    async def add_documents(self, documents: list[Document]) -> list[str]:
        """Add documents to both dense and sparse indexes.
        
        Args:
            documents: Documents to add.
            
        Returns:
            List of document IDs.
        """
        # Add to dense (vectorstore)
        ids = await self._dense.add_documents(documents)
        
        # Add to sparse (BM25)
        self._sparse.add_documents(documents)
        
        return ids
    
    async def add_texts(
        self,
        texts: list[str],
        metadatas: list[dict[str, Any]] | None = None,
    ) -> list[str]:
        """Add texts to both indexes.
        
        Args:
            texts: Texts to add.
            metadatas: Optional metadata for each text.
            
        Returns:
            List of document IDs.
        """
        # Create documents
        metadatas = metadatas or [{}] * len(texts)
        documents = [
            Document(text=text, metadata=meta)
            for text, meta in zip(texts, metadatas)
        ]
        
        return await self.add_documents(documents)
    
    async def _ensure_sparse_synced(self) -> None:
        """Ensure sparse index has all documents from vectorstore."""
        if self._sparse_synced:
            return
        
        # This is a basic sync - in production you'd want better tracking
        # For now, we just mark as synced and rely on user calling add_documents
        self._sparse_synced = True
    
    async def retrieve_with_scores(
        self,
        query: str,
        k: int = 4,
        filter: dict[str, Any] | None = None,
    ) -> list[RetrievalResult]:
        """Retrieve using hybrid dense + sparse search.
        
        Args:
            query: The search query.
            k: Number of documents to retrieve.
            filter: Optional metadata filter.
            
        Returns:
            Fused results from both retrievers.
        """
        await self._ensure_sparse_synced()
        
        # Retrieve more from each to have enough after fusion
        fetch_k = k * 2
        
        # Get results from both retrievers
        dense_results = await self._dense.retrieve_with_scores(
            query, k=fetch_k, filter=filter
        )
        sparse_results = await self._sparse.retrieve_with_scores(
            query, k=fetch_k, filter=filter
        )
        
        # Fuse results
        weights = [self._dense_weight, self._sparse_weight]
        fused = fuse_results(
            [dense_results, sparse_results],
            strategy=self._fusion,
            weights=weights,
            k=k,
        )
        
        # Update retriever name
        for result in fused:
            result.retriever_name = self.name
            result.metadata["dense_weight"] = self._dense_weight
            result.metadata["sparse_weight"] = self._sparse_weight
        
        return fused
    
    @property
    def dense_retriever(self) -> DenseRetriever:
        """Access the underlying dense retriever."""
        return self._dense
    
    @property
    def sparse_retriever(self) -> BM25Retriever:
        """Access the underlying sparse retriever."""
        return self._sparse


def create_hybrid_retriever(
    texts: list[str] | None = None,
    documents: list[Document] | None = None,
    metadatas: list[dict[str, Any]] | None = None,
    dense_weight: float = 0.7,
    sparse_weight: float = 0.3,
    fusion: str = "rrf",
    **vectorstore_kwargs: Any,
) -> HybridRetriever:
    """Create a hybrid retriever with optional initial documents.
    
    Convenience function that creates both vectorstore and retriever.
    
    Args:
        texts: Initial texts to add.
        documents: Initial documents to add (alternative to texts).
        metadatas: Metadata for texts.
        dense_weight: Weight for dense retrieval.
        sparse_weight: Weight for sparse retrieval.
        fusion: Fusion strategy.
        **vectorstore_kwargs: Arguments for VectorStore.
        
    Returns:
        Configured HybridRetriever.
        
    Example:
        >>> retriever = create_hybrid_retriever(
        ...     texts=["doc1", "doc2", "doc3"],
        ...     dense_weight=0.6,
        ...     sparse_weight=0.4,
        ... )
        >>> docs = await retriever.retrieve("query")
    """
    from agenticflow.vectorstore import VectorStore
    
    store = VectorStore(**vectorstore_kwargs)
    retriever = HybridRetriever(
        vectorstore=store,
        dense_weight=dense_weight,
        sparse_weight=sparse_weight,
        fusion=FusionStrategy(fusion),
    )
    
    # Add initial documents if provided
    if documents:
        import asyncio
        asyncio.get_event_loop().run_until_complete(
            retriever.add_documents(documents)
        )
    elif texts:
        import asyncio
        asyncio.get_event_loop().run_until_complete(
            retriever.add_texts(texts, metadatas=metadatas)
        )
    
    return retriever
