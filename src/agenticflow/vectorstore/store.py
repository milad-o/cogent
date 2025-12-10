"""High-level VectorStore class.

Provides a unified interface over different backends and embedding providers.
Simple by default, powerful when configured.

Example:
    >>> from agenticflow.vectorstore import VectorStore
    >>> store = VectorStore()
    >>> await store.add_texts(["Python is great", "JavaScript is popular"])
    >>> results = await store.search("programming language")
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Literal

from agenticflow.vectorstore.backends.inmemory import InMemoryBackend, SimilarityMetric
from agenticflow.vectorstore.base import (
    EmbeddingProvider,
    SearchResult,
    VectorStoreBackend,
)
from agenticflow.vectorstore.document import Document, create_documents
from agenticflow.vectorstore.embeddings import MockEmbeddings
from agenticflow.models.openai import OpenAIEmbedding


logger = logging.getLogger(__name__)
BackendType = Literal["inmemory", "faiss", "chroma"]


async def _emit_event(event_type: str, data: dict[str, Any]) -> None:
    """Emit an event to the global event bus if available."""
    try:
        from agenticflow.observability.bus import get_event_bus
        from agenticflow.observability.event import Event, EventType
        
        bus = get_event_bus()
        event = Event(
            type=EventType(event_type),
            data=data,
            source="vectorstore",
        )
        await bus.publish(event)
    except (ImportError, ValueError, Exception) as e:
        logger.debug("Failed to emit event %s: %s", event_type, e)


@dataclass
class VectorStore:
    """High-level vector store with embedding generation.
    
    Combines a storage backend with an embedding provider for a complete
    semantic search solution.
    
    Attributes:
        embeddings: Embedding provider (default: OpenAIEmbeddings).
        backend: Storage backend (default: InMemoryBackend).
        metric: Similarity metric for default InMemory backend.
        
    Example:
        >>> store = VectorStore()  # Uses OpenAI + InMemory + Cosine
        >>> await store.add_texts(["Hello", "World"])
        >>> results = await store.search("greeting")
        
        >>> # With Euclidean distance
        >>> store = VectorStore(metric="euclidean")
        
        >>> # With custom embeddings
        >>> from agenticflow.vectorstore import OllamaEmbeddings
        >>> store = VectorStore(embeddings=OllamaEmbeddings())
    """
    
    embeddings: EmbeddingProvider | None = None
    backend: VectorStoreBackend | None = None
    metric: SimilarityMetric | str = SimilarityMetric.COSINE
    _initialized: bool = field(default=False, init=False)
    
    def __post_init__(self) -> None:
        """Initialize default embeddings and backend if not provided."""
        if self.embeddings is None:
            self.embeddings = OpenAIEmbedding()
        
        if self.backend is None:
            # Use metric parameter for default InMemory backend
            self.backend = InMemoryBackend(metric=self.metric)
        
        self._initialized = True
    
    async def add_texts(
        self,
        texts: list[str],
        metadatas: list[dict[str, Any]] | None = None,
        ids: list[str] | None = None,
    ) -> list[str]:
        """Add texts to the vector store.
        
        Automatically generates embeddings and stores documents.
        
        Args:
            texts: List of text contents to add.
            metadatas: Optional metadata for each text.
            ids: Optional IDs for each text (auto-generated if not provided).
            
        Returns:
            List of document IDs.
            
        Example:
            >>> ids = await store.add_texts(
            ...     texts=["Python is great", "JavaScript rocks"],
            ...     metadatas=[{"lang": "python"}, {"lang": "js"}],
            ... )
        """
        # Create documents
        documents = create_documents(texts, metadatas, ids)
        doc_ids = [doc.id for doc in documents]
        
        # Generate embeddings
        embeddings = await self.embeddings.aembed_texts(texts)  # type: ignore
        
        # Store in backend
        await self.backend.add(doc_ids, embeddings, documents)  # type: ignore
        
        # Emit event
        await _emit_event("vectorstore.add", {
            "count": len(doc_ids),
            "ids": doc_ids[:5],  # First 5 IDs for brevity
        })
        
        return doc_ids
    
    async def add_documents(
        self,
        documents: list[Document],
    ) -> list[str]:
        """Add Document objects to the vector store.
        
        If documents have pre-computed embeddings, uses those.
        Otherwise, generates embeddings from text.
        
        Args:
            documents: List of Document objects.
            
        Returns:
            List of document IDs.
        """
        # Check which documents need embeddings
        docs_needing_embeddings: list[tuple[int, Document]] = []
        embeddings: list[list[float]] = [[] for _ in documents]
        
        for i, doc in enumerate(documents):
            if doc.embedding:
                embeddings[i] = doc.embedding
            else:
                docs_needing_embeddings.append((i, doc))
        
        # Generate missing embeddings
        if docs_needing_embeddings:
            texts = [doc.text for _, doc in docs_needing_embeddings]
            new_embeddings = await self.embeddings.aembed_texts(texts)  # type: ignore
            
            for (idx, _), embedding in zip(docs_needing_embeddings, new_embeddings):
                embeddings[idx] = embedding
        
        # Store
        doc_ids = [doc.id for doc in documents]
        await self.backend.add(doc_ids, embeddings, documents)  # type: ignore
        
        # Emit event
        await _emit_event("vectorstore.add", {
            "count": len(doc_ids),
            "ids": doc_ids[:5],  # First 5 IDs for brevity
            "embedded": len(docs_needing_embeddings),
        })
        
        return doc_ids
    
    async def search(
        self,
        query: str,
        k: int = 4,
        filter: dict[str, Any] | None = None,
    ) -> list[SearchResult]:
        """Search for similar documents.
        
        Args:
            query: Query text.
            k: Number of results to return.
            filter: Optional metadata filter (exact match).
            
        Returns:
            List of SearchResult objects sorted by similarity.
            
        Example:
            >>> results = await store.search("web development", k=5)
            >>> for r in results:
            ...     print(f"{r.score:.2f}: {r.document.text[:50]}")
        """
        # Generate query embedding
        query_embedding = await self.embeddings.aembed_query(query)  # type: ignore
        
        # Search backend
        results = await self.backend.search(query_embedding, k, filter)  # type: ignore
        
        # Emit event
        await _emit_event("vectorstore.search", {
            "query": query[:100],  # Truncate long queries
            "k": k,
            "results": len(results),
            "top_score": results[0].score if results else None,
        })
        
        return results
    
    async def similarity_search(
        self,
        query: str,
        k: int = 4,
        filter: dict[str, Any] | None = None,
    ) -> list[Document]:
        """Search and return just the documents (alias for compatibility).
        
        Args:
            query: Query text.
            k: Number of results to return.
            filter: Optional metadata filter.
            
        Returns:
            List of Document objects sorted by similarity.
        """
        results = await self.search(query, k, filter)
        return [r.document for r in results]
    
    async def delete(self, ids: list[str]) -> bool:
        """Delete documents by ID.
        
        Args:
            ids: List of document IDs to delete.
            
        Returns:
            True if any documents were deleted.
        """
        result = await self.backend.delete(ids)  # type: ignore
        
        # Emit event
        await _emit_event("vectorstore.delete", {
            "count": len(ids),
            "ids": ids[:5],  # First 5 IDs for brevity
            "success": result,
        })
        
        return result
    
    async def clear(self) -> None:
        """Remove all documents from the store."""
        count = self.count()
        await self.backend.clear()  # type: ignore
        
        # Emit event
        await _emit_event("vectorstore.delete", {
            "count": count,
            "clear": True,
            "success": True,
        })
    
    async def get(self, ids: list[str]) -> list[Document]:
        """Get documents by ID.
        
        Args:
            ids: List of document IDs to retrieve.
            
        Returns:
            List of Document objects.
        """
        return await self.backend.get(ids)  # type: ignore
    
    def count(self) -> int:
        """Return the number of documents in the store."""
        return self.backend.count()  # type: ignore
    
    @classmethod
    def from_texts(
        cls,
        texts: list[str],
        metadatas: list[dict[str, Any]] | None = None,
        embeddings: EmbeddingProvider | None = None,
        backend: BackendType = "inmemory",
    ) -> VectorStore:
        """Create a VectorStore from texts (sync factory for convenience).
        
        Note: This creates the store but doesn't add texts yet.
        Use `await store.add_texts(texts)` after creation.
        
        Args:
            texts: Texts to add (must call add_texts separately).
            metadatas: Metadata for texts.
            embeddings: Embedding provider.
            backend: Backend type.
            
        Returns:
            VectorStore instance (texts not yet added).
        """
        backend_instance: VectorStoreBackend
        if backend == "inmemory":
            backend_instance = InMemoryBackend()
        else:
            msg = f"Backend '{backend}' not yet implemented"
            raise ValueError(msg)
        
        return cls(embeddings=embeddings, backend=backend_instance)
    
    @classmethod
    def with_mock_embeddings(cls, dimension: int = 384) -> VectorStore:
        """Create a VectorStore with mock embeddings for testing.
        
        Args:
            dimension: Embedding dimension.
            
        Returns:
            VectorStore with MockEmbeddings backend.
        """
        return cls(embeddings=MockEmbeddings(dimensions=dimension))
    
    def as_retriever(
        self,
        *,
        name: str | None = None,
        score_threshold: float | None = None,
    ):
        """Convert this VectorStore to a DenseRetriever.
        
        Provides a convenient interface for using the vectorstore as a retriever
        in RAG pipelines or with other retrieval components.
        
        Args:
            name: Optional custom name for the retriever.
            score_threshold: Minimum score threshold (0-1). Results below
                this score are filtered out.
                
        Returns:
            DenseRetriever instance wrapping this vectorstore.
            
        Example:
            >>> store = VectorStore()
            >>> await store.add_texts(["Python is great", "JavaScript rocks"])
            >>> 
            >>> # Use as retriever
            >>> retriever = store.as_retriever()
            >>> docs = await retriever.retrieve("programming language", k=3)
            >>> 
            >>> # Or expose as agent tool
            >>> tool = store.as_retriever().as_tool(
            ...     name="search_kb",
            ...     description="Search knowledge base",
            ... )
        """
        from agenticflow.retriever.dense import DenseRetriever
        
        return DenseRetriever(
            vectorstore=self,
            name=name,
            score_threshold=score_threshold,
        )


# ============================================================
# Convenience Functions
# ============================================================

async def create_vectorstore(
    texts: list[str] | None = None,
    metadatas: list[dict[str, Any]] | None = None,
    embeddings: EmbeddingProvider | None = None,
    backend: BackendType = "inmemory",
) -> VectorStore:
    """Create and optionally populate a VectorStore.
    
    Args:
        texts: Optional texts to add immediately.
        metadatas: Optional metadata for texts.
        embeddings: Embedding provider.
        backend: Backend type.
        
    Returns:
        Initialized VectorStore.
        
    Example:
        >>> store = await create_vectorstore(
        ...     texts=["Hello", "World"],
        ...     metadatas=[{"type": "greeting"}, {"type": "noun"}],
        ... )
    """
    backend_instance: VectorStoreBackend
    if backend == "inmemory":
        backend_instance = InMemoryBackend()
    else:
        msg = f"Backend '{backend}' not yet implemented"
        raise ValueError(msg)
    
    store = VectorStore(embeddings=embeddings, backend=backend_instance)
    
    if texts:
        await store.add_texts(texts, metadatas)
    
    return store
