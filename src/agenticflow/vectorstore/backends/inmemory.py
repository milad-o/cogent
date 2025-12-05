"""In-memory vector store backend using NumPy.

A simple, zero-dependency backend suitable for small to medium datasets (<10k documents).
Supports multiple similarity metrics: cosine, euclidean, dot product.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from agenticflow.vectorstore.base import SearchResult
from agenticflow.vectorstore.document import Document


class SimilarityMetric(Enum):
    """Similarity metrics for vector search.
    
    COSINE: Cosine similarity (default) - measures angle between vectors.
            Best for: normalized embeddings, semantic similarity.
            Range: -1 to 1 (normalized to 0-1 in results).
            
    EUCLIDEAN: Euclidean distance converted to similarity.
               Best for: absolute distances matter, clustering.
               Range: 0 to 1 (closer = higher score).
               
    DOT_PRODUCT: Raw dot product (inner product).
                 Best for: pre-normalized embeddings, maximum inner product.
                 Range: unbounded (higher = more similar).
                 
    MANHATTAN: Manhattan (L1) distance converted to similarity.
               Best for: high-dimensional sparse vectors.
               Range: 0 to 1 (closer = higher score).
    """
    
    COSINE = "cosine"
    EUCLIDEAN = "euclidean"
    DOT_PRODUCT = "dot_product"
    MANHATTAN = "manhattan"


@dataclass
class StoredDocument:
    """Internal representation of a stored document.
    
    Attributes:
        id: Unique identifier.
        embedding: Vector embedding.
        document: The original document.
    """
    
    id: str
    embedding: list[float]
    document: Document


@dataclass
class InMemoryBackend:
    """In-memory vector store backend using NumPy-style operations.
    
    Uses pure Python with optional NumPy acceleration for similarity search.
    Good for datasets up to ~10k documents.
    
    Attributes:
        metric: Similarity metric to use (default: COSINE).
        normalize: Whether to normalize embeddings (auto-set based on metric).
    
    Example:
        >>> # Default: cosine similarity
        >>> backend = InMemoryBackend()
        >>> 
        >>> # Euclidean distance
        >>> backend = InMemoryBackend(metric=SimilarityMetric.EUCLIDEAN)
        >>> 
        >>> # Dot product (for pre-normalized embeddings)
        >>> backend = InMemoryBackend(metric=SimilarityMetric.DOT_PRODUCT)
        >>> 
        >>> await backend.add(["id1"], [[0.1, 0.2, 0.3]], [doc1])
        >>> results = await backend.search([0.1, 0.2, 0.3], k=5)
    """
    
    metric: SimilarityMetric | str = SimilarityMetric.COSINE
    normalize: bool | None = None  # Auto-set based on metric if None
    _storage: dict[str, StoredDocument] = field(default_factory=dict)
    _numpy_available: bool = field(default=False, init=False)
    
    def __post_init__(self) -> None:
        """Initialize metric and check for NumPy."""
        # Convert string to enum
        if isinstance(self.metric, str):
            self.metric = SimilarityMetric(self.metric)
        
        # Auto-set normalization based on metric
        if self.normalize is None:
            # Cosine and dot product benefit from normalized vectors
            self.normalize = self.metric in (
                SimilarityMetric.COSINE,
                SimilarityMetric.DOT_PRODUCT,
            )
        
        # Check NumPy availability
        try:
            import numpy as np  # noqa: F401
            self._numpy_available = True
        except ImportError:
            self._numpy_available = False
    
    async def add(
        self,
        ids: list[str],
        embeddings: list[list[float]],
        documents: list[Document],
    ) -> None:
        """Add documents with their embeddings.
        
        Args:
            ids: Unique identifiers for each document.
            embeddings: Embedding vectors for each document.
            documents: Document objects to store.
            
        Raises:
            ValueError: If lengths don't match.
        """
        if not (len(ids) == len(embeddings) == len(documents)):
            msg = f"Lengths must match: ids={len(ids)}, embeddings={len(embeddings)}, documents={len(documents)}"
            raise ValueError(msg)
        
        for doc_id, embedding, document in zip(ids, embeddings, documents):
            # Normalize if requested
            if self.normalize:
                embedding = self._normalize_vector(embedding)
            
            self._storage[doc_id] = StoredDocument(
                id=doc_id,
                embedding=embedding,
                document=document,
            )
    
    async def search(
        self,
        embedding: list[float],
        k: int = 4,
        filter: dict[str, Any] | None = None,
    ) -> list[SearchResult]:
        """Search for similar documents using configured similarity metric.
        
        Args:
            embedding: Query embedding vector.
            k: Number of results to return.
            filter: Optional metadata filter (exact match).
            
        Returns:
            List of SearchResult objects sorted by similarity (highest first).
        """
        if not self._storage:
            return []
        
        # Normalize query if needed
        if self.normalize:
            embedding = self._normalize_vector(embedding)
        
        # Calculate similarities
        if self._numpy_available:
            results = self._search_numpy(embedding, k, filter)
        else:
            results = self._search_pure_python(embedding, k, filter)
        
        return results
    
    def _compute_similarity(
        self,
        query: list[float],
        doc_embedding: list[float],
    ) -> float:
        """Compute similarity score based on configured metric.
        
        Args:
            query: Query embedding.
            doc_embedding: Document embedding.
            
        Returns:
            Similarity score (higher = more similar).
        """
        if self.metric == SimilarityMetric.COSINE:
            # Cosine similarity (dot product of normalized vectors)
            return self._dot_product(query, doc_embedding)
        
        elif self.metric == SimilarityMetric.DOT_PRODUCT:
            # Raw dot product
            return self._dot_product(query, doc_embedding)
        
        elif self.metric == SimilarityMetric.EUCLIDEAN:
            # Convert distance to similarity: 1 / (1 + distance)
            distance = self._euclidean_distance(query, doc_embedding)
            return 1.0 / (1.0 + distance)
        
        elif self.metric == SimilarityMetric.MANHATTAN:
            # Convert distance to similarity: 1 / (1 + distance)
            distance = self._manhattan_distance(query, doc_embedding)
            return 1.0 / (1.0 + distance)
        
        else:
            # Fallback to cosine
            return self._dot_product(query, doc_embedding)
    
    def _search_pure_python(
        self,
        embedding: list[float],
        k: int,
        filter: dict[str, Any] | None,
    ) -> list[SearchResult]:
        """Pure Python search implementation."""
        scores: list[tuple[float, StoredDocument]] = []
        
        for stored in self._storage.values():
            # Apply filter
            if filter and not self._matches_filter(stored.document, filter):
                continue
            
            # Compute similarity using configured metric
            score = self._compute_similarity(embedding, stored.embedding)
            scores.append((score, stored))
        
        # Sort by score descending
        scores.sort(key=lambda x: x[0], reverse=True)
        
        # Return top k
        results = []
        for score, stored in scores[:k]:
            results.append(SearchResult(
                document=stored.document,
                score=score,
                id=stored.id,
            ))
        
        return results
    
    def _search_numpy(
        self,
        embedding: list[float],
        k: int,
        filter: dict[str, Any] | None,
    ) -> list[SearchResult]:
        """NumPy-accelerated search implementation."""
        import numpy as np
        
        # Filter documents first
        filtered_docs = []
        for stored in self._storage.values():
            if filter and not self._matches_filter(stored.document, filter):
                continue
            filtered_docs.append(stored)
        
        if not filtered_docs:
            return []
        
        # Build embedding matrix
        query = np.array(embedding)
        matrix = np.array([doc.embedding for doc in filtered_docs])
        
        # Compute similarity based on metric
        if self.metric in (SimilarityMetric.COSINE, SimilarityMetric.DOT_PRODUCT):
            # Dot product (cosine for normalized vectors)
            scores = np.dot(matrix, query)
        
        elif self.metric == SimilarityMetric.EUCLIDEAN:
            # Euclidean distance converted to similarity
            distances = np.linalg.norm(matrix - query, axis=1)
            scores = 1.0 / (1.0 + distances)
        
        elif self.metric == SimilarityMetric.MANHATTAN:
            # Manhattan distance converted to similarity
            distances = np.sum(np.abs(matrix - query), axis=1)
            scores = 1.0 / (1.0 + distances)
        
        else:
            # Fallback to dot product
            scores = np.dot(matrix, query)
        
        # Get top k indices
        top_k = min(k, len(scores))
        top_indices = np.argsort(scores)[-top_k:][::-1]
        
        # Build results
        results = []
        for idx in top_indices:
            stored = filtered_docs[idx]
            results.append(SearchResult(
                document=stored.document,
                score=float(scores[idx]),
                id=stored.id,
            ))
        
        return results
    
    async def delete(self, ids: list[str]) -> bool:
        """Delete documents by ID.
        
        Args:
            ids: List of document IDs to delete.
            
        Returns:
            True if any documents were deleted.
        """
        deleted = False
        for doc_id in ids:
            if doc_id in self._storage:
                del self._storage[doc_id]
                deleted = True
        return deleted
    
    async def clear(self) -> None:
        """Remove all documents from the store."""
        self._storage.clear()
    
    async def get(self, ids: list[str]) -> list[Document]:
        """Get documents by ID.
        
        Args:
            ids: List of document IDs to retrieve.
            
        Returns:
            List of Document objects (in order, None for missing IDs).
        """
        results = []
        for doc_id in ids:
            if doc_id in self._storage:
                results.append(self._storage[doc_id].document)
        return results
    
    def count(self) -> int:
        """Return the number of documents in the store."""
        return len(self._storage)
    
    # ============================================================
    # Helper Methods
    # ============================================================
    
    @staticmethod
    def _normalize_vector(vec: list[float]) -> list[float]:
        """Normalize vector to unit length."""
        magnitude = math.sqrt(sum(x * x for x in vec))
        if magnitude == 0:
            return vec
        return [x / magnitude for x in vec]
    
    @staticmethod
    def _dot_product(a: list[float], b: list[float]) -> float:
        """Compute dot product of two vectors."""
        return sum(x * y for x, y in zip(a, b))
    
    @staticmethod
    def _euclidean_distance(a: list[float], b: list[float]) -> float:
        """Compute Euclidean (L2) distance between two vectors."""
        return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))
    
    @staticmethod
    def _manhattan_distance(a: list[float], b: list[float]) -> float:
        """Compute Manhattan (L1) distance between two vectors."""
        return sum(abs(x - y) for x, y in zip(a, b))
    
    @staticmethod
    def _matches_filter(doc: Document, filter: dict[str, Any]) -> bool:
        """Check if document metadata matches filter.
        
        Uses exact match for all filter keys.
        """
        for key, value in filter.items():
            if key not in doc.metadata:
                return False
            if doc.metadata[key] != value:
                return False
        return True
