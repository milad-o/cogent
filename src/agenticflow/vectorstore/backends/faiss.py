"""FAISS vector store backend.

High-performance vector similarity search using Facebook AI Similarity Search.
Requires: pip install faiss-cpu (or faiss-gpu for CUDA support).

Best for:
- Large datasets (100k+ documents)
- Fast similarity search
- When you need ANN (Approximate Nearest Neighbor) search
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from agenticflow.vectorstore.base import SearchResult
from agenticflow.vectorstore.document import Document


@dataclass
class FAISSBackend:
    """FAISS vector store backend.

    Uses FAISS for efficient similarity search on large datasets.
    Supports both exact (IndexFlatIP) and approximate (IndexIVFFlat) search.

    Attributes:
        dimension: Embedding dimension (required).
        index_type: Type of index ("flat", "ivf", "hnsw"). Default: "flat".
        nlist: Number of clusters for IVF index. Default: 100.
        nprobe: Number of clusters to search. Default: 10.
        persist_directory: Directory for saving/loading index. Optional.

    Example:
        backend = FAISSBackend(dimension=1536)
        await backend.add(ids, embeddings, documents)
        results = await backend.search(query_embedding, k=10)
    """

    dimension: int
    index_type: str = "flat"
    nlist: int = 100
    nprobe: int = 10
    persist_directory: str | Path | None = None

    _index: Any = field(default=None, init=False, repr=False)
    _id_to_doc: dict[str, Document] = field(default_factory=dict, repr=False)
    _id_to_idx: dict[str, int] = field(default_factory=dict, repr=False)
    _idx_to_id: dict[int, str] = field(default_factory=dict, repr=False)
    _faiss: Any = field(default=None, init=False, repr=False)
    _np: Any = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        """Initialize FAISS index."""
        try:
            import faiss
            import numpy as np
            self._faiss = faiss
            self._np = np
        except ImportError as e:
            msg = "FAISS not installed. Install with: pip install faiss-cpu"
            raise ImportError(msg) from e

        self._create_index()

        # Load from disk if directory exists
        if self.persist_directory:
            self.persist_directory = Path(self.persist_directory)
            self._load_if_exists()

    def _create_index(self) -> None:
        """Create the FAISS index based on index_type."""
        faiss = self._faiss

        if self.index_type == "flat":
            # Exact search with inner product (for normalized vectors = cosine)
            self._index = faiss.IndexFlatIP(self.dimension)
        elif self.index_type == "ivf":
            # Approximate search with IVF
            quantizer = faiss.IndexFlatIP(self.dimension)
            self._index = faiss.IndexIVFFlat(quantizer, self.dimension, self.nlist)
            self._index.nprobe = self.nprobe
        elif self.index_type == "hnsw":
            # HNSW index for fast approximate search
            self._index = faiss.IndexHNSWFlat(self.dimension, 32)  # 32 neighbors
        else:
            msg = f"Unknown index_type: {self.index_type}. Use 'flat', 'ivf', or 'hnsw'."
            raise ValueError(msg)

    def _normalize(self, vectors: Any) -> Any:
        """Normalize vectors for cosine similarity."""
        np = self._np
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1, norms)  # Avoid division by zero
        return vectors / norms

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
        """
        if not ids:
            return

        np = self._np
        faiss = self._faiss

        # Convert to numpy and normalize
        vectors = np.array(embeddings, dtype=np.float32)
        vectors = self._normalize(vectors)

        # Train IVF index if needed and not trained
        if self.index_type == "ivf" and not self._index.is_trained:
            if len(vectors) < self.nlist:
                # Not enough vectors to train, use flat index temporarily
                self._index = faiss.IndexFlatIP(self.dimension)
            else:
                self._index.train(vectors)

        # Get current index size for new indices
        start_idx = self._index.ntotal

        # Add to index
        self._index.add(vectors)

        # Store document mappings
        for i, (doc_id, doc) in enumerate(zip(ids, documents, strict=False)):
            idx = start_idx + i
            self._id_to_doc[doc_id] = doc
            self._id_to_idx[doc_id] = idx
            self._idx_to_id[idx] = doc_id

        # Auto-save if persist directory is set
        if self.persist_directory:
            self._save()

    async def search(
        self,
        embedding: list[float],
        k: int = 4,
        filter: dict[str, Any] | None = None,
    ) -> list[SearchResult]:
        """Search for similar documents.

        Args:
            embedding: Query embedding vector.
            k: Number of results to return.
            filter: Optional metadata filter (post-filtering).

        Returns:
            List of SearchResult objects sorted by similarity.
        """
        if self._index.ntotal == 0:
            return []

        np = self._np

        # Convert and normalize query
        query = np.array([embedding], dtype=np.float32)
        query = self._normalize(query)

        # Search more results if filtering (for post-filtering)
        search_k = k * 4 if filter else k
        search_k = min(search_k, self._index.ntotal)

        # Search
        scores, indices = self._index.search(query, search_k)

        # Build results with optional filtering
        results: list[SearchResult] = []
        for score, idx in zip(scores[0], indices[0], strict=False):
            if idx == -1:  # FAISS returns -1 for empty slots
                continue

            doc_id = self._idx_to_id.get(int(idx))
            if doc_id is None:
                continue

            doc = self._id_to_doc.get(doc_id)
            if doc is None:
                continue

            # Apply filter
            if filter and not self._matches_filter(doc, filter):
                continue

            results.append(SearchResult(
                document=doc,
                score=float(score),
                id=doc_id,
            ))

            if len(results) >= k:
                break

        return results

    async def delete(self, ids: list[str]) -> bool:
        """Delete documents by ID.

        Note: FAISS doesn't support efficient deletion. This marks documents
        as deleted but doesn't remove them from the index. Consider rebuilding
        the index periodically for cleanup.

        Args:
            ids: List of document IDs to delete.

        Returns:
            True if any documents were deleted.
        """
        deleted = False
        for doc_id in ids:
            if doc_id in self._id_to_doc:
                # Remove from mappings (vector stays in index but won't be returned)
                idx = self._id_to_idx.pop(doc_id, None)
                if idx is not None:
                    self._idx_to_id.pop(idx, None)
                self._id_to_doc.pop(doc_id, None)
                deleted = True

        if deleted and self.persist_directory:
            self._save()

        return deleted

    async def clear(self) -> None:
        """Remove all documents from the store."""
        self._create_index()  # Reset index
        self._id_to_doc.clear()
        self._id_to_idx.clear()
        self._idx_to_id.clear()

        if self.persist_directory:
            self._save()

    async def get(self, ids: list[str]) -> list[Document]:
        """Get documents by ID.

        Args:
            ids: List of document IDs to retrieve.

        Returns:
            List of Document objects.
        """
        results = []
        for doc_id in ids:
            doc = self._id_to_doc.get(doc_id)
            if doc:
                results.append(doc)
        return results

    def count(self) -> int:
        """Return the number of documents in the store."""
        return len(self._id_to_doc)

    # ============================================================
    # Persistence
    # ============================================================

    def _save(self) -> None:
        """Save index and metadata to disk."""
        import json

        if not self.persist_directory:
            return

        self.persist_directory.mkdir(parents=True, exist_ok=True)

        # Save FAISS index
        index_path = self.persist_directory / "index.faiss"
        self._faiss.write_index(self._index, str(index_path))

        # Save metadata
        metadata = {
            "id_to_idx": self._id_to_idx,
            "idx_to_id": {str(k): v for k, v in self._idx_to_id.items()},
            "documents": {
                doc_id: doc.to_dict() for doc_id, doc in self._id_to_doc.items()
            },
        }
        metadata_path = self.persist_directory / "metadata.json"
        with open(metadata_path, "w") as f:
            json.dump(metadata, f)

    def _load_if_exists(self) -> None:
        """Load index and metadata from disk if they exist."""
        import json

        if not self.persist_directory:
            return

        index_path = self.persist_directory / "index.faiss"
        metadata_path = self.persist_directory / "metadata.json"

        if not index_path.exists() or not metadata_path.exists():
            return

        # Load FAISS index
        self._index = self._faiss.read_index(str(index_path))

        # Load metadata
        with open(metadata_path) as f:
            metadata = json.load(f)

        self._id_to_idx = metadata["id_to_idx"]
        self._idx_to_id = {int(k): v for k, v in metadata["idx_to_id"].items()}
        self._id_to_doc = {
            doc_id: Document.from_dict(doc_data)
            for doc_id, doc_data in metadata["documents"].items()
        }

    # ============================================================
    # Helpers
    # ============================================================

    @staticmethod
    def _matches_filter(doc: Document, filter: dict[str, Any]) -> bool:
        """Check if document metadata matches filter."""
        for key, value in filter.items():
            if key not in doc.metadata:
                return False
            if doc.metadata[key] != value:
                return False
        return True
