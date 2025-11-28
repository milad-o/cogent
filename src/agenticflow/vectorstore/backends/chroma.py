"""Chroma vector store backend.

Persistent vector database with built-in embedding and metadata filtering.
Requires: pip install chromadb

Best for:
- Persistent storage with metadata filtering
- Built-in embedding support
- Local development and small-to-medium datasets
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from agenticflow.vectorstore.base import SearchResult
from agenticflow.vectorstore.document import Document


@dataclass
class ChromaBackend:
    """Chroma vector store backend.
    
    Uses ChromaDB for persistent vector storage with metadata filtering.
    
    Attributes:
        collection_name: Name of the collection. Default: "default".
        persist_directory: Directory for persistent storage. Optional.
        distance_fn: Distance function ("cosine", "l2", "ip"). Default: "cosine".
        
    Example:
        backend = ChromaBackend(
            collection_name="my_docs",
            persist_directory="./chroma_db"
        )
        await backend.add(ids, embeddings, documents)
        results = await backend.search(query_embedding, k=10, filter={"type": "article"})
    """
    
    collection_name: str = "default"
    persist_directory: str | Path | None = None
    distance_fn: str = "cosine"
    
    _client: Any = field(default=None, init=False, repr=False)
    _collection: Any = field(default=None, init=False, repr=False)
    
    def __post_init__(self) -> None:
        """Initialize Chroma client and collection."""
        try:
            import chromadb
            from chromadb.config import Settings
        except ImportError as e:
            msg = "ChromaDB not installed. Install with: pip install chromadb"
            raise ImportError(msg) from e
        
        # Create client
        if self.persist_directory:
            self.persist_directory = Path(self.persist_directory)
            self.persist_directory.mkdir(parents=True, exist_ok=True)
            self._client = chromadb.PersistentClient(
                path=str(self.persist_directory),
                settings=Settings(anonymized_telemetry=False),
            )
        else:
            self._client = chromadb.Client(
                settings=Settings(anonymized_telemetry=False),
            )
        
        # Get or create collection
        self._collection = self._client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": self.distance_fn},
        )
    
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
        
        # Prepare data for Chroma
        texts = [doc.text for doc in documents]
        metadatas = [doc.metadata for doc in documents]
        
        # Sanitize metadata (Chroma only supports str, int, float, bool)
        sanitized_metadatas = [self._sanitize_metadata(m) for m in metadatas]
        
        # Add to collection
        self._collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=texts,
            metadatas=sanitized_metadatas,
        )
    
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
            filter: Optional metadata filter (Chroma where clause).
            
        Returns:
            List of SearchResult objects sorted by similarity.
        """
        # Build where clause for Chroma
        where = self._build_where_clause(filter) if filter else None
        
        # Query collection
        results = self._collection.query(
            query_embeddings=[embedding],
            n_results=k,
            where=where,
            include=["documents", "metadatas", "distances"],
        )
        
        # Build SearchResult objects
        search_results: list[SearchResult] = []
        
        if results["ids"] and results["ids"][0]:
            for i, doc_id in enumerate(results["ids"][0]):
                text = results["documents"][0][i] if results["documents"] else ""
                metadata = results["metadatas"][0][i] if results["metadatas"] else {}
                distance = results["distances"][0][i] if results["distances"] else 0
                
                # Convert distance to similarity score
                # Chroma returns distance, we want similarity (higher is better)
                if self.distance_fn == "cosine":
                    score = 1 - distance  # Cosine distance to similarity
                elif self.distance_fn == "l2":
                    score = 1 / (1 + distance)  # L2 distance to similarity
                else:  # ip (inner product)
                    score = distance  # Already a similarity
                
                doc = Document(
                    text=text,
                    metadata=metadata,
                    id=doc_id,
                )
                
                search_results.append(SearchResult(
                    document=doc,
                    score=float(score),
                    id=doc_id,
                ))
        
        return search_results
    
    async def delete(self, ids: list[str]) -> bool:
        """Delete documents by ID.
        
        Args:
            ids: List of document IDs to delete.
            
        Returns:
            True if operation completed (Chroma doesn't return delete status).
        """
        if not ids:
            return False
        
        try:
            self._collection.delete(ids=ids)
            return True
        except Exception:
            return False
    
    async def clear(self) -> None:
        """Remove all documents from the store."""
        # Delete and recreate collection
        self._client.delete_collection(self.collection_name)
        self._collection = self._client.create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": self.distance_fn},
        )
    
    async def get(self, ids: list[str]) -> list[Document]:
        """Get documents by ID.
        
        Args:
            ids: List of document IDs to retrieve.
            
        Returns:
            List of Document objects.
        """
        if not ids:
            return []
        
        results = self._collection.get(
            ids=ids,
            include=["documents", "metadatas"],
        )
        
        documents = []
        if results["ids"]:
            for i, doc_id in enumerate(results["ids"]):
                text = results["documents"][i] if results["documents"] else ""
                metadata = results["metadatas"][i] if results["metadatas"] else {}
                
                documents.append(Document(
                    text=text,
                    metadata=metadata,
                    id=doc_id,
                ))
        
        return documents
    
    def count(self) -> int:
        """Return the number of documents in the store."""
        return self._collection.count()
    
    # ============================================================
    # Helpers
    # ============================================================
    
    @staticmethod
    def _sanitize_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
        """Sanitize metadata for Chroma (only str, int, float, bool allowed)."""
        sanitized = {}
        for key, value in metadata.items():
            if isinstance(value, (str, int, float, bool)):
                sanitized[key] = value
            elif value is None:
                continue  # Skip None values
            else:
                # Convert to string
                sanitized[key] = str(value)
        return sanitized
    
    @staticmethod
    def _build_where_clause(filter: dict[str, Any]) -> dict[str, Any]:
        """Build Chroma where clause from simple filter dict.
        
        Supports:
        - Simple equality: {"key": "value"}
        - Operators: {"key": {"$gt": 5}}
        """
        if not filter:
            return {}
        
        # Check if it's already a Chroma-style filter
        for value in filter.values():
            if isinstance(value, dict):
                return filter  # Already Chroma format
        
        # Convert simple equality to Chroma format
        if len(filter) == 1:
            return filter
        else:
            # Multiple conditions use $and
            return {"$and": [{k: v} for k, v in filter.items()]}
