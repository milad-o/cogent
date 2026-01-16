"""Qdrant vector store backend.

Production-grade vector database with advanced filtering and scalability.
Requires: pip install qdrant-client

Best for:
- Production deployments
- Advanced filtering with payload
- Scalable cloud deployments
- High availability requirements
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from agenticflow.vectorstore.base import SearchResult
from agenticflow.vectorstore.document import Document


@dataclass
class QdrantBackend:
    """Qdrant vector store backend.

    Uses Qdrant for production-grade vector storage and search.
    Supports both local (in-memory/disk) and remote (cloud) deployments.

    Attributes:
        collection_name: Name of the collection. Default: "default".
        url: Qdrant server URL. Default: None (uses in-memory).
        api_key: API key for Qdrant Cloud. Optional.
        dimension: Embedding dimension (required for collection creation).
        distance: Distance metric ("cosine", "euclid", "dot"). Default: "cosine".
        path: Path for local disk persistence. Optional.

    Example:
        # In-memory (for testing)
        backend = QdrantBackend(collection_name="docs", dimension=1536)

        # Local persistent
        backend = QdrantBackend(
            collection_name="docs",
            dimension=1536,
            path="./qdrant_data"
        )

        # Remote (Qdrant Cloud)
        backend = QdrantBackend(
            collection_name="docs",
            dimension=1536,
            url="https://xxx.qdrant.io",
            api_key="your-api-key"
        )
    """

    collection_name: str = "default"
    url: str | None = None
    api_key: str | None = None
    dimension: int = 1536
    distance: str = "cosine"
    path: str | None = None

    _client: Any = field(default=None, init=False, repr=False)
    _models: Any = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        """Initialize Qdrant client and collection."""
        try:
            from qdrant_client import QdrantClient, models
            self._models = models
        except ImportError as e:
            msg = "Qdrant client not installed. Install with: pip install qdrant-client"
            raise ImportError(msg) from e

        # Create client based on configuration
        if self.url:
            # Remote Qdrant server or cloud
            self._client = QdrantClient(
                url=self.url,
                api_key=self.api_key,
            )
        elif self.path:
            # Local persistent storage
            self._client = QdrantClient(path=self.path)
        else:
            # In-memory (for testing)
            self._client = QdrantClient(":memory:")

        # Create collection if it doesn't exist
        self._ensure_collection()

    def _ensure_collection(self) -> None:
        """Ensure collection exists with correct configuration."""
        models = self._models

        # Map distance string to Qdrant Distance enum
        distance_map = {
            "cosine": models.Distance.COSINE,
            "euclid": models.Distance.EUCLID,
            "dot": models.Distance.DOT,
        }
        distance = distance_map.get(self.distance, models.Distance.COSINE)

        # Check if collection exists
        collections = self._client.get_collections().collections
        exists = any(c.name == self.collection_name for c in collections)

        if not exists:
            self._client.create_collection(
                collection_name=self.collection_name,
                vectors_config=models.VectorParams(
                    size=self.dimension,
                    distance=distance,
                ),
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

        models = self._models

        # Build points
        points = []
        for doc_id, embedding, doc in zip(ids, embeddings, documents, strict=False):
            # Build payload (metadata + text)
            payload = {
                "text": doc.text,
                **doc.metadata,
            }

            points.append(models.PointStruct(
                id=doc_id,
                vector=embedding,
                payload=payload,
            ))

        # Upsert points
        self._client.upsert(
            collection_name=self.collection_name,
            points=points,
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
            filter: Optional metadata filter (Qdrant filter format).

        Returns:
            List of SearchResult objects sorted by similarity.
        """

        # Build filter
        query_filter = self._build_filter(filter) if filter else None

        # Search
        results = self._client.search(
            collection_name=self.collection_name,
            query_vector=embedding,
            limit=k,
            query_filter=query_filter,
        )

        # Build SearchResult objects
        search_results: list[SearchResult] = []

        for result in results:
            payload = result.payload or {}
            text = payload.pop("text", "")

            doc = Document(
                text=text,
                metadata=payload,
                id=str(result.id),
            )

            search_results.append(SearchResult(
                document=doc,
                score=float(result.score),
                id=str(result.id),
            ))

        return search_results

    async def delete(self, ids: list[str]) -> bool:
        """Delete documents by ID.

        Args:
            ids: List of document IDs to delete.

        Returns:
            True if operation completed.
        """
        if not ids:
            return False

        models = self._models

        self._client.delete(
            collection_name=self.collection_name,
            points_selector=models.PointIdsList(points=ids),
        )

        return True

    async def clear(self) -> None:
        """Remove all documents from the store."""
        # Delete and recreate collection
        self._client.delete_collection(self.collection_name)
        self._ensure_collection()

    async def get(self, ids: list[str]) -> list[Document]:
        """Get documents by ID.

        Args:
            ids: List of document IDs to retrieve.

        Returns:
            List of Document objects.
        """
        if not ids:
            return []

        results = self._client.retrieve(
            collection_name=self.collection_name,
            ids=ids,
        )

        documents = []
        for result in results:
            payload = result.payload or {}
            text = payload.pop("text", "")

            documents.append(Document(
                text=text,
                metadata=payload,
                id=str(result.id),
            ))

        return documents

    def count(self) -> int:
        """Return the number of documents in the store."""
        info = self._client.get_collection(self.collection_name)
        return info.points_count

    # ============================================================
    # Helpers
    # ============================================================

    def _build_filter(self, filter: dict[str, Any]) -> Any:
        """Build Qdrant filter from simple filter dict.

        Supports:
        - Simple equality: {"key": "value"}
        - Numeric comparisons via special keys
        """
        models = self._models

        if not filter:
            return None

        conditions = []
        for key, value in filter.items():
            if isinstance(value, dict):
                # Handle operator-style filters
                for op, val in value.items():
                    if op == "$gt":
                        conditions.append(models.FieldCondition(
                            key=key,
                            range=models.Range(gt=val),
                        ))
                    elif op == "$gte":
                        conditions.append(models.FieldCondition(
                            key=key,
                            range=models.Range(gte=val),
                        ))
                    elif op == "$lt":
                        conditions.append(models.FieldCondition(
                            key=key,
                            range=models.Range(lt=val),
                        ))
                    elif op == "$lte":
                        conditions.append(models.FieldCondition(
                            key=key,
                            range=models.Range(lte=val),
                        ))
                    elif op == "$in":
                        conditions.append(models.FieldCondition(
                            key=key,
                            match=models.MatchAny(any=val),
                        ))
            else:
                # Simple equality match
                conditions.append(models.FieldCondition(
                    key=key,
                    match=models.MatchValue(value=value),
                ))

        if len(conditions) == 1:
            return models.Filter(must=conditions)
        else:
            return models.Filter(must=conditions)
