"""Base protocols and types for vector store module.

Defines the core abstractions:
- VectorStoreBackend: Protocol for storage backends
- EmbeddingProvider: Protocol for embedding generation
- SearchResult: Result from similarity search
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

if TYPE_CHECKING:
    from agenticflow.vectorstore.document import Document


@dataclass
class SearchResult:
    """Result from a similarity search.

    Attributes:
        document: The matched document.
        score: Similarity score (higher is more similar, typically 0-1 for cosine).
        id: Document ID.
    """

    document: Document
    score: float
    id: str = ""

    def __post_init__(self) -> None:
        """Set id from document if not provided."""
        if not self.id and self.document.id:
            self.id = self.document.id


@dataclass
class EmbeddingResult:
    """Result from embedding generation.

    Attributes:
        embeddings: List of embedding vectors.
        model: Model used to generate embeddings.
        usage: Token usage information.
    """

    embeddings: list[list[float]]
    model: str = ""
    usage: dict[str, int] = field(default_factory=dict)


@runtime_checkable
class EmbeddingProvider(Protocol):
    """Protocol for embedding providers.

    Implementations must provide:
    - embed_texts: Generate embeddings for multiple texts
    - embed_query: Generate embedding for a single query (may differ from document embedding)
    - dimension: The dimension of embeddings produced
    """

    @property
    def dimension(self) -> int:
        """Return the dimension of embeddings."""
        ...

    async def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for multiple texts.

        Args:
            texts: List of texts to embed.

        Returns:
            List of embedding vectors.
        """
        ...

    async def embed_query(self, text: str) -> list[float]:
        """Generate embedding for a query.

        Some models use different embeddings for queries vs documents.
        Default implementation calls embed_texts with a single text.

        Args:
            text: Query text to embed.

        Returns:
            Embedding vector.
        """
        ...


@runtime_checkable
class VectorStoreBackend(Protocol):
    """Protocol for vector store backends.

    Implementations must provide:
    - add: Add documents with embeddings
    - search: Find similar documents
    - delete: Remove documents
    - clear: Remove all documents
    """

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
        ...

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
            filter: Optional metadata filter.

        Returns:
            List of SearchResult objects sorted by similarity.
        """
        ...

    async def delete(self, ids: list[str]) -> bool:
        """Delete documents by ID.

        Args:
            ids: List of document IDs to delete.

        Returns:
            True if any documents were deleted.
        """
        ...

    async def clear(self) -> None:
        """Remove all documents from the store."""
        ...

    async def get(self, ids: list[str]) -> list[Document]:
        """Get documents by ID.

        Args:
            ids: List of document IDs to retrieve.

        Returns:
            List of Document objects (empty for missing IDs).
        """
        ...

    def count(self) -> int:
        """Return the number of documents in the store."""
        ...


@dataclass
class VectorStoreConfig:
    """Configuration for vector store.

    Attributes:
        backend: Backend type ("inmemory", "faiss", "chroma").
        embedding_model: Embedding model name.
        embedding_provider: Provider ("openai", "ollama").
        collection_name: Name for the collection/index.
        persist_directory: Directory for persistence (optional).
    """

    backend: str = "inmemory"
    embedding_model: str = "text-embedding-3-small"
    embedding_provider: str = "openai"
    collection_name: str = "default"
    persist_directory: str | None = None
