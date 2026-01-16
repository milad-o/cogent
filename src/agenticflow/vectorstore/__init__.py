"""Vector store module for semantic search and RAG.

Provides:
- Document storage with embeddings
- Similarity search
- Multiple backend support (InMemory, FAISS, Chroma, Qdrant, pgvector)
- Multiple embedding providers (OpenAI, Ollama, etc.)

Simple by default, powerful when needed.

Example:
    >>> from agenticflow.vectorstore import VectorStore
    >>> store = VectorStore()
    >>> await store.add_texts(["Python is great", "JavaScript is popular"])
    >>> results = await store.search("programming language")

Available backends (optional dependencies):
    - InMemoryBackend: NumPy-based, default (no extra deps)
    - FAISSBackend: Large-scale search (pip install faiss-cpu)
    - ChromaBackend: Persistent vector DB (pip install chromadb)
    - QdrantBackend: Production vector DB (pip install qdrant-client)
    - PgVectorBackend: PostgreSQL (pip install psycopg[pool])
"""

# Import backends submodule for lazy access
from agenticflow.vectorstore import backends

# Import backends submodule for lazy access
from agenticflow.vectorstore.backends.inmemory import SimilarityMetric
from agenticflow.vectorstore.base import (
    EmbeddingProvider,
    SearchResult,
    VectorStoreBackend,
    VectorStoreConfig,
)
from agenticflow.vectorstore.document import (
    Document,
    create_documents,
    split_documents,
    split_text,
)
from agenticflow.vectorstore.embeddings import (
    MockEmbeddings,
    OllamaEmbeddings,
    OpenAIEmbeddings,
)
from agenticflow.vectorstore.store import VectorStore, create_vectorstore

__all__ = [
    # Core
    "VectorStore",
    "Document",
    "SearchResult",
    # Factory
    "create_vectorstore",
    # Protocols
    "VectorStoreBackend",
    "EmbeddingProvider",
    "VectorStoreConfig",
    # Similarity metrics
    "SimilarityMetric",
    # Embeddings
    "OpenAIEmbeddings",
    "OllamaEmbeddings",
    "MockEmbeddings",
    # Document utilities
    "create_documents",
    "split_text",
    "split_documents",
    # Backends module
    "backends",
]
