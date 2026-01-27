"""Vector store backends package.

Provides different storage backends:
- InMemoryBackend: NumPy-based, good for <10k documents (default)
- FAISSBackend: Large-scale similarity search
- ChromaBackend: Persistent with metadata filtering
- QdrantBackend: Production vector database
- PgVectorBackend: PostgreSQL with pgvector
"""

from cogent.vectorstore.backends.inmemory import InMemoryBackend

# Lazy imports for optional backends
__all__ = [
    "InMemoryBackend",
    "FAISSBackend",
    "ChromaBackend",
    "QdrantBackend",
    "PgVectorBackend",
]


def __getattr__(name: str):
    """Lazy import optional backends."""
    if name == "FAISSBackend":
        from cogent.vectorstore.backends.faiss import FAISSBackend

        return FAISSBackend
    elif name == "ChromaBackend":
        from cogent.vectorstore.backends.chroma import ChromaBackend

        return ChromaBackend
    elif name == "QdrantBackend":
        from cogent.vectorstore.backends.qdrant import QdrantBackend

        return QdrantBackend
    elif name == "PgVectorBackend":
        from cogent.vectorstore.backends.pgvector import PgVectorBackend

        return PgVectorBackend
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
