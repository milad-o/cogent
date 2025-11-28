"""Vector store backends package.

Provides different storage backends:
- InMemoryBackend: NumPy-based, good for <10k documents (default)
- FAISSBackend: Large-scale similarity search
- ChromaBackend: Persistent with metadata filtering  
- QdrantBackend: Production vector database
- PgVectorBackend: PostgreSQL with pgvector
"""

from agenticflow.vectorstore.backends.inmemory import InMemoryBackend

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
        from agenticflow.vectorstore.backends.faiss import FAISSBackend
        return FAISSBackend
    elif name == "ChromaBackend":
        from agenticflow.vectorstore.backends.chroma import ChromaBackend
        return ChromaBackend
    elif name == "QdrantBackend":
        from agenticflow.vectorstore.backends.qdrant import QdrantBackend
        return QdrantBackend
    elif name == "PgVectorBackend":
        from agenticflow.vectorstore.backends.pgvector import PgVectorBackend
        return PgVectorBackend
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
