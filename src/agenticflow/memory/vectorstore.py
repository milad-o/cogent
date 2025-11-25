"""Vector store wrappers for semantic memory.

Thin wrappers around LangChain vector stores for
semantic search and retrieval.

Semantic memory = Vector Stores:
- Embedding-based similarity search
- Document storage and retrieval
- RAG (Retrieval Augmented Generation) support
"""

from __future__ import annotations

from typing import Any

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import InMemoryVectorStore, VectorStore

from agenticflow.providers import EmbeddingSpec, create_embeddings


def create_vectorstore(
    backend: str = "memory",
    *,
    embeddings: str | EmbeddingSpec | Embeddings | None = None,
    **kwargs: Any,
) -> VectorStore:
    """Create a vector store for semantic memory.

    Vector stores enable semantic search over documents
    using embeddings.

    Args:
        backend: Storage backend type:
            - "memory": In-memory (ephemeral)
            - "faiss": FAISS (local, persistent)
            - "chroma": ChromaDB (local/cloud)
            - "pgvector": PostgreSQL with pgvector
        embeddings: Embeddings specification - can be:
            - String: "openai/text-embedding-3-small", "cohere/embed-english-v3.0"
            - EmbeddingSpec: Explicit specification object
            - Embeddings: Direct LangChain embeddings instance
        **kwargs: Backend-specific configuration

    Returns:
        A configured vector store instance.

    Examples:
        # In-memory with string spec (simplest)
        >>> vectorstore = create_vectorstore(
        ...     "memory",
        ...     embeddings="openai/text-embedding-3-small"
        ... )

        # Using EmbeddingSpec for more control
        >>> from agenticflow.providers import EmbeddingSpec
        >>> spec = EmbeddingSpec(
        ...     provider="openai",
        ...     model="text-embedding-3-large",
        ...     dimensions=1024,  # Reduce dimensions
        ... )
        >>> vectorstore = create_vectorstore("memory", embeddings=spec)

        # Direct LangChain embeddings (legacy)
        >>> from langchain_openai import OpenAIEmbeddings
        >>> embeddings = OpenAIEmbeddings()
        >>> vectorstore = create_vectorstore("memory", embeddings=embeddings)

        # Add documents
        >>> docs = [Document(page_content="Hello world")]
        >>> vectorstore.add_documents(docs)

        # Search
        >>> results = vectorstore.similarity_search("greeting", k=1)
    """
    # Resolve embeddings from string or spec
    resolved_embeddings = _resolve_embeddings(embeddings)

    if backend == "memory":
        if resolved_embeddings is None:
            raise ValueError("embeddings required for memory backend")
        return InMemoryVectorStore(embedding=resolved_embeddings, **kwargs)

    elif backend == "faiss":
        try:
            from langchain_community.vectorstores import FAISS  # type: ignore[import-untyped]
        except ImportError as e:
            raise ImportError(
                "FAISS backend requires langchain-community and faiss-cpu. "
                "Install with: uv add langchain-community faiss-cpu"
            ) from e

        if resolved_embeddings is None:
            raise ValueError("embeddings required for faiss backend")

        # Check if loading from existing index
        if "index_path" in kwargs:
            index_path = kwargs.pop("index_path")
            return FAISS.load_local(
                index_path,
                resolved_embeddings,
                allow_dangerous_deserialization=kwargs.pop(
                    "allow_dangerous_deserialization", True
                ),
                **kwargs,
            )

        # Create new empty index
        return FAISS.from_documents([], resolved_embeddings, **kwargs)

    elif backend == "chroma":
        try:
            from langchain_chroma import Chroma  # type: ignore[import-not-found]
        except ImportError as e:
            raise ImportError(
                "Chroma backend requires langchain-chroma. "
                "Install with: uv add langchain-chroma"
            ) from e

        return Chroma(embedding_function=resolved_embeddings, **kwargs)

    elif backend == "pgvector":
        try:
            from langchain_postgres import PGVector  # type: ignore[import-not-found]
        except ImportError as e:
            raise ImportError(
                "PGVector backend requires langchain-postgres. "
                "Install with: uv add langchain-postgres"
            ) from e

        connection = kwargs.pop("connection")
        return PGVector(
            embeddings=resolved_embeddings,
            connection=connection,
            **kwargs,
        )

    else:
        raise ValueError(
            f"Unknown backend: {backend}. "
            "Supported: 'memory', 'faiss', 'chroma', 'pgvector'"
        )


def _resolve_embeddings(
    embeddings: str | EmbeddingSpec | Embeddings | None,
) -> Embeddings | None:
    """Resolve embeddings from various input types.

    Args:
        embeddings: String spec, EmbeddingSpec, or Embeddings instance.

    Returns:
        Resolved Embeddings instance or None.
    """
    if embeddings is None:
        return None

    if isinstance(embeddings, Embeddings):
        return embeddings

    if isinstance(embeddings, EmbeddingSpec):
        return embeddings.create()

    if isinstance(embeddings, str):
        return create_embeddings(embeddings)

    raise TypeError(
        f"embeddings must be str, EmbeddingSpec, or Embeddings, got {type(embeddings)}"
    )


def memory_vectorstore(
    embeddings: str | EmbeddingSpec | Embeddings,
    **kwargs: Any,
) -> InMemoryVectorStore:
    """Create an in-memory vector store.

    Args:
        embeddings: Embeddings specification - string, EmbeddingSpec, or instance.
        **kwargs: Additional configuration

    Returns:
        InMemoryVectorStore instance.

    Example:
        # Using string spec (simplest)
        >>> vectorstore = memory_vectorstore("openai/text-embedding-3-small")
        >>> vectorstore.add_texts(["Hello", "World"])
        >>> vectorstore.similarity_search("greeting")

        # Using EmbeddingSpec
        >>> from agenticflow.providers import EmbeddingSpec
        >>> spec = EmbeddingSpec(provider="openai", model="text-embedding-3-small")
        >>> vectorstore = memory_vectorstore(spec)
    """
    resolved = _resolve_embeddings(embeddings)
    if resolved is None:
        raise ValueError("embeddings is required")
    return InMemoryVectorStore(embedding=resolved, **kwargs)


def faiss_vectorstore(
    embeddings: str | EmbeddingSpec | Embeddings,
    *,
    index_path: str | None = None,
    **kwargs: Any,
) -> VectorStore:
    """Create a FAISS vector store.

    Requires: uv add langchain-community faiss-cpu

    Args:
        embeddings: Embeddings specification - string, EmbeddingSpec, or instance.
        index_path: Path to load existing index from
        **kwargs: Additional configuration

    Returns:
        FAISS instance.

    Example:
        # Using string spec
        >>> vectorstore = faiss_vectorstore("openai/text-embedding-3-small")
        >>> vectorstore.add_texts(["Hello", "World"])
        >>> vectorstore.save_local("my_index")
        >>>
        >>> # Load later
        >>> vectorstore = faiss_vectorstore(
        ...     "openai/text-embedding-3-small",
        ...     index_path="my_index"
        ... )
    """
    return create_vectorstore("faiss", embeddings=embeddings, index_path=index_path, **kwargs)


def chroma_vectorstore(
    embeddings: str | EmbeddingSpec | Embeddings | None = None,
    *,
    collection_name: str = "langchain",
    persist_directory: str | None = None,
    **kwargs: Any,
) -> VectorStore:
    """Create a ChromaDB vector store.

    Requires: uv add langchain-chroma

    Args:
        embeddings: Embeddings specification - string, EmbeddingSpec, or instance.
        collection_name: Name of the collection
        persist_directory: Directory for persistence
        **kwargs: Additional configuration

    Returns:
        Chroma instance.

    Example:
        # Using string spec
        >>> vectorstore = chroma_vectorstore(
        ...     "openai/text-embedding-3-small",
        ...     collection_name="my_docs",
        ...     persist_directory="./chroma_db"
        ... )
    """
    return create_vectorstore(
        "chroma",
        embeddings=embeddings,
        collection_name=collection_name,
        persist_directory=persist_directory,
        **kwargs,
    )


# Re-export base types
__all__ = [
    "create_vectorstore",
    "memory_vectorstore",
    "faiss_vectorstore",
    "chroma_vectorstore",
    "VectorStore",
    "InMemoryVectorStore",
    "Document",
    "EmbeddingSpec",
]
