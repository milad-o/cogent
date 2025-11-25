"""Vector store wrappers for semantic memory.

Thin wrappers around LangChain vector stores for
semantic search and retrieval.

Semantic memory = Vector Stores:
- Embedding-based similarity search
- Document storage and retrieval
- RAG (Retrieval Augmented Generation) support
"""

from __future__ import annotations

from typing import Any, TYPE_CHECKING

from langchain_core.documents import Document
from langchain_core.vectorstores import InMemoryVectorStore, VectorStore
from langchain_core.embeddings import Embeddings

if TYPE_CHECKING:
    pass


def create_vectorstore(
    backend: str = "memory",
    *,
    embeddings: Embeddings | None = None,
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
        embeddings: Embeddings model instance
        **kwargs: Backend-specific configuration

    Returns:
        A configured vector store instance.

    Examples:
        # In-memory with OpenAI embeddings
        >>> from langchain_openai import OpenAIEmbeddings
        >>> embeddings = OpenAIEmbeddings()
        >>> vectorstore = create_vectorstore("memory", embeddings=embeddings)

        # Add documents
        >>> docs = [Document(page_content="Hello world")]
        >>> vectorstore.add_documents(docs)

        # Search
        >>> results = vectorstore.similarity_search("greeting", k=1)
    """
    if backend == "memory":
        if embeddings is None:
            raise ValueError("embeddings required for memory backend")
        return InMemoryVectorStore(embedding=embeddings, **kwargs)

    elif backend == "faiss":
        try:
            from langchain_community.vectorstores import FAISS  # type: ignore[import-untyped]
        except ImportError as e:
            raise ImportError(
                "FAISS backend requires langchain-community and faiss-cpu. "
                "Install with: pip install langchain-community faiss-cpu"
            ) from e

        if embeddings is None:
            raise ValueError("embeddings required for faiss backend")

        # Check if loading from existing index
        if "index_path" in kwargs:
            index_path = kwargs.pop("index_path")
            return FAISS.load_local(
                index_path,
                embeddings,
                allow_dangerous_deserialization=kwargs.pop(
                    "allow_dangerous_deserialization", True
                ),
                **kwargs,
            )

        # Create new empty index
        return FAISS.from_documents([], embeddings, **kwargs)

    elif backend == "chroma":
        try:
            from langchain_chroma import Chroma  # type: ignore[import-not-found]
        except ImportError as e:
            raise ImportError(
                "Chroma backend requires langchain-chroma. "
                "Install with: pip install langchain-chroma"
            ) from e

        return Chroma(embedding_function=embeddings, **kwargs)

    elif backend == "pgvector":
        try:
            from langchain_postgres import PGVector  # type: ignore[import-not-found]
        except ImportError as e:
            raise ImportError(
                "PGVector backend requires langchain-postgres. "
                "Install with: pip install langchain-postgres"
            ) from e

        connection = kwargs.pop("connection")
        return PGVector(
            embeddings=embeddings,
            connection=connection,
            **kwargs,
        )

    else:
        raise ValueError(
            f"Unknown backend: {backend}. "
            "Supported: 'memory', 'faiss', 'chroma', 'pgvector'"
        )


def memory_vectorstore(embeddings: Embeddings, **kwargs: Any) -> InMemoryVectorStore:
    """Create an in-memory vector store.

    Args:
        embeddings: Embeddings model
        **kwargs: Additional configuration

    Returns:
        InMemoryVectorStore instance.

    Example:
        >>> from langchain_openai import OpenAIEmbeddings
        >>> embeddings = OpenAIEmbeddings()
        >>> vectorstore = memory_vectorstore(embeddings)
        >>> vectorstore.add_texts(["Hello", "World"])
        >>> vectorstore.similarity_search("greeting")
    """
    return InMemoryVectorStore(embedding=embeddings, **kwargs)


def faiss_vectorstore(
    embeddings: Embeddings,
    *,
    index_path: str | None = None,
    **kwargs: Any,
) -> VectorStore:
    """Create a FAISS vector store.

    Requires: pip install langchain-community faiss-cpu

    Args:
        embeddings: Embeddings model
        index_path: Path to load existing index from
        **kwargs: Additional configuration

    Returns:
        FAISS instance.

    Example:
        >>> from langchain_openai import OpenAIEmbeddings
        >>> embeddings = OpenAIEmbeddings()
        >>> vectorstore = faiss_vectorstore(embeddings)
        >>> vectorstore.add_texts(["Hello", "World"])
        >>> vectorstore.save_local("my_index")
        >>>
        >>> # Load later
        >>> vectorstore = faiss_vectorstore(embeddings, index_path="my_index")
    """
    return create_vectorstore("faiss", embeddings=embeddings, index_path=index_path, **kwargs)


def chroma_vectorstore(
    embeddings: Embeddings | None = None,
    *,
    collection_name: str = "langchain",
    persist_directory: str | None = None,
    **kwargs: Any,
) -> VectorStore:
    """Create a ChromaDB vector store.

    Requires: pip install langchain-chroma

    Args:
        embeddings: Embeddings model (optional for Chroma)
        collection_name: Name of the collection
        persist_directory: Directory for persistence
        **kwargs: Additional configuration

    Returns:
        Chroma instance.

    Example:
        >>> from langchain_openai import OpenAIEmbeddings
        >>> embeddings = OpenAIEmbeddings()
        >>> vectorstore = chroma_vectorstore(
        ...     embeddings,
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
]
