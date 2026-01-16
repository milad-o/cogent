"""Dense retriever using vector similarity search.

Wraps a VectorStore to provide the Retriever interface.
This is the most common retriever type for semantic search.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from agenticflow.retriever.base import BaseRetriever, RetrievalResult

if TYPE_CHECKING:
    from agenticflow.vectorstore import Document, VectorStore
    from agenticflow.vectorstore.base import EmbeddingProvider


class DenseRetriever(BaseRetriever):
    """Dense retriever using vector embeddings.

    Uses a VectorStore for semantic similarity search.
    Documents are matched based on embedding similarity.

    Attributes:
        vectorstore: The underlying vector store.

    Example:
        >>> from agenticflow.vectorstore import VectorStore
        >>> from agenticflow.retriever import DenseRetriever
        >>>
        >>> store = VectorStore()
        >>> await store.add_texts(["Python is great", "JavaScript rocks"])
        >>>
        >>> retriever = DenseRetriever(store)
        >>> docs = await retriever.retrieve("programming language")
    """

    _name: str = "dense"

    def __init__(
        self,
        vectorstore: VectorStore,
        *,
        name: str | None = None,
        score_threshold: float | None = None,
    ) -> None:
        """Create a dense retriever.

        Args:
            vectorstore: VectorStore for similarity search.
            name: Optional custom name for this retriever.
            score_threshold: Minimum score threshold (0-1). Results below
                this score are filtered out.
        """
        self.vectorstore = vectorstore
        self.score_threshold = score_threshold
        if name:
            self._name = name

    async def retrieve_with_scores(
        self,
        query: str,
        k: int = 4,
        filter: dict[str, Any] | None = None,
    ) -> list[RetrievalResult]:
        """Retrieve documents using vector similarity.

        Args:
            query: The search query (will be embedded).
            k: Number of documents to retrieve.
            filter: Optional metadata filter.

        Returns:
            List of RetrievalResult ordered by similarity score.
        """
        # Use vectorstore's search which returns SearchResult
        search_results = await self.vectorstore.search(query, k=k, filter=filter)

        results = []
        for sr in search_results:
            # Apply score threshold if set
            if self.score_threshold is not None and sr.score < self.score_threshold:
                continue

            results.append(
                RetrievalResult(
                    document=sr.document,
                    score=sr.score,
                    retriever_name=self.name,
                    metadata={"search_type": "dense"},
                )
            )

        return results

    async def add_documents(self, documents: list[Document]) -> list[str]:
        """Add documents to the underlying vector store.

        Convenience method for adding documents through the retriever.

        Args:
            documents: Documents to add.

        Returns:
            List of document IDs.
        """
        return await self.vectorstore.add_documents(documents)

    async def add_texts(
        self,
        texts: list[str],
        metadatas: list[dict[str, Any]] | None = None,
    ) -> list[str]:
        """Add texts to the underlying vector store.

        Args:
            texts: Texts to add.
            metadatas: Optional metadata for each text.

        Returns:
            List of document IDs.
        """
        return await self.vectorstore.add_texts(texts, metadatas=metadatas)


def create_dense_retriever(
    embeddings: EmbeddingProvider | None = None,
    backend: str = "inmemory",
    **kwargs: Any,
) -> DenseRetriever:
    """Create a dense retriever with a new vector store.

    Convenience function for creating a retriever without
    manually creating a VectorStore.

    Args:
        embeddings: Embedding provider (default: OpenAI).
        backend: Backend type ("inmemory", "faiss", "chroma", etc.).
        **kwargs: Additional arguments for VectorStore.

    Returns:
        Configured DenseRetriever.

    Example:
        >>> retriever = create_dense_retriever()
        >>> await retriever.add_texts(["doc1", "doc2"])
        >>> docs = await retriever.retrieve("query")
    """
    from agenticflow.vectorstore import VectorStore

    store = VectorStore(embeddings=embeddings, **kwargs)
    return DenseRetriever(store)
