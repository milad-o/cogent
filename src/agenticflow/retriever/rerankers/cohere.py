"""Cohere reranker using the Cohere Rerank API.

Cohere provides a high-quality reranking service that's
easy to use and doesn't require local GPU resources.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from agenticflow.retriever.base import RetrievalResult
from agenticflow.retriever.rerankers.base import BaseReranker

if TYPE_CHECKING:
    from agenticflow.vectorstore import Document


class CohereReranker(BaseReranker):
    """Reranker using Cohere's Rerank API.
    
    Cohere provides state-of-the-art reranking as a service.
    No local resources required, just an API key.
    
    Requires: `uv add cohere`
    Set COHERE_API_KEY environment variable.
    
    Models:
    - rerank-english-v3.0 (English, best quality)
    - rerank-multilingual-v3.0 (100+ languages)
    - rerank-english-v2.0 (English, legacy)
    
    Example:
        >>> reranker = CohereReranker(model="rerank-english-v3.0")
        >>> reranked = await reranker.rerank(query, documents, top_n=5)
    """
    
    _name: str = "cohere"
    
    def __init__(
        self,
        model: str = "rerank-english-v3.0",
        *,
        api_key: str | None = None,
        name: str | None = None,
    ) -> None:
        """Create a Cohere reranker.
        
        Args:
            model: Cohere rerank model name.
            api_key: Cohere API key (or set COHERE_API_KEY env var).
            name: Optional custom name.
        """
        self._model = model
        self._api_key = api_key
        self._client = None  # Lazy loaded
        
        if name:
            self._name = name
    
    def _get_client(self):
        """Lazy load the Cohere client."""
        if self._client is None:
            try:
                import cohere
            except ImportError as e:
                raise ImportError(
                    "CohereReranker requires cohere. "
                    "Install with: uv add cohere"
                ) from e
            
            import os
            api_key = self._api_key or os.environ.get("COHERE_API_KEY")
            if not api_key:
                raise ValueError(
                    "Cohere API key required. Set COHERE_API_KEY or pass api_key."
                )
            
            self._client = cohere.AsyncClient(api_key=api_key)
        return self._client
    
    async def rerank(
        self,
        query: str,
        documents: list[Document],
        top_n: int | None = None,
    ) -> list[RetrievalResult]:
        """Rerank documents using Cohere API.
        
        Args:
            query: The search query.
            documents: Documents to rerank.
            top_n: Number of top documents to return.
            
        Returns:
            Reranked results sorted by relevance score.
        """
        if not documents:
            return []
        
        client = self._get_client()
        
        # Extract texts for API
        texts = [doc.text for doc in documents]
        
        # Call Cohere rerank API
        response = await client.rerank(
            model=self._model,
            query=query,
            documents=texts,
            top_n=top_n or len(documents),
        )
        
        # Build results
        results = []
        for result in response.results:
            doc = documents[result.index]
            results.append(
                RetrievalResult(
                    document=doc,
                    score=result.relevance_score,
                    retriever_name=self.name,
                    metadata={
                        "reranker": "cohere",
                        "model": self._model,
                        "original_index": result.index,
                    },
                )
            )
        
        return results
