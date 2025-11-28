"""FlashRank lightweight reranker.

FlashRank provides fast, lightweight neural reranking without
heavy dependencies. Good balance of speed and quality.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

from agenticflow.retriever.base import RetrievalResult
from agenticflow.retriever.rerankers.base import BaseReranker

if TYPE_CHECKING:
    from agenticflow.vectorstore import Document


class FlashRankReranker(BaseReranker):
    """Reranker using FlashRank library.
    
    FlashRank offers fast, lightweight neural reranking models.
    Much smaller and faster than full cross-encoders while
    maintaining good quality.
    
    Requires: `uv add flashrank`
    
    Available models:
    - ms-marco-TinyBERT-L-2-v2 (default, fastest, ~4MB)
    - ms-marco-MiniLM-L-12-v2 (balanced, ~33MB)
    - rank-T5-flan (larger, ~110MB)
    - ms-marco-MultiBERT-L-12 (multilingual)
    
    Example:
        >>> reranker = FlashRankReranker()
        >>> reranked = await reranker.rerank(query, documents, top_n=5)
    """
    
    _name: str = "flashrank"
    
    def __init__(
        self,
        model: str = "ms-marco-TinyBERT-L-2-v2",
        *,
        cache_dir: str | None = None,
        name: str | None = None,
    ) -> None:
        """Create a FlashRank reranker.
        
        Args:
            model: FlashRank model name.
            cache_dir: Directory to cache models.
            name: Optional custom name.
        """
        self._model_name = model
        self._cache_dir = cache_dir
        self._ranker = None  # Lazy loaded
        
        if name:
            self._name = name
    
    def _get_ranker(self):
        """Lazy load the FlashRank ranker."""
        if self._ranker is None:
            try:
                from flashrank import Ranker, RerankRequest  # noqa: F401
            except ImportError as e:
                raise ImportError(
                    "FlashRankReranker requires flashrank. "
                    "Install with: uv add flashrank"
                ) from e
            
            kwargs = {"model_name": self._model_name}
            if self._cache_dir:
                kwargs["cache_dir"] = self._cache_dir
            
            self._ranker = Ranker(**kwargs)
        return self._ranker
    
    async def rerank(
        self,
        query: str,
        documents: list[Document],
        top_n: int | None = None,
    ) -> list[RetrievalResult]:
        """Rerank documents using FlashRank.
        
        Args:
            query: The search query.
            documents: Documents to rerank.
            top_n: Number of top documents to return.
            
        Returns:
            Reranked results sorted by relevance score.
        """
        if not documents:
            return []
        
        try:
            from flashrank import RerankRequest
        except ImportError as e:
            raise ImportError(
                "FlashRankReranker requires flashrank. "
                "Install with: uv add flashrank"
            ) from e
        
        ranker = self._get_ranker()
        
        # Create passages in FlashRank format
        # FlashRank expects list of dicts with 'id', 'text', and optional 'meta'
        passages = []
        for i, doc in enumerate(documents):
            passage = {
                "id": doc.metadata.get("id", str(i)),
                "text": doc.text,
                "meta": doc.metadata,
            }
            passages.append(passage)
        
        # Create rerank request
        request = RerankRequest(query=query, passages=passages)
        
        # Run reranking in thread pool to not block async
        loop = asyncio.get_event_loop()
        rerank_results = await loop.run_in_executor(
            None,
            lambda: ranker.rerank(request),
        )
        
        # Apply top_n limit
        if top_n:
            rerank_results = rerank_results[:top_n]
        
        # Convert to RetrievalResult
        # FlashRank results have 'text', 'score', 'id', and 'meta'
        results = []
        for result in rerank_results:
            # Find the original document
            doc_idx = next(
                (i for i, d in enumerate(documents) 
                 if d.text == result["text"]),
                None
            )
            
            if doc_idx is not None:
                doc = documents[doc_idx]
            else:
                # Fallback: create new document from result
                from agenticflow.vectorstore import Document as DocClass
                doc = DocClass(
                    text=result["text"],
                    metadata=result.get("meta", {}),
                )
            
            results.append(
                RetrievalResult(
                    document=doc,
                    score=float(result["score"]),
                    retriever_name=self.name,
                    metadata={
                        "reranker": "flashrank",
                        "model": self._model_name,
                    },
                )
            )
        
        return results


class FlashRankRerankerLite(BaseReranker):
    """Ultra-lightweight FlashRank reranker for resource-constrained environments.
    
    Uses the smallest FlashRank model with additional optimizations
    for minimum memory footprint and fastest inference.
    
    Example:
        >>> reranker = FlashRankRerankerLite()
        >>> reranked = await reranker.rerank(query, documents, top_n=5)
    """
    
    _name: str = "flashrank_lite"
    
    def __init__(self, name: str | None = None) -> None:
        """Create lightweight FlashRank reranker."""
        # Use the smallest, fastest model
        self._inner = FlashRankReranker(
            model="ms-marco-TinyBERT-L-2-v2",
            name=name or self._name,
        )
    
    async def rerank(
        self,
        query: str,
        documents: list[Document],
        top_n: int | None = None,
    ) -> list[RetrievalResult]:
        """Rerank using lightweight model."""
        return await self._inner.rerank(query, documents, top_n)
    
    @property
    def name(self) -> str:
        """Get reranker name."""
        return self._inner.name
