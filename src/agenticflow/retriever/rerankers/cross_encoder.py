"""Cross-encoder reranker using sentence-transformers.

Cross-encoders jointly encode query and document, providing
more accurate relevance scores than bi-encoders (embeddings).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from agenticflow.retriever.base import RetrievalResult
from agenticflow.retriever.rerankers.base import BaseReranker

if TYPE_CHECKING:
    from agenticflow.vectorstore import Document


class CrossEncoderReranker(BaseReranker):
    """Reranker using sentence-transformers cross-encoder.

    Cross-encoders provide high-quality relevance scores by
    jointly encoding query and document. They're slower than
    bi-encoders but more accurate for reranking.

    Requires: `uv add sentence-transformers`

    Popular models:
    - cross-encoder/ms-marco-MiniLM-L-6-v2 (fast, good quality)
    - cross-encoder/ms-marco-TinyBERT-L-2-v2 (fastest, slightly lower quality)
    - BAAI/bge-reranker-base (good balance)
    - BAAI/bge-reranker-large (best quality, slower)

    Example:
        >>> reranker = CrossEncoderReranker(model="cross-encoder/ms-marco-MiniLM-L-6-v2")
        >>> reranked = await reranker.rerank(query, documents, top_n=5)
    """

    _name: str = "cross_encoder"

    def __init__(
        self,
        model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        *,
        device: str | None = None,
        batch_size: int = 32,
        name: str | None = None,
    ) -> None:
        """Create a cross-encoder reranker.

        Args:
            model: HuggingFace model name or path.
            device: Device to run on ("cpu", "cuda", "mps"). Auto-detected if None.
            batch_size: Batch size for inference.
            name: Optional custom name.
        """
        self._model_name = model
        self._device = device
        self._batch_size = batch_size
        self._model = None  # Lazy loaded

        if name:
            self._name = name

    def _get_model(self):
        """Lazy load the cross-encoder model."""
        if self._model is None:
            try:
                from sentence_transformers import CrossEncoder
            except ImportError as e:
                raise ImportError(
                    "CrossEncoderReranker requires sentence-transformers. "
                    "Install with: uv add sentence-transformers"
                ) from e

            self._model = CrossEncoder(
                self._model_name,
                device=self._device,
            )
        return self._model

    async def rerank(
        self,
        query: str,
        documents: list[Document],
        top_n: int | None = None,
    ) -> list[RetrievalResult]:
        """Rerank documents using cross-encoder.

        Args:
            query: The search query.
            documents: Documents to rerank.
            top_n: Number of top documents to return.

        Returns:
            Reranked results sorted by relevance score.
        """
        if not documents:
            return []

        model = self._get_model()

        # Create query-document pairs
        pairs = [(query, doc.text) for doc in documents]

        # Get scores (run in thread pool to not block async)
        import asyncio
        loop = asyncio.get_event_loop()
        scores = await loop.run_in_executor(
            None,
            lambda: model.predict(pairs, batch_size=self._batch_size).tolist(),
        )

        # Combine with documents and sort
        doc_scores = list(zip(documents, scores, strict=False))
        doc_scores.sort(key=lambda x: x[1], reverse=True)

        # Apply top_n limit
        if top_n:
            doc_scores = doc_scores[:top_n]

        # Normalize scores to 0-1 using sigmoid
        import math
        def sigmoid(x: float) -> float:
            return 1 / (1 + math.exp(-x))

        results = []
        for doc, score in doc_scores:
            results.append(
                RetrievalResult(
                    document=doc,
                    score=sigmoid(score),  # Normalize to 0-1
                    retriever_name=self.name,
                    metadata={
                        "reranker": "cross_encoder",
                        "model": self._model_name,
                        "raw_score": score,
                    },
                )
            )

        return results
