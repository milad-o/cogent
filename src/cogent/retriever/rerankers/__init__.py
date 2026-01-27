"""Rerankers for improving retrieval quality.

Rerankers take initial retrieval results and re-score them
using more sophisticated models for better relevance ranking.

Available rerankers:
- CrossEncoderReranker: Sentence-transformers cross-encoder (local)
- CohereReranker: Cohere Rerank API (cloud service)
- LLMReranker: Any LLM for scoring (flexible)
- ListwiseLLMReranker: LLM ranking all docs at once (efficient)
- FlashRankReranker: Lightweight neural reranker (local, fast)
"""

from cogent.retriever.rerankers.base import BaseReranker, Reranker
from cogent.retriever.rerankers.cohere import CohereReranker
from cogent.retriever.rerankers.cross_encoder import CrossEncoderReranker
from cogent.retriever.rerankers.flashrank import (
    FlashRankReranker,
    FlashRankRerankerLite,
)
from cogent.retriever.rerankers.llm import ListwiseLLMReranker, LLMReranker

__all__ = [
    # Protocols
    "Reranker",
    "BaseReranker",
    # Implementations
    "CrossEncoderReranker",
    "CohereReranker",
    "LLMReranker",
    "ListwiseLLMReranker",
    "FlashRankReranker",
    "FlashRankRerankerLite",
]
