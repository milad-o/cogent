"""Retriever module for advanced document retrieval.

This module provides a comprehensive retrieval system with:

**Core Retrievers:**
- DenseRetriever: Vector similarity search
- BM25Retriever: Sparse lexical retrieval (BM25 algorithm)
- HybridRetriever: Combines dense and sparse for best of both
- EnsembleRetriever: Combine any N retrievers with fusion

**Contextual Retrievers:**
- ParentDocumentRetriever: Index chunks, retrieve full documents
- SentenceWindowRetriever: Index sentences, return with context

**Advanced Retrievers:**
- SelfQueryRetriever: LLM-parsed filters from natural language

**Rerankers:**
- CrossEncoderReranker: Local cross-encoder models
- CohereReranker: Cohere Rerank API
- LLMReranker: Any LLM for scoring
- ListwiseLLMReranker: LLM ranking all docs at once

**Utilities:**
- FusionStrategy: RRF, linear, max, voting
- fuse_results: Combine results from multiple retrievers

Example:
    >>> from agenticflow.retriever import (
    ...     DenseRetriever,
    ...     BM25Retriever,
    ...     HybridRetriever,
    ...     CrossEncoderReranker,
    ... )
    >>> from agenticflow.vectorstore import VectorStore
    >>> 
    >>> # Create retrievers
    >>> vs = VectorStore()
    >>> dense = DenseRetriever(vs)
    >>> sparse = BM25Retriever()
    >>> 
    >>> # Combine into hybrid
    >>> hybrid = HybridRetriever(dense, sparse, dense_weight=0.7)
    >>> 
    >>> # Add reranking
    >>> reranker = CrossEncoderReranker()
    >>> results = await hybrid.retrieve(query, k=20)
    >>> reranked = await reranker.rerank(query, results, top_n=5)
"""

from agenticflow.retriever.base import (
    BaseRetriever,
    FusionStrategy,
    RetrievalResult,
    Retriever,
)
from agenticflow.retriever.contextual import (
    ParentDocumentRetriever,
    SentenceWindowRetriever,
)
from agenticflow.retriever.dense import DenseRetriever
from agenticflow.retriever.ensemble import EnsembleRetriever
from agenticflow.retriever.hybrid import HybridRetriever
from agenticflow.retriever.rerankers import (
    BaseReranker,
    CohereReranker,
    CrossEncoderReranker,
    ListwiseLLMReranker,
    LLMReranker,
    Reranker,
)
from agenticflow.retriever.self_query import (
    AttributeInfo,
    ParsedQuery,
    SelfQueryRetriever,
)
from agenticflow.retriever.sparse import BM25Retriever
from agenticflow.retriever.utils import deduplicate_results, fuse_results, normalize_scores

__all__ = [
    # Core protocols
    "Retriever",
    "BaseRetriever",
    "FusionStrategy",
    "RetrievalResult",
    # Core retrievers
    "DenseRetriever",
    "BM25Retriever",
    "HybridRetriever",
    "EnsembleRetriever",
    # Contextual retrievers
    "ParentDocumentRetriever",
    "SentenceWindowRetriever",
    # Advanced retrievers
    "SelfQueryRetriever",
    "AttributeInfo",
    "ParsedQuery",
    # Rerankers
    "Reranker",
    "BaseReranker",
    "CrossEncoderReranker",
    "CohereReranker",
    "LLMReranker",
    "ListwiseLLMReranker",
    # Utilities
    "fuse_results",
    "normalize_scores",
    "deduplicate_results",
]
