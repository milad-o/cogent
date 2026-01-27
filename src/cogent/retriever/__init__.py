"""Retriever module for advanced document retrieval.

This module provides a comprehensive retrieval system with multiple strategies.

**Core Retrievers:**
- DenseRetriever: Vector similarity search (wraps VectorStore)
- BM25Retriever: Sparse lexical retrieval (BM25 algorithm)
- EnsembleRetriever: Combine N retrievers with fusion strategies
- HybridRetriever: Combine metadata search + content search

**Contextual Retrievers:**
- ParentDocumentRetriever: Index chunks, retrieve full parent documents
- SentenceWindowRetriever: Index sentences, return with surrounding context

**Specialized Indexes:**
- SummaryIndex: LLM-generated summaries for efficient retrieval
- TreeIndex: Hierarchical tree of summaries for large documents
- KeywordTableIndex: Inverted keyword index for fast lookup
- KnowledgeGraphIndex: Graph-based retrieval with KG integration
- HierarchicalIndex: Multi-level structure (doc → section → chunk)
- TimeBasedIndex: Time-decay scoring for recency-aware retrieval
- MultiRepresentationIndex: Multiple embeddings per document

**Advanced Retrievers:**
- SelfQueryRetriever: LLM-parsed natural language to structured filters

**Rerankers:**
- CrossEncoderReranker: Local cross-encoder models
- FlashRankReranker: Lightweight neural reranker
- CohereReranker: Cohere Rerank API
- LLMReranker: Any LLM for pointwise scoring
- ListwiseLLMReranker: LLM ranking all docs at once

**Utilities:**
- FusionStrategy: RRF, linear, max, voting strategies
- fuse_results: Combine results from multiple retrievers

Example:
    >>> from cogent.retriever import DenseRetriever, HybridRetriever
    >>> from cogent.documents import DocumentLoader, RecursiveCharacterSplitter
    >>> from cogent.vectorstore import VectorStore
    >>>
    >>> # Load and split documents
    >>> docs = await DocumentLoader().load_directory("./documents")
    >>> chunks = RecursiveCharacterSplitter(chunk_size=1000).split_documents(docs)
    >>>
    >>> # Create vector store and retriever
    >>> vs = VectorStore()
    >>> await vs.add_documents(chunks)
    >>> retriever = DenseRetriever(vs)
    >>>
    >>> # Retrieve
    >>> results = await retriever.retrieve("search query", k=5)
"""

# Re-export document types for backward compatibility
from cogent.documents import (
    LOADERS,
    CharacterSplitter,
    CodeSplitter,
    Document,
    DocumentLoader,
    HTMLSplitter,
    MarkdownSplitter,
    # Splitters
    RecursiveCharacterSplitter,
    SemanticSplitter,
    SentenceSplitter,
    TokenSplitter,
    load_documents,
    load_documents_sync,
    split_text,
)
from cogent.documents import (
    BaseSplitter as TextSplitter,
)
from cogent.retriever.base import (
    BaseRetriever,
    FusionStrategy,
    RetrievalResult,
    Retriever,
)
from cogent.retriever.contextual import (
    ParentDocumentRetriever,
    SentenceWindowRetriever,
)
from cogent.retriever.dense import DenseRetriever
from cogent.retriever.ensemble import EnsembleRetriever
from cogent.retriever.hierarchical import (
    HierarchicalIndex,
    HierarchyLevel,
    HierarchyNode,
)
from cogent.retriever.hybrid import (
    HybridRetriever,
    MetadataMatchMode,
    MetadataWeight,
)
from cogent.retriever.hyde import HyDERetriever
from cogent.retriever.multi_representation import (
    MultiRepresentationIndex,
    QueryType,
    RepresentationType,
)
from cogent.retriever.rerankers import (
    BaseReranker,
    CohereReranker,
    CrossEncoderReranker,
    ListwiseLLMReranker,
    LLMReranker,
    Reranker,
)
from cogent.retriever.self_query import (
    AttributeInfo,
    ParsedQuery,
    SelfQueryRetriever,
)
from cogent.retriever.sparse import BM25Retriever
from cogent.retriever.summary import (
    DocumentSummary,
    KeywordTableIndex,
    KnowledgeGraphIndex,
    SummaryIndex,
    TreeIndex,
)
from cogent.retriever.temporal import (
    DecayFunction,
    TimeBasedIndex,
    TimeRange,
)
from cogent.retriever.utils import (
    add_citations,
    deduplicate_results,
    filter_by_score,
    format_citations_reference,
    format_context,
    fuse_results,
    normalize_scores,
    top_k,
)
from cogent.retriever.utils.llm_adapter import (
    ChatModelAdapter,
    LLMProtocol,
    adapt_llm,
)

__all__ = [
    # Document types (re-exported from cogent.documents)
    "Document",
    "DocumentLoader",
    "LOADERS",
    "load_documents",
    "load_documents_sync",
    # Text Splitting (re-exported from cogent.documents)
    "TextSplitter",
    "RecursiveCharacterSplitter",
    "CharacterSplitter",
    "SentenceSplitter",
    "MarkdownSplitter",
    "HTMLSplitter",
    "CodeSplitter",
    "SemanticSplitter",
    "TokenSplitter",
    "split_text",
    # Core protocols
    "Retriever",
    "BaseRetriever",
    "FusionStrategy",
    "RetrievalResult",
    # Core retrievers
    "DenseRetriever",
    "BM25Retriever",
    "HybridRetriever",
    "MetadataMatchMode",
    "MetadataWeight",
    "EnsembleRetriever",
    "HyDERetriever",
    # Contextual retrievers
    "ParentDocumentRetriever",
    "SentenceWindowRetriever",
    # Summary-based indexes
    "SummaryIndex",
    "TreeIndex",
    "KeywordTableIndex",
    "KnowledgeGraphIndex",
    "DocumentSummary",
    # Hierarchical index
    "HierarchicalIndex",
    "HierarchyLevel",
    "HierarchyNode",
    # Temporal index
    "TimeBasedIndex",
    "DecayFunction",
    "TimeRange",
    # Multi-representation index
    "MultiRepresentationIndex",
    "QueryType",
    "RepresentationType",
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
    "add_citations",
    "format_context",
    "format_citations_reference",
    "filter_by_score",
    "top_k",
    # LLM adaptation (advanced usage)
    "ChatModelAdapter",
    "LLMProtocol",
    "adapt_llm",
]
