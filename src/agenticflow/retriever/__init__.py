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
    >>> from agenticflow.retriever import DenseRetriever, HybridRetriever
    >>> from agenticflow.document import DocumentLoader, RecursiveCharacterSplitter
    >>> from agenticflow.vectorstore import VectorStore
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
from agenticflow.retriever.hybrid import HybridRetriever, MetadataMatchMode, MetadataWeight

# Re-export document types for backward compatibility
from agenticflow.document import (
    Document,
    DocumentLoader,
    TextChunk,
    load_documents,
    load_documents_sync,
    # Splitters
    RecursiveCharacterSplitter,
    CharacterSplitter,
    SentenceSplitter,
    MarkdownSplitter,
    HTMLSplitter,
    CodeSplitter,
    SemanticSplitter,
    TokenSplitter,
    BaseSplitter as TextSplitter,
    split_text,
    LOADERS,
)
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
from agenticflow.retriever.summary import (
    DocumentSummary,
    KeywordTableIndex,
    KnowledgeGraphIndex,
    SummaryIndex,
    TreeIndex,
)
from agenticflow.retriever.hierarchical import (
    HierarchicalIndex,
    HierarchyLevel,
    HierarchyNode,
)
from agenticflow.retriever.temporal import (
    DecayFunction,
    TimeBasedIndex,
    TimeRange,
)
from agenticflow.retriever.multi_representation import (
    MultiRepresentationIndex,
    QueryType,
    RepresentationType,
)
from agenticflow.retriever.utils import deduplicate_results, fuse_results, normalize_scores, add_citations, format_context, format_citations_reference, filter_by_score, top_k

__all__ = [
    # Document types (re-exported from agenticflow.document)
    "Document",
    "DocumentLoader",
    "LOADERS",
    "load_documents",
    "load_documents_sync",
    # Text Splitting (re-exported from agenticflow.document)
    "TextChunk",
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
]
