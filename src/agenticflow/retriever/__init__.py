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
    ...     HybridRetriever,
    ... )
    >>> from agenticflow.document import (
    ...     DocumentLoader,
    ...     RecursiveCharacterSplitter,
    ... )
    >>> from agenticflow.vectorstore import VectorStore
    >>> 
    >>> # Load and split documents
    >>> loader = DocumentLoader()
    >>> docs = await loader.load_directory("./documents")
    >>> 
    >>> splitter = RecursiveCharacterSplitter(chunk_size=1000)
    >>> chunks = splitter.split_documents(docs)
    >>> 
    >>> # Create retrievers
    >>> vs = VectorStore()
    >>> vs.add_texts([c.content for c in chunks])
    >>> retriever = DenseRetriever(vs)
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
from agenticflow.retriever.utils import deduplicate_results, fuse_results, normalize_scores

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
