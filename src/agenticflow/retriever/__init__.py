"""Retriever module for advanced document retrieval.

This module provides a comprehensive retrieval system with:

**Document Loading:**
- DocumentLoader: Universal loader for multiple file types
- Supports: PDF, DOCX, TXT, MD, HTML, CSV, JSON, XLSX, code files

**Text Splitting:**
- RecursiveCharacterSplitter: Hierarchical separator-based splitting
- SentenceSplitter: Sentence boundary detection
- MarkdownSplitter: Markdown structure-aware splitting
- HTMLSplitter: HTML tag-based splitting  
- CodeSplitter: Language-aware code splitting
- SemanticSplitter: Embedding-based semantic chunking
- TokenSplitter: Token count-based splitting

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
    ...     DocumentLoader,
    ...     RecursiveCharacterSplitter,
    ...     DenseRetriever,
    ...     HybridRetriever,
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
from agenticflow.retriever.loaders import (
    Document,
    DocumentLoader,
    LOADERS,
    load_documents,
    load_documents_sync,
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
from agenticflow.retriever.splitters import (
    CharacterSplitter,
    CodeSplitter,
    HTMLSplitter,
    MarkdownSplitter,
    RecursiveCharacterSplitter,
    SemanticSplitter,
    SentenceSplitter,
    TextChunk,
    TextSplitter,
    TokenSplitter,
    split_text,
)
from agenticflow.retriever.utils import deduplicate_results, fuse_results, normalize_scores

__all__ = [
    # Document Loading
    "Document",
    "DocumentLoader",
    "LOADERS",
    "load_documents",
    "load_documents_sync",
    # Text Splitting
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
