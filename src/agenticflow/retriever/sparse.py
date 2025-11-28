"""Sparse retriever using BM25 algorithm.

BM25 (Best Matching 25) is a lexical retrieval algorithm that
ranks documents based on term frequency and inverse document frequency.
It's fast, interpretable, and works well for keyword-based queries.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Any

from agenticflow.retriever.base import BaseRetriever, RetrievalResult
from agenticflow.vectorstore import Document


@dataclass
class BM25Index:
    """In-memory BM25 index for sparse retrieval.
    
    Implements the Okapi BM25 ranking function.
    
    Attributes:
        k1: Term frequency saturation parameter (default: 1.5).
        b: Length normalization parameter (default: 0.75).
    """
    
    k1: float = 1.5
    b: float = 0.75
    
    # Internal state
    _documents: list[Document] = field(default_factory=list)
    _doc_tokens: list[list[str]] = field(default_factory=list)
    _doc_freqs: dict[str, int] = field(default_factory=dict)
    _avg_doc_len: float = 0.0
    _idf: dict[str, float] = field(default_factory=dict)
    
    def _tokenize(self, text: str) -> list[str]:
        """Tokenize text into lowercase words.
        
        Simple whitespace tokenization with basic normalization.
        For production, consider using a proper tokenizer.
        """
        # Lowercase and split on non-alphanumeric
        text = text.lower()
        tokens = re.findall(r'\b\w+\b', text)
        return tokens
    
    def add_documents(self, documents: list[Document]) -> None:
        """Add documents to the index.
        
        Args:
            documents: Documents to index.
        """
        for doc in documents:
            tokens = self._tokenize(doc.text)
            self._documents.append(doc)
            self._doc_tokens.append(tokens)
            
            # Update document frequencies
            seen = set()
            for token in tokens:
                if token not in seen:
                    self._doc_freqs[token] = self._doc_freqs.get(token, 0) + 1
                    seen.add(token)
        
        # Recalculate average document length
        total_len = sum(len(tokens) for tokens in self._doc_tokens)
        self._avg_doc_len = total_len / len(self._doc_tokens) if self._doc_tokens else 0
        
        # Recalculate IDF values
        n = len(self._documents)
        for term, df in self._doc_freqs.items():
            # IDF with smoothing to avoid negative values
            self._idf[term] = math.log((n - df + 0.5) / (df + 0.5) + 1)
    
    def search(
        self,
        query: str,
        k: int = 4,
        filter: dict[str, Any] | None = None,
    ) -> list[tuple[Document, float]]:
        """Search for documents matching the query.
        
        Args:
            query: Search query.
            k: Number of results to return.
            filter: Optional metadata filter.
            
        Returns:
            List of (document, score) tuples.
        """
        if not self._documents:
            return []
        
        query_tokens = self._tokenize(query)
        scores: list[tuple[int, float]] = []
        
        for idx, doc_tokens in enumerate(self._doc_tokens):
            doc = self._documents[idx]
            
            # Apply metadata filter
            if filter:
                match = all(
                    doc.metadata.get(k) == v
                    for k, v in filter.items()
                )
                if not match:
                    continue
            
            # Calculate BM25 score
            score = self._score_document(query_tokens, doc_tokens)
            if score > 0:
                scores.append((idx, score))
        
        # Sort by score descending
        scores.sort(key=lambda x: x[1], reverse=True)
        
        # Return top k
        results = []
        for idx, score in scores[:k]:
            results.append((self._documents[idx], score))
        
        return results
    
    def _score_document(
        self,
        query_tokens: list[str],
        doc_tokens: list[str],
    ) -> float:
        """Calculate BM25 score for a document.
        
        Args:
            query_tokens: Tokenized query.
            doc_tokens: Tokenized document.
            
        Returns:
            BM25 score.
        """
        doc_len = len(doc_tokens)
        doc_term_freqs = Counter(doc_tokens)
        
        score = 0.0
        for term in query_tokens:
            if term not in self._idf:
                continue
            
            tf = doc_term_freqs.get(term, 0)
            idf = self._idf[term]
            
            # BM25 formula
            numerator = tf * (self.k1 + 1)
            denominator = tf + self.k1 * (
                1 - self.b + self.b * (doc_len / self._avg_doc_len)
            )
            
            score += idf * (numerator / denominator)
        
        return score
    
    def clear(self) -> None:
        """Clear the index."""
        self._documents = []
        self._doc_tokens = []
        self._doc_freqs = {}
        self._avg_doc_len = 0.0
        self._idf = {}
    
    def __len__(self) -> int:
        return len(self._documents)


class BM25Retriever(BaseRetriever):
    """Sparse retriever using BM25 algorithm.
    
    BM25 is effective for:
    - Keyword-based queries
    - When exact term matching is important
    - Fast retrieval without embeddings
    - Combining with dense retrieval (hybrid)
    
    Example:
        >>> retriever = BM25Retriever()
        >>> retriever.add_documents([
        ...     Document(text="Python is a programming language"),
        ...     Document(text="JavaScript runs in browsers"),
        ... ])
        >>> docs = await retriever.retrieve("Python programming")
    """
    
    _name: str = "bm25"
    
    def __init__(
        self,
        documents: list[Document] | None = None,
        *,
        k1: float = 1.5,
        b: float = 0.75,
        name: str | None = None,
        score_threshold: float | None = None,
    ) -> None:
        """Create a BM25 retriever.
        
        Args:
            documents: Initial documents to index.
            k1: Term frequency saturation (default: 1.5).
            b: Length normalization (default: 0.75).
            name: Optional custom name.
            score_threshold: Minimum score threshold.
        """
        self._index = BM25Index(k1=k1, b=b)
        self.score_threshold = score_threshold
        
        if name:
            self._name = name
        
        if documents:
            self._index.add_documents(documents)
    
    def add_documents(self, documents: list[Document]) -> None:
        """Add documents to the index.
        
        Args:
            documents: Documents to add.
        """
        self._index.add_documents(documents)
    
    async def index_documents(self, documents: list[Document]) -> None:
        """Async wrapper for adding documents to the index.
        
        Args:
            documents: Documents to index.
        """
        self._index.add_documents(documents)
    
    def add_texts(
        self,
        texts: list[str],
        metadatas: list[dict[str, Any]] | None = None,
    ) -> None:
        """Add texts to the index.
        
        Args:
            texts: Texts to add.
            metadatas: Optional metadata for each text.
        """
        metadatas = metadatas or [{}] * len(texts)
        documents = [
            Document(text=text, metadata=meta)
            for text, meta in zip(texts, metadatas)
        ]
        self._index.add_documents(documents)
    
    async def retrieve_with_scores(
        self,
        query: str,
        k: int = 4,
        filter: dict[str, Any] | None = None,
    ) -> list[RetrievalResult]:
        """Retrieve documents using BM25.
        
        Args:
            query: The search query.
            k: Number of documents to retrieve.
            filter: Optional metadata filter.
            
        Returns:
            List of RetrievalResult ordered by BM25 score.
        """
        search_results = self._index.search(query, k=k, filter=filter)
        
        # Normalize scores to 0-1 range if we have results
        max_score = max((s for _, s in search_results), default=1.0)
        if max_score == 0:
            max_score = 1.0
        
        results = []
        for doc, score in search_results:
            normalized_score = score / max_score
            
            # Apply threshold
            if self.score_threshold is not None and normalized_score < self.score_threshold:
                continue
            
            results.append(
                RetrievalResult(
                    document=doc,
                    score=normalized_score,
                    retriever_name=self.name,
                    metadata={"search_type": "sparse", "raw_score": score},
                )
            )
        
        return results
    
    def clear(self) -> None:
        """Clear the index."""
        self._index.clear()
    
    @property
    def document_count(self) -> int:
        """Number of indexed documents."""
        return len(self._index)


class TFIDFRetriever(BaseRetriever):
    """Sparse retriever using TF-IDF.
    
    TF-IDF (Term Frequency-Inverse Document Frequency) is a simpler
    alternative to BM25. Uses cosine similarity between TF-IDF vectors.
    
    Note: For most use cases, BM25Retriever is recommended as it
    generally performs better.
    """
    
    _name: str = "tfidf"
    
    def __init__(
        self,
        documents: list[Document] | None = None,
        *,
        name: str | None = None,
    ) -> None:
        """Create a TF-IDF retriever.
        
        Args:
            documents: Initial documents to index.
            name: Optional custom name.
        """
        self._documents: list[Document] = []
        self._doc_vectors: list[dict[str, float]] = []
        self._idf: dict[str, float] = {}
        
        if name:
            self._name = name
        
        if documents:
            self.add_documents(documents)
    
    def _tokenize(self, text: str) -> list[str]:
        """Tokenize text."""
        return re.findall(r'\b\w+\b', text.lower())
    
    def add_documents(self, documents: list[Document]) -> None:
        """Add documents to the index."""
        # First pass: count document frequencies
        doc_freqs: dict[str, int] = {}
        all_tokens: list[list[str]] = []
        
        for doc in documents:
            tokens = self._tokenize(doc.text)
            all_tokens.append(tokens)
            seen = set(tokens)
            for token in seen:
                doc_freqs[token] = doc_freqs.get(token, 0) + 1
        
        # Calculate IDF
        n = len(documents) + len(self._documents)
        for term, df in doc_freqs.items():
            self._idf[term] = math.log(n / (df + 1)) + 1
        
        # Second pass: create TF-IDF vectors
        for doc, tokens in zip(documents, all_tokens):
            term_freqs = Counter(tokens)
            doc_len = len(tokens)
            
            # TF-IDF vector
            vector: dict[str, float] = {}
            for term, tf in term_freqs.items():
                tfidf = (tf / doc_len) * self._idf.get(term, 0)
                if tfidf > 0:
                    vector[term] = tfidf
            
            self._documents.append(doc)
            self._doc_vectors.append(vector)
    
    async def retrieve_with_scores(
        self,
        query: str,
        k: int = 4,
        filter: dict[str, Any] | None = None,
    ) -> list[RetrievalResult]:
        """Retrieve using TF-IDF cosine similarity."""
        if not self._documents:
            return []
        
        # Create query vector
        tokens = self._tokenize(query)
        term_freqs = Counter(tokens)
        query_len = len(tokens)
        
        query_vector: dict[str, float] = {}
        for term, tf in term_freqs.items():
            tfidf = (tf / query_len) * self._idf.get(term, 0)
            if tfidf > 0:
                query_vector[term] = tfidf
        
        # Calculate cosine similarity with each document
        scores: list[tuple[int, float]] = []
        
        for idx, doc_vector in enumerate(self._doc_vectors):
            doc = self._documents[idx]
            
            # Apply filter
            if filter:
                match = all(doc.metadata.get(k) == v for k, v in filter.items())
                if not match:
                    continue
            
            # Cosine similarity
            score = self._cosine_similarity(query_vector, doc_vector)
            if score > 0:
                scores.append((idx, score))
        
        # Sort and return top k
        scores.sort(key=lambda x: x[1], reverse=True)
        
        results = []
        for idx, score in scores[:k]:
            results.append(
                RetrievalResult(
                    document=self._documents[idx],
                    score=score,
                    retriever_name=self.name,
                    metadata={"search_type": "tfidf"},
                )
            )
        
        return results
    
    def _cosine_similarity(
        self,
        vec1: dict[str, float],
        vec2: dict[str, float],
    ) -> float:
        """Calculate cosine similarity between two sparse vectors."""
        # Dot product
        dot = sum(vec1.get(k, 0) * vec2.get(k, 0) for k in vec1)
        
        # Magnitudes
        mag1 = math.sqrt(sum(v * v for v in vec1.values()))
        mag2 = math.sqrt(sum(v * v for v in vec2.values()))
        
        if mag1 == 0 or mag2 == 0:
            return 0.0
        
        return dot / (mag1 * mag2)
