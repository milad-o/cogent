"""Contextual retrieval strategies.

These retrievers enhance retrieval by leveraging document context:
- ParentDocumentRetriever: Store chunks, retrieve parent documents
- SentenceWindowRetriever: Retrieve sentences with surrounding context
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from agenticflow.retriever.base import BaseRetriever, RetrievalResult
from agenticflow.vectorstore import Document

if TYPE_CHECKING:
    from agenticflow.vectorstore import VectorStore


@dataclass
class DocumentChunk:
    """A chunk with reference to its parent document."""
    
    chunk_id: str
    parent_id: str
    chunk_index: int
    text: str
    metadata: dict = field(default_factory=dict)


class ParentDocumentRetriever(BaseRetriever):
    """Retriever that indexes chunks but returns parent documents.
    
    This strategy splits documents into small chunks for precise
    matching, but returns the full parent document for more context.
    
    Use case: When you need precise retrieval (small chunks match well)
    but your downstream task needs more context (full documents).
    
    Example:
        >>> retriever = ParentDocumentRetriever(
        ...     vectorstore=vectorstore,
        ...     chunk_size=500,
        ...     chunk_overlap=50,
        ... )
        >>> await retriever.add_documents(large_documents)
        >>> results = await retriever.retrieve("query", k=3)
        >>> # Results contain full parent documents, not chunks
    """
    
    _name: str = "parent_document"
    
    def __init__(
        self,
        vectorstore: VectorStore,
        *,
        chunk_size: int = 500,
        chunk_overlap: int = 50,
        k: int = 4,
        name: str | None = None,
    ) -> None:
        """Create a parent document retriever.
        
        Args:
            vectorstore: Vector store for chunk storage.
            chunk_size: Size of chunks in characters.
            chunk_overlap: Overlap between chunks.
            k: Default number of results.
            name: Optional custom name.
        """
        self._vectorstore = vectorstore
        self._chunk_size = chunk_size
        self._chunk_overlap = chunk_overlap
        self._k = k
        
        # Parent document store (in-memory for now)
        self._parents: dict[str, Document] = {}
        
        if name:
            self._name = name
    
    def _chunk_text(self, text: str) -> list[str]:
        """Split text into overlapping chunks."""
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + self._chunk_size
            chunk = text[start:end]
            
            # Try to break at sentence/word boundary
            if end < len(text):
                # Look for sentence end
                for sep in [". ", ".\n", "! ", "? ", "\n\n"]:
                    last_sep = chunk.rfind(sep)
                    if last_sep > self._chunk_size // 2:
                        chunk = chunk[:last_sep + len(sep)]
                        end = start + len(chunk)
                        break
            
            if chunk.strip():
                chunks.append(chunk.strip())
            
            start = end - self._chunk_overlap
            if start <= 0:
                start = end
        
        return chunks
    
    async def add_documents(
        self,
        documents: list[Document],
    ) -> None:
        """Add documents by chunking and indexing.
        
        Stores full documents and indexes chunks pointing to them.
        
        Args:
            documents: Documents to add.
        """
        import uuid
        
        chunks_to_add = []
        
        for doc in documents:
            # Generate parent ID
            parent_id = doc.metadata.get("id") or str(uuid.uuid4())
            
            # Store parent
            self._parents[parent_id] = doc
            
            # Create chunks
            text_chunks = self._chunk_text(doc.text)
            
            for i, chunk_text in enumerate(text_chunks):
                chunk_doc = Document(
                    text=chunk_text,
                    metadata={
                        **doc.metadata,
                        "parent_id": parent_id,
                        "chunk_index": i,
                        "total_chunks": len(text_chunks),
                    },
                )
                chunks_to_add.append(chunk_doc)
        
        # Add chunks to vector store
        await self._vectorstore.add_documents(chunks_to_add)
    
    async def retrieve(
        self,
        query: str,
        k: int | None = None,
        filter: dict | None = None,
    ) -> list[Document]:
        """Retrieve parent documents based on chunk matches.
        
        Args:
            query: Search query.
            k: Number of parents to return.
            filter: Optional metadata filter.
            
        Returns:
            Parent documents (not chunks).
        """
        k = k or self._k
        
        # Retrieve more chunks to find diverse parents
        chunk_results = await self._vectorstore.search(
            query=query,
            k=k * 3,  # Get more chunks
            filter=filter,
        )
        
        # Collect unique parents
        seen_parents = set()
        parents = []
        
        for result in chunk_results:
            chunk_doc = result.document
            parent_id = chunk_doc.metadata.get("parent_id")
            if parent_id and parent_id not in seen_parents:
                seen_parents.add(parent_id)
                if parent_id in self._parents:
                    parents.append(self._parents[parent_id])
                    
                    if len(parents) >= k:
                        break
        
        return parents
    
    async def retrieve_with_scores(
        self,
        query: str,
        k: int | None = None,
        filter: dict | None = None,
    ) -> list[RetrievalResult]:
        """Retrieve parent documents with aggregated scores.
        
        Args:
            query: Search query.
            k: Number of parents to return.
            filter: Optional metadata filter.
            
        Returns:
            Retrieval results with parent documents.
        """
        k = k or self._k
        
        # Retrieve chunks with scores
        chunk_results = await self._vectorstore.search(
            query=query,
            k=k * 3,
            filter=filter,
        )
        
        # Aggregate scores by parent
        parent_scores: dict[str, list[float]] = {}
        
        for result in chunk_results:
            chunk_doc = result.document
            score = result.score
            parent_id = chunk_doc.metadata.get("parent_id")
            if parent_id:
                if parent_id not in parent_scores:
                    parent_scores[parent_id] = []
                parent_scores[parent_id].append(score)
        
        # Build results with max score per parent
        results = []
        for parent_id, scores in parent_scores.items():
            if parent_id in self._parents:
                max_score = max(scores)
                results.append(
                    RetrievalResult(
                        document=self._parents[parent_id],
                        score=max_score,
                        retriever_name=self.name,
                        metadata={
                            "parent_id": parent_id,
                            "matching_chunks": len(scores),
                            "avg_chunk_score": sum(scores) / len(scores),
                        },
                    )
                )
        
        # Sort by score and limit
        results.sort(key=lambda r: r.score, reverse=True)
        return results[:k]


class SentenceWindowRetriever(BaseRetriever):
    """Retriever that indexes sentences but returns with context.
    
    Indexes individual sentences for precise matching, then
    returns surrounding context (window) for better understanding.
    
    Use case: When you need precise sentence-level matching
    but want to return paragraphs or larger context.
    
    Example:
        >>> retriever = SentenceWindowRetriever(
        ...     vectorstore=vectorstore,
        ...     window_size=2,  # 2 sentences before and after
        ... )
        >>> await retriever.add_documents(documents)
        >>> results = await retriever.retrieve("query")
        >>> # Results contain sentences with surrounding context
    """
    
    _name: str = "sentence_window"
    
    def __init__(
        self,
        vectorstore: VectorStore,
        *,
        window_size: int = 2,
        k: int = 4,
        name: str | None = None,
    ) -> None:
        """Create a sentence window retriever.
        
        Args:
            vectorstore: Vector store for sentence storage.
            window_size: Number of sentences before/after to include.
            k: Default number of results.
            name: Optional custom name.
        """
        self._vectorstore = vectorstore
        self._window_size = window_size
        self._k = k
        
        # Store sentences by document for window retrieval
        self._doc_sentences: dict[str, list[str]] = {}
        
        if name:
            self._name = name
    
    def _split_sentences(self, text: str) -> list[str]:
        """Split text into sentences."""
        import re
        
        # Simple sentence splitting
        # More sophisticated: use nltk or spacy
        sentences = re.split(r'(?<=[.!?])\s+', text)
        return [s.strip() for s in sentences if s.strip()]
    
    async def add_documents(
        self,
        documents: list[Document],
    ) -> None:
        """Add documents by indexing sentences.
        
        Args:
            documents: Documents to add.
        """
        import uuid
        
        sentences_to_add = []
        
        for doc in documents:
            doc_id = doc.metadata.get("id") or str(uuid.uuid4())
            
            # Split into sentences
            sentences = self._split_sentences(doc.text)
            
            # Store for window retrieval
            self._doc_sentences[doc_id] = sentences
            
            # Create sentence documents
            for i, sentence in enumerate(sentences):
                sent_doc = Document(
                    text=sentence,
                    metadata={
                        **doc.metadata,
                        "doc_id": doc_id,
                        "sentence_index": i,
                        "total_sentences": len(sentences),
                    },
                )
                sentences_to_add.append(sent_doc)
        
        await self._vectorstore.add_documents(sentences_to_add)
    
    def _get_window(
        self,
        doc_id: str,
        sentence_index: int,
    ) -> str:
        """Get sentence with surrounding window context."""
        sentences = self._doc_sentences.get(doc_id, [])
        if not sentences:
            return ""
        
        start = max(0, sentence_index - self._window_size)
        end = min(len(sentences), sentence_index + self._window_size + 1)
        
        return " ".join(sentences[start:end])
    
    async def retrieve(
        self,
        query: str,
        k: int | None = None,
        filter: dict | None = None,
    ) -> list[Document]:
        """Retrieve sentences with surrounding context.
        
        Args:
            query: Search query.
            k: Number of results.
            filter: Optional metadata filter.
            
        Returns:
            Documents containing sentence windows.
        """
        k = k or self._k
        
        # Search for matching sentences
        results = await self._vectorstore.search(
            query=query,
            k=k,
            filter=filter,
        )
        
        # Build windowed documents
        documents = []
        for result in results:
            sent_doc = result.document
            doc_id = sent_doc.metadata.get("doc_id")
            sent_idx = sent_doc.metadata.get("sentence_index", 0)
            
            window_text = self._get_window(doc_id, sent_idx)
            
            documents.append(
                Document(
                    text=window_text,
                    metadata={
                        **sent_doc.metadata,
                        "window_size": self._window_size,
                        "matched_sentence": sent_doc.text,
                    },
                )
            )
        
        return documents
    
    async def retrieve_with_scores(
        self,
        query: str,
        k: int | None = None,
        filter: dict | None = None,
    ) -> list[RetrievalResult]:
        """Retrieve sentence windows with scores.
        
        Args:
            query: Search query.
            k: Number of results.
            filter: Optional metadata filter.
            
        Returns:
            Retrieval results with windowed text.
        """
        k = k or self._k
        
        results = await self._vectorstore.search(
            query=query,
            k=k,
            filter=filter,
        )
        
        retrieval_results = []
        for result in results:
            sent_doc = result.document
            score = result.score
            doc_id = sent_doc.metadata.get("doc_id")
            sent_idx = sent_doc.metadata.get("sentence_index", 0)
            
            window_text = self._get_window(doc_id, sent_idx)
            
            retrieval_results.append(
                RetrievalResult(
                    document=Document(
                        text=window_text,
                        metadata={
                            **sent_doc.metadata,
                            "window_size": self._window_size,
                            "matched_sentence": sent_doc.text,
                        },
                    ),
                    score=score,
                    retriever_name=self.name,
                    metadata={
                        "doc_id": doc_id,
                        "sentence_index": sent_idx,
                    },
                )
            )
        
        return retrieval_results
