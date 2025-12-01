"""Semantic similarity-based text splitter."""

from __future__ import annotations

import math
from typing import Any

from agenticflow.document.splitters.base import BaseSplitter
from agenticflow.document.splitters.character import RecursiveCharacterSplitter
from agenticflow.document.splitters.sentence import SentenceSplitter
from agenticflow.document.types import TextChunk


class SemanticSplitter(BaseSplitter):
    """Split text by semantic similarity using embeddings.
    
    Groups semantically similar sentences together and splits
    when semantic similarity drops below threshold.
    
    Args:
        embedding_model: Embedding model for computing similarity.
        chunk_size: Target chunk size.
        breakpoint_threshold: Similarity threshold for splits (0-1).
        buffer_size: Number of sentences to compare.
        
    Example:
        >>> from agenticflow.models import EmbeddingModel
        >>> embedder = EmbeddingModel()
        >>> splitter = SemanticSplitter(embedding_model=embedder)
        >>> chunks = await splitter.asplit_text(text)
    """
    
    def __init__(
        self,
        embedding_model: Any = None,
        chunk_size: int = 1000,
        chunk_overlap: int = 0,
        breakpoint_threshold: float = 0.5,
        buffer_size: int = 1,
        **kwargs: Any,
    ):
        super().__init__(chunk_size=chunk_size, chunk_overlap=chunk_overlap, **kwargs)
        self._embedding_model = embedding_model
        self.breakpoint_threshold = breakpoint_threshold
        self.buffer_size = buffer_size
    
    @property
    def embedding_model(self) -> Any:
        """Get or create embedding model."""
        if self._embedding_model is None:
            from agenticflow.models import EmbeddingModel
            self._embedding_model = EmbeddingModel()
        return self._embedding_model
    
    def split_text(self, text: str) -> list[TextChunk]:
        """Synchronous split - falls back to sentence splitting.
        
        For true semantic splitting, use asplit_text() instead.
        """
        sentence_splitter = SentenceSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
        )
        return sentence_splitter.split_text(text)
    
    async def asplit_text(self, text: str) -> list[TextChunk]:
        """Asynchronously split text by semantic similarity."""
        # First split into sentences
        sentence_splitter = SentenceSplitter(min_sentence_length=5)
        sentence_chunks = sentence_splitter.split_text(text)
        sentences = [c.content for c in sentence_chunks]
        
        if len(sentences) <= 1:
            return [TextChunk(text=text, metadata={"chunk_index": 0})]
        
        # Get embeddings for all sentences
        embeddings = await self.embedding_model.aembed(sentences)
        
        # Find breakpoints based on similarity
        breakpoints = self._find_breakpoints(embeddings)
        
        # Group sentences by breakpoints
        chunks: list[TextChunk] = []
        current_sentences: list[str] = []
        
        for i, sentence in enumerate(sentences):
            current_sentences.append(sentence)
            
            if i in breakpoints or i == len(sentences) - 1:
                content = " ".join(current_sentences)
                
                # Check size and split if needed
                if self.length_function(content) > self.chunk_size:
                    # Split large chunk with recursive splitter
                    sub_splitter = RecursiveCharacterSplitter(
                        chunk_size=self.chunk_size,
                        chunk_overlap=self.chunk_overlap,
                    )
                    sub_chunks = sub_splitter.split_text(content)
                    chunks.extend(sub_chunks)
                else:
                    chunks.append(TextChunk(
                        text=content,
                        metadata={"chunk_index": len(chunks)},
                    ))
                
                current_sentences = []
        
        # Renumber
        for i, chunk in enumerate(chunks):
            chunk.metadata["chunk_index"] = i
        
        return chunks
    
    def _find_breakpoints(self, embeddings: list[list[float]]) -> set[int]:
        """Find indices where semantic similarity drops."""
        breakpoints: set[int] = set()
        
        for i in range(1, len(embeddings)):
            # Compare with previous buffer
            start = max(0, i - self.buffer_size)
            prev_embeddings = embeddings[start:i]
            curr_embedding = embeddings[i]
            
            # Compute average similarity with previous sentences
            similarities = []
            for prev_emb in prev_embeddings:
                sim = self._cosine_similarity(prev_emb, curr_embedding)
                similarities.append(sim)
            
            avg_similarity = sum(similarities) / len(similarities) if similarities else 1.0
            
            if avg_similarity < self.breakpoint_threshold:
                breakpoints.add(i - 1)  # Break after previous sentence
        
        return breakpoints
    
    def _cosine_similarity(self, a: list[float], b: list[float]) -> float:
        """Compute cosine similarity between two vectors."""
        dot_product = sum(x * y for x, y in zip(a, b))
        norm_a = math.sqrt(sum(x * x for x in a))
        norm_b = math.sqrt(sum(y * y for y in b))
        
        if norm_a == 0 or norm_b == 0:
            return 0.0
        
        return dot_product / (norm_a * norm_b)


__all__ = ["SemanticSplitter"]
