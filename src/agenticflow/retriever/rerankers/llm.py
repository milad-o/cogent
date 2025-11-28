"""LLM-based reranker using language models for scoring.

Uses an LLM to score document relevance to a query.
Flexible and customizable but slower than specialized rerankers.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

from agenticflow.retriever.base import RetrievalResult
from agenticflow.retriever.rerankers.base import BaseReranker

if TYPE_CHECKING:
    from agenticflow.models import Model
    from agenticflow.vectorstore import Document


# Default prompt for relevance scoring
DEFAULT_RERANK_PROMPT = """Rate the relevance of the following document to the query.

Query: {query}

Document: {document}

On a scale of 0 to 10, where 0 means completely irrelevant and 10 means perfectly relevant, 
provide only a single number as your response. Do not include any explanation."""


class LLMReranker(BaseReranker):
    """Reranker using LLM for relevance scoring.
    
    Uses a language model to score each document's relevance
    to the query. Flexible but slower than specialized models.
    
    Supports batched scoring for efficiency and custom prompts.
    
    Example:
        >>> from agenticflow.models import OpenAIModel
        >>> model = OpenAIModel(model="gpt-4o-mini")
        >>> reranker = LLMReranker(model=model)
        >>> reranked = await reranker.rerank(query, documents, top_n=5)
    
    Custom prompt example:
        >>> reranker = LLMReranker(
        ...     model=model,
        ...     prompt="Score relevance 0-10 for query: {query}\\nDoc: {document}"
        ... )
    """
    
    _name: str = "llm"
    
    def __init__(
        self,
        model: Model,
        *,
        prompt: str | None = None,
        batch_size: int = 5,
        max_concurrent: int = 3,
        name: str | None = None,
    ) -> None:
        """Create an LLM reranker.
        
        Args:
            model: LLM model to use for scoring.
            prompt: Custom prompt template with {query} and {document}.
            batch_size: Number of documents per prompt (for batched mode).
            max_concurrent: Max concurrent API calls.
            name: Optional custom name.
        """
        self._model = model
        self._prompt = prompt or DEFAULT_RERANK_PROMPT
        self._batch_size = batch_size
        self._max_concurrent = max_concurrent
        
        if name:
            self._name = name
    
    async def _score_document(
        self,
        query: str,
        document: Document,
        semaphore: asyncio.Semaphore,
    ) -> float:
        """Score a single document's relevance."""
        async with semaphore:
            prompt = self._prompt.format(
                query=query,
                document=document.text[:2000],  # Truncate for token limits
            )
            
            response = await self._model.generate(prompt)
            
            # Parse numeric score from response
            try:
                # Extract first number from response
                import re
                match = re.search(r"(\d+(?:\.\d+)?)", response.strip())
                if match:
                    score = float(match.group(1))
                    # Normalize to 0-1 range
                    return min(score / 10.0, 1.0)
                return 0.5  # Default if parsing fails
            except (ValueError, AttributeError):
                return 0.5
    
    async def rerank(
        self,
        query: str,
        documents: list[Document],
        top_n: int | None = None,
    ) -> list[RetrievalResult]:
        """Rerank documents using LLM scoring.
        
        Args:
            query: The search query.
            documents: Documents to rerank.
            top_n: Number of top documents to return.
            
        Returns:
            Reranked results sorted by LLM relevance scores.
        """
        if not documents:
            return []
        
        # Score all documents concurrently with rate limiting
        semaphore = asyncio.Semaphore(self._max_concurrent)
        
        tasks = [
            self._score_document(query, doc, semaphore)
            for doc in documents
        ]
        
        scores = await asyncio.gather(*tasks)
        
        # Build results
        results = [
            RetrievalResult(
                document=doc,
                score=score,
                retriever_name=self.name,
                metadata={
                    "reranker": "llm",
                    "model": getattr(self._model, "model", "unknown"),
                },
            )
            for doc, score in zip(documents, scores)
        ]
        
        # Sort by score descending
        results.sort(key=lambda r: r.score, reverse=True)
        
        # Return top_n
        if top_n:
            results = results[:top_n]
        
        return results


class ListwiseLLMReranker(BaseReranker):
    """Reranker using LLM to rank all documents at once.
    
    More efficient than scoring each document individually.
    Asks the LLM to rank all documents in a single call.
    
    Best for small document sets (< 20 documents).
    
    Example:
        >>> from agenticflow.models import OpenAIModel
        >>> model = OpenAIModel(model="gpt-4o")
        >>> reranker = ListwiseLLMReranker(model=model)
        >>> reranked = await reranker.rerank(query, documents, top_n=5)
    """
    
    _name: str = "listwise_llm"
    
    def __init__(
        self,
        model: Model,
        *,
        max_documents: int = 20,
        name: str | None = None,
    ) -> None:
        """Create a listwise LLM reranker.
        
        Args:
            model: LLM model to use for ranking.
            max_documents: Maximum documents to rank at once.
            name: Optional custom name.
        """
        self._model = model
        self._max_documents = max_documents
        
        if name:
            self._name = name
    
    async def rerank(
        self,
        query: str,
        documents: list[Document],
        top_n: int | None = None,
    ) -> list[RetrievalResult]:
        """Rerank documents using listwise LLM ranking.
        
        Args:
            query: The search query.
            documents: Documents to rerank.
            top_n: Number of top documents to return.
            
        Returns:
            Reranked results based on LLM ordering.
        """
        if not documents:
            return []
        
        # Limit to max documents
        docs_to_rank = documents[:self._max_documents]
        
        # Build prompt with numbered documents
        doc_list = "\n\n".join(
            f"[{i+1}] {doc.text[:500]}"  # Truncate for token limits
            for i, doc in enumerate(docs_to_rank)
        )
        
        prompt = f"""Rank the following documents by relevance to the query.

Query: {query}

Documents:
{doc_list}

Return only the document numbers in order of relevance, most relevant first.
Format: comma-separated numbers (e.g., 3,1,5,2,4)"""
        
        response = await self._model.generate(prompt)
        
        # Parse ranking
        import re
        numbers = re.findall(r"\d+", response)
        
        # Build ordered results
        results = []
        seen = set()
        
        for i, num_str in enumerate(numbers):
            try:
                idx = int(num_str) - 1  # Convert to 0-indexed
                if 0 <= idx < len(docs_to_rank) and idx not in seen:
                    seen.add(idx)
                    # Score decreases with rank
                    score = 1.0 - (i / len(numbers))
                    results.append(
                        RetrievalResult(
                            document=docs_to_rank[idx],
                            score=score,
                            retriever_name=self.name,
                            metadata={
                                "reranker": "listwise_llm",
                                "rank": i + 1,
                                "model": getattr(self._model, "model", "unknown"),
                            },
                        )
                    )
            except (ValueError, IndexError):
                continue
        
        # Add any documents not in ranking at the end
        for i, doc in enumerate(docs_to_rank):
            if i not in seen:
                results.append(
                    RetrievalResult(
                        document=doc,
                        score=0.0,
                        retriever_name=self.name,
                        metadata={"reranker": "listwise_llm", "rank": None},
                    )
                )
        
        # Return top_n
        if top_n:
            results = results[:top_n]
        
        return results
