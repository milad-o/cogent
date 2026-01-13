"""HyDE (Hypothetical Document Embeddings) Retriever.

HyDE improves retrieval by generating a hypothetical document that would
answer the query, then using that document's embedding for search.

This bridges the semantic gap between short queries and long documents:
- Query: "What is photosynthesis?"
- Hypothetical doc: "Photosynthesis is the process by which plants convert
  sunlight into chemical energy..."
- Search using the hypothetical doc's embedding finds better matches.

Reference: Gao et al. "Precise Zero-Shot Dense Retrieval without Relevance Labels"
https://arxiv.org/abs/2212.10496
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from agenticflow.retriever.base import BaseRetriever, RetrievalResult
from agenticflow.retriever.utils.llm_adapter import adapt_llm

if TYPE_CHECKING:
    from agenticflow.models import Model
    from agenticflow.retriever.utils.llm_adapter import LLMProtocol


# Default prompt for generating hypothetical documents
DEFAULT_HYDE_PROMPT = """Write a passage that would answer this question. 
Write as if you're writing a section of a document, not answering directly.
Be detailed and informative. Do not include phrases like "This passage discusses" or "This document explains".

Question: {query}

Passage:"""


class HyDERetriever(BaseRetriever):
    """HyDE (Hypothetical Document Embeddings) retriever.
    
    Improves retrieval by first generating a hypothetical document that
    would answer the query, then searching using that document's embedding.
    
    This is especially effective for:
    - Abstract or conceptual queries
    - Queries phrased differently than document content
    - Zero-shot retrieval without training data
    
    Attributes:
        base_retriever: The underlying retriever to use for search.
        model: LLM for generating hypothetical documents.
        prompt_template: Template for the generation prompt.
        n_hypotheticals: Number of hypothetical docs to generate (for ensemble).
        
    Example:
        >>> from agenticflow.retriever import DenseRetriever, HyDERetriever
        >>> from agenticflow.models import create_chat
        >>> 
        >>> base = DenseRetriever(vectorstore)
        >>> model = create_chat("openai", model="gpt-4o-mini")
        >>> hyde = HyDERetriever(base, model)
        >>> 
        >>> # Query is transformed into a hypothetical document before search
        >>> docs = await hyde.retrieve("What are the benefits of exercise?")
    """
    
    _name: str = "hyde"
    
    def __init__(
        self,
        base_retriever: BaseRetriever,
        model: Model,
        *,
        prompt_template: str | None = None,
        n_hypotheticals: int = 1,
        name: str | None = None,
        include_original_query: bool = False,
    ) -> None:
        """Create a HyDE retriever.
        
        Args:
            base_retriever: Underlying retriever for the actual search.
            model: LLM for generating hypothetical documents.
            prompt_template: Custom prompt template with {query} placeholder.
                Defaults to a general-purpose passage generation prompt.
            n_hypotheticals: Number of hypothetical documents to generate.
                If > 1, results from each are fused. Default is 1.
            name: Optional custom name for this retriever.
            include_original_query: If True, also search with original query
                and fuse results. Default is False.
        """
        self.base_retriever = base_retriever
        # Keep the original model for introspection/backward compatibility,
        # but use the internal adapter for consistent generation.
        self.model = model
        self._llm: LLMProtocol = adapt_llm(model)
        self.prompt_template = prompt_template or DEFAULT_HYDE_PROMPT
        self.n_hypotheticals = n_hypotheticals
        self.include_original_query = include_original_query
        if name:
            self._name = name
    
    async def generate_hypothetical(self, query: str) -> str:
        """Generate a hypothetical document for the query.
        
        Args:
            query: The user's query.
            
        Returns:
            A hypothetical document passage.
        """
        prompt = self.prompt_template.format(query=query)
        return await self._llm.generate(prompt)
    
    async def retrieve_with_scores(
        self,
        query: str,
        k: int = 4,
        filter: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> list[RetrievalResult]:
        """Retrieve documents using HyDE.
        
        1. Generate hypothetical document(s) from the query
        2. Search using hypothetical doc embedding(s)
        3. Optionally include original query results
        4. Fuse and deduplicate results
        
        Args:
            query: The search query.
            k: Number of documents to retrieve.
            filter: Optional metadata filter.
            **kwargs: Additional arguments passed to base retriever.
            
        Returns:
            List of RetrievalResult ordered by relevance.
        """
        import asyncio
        from agenticflow.retriever.utils import fuse_results, deduplicate_results
        from agenticflow.retriever.base import FusionStrategy
        
        # Generate hypothetical documents
        if self.n_hypotheticals == 1:
            hypotheticals = [await self.generate_hypothetical(query)]
        else:
            tasks = [self.generate_hypothetical(query) for _ in range(self.n_hypotheticals)]
            hypotheticals = await asyncio.gather(*tasks)
        
        # Search with each hypothetical document
        search_tasks = [
            self.base_retriever.retrieve(
                hypo, k=k, filter=filter, include_scores=True, **kwargs
            )
            for hypo in hypotheticals
        ]
        
        # Optionally include original query
        if self.include_original_query:
            search_tasks.append(
                self.base_retriever.retrieve(
                    query, k=k, filter=filter, include_scores=True, **kwargs
                )
            )
        
        all_results = await asyncio.gather(*search_tasks)
        
        # If only one search, just return those results
        if len(all_results) == 1:
            results = all_results[0]
            # Update retriever name
            for r in results:
                r.retriever_name = self.name
            return results
        
        # Fuse results from multiple searches using RRF
        fused = fuse_results(
            list(all_results),
            strategy=FusionStrategy.RRF,
            k=k,
        )
        
        # Deduplicate by document content
        deduped = deduplicate_results(fused)
        
        # Update retriever name and return top k
        for r in deduped:
            r.retriever_name = self.name
        
        return deduped[:k]
    
    def __repr__(self) -> str:
        return (
            f"HyDERetriever(base={self.base_retriever.name!r}, "
            f"n_hypotheticals={self.n_hypotheticals})"
        )
