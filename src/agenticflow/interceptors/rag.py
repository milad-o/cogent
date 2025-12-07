"""
RAG Interceptor - Automatic context injection for retrieval-augmented generation.

Injects retrieved context into the agent's messages before the first model call,
enabling "naive RAG" without the agent needing to call search tools.

Example:
    from agenticflow import Agent
    from agenticflow.interceptors import RAGInterceptor
    from agenticflow.retriever import DenseRetriever
    
    retriever = DenseRetriever(store)
    
    agent = Agent(
        name="assistant",
        model=model,
        intercept=[RAGInterceptor(retriever, k=5)],
    )
    
    # Context is automatically retrieved and injected
    answer = await agent.run("What are the key findings?")
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from agenticflow.interceptors.base import (
    Interceptor,
    InterceptContext,
    InterceptResult,
)
from agenticflow.retriever.utils.results import (
    add_citations,
    filter_by_score,
    format_context,
    format_citations_reference,
)

if TYPE_CHECKING:
    from agenticflow.retriever.base import Retriever


@dataclass
class RAGInterceptor(Interceptor):
    """Automatically retrieve and inject context before agent thinks.
    
    This interceptor enables "naive RAG" - retrieval happens automatically
    on the first model call, without the agent needing to decide to search.
    
    Benefits:
    - Single LLM call (cheaper than agentic RAG)
    - Deterministic retrieval (always retrieves)
    - Uses citation markers («1», «2») for source tracking
    
    Attributes:
        retriever: The retriever to use for context retrieval.
        k: Number of results to retrieve.
        min_score: Minimum score threshold (filters low-quality results).
        include_sources: Whether to append a sources reference section.
        context_template: Template for injecting context into the message.
        
    Example:
        ```python
        from agenticflow import Agent
        from agenticflow.interceptors import RAGInterceptor
        from agenticflow.retriever import DenseRetriever
        
        retriever = DenseRetriever(store)
        
        agent = Agent(
            name="assistant",
            model=model,
            intercept=[
                RAGInterceptor(
                    retriever=retriever,
                    k=5,
                    min_score=0.5,
                ),
            ],
        )
        
        # Context automatically injected - agent doesn't call search tools
        answer = await agent.run("What are the key findings?")
        ```
    """
    
    retriever: Retriever
    k: int = 5
    min_score: float | None = None
    include_sources: bool = True
    context_template: str = """Use the following context to answer the question.
Cite sources using markers like «1», «2» when referencing information.

Context:
{context}

{sources}
Question: {question}"""
    
    async def pre_think(self, ctx: InterceptContext) -> InterceptResult:
        """Retrieve and inject context before the first model call."""
        # Only inject on first model call
        if ctx.model_calls > 0:
            return InterceptResult.ok()
        
        # Retrieve relevant context
        results = await self.retriever.retrieve(
            ctx.task,
            k=self.k,
            include_scores=True,
        )
        
        # Filter by score if threshold set
        if self.min_score is not None:
            results = filter_by_score(results, self.min_score)
        
        # No results found
        if not results:
            return InterceptResult.ok()
        
        # Add citation markers
        results = add_citations(results)
        
        # Store results in state for potential post-processing
        ctx.state["rag_results"] = results
        ctx.state["rag_query"] = ctx.task
        
        # Format context
        context = format_context(results)
        sources = format_citations_reference(results) if self.include_sources else ""
        
        # Build the injected message
        injected_content = self.context_template.format(
            context=context,
            sources=sources,
            question=ctx.task,
        )
        
        # Replace the user message with context-injected version
        new_messages = []
        for msg in ctx.messages:
            if msg.get("role") == "user" and msg.get("content") == ctx.task:
                new_messages.append({
                    "role": "user",
                    "content": injected_content,
                })
            else:
                new_messages.append(msg)
        
        # If no user message was found, add one
        if not any(m.get("content") == injected_content for m in new_messages):
            new_messages.append({
                "role": "user",
                "content": injected_content,
            })
        
        return InterceptResult.modify_messages(new_messages)


@dataclass
class RAGPostProcessor(Interceptor):
    """Post-process agent output to append citation references.
    
    Use this with RAGInterceptor to automatically append a sources
    section to the agent's response.
    
    Example:
        ```python
        agent = Agent(
            model=model,
            intercept=[
                RAGInterceptor(retriever, k=5),
                RAGPostProcessor(),
            ],
        )
        ```
    """
    
    header: str = "\n\n---\nSources:"
    
    async def post_run(self, ctx: InterceptContext) -> InterceptResult:
        """Append sources to the final response."""
        results = ctx.state.get("rag_results")
        
        if not results:
            return InterceptResult.ok()
        
        # Format sources reference
        sources = format_citations_reference(results)
        
        # Note: post_run can't easily modify the final response string
        # This is stored for the caller to access if needed
        ctx.state["rag_sources_reference"] = sources
        
        return InterceptResult.ok()


__all__ = [
    "RAGInterceptor",
    "RAGPostProcessor",
]
