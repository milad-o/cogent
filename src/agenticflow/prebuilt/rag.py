"""
Prebuilt RAG (Retrieval-Augmented Generation) Agent.

RAGAgent is Agent + RAG capability with a unified run() API.

Example:
    ```python
    from agenticflow.prebuilt import RAGAgent
    
    rag = RAGAgent(model=model, embeddings=embeddings)
    await rag.load("docs/", "report.pdf")
    
    # Unified run() API
    answer = await rag.run("What are the key findings?")
    
    # With structured citations
    response = await rag.run("What is X?", citations=True)
    print(response.answer)
    print(response.format_bibliography())
    
    # Direct vectorstore access
    results = await rag.vectorstore.search("query", k=5)
    ```
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any, overload

from agenticflow import Agent
from agenticflow.capabilities.rag import (
    RAG,
    RAGConfig,
    RAGResponse,
    CitedPassage,
    DocumentPipeline,
    PipelineRegistry,
)

if TYPE_CHECKING:
    from agenticflow.agent.memory import AgentMemory, MemorySaver, MemoryStore
    from agenticflow.agent.resilience import ResilienceConfig
    from agenticflow.context import RunContext
    from agenticflow.models.base import BaseChatModel
    from agenticflow.retriever.base import Retriever
    from agenticflow.retriever.rerankers.base import Reranker
    from agenticflow.vectorstore import VectorStore
    from agenticflow.vectorstore.base import EmbeddingProvider


class RAGAgent(Agent):
    """
    RAG Agent with unified run() API.
    
    Inherits from Agent with RAG capability pre-attached.
    
    Unified API:
        - `run(query)` → str (agent uses tools intelligently)
        - `run(query, citations=True)` → RAGResponse (structured)
    
    Direct access to components:
        - `rag.vectorstore` - VectorStore for direct search
        - `rag.retriever` - Retriever with optional reranking
        - `rag.citations` - Citations from last search
    
    Example:
        ```python
        rag = RAGAgent(model=model, embeddings=embeddings)
        await rag.load("docs/")
        
        # Agent decides how to search
        answer = await rag.run("What is the main topic?")
        
        # Get structured citations
        response = await rag.run("Key findings?", citations=True)
        print(response.answer)
        for c in response.citations:
            print(c.format_full())
        
        # Direct vectorstore search
        results = await rag.vectorstore.search("query", k=10)
        ```
    """
    
    def __init__(
        self,
        model: BaseChatModel,
        *,
        embeddings: EmbeddingProvider,
        pipelines: PipelineRegistry | None = None,
        vectorstore: VectorStore | None = None,
        retriever: Retriever | None = None,
        reranker: Reranker | None = None,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        top_k: int = 4,
        backend: str = "inmemory",
        show_progress: bool = True,
        # Agent parameters
        name: str = "RAG_Assistant",
        instructions: str | None = None,
        tools: Sequence[Any] | None = None,
        capabilities: Sequence[Any] | None = None,
        memory: bool | MemorySaver | AgentMemory | None = None,
        store: MemoryStore | None = None,
        intercept: Sequence[Any] | None = None,
        stream: bool = False,
        reasoning: bool | Any = False,
        output: type | dict | None = None,
        verbose: bool | str = False,
        resilience: ResilienceConfig | None = None,
        interrupt_on: dict[str, Any] | None = None,
        observer: Any | None = None,
        taskboard: bool | Any = None,
    ) -> None:
        """
        Create a RAG agent.
        
        Args:
            model: Chat model for generation.
            embeddings: Embedding provider for vector search.
            
            **RAG Configuration:**
            pipelines: Per-file-type processing pipelines.
            vectorstore: Pre-configured VectorStore.
            retriever: Pre-configured retriever.
            reranker: Reranker for two-stage retrieval.
            chunk_size: Chunk size for splitting (default: 1000).
            chunk_overlap: Chunk overlap (default: 200).
            top_k: Passages to retrieve (default: 4).
            backend: Vector backend ("inmemory", "faiss", "chroma").
            show_progress: Print progress during loading.
            
            **Agent parameters:** All standard Agent params supported.
        """
        config = RAGConfig(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            top_k=top_k,
            backend=backend,
            show_progress=show_progress,
        )
        
        self._rag = RAG(
            embeddings=embeddings,
            config=config,
            pipelines=pipelines,
            vectorstore=vectorstore,
            retriever=retriever,
            reranker=reranker,
        )
        
        all_capabilities = [self._rag]
        if capabilities:
            all_capabilities.extend(capabilities)
        
        super().__init__(
            name=name,
            model=model,
            instructions=instructions or RAG.DEFAULT_INSTRUCTIONS,
            tools=tools or [],
            capabilities=all_capabilities,
            memory=memory,
            store=store,
            intercept=intercept,
            stream=stream,
            reasoning=reasoning,
            output=output,
            verbose=verbose,
            resilience=resilience,
            interrupt_on=interrupt_on,
            observer=observer,
            taskboard=taskboard,
        )
    
    async def _ensure_initialized(self) -> None:
        """Ensure RAG capability is initialized."""
        if self._rag._agent is None:
            await self._rag.initialize(self)
    
    # ================================================================
    # Document Loading
    # ================================================================
    
    async def load(self, *sources: str | Path, glob: str | None = None) -> None:
        """Load documents from files, directories, or URLs.
        
        Args:
            *sources: File paths, directory paths, or URLs.
            glob: Glob pattern for directories.
        """
        await self._ensure_initialized()
        await self._rag.load(*sources, glob=glob)
    
    async def load_text(self, text: str, source: str = "text") -> None:
        """Load text content directly."""
        await self._ensure_initialized()
        await self._rag.load_text(text, source)
    
    # Legacy aliases
    async def load_documents(self, paths: list[str | Path], **kwargs) -> None:
        """Legacy: Use load() instead."""
        await self.load(*paths, **kwargs)
    
    async def load_url(self, url: str, **kwargs) -> None:
        """Legacy: Use load(url) instead."""
        await self.load(url)
    
    # ================================================================
    # Unified run() API
    # ================================================================
    
    @overload
    async def run(
        self,
        prompt: str,
        *,
        citations: bool = False,
        context: dict[str, Any] | RunContext | None = None,
        **kwargs,
    ) -> str: ...
    
    @overload
    async def run(
        self,
        prompt: str,
        *,
        citations: bool = True,
        context: dict[str, Any] | RunContext | None = None,
        **kwargs,
    ) -> RAGResponse: ...
    
    async def run(
        self,
        prompt: str,
        *,
        citations: bool = False,
        k: int | None = None,
        context: dict[str, Any] | RunContext | None = None,
        **kwargs,
    ) -> str | RAGResponse:
        """Run a query against the loaded documents.
        
        The agent uses tools intelligently to search and answer.
        
        Args:
            prompt: Your question or instruction.
            citations: If True, return RAGResponse with structured citations.
            k: Number of passages to retrieve (for citations mode).
            context: Optional context for tools/interceptors.
            **kwargs: Additional args passed to Agent.run().
            
        Returns:
            str if citations=False (default)
            RAGResponse if citations=True
            
        Example:
            >>> answer = await rag.run("What is X?")
            >>> response = await rag.run("What is X?", citations=True)
            >>> print(response.format_full())
        """
        await self._ensure_initialized()
        
        if not citations:
            # Standard agent run - uses tools
            return await super().run(prompt, context=context, **kwargs)
        
        # Structured response with citations
        k = k or self._rag.top_k
        
        # Get relevant passages
        passages = await self._rag.search(prompt, k=k)
        
        if not passages:
            return RAGResponse(
                answer="No relevant information found in the documents.",
                citations=[],
                sources_used=[],
                query=prompt,
            )
        
        # Build context
        context_parts = []
        for p in passages:
            header = f"[{p.citation_id}] Source: {p.source}"
            if p.page:
                header += f", Page {p.page}"
            context_parts.append(f"{header}\n{p.text}")
        
        context_str = "\n\n---\n\n".join(context_parts)
        
        augmented_prompt = f"""Based on the following retrieved passages, answer the question.
Cite sources using [1], [2], etc. If page numbers are available, use [n, p.X].

PASSAGES:
{context_str}

QUESTION: {prompt}

ANSWER:"""
        
        answer = await super().run(augmented_prompt, context=context, **kwargs)
        
        sources_used = list(dict.fromkeys(p.source for p in passages))
        
        return RAGResponse(
            answer=answer,
            citations=passages,
            sources_used=sources_used,
            query=prompt,
        )
    
    # ================================================================
    # Component Access
    # ================================================================
    
    @property
    def rag(self) -> RAG:
        """The RAG capability."""
        return self._rag
    
    @property
    def vectorstore(self) -> VectorStore:
        """Direct access to vector store for search.
        
        Example:
            >>> results = await rag.vectorstore.search("query", k=10)
        """
        return self._rag.vectorstore
    
    @property
    def retriever(self):
        """Direct access to retriever."""
        return self._rag.retriever
    
    @property
    def document_count(self) -> int:
        return self._rag.document_count
    
    @property
    def sources(self) -> list[str]:
        return self._rag.sources
    
    @property
    def citations(self) -> list[CitedPassage]:
        """Citations from last search operation."""
        return self._rag.citations
    
    @property
    def is_ready(self) -> bool:
        return self._rag.is_ready
    
    def register_pipeline(self, extension: str, pipeline: DocumentPipeline) -> None:
        """Register a processing pipeline for a file extension."""
        self._rag.register_pipeline(extension, pipeline)


def create_rag_agent(
    model: BaseChatModel,
    *,
    embeddings: EmbeddingProvider,
    **kwargs,
) -> RAGAgent:
    """Factory function to create a RAGAgent."""
    return RAGAgent(model=model, embeddings=embeddings, **kwargs)


# Re-export for convenience
from agenticflow.vectorstore import VectorStore

__all__ = [
    "RAGAgent",
    "RAGResponse",
    "CitedPassage",
    "DocumentPipeline",
    "PipelineRegistry",
    "VectorStore",
    "create_rag_agent",
]
