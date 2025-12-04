"""
Prebuilt RAG (Retrieval-Augmented Generation) Agent.

A convenience wrapper that creates an Agent with RAG capability pre-attached.
For composable RAG functionality that can be added to any agent, use the
RAG capability directly from `agenticflow.capabilities`.

Example - Using RAGAgent (convenience wrapper):
    ```python
    from agenticflow.models import ChatModel
    from agenticflow.vectorstore import OpenAIEmbeddings
    from agenticflow.prebuilt import RAGAgent
    
    rag = RAGAgent(
        model=ChatModel(model="gpt-4o-mini"),
        embeddings=OpenAIEmbeddings(),
    )
    await rag.load("report.pdf", "notes.md")
    response = await rag.query("What are the key findings?")
    print(response.answer)
    print(response.format_bibliography())
    ```

Example - Using RAG capability (composable approach):
    ```python
    from agenticflow import Agent
    from agenticflow.capabilities import RAG
    from agenticflow.vectorstore import OpenAIEmbeddings
    
    agent = Agent(
        name="ResearchAssistant",
        model=model,
        capabilities=[RAG(embeddings=OpenAIEmbeddings())],
    )
    await agent.rag.load("docs/")
    response = await agent.rag.query("What is the main topic?")
    ```

The RAGAgent is simply an Agent + RAG capability, providing a familiar API
for document Q&A while supporting all standard Agent features.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

from agenticflow import Agent
from agenticflow.capabilities.rag import (
    RAG,
    RAGConfig,
    RAGResponse,
    CitedPassage,
    DocumentPipeline,
    PipelineRegistry,
)
from agenticflow.vectorstore import SearchResult

if TYPE_CHECKING:
    from agenticflow.agent.memory import AgentMemory, MemorySaver, MemoryStore
    from agenticflow.agent.resilience import ResilienceConfig
    from agenticflow.context import RunContext
    from agenticflow.document import BaseSplitter, DocumentLoader
    from agenticflow.models.base import BaseChatModel
    from agenticflow.retriever.base import Retriever
    from agenticflow.retriever.rerankers.base import Reranker
    from agenticflow.vectorstore import VectorStore
    from agenticflow.vectorstore.base import EmbeddingProvider


class RAGAgent(Agent):
    """
    A RAG (Retrieval-Augmented Generation) agent for document Q&A.
    
    This is a convenience wrapper around Agent + RAG capability. It provides
    the familiar RAGAgent API while leveraging the composable RAG capability.
    
    Inherits from Agent, so you get full access to all agent capabilities:
    - tools, capabilities, streaming, reasoning, structured output
    - memory, store, interceptors, resilience, observability
    - run(), chat(), think(), and all other Agent methods
    
    RAG-specific functionality:
    - load(): Load and index documents (files, directories, URLs)
    - query(): Ask questions with structured citations → RAGResponse
    - search(): Direct semantic search → list[CitedPassage]
    
    **Unified API:**
    - `query()` always returns `RAGResponse` with citations
    - `search()` always returns `list[CitedPassage]`
    
    **Per-File-Type Processing:**
    Different file types often need different processing:
    - Markdown → MarkdownSplitter (preserves headers)
    - Code → CodeSplitter (preserves functions)
    - PDF → SemanticSplitter (handles mixed content)
    - HTML → HTMLSplitter (strips tags intelligently)
    
    Example - Simple:
        ```python
        from agenticflow.models import ChatModel
        from agenticflow.vectorstore import OpenAIEmbeddings
        from agenticflow.prebuilt import RAGAgent
        
        rag = RAGAgent(
            model=ChatModel(model="gpt-4o-mini"),
            embeddings=OpenAIEmbeddings(),
        )
        await rag.load("report.pdf", "notes.md")
        response = await rag.query("What are the key findings?")
        print(response.answer)
        for cite in response.citations:
            print(cite.format_full())
        ```
    
    Example - Per-file-type pipelines:
        ```python
        from agenticflow.document import MarkdownSplitter, CodeSplitter, SemanticSplitter
        from agenticflow.prebuilt.rag import PipelineRegistry, DocumentPipeline
        
        pipelines = PipelineRegistry(
            pipelines={
                ".md": DocumentPipeline(splitter=MarkdownSplitter(chunk_size=500)),
                ".py": DocumentPipeline(splitter=CodeSplitter(language="python")),
                ".pdf": DocumentPipeline(splitter=SemanticSplitter(model=model)),
            },
        )
        
        rag = RAGAgent(
            model=model,
            embeddings=embeddings,
            pipelines=pipelines,
        )
        ```
    
    Example - With additional tools:
        ```python
        from agenticflow.tools import tool
        
        @tool
        def calculate(expression: str) -> str:
            '''Evaluate a math expression.'''
            return str(eval(expression))
        
        rag = RAGAgent(
            model=model,
            embeddings=embeddings,
            tools=[calculate],  # Add custom tools alongside RAG tools!
        )
        await rag.load("financial_report.pdf")
        # RAG agent can now search docs AND do calculations
        ```
    """
    
    def __init__(
        self,
        model: BaseChatModel,
        *,
        # RAG configuration
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
            model: Chat model for answer generation.
            embeddings: Embedding provider for vector search. Required.
            
            **RAG Configuration:**
            pipelines: PipelineRegistry for per-file-type processing.
            vectorstore: Pre-configured VectorStore (optional).
            retriever: Pre-configured retriever (optional).
            reranker: Pre-configured reranker for two-stage retrieval.
            chunk_size: Chunk size for text splitting (default: 1000).
            chunk_overlap: Chunk overlap (default: 200).
            top_k: Number of passages to retrieve (default: 4).
            backend: Vector store backend ("inmemory", "faiss", "chroma", etc.).
            show_progress: Print progress messages during loading.
            
            **All Agent parameters are supported:**
            name: Agent name.
            instructions: Custom system prompt.
            tools: Additional tools beyond the built-in RAG tools.
            capabilities: Additional capabilities to attach.
            memory: Enable conversation persistence.
            store: Long-term memory store.
            intercept: Interceptors (gates, guards, prompt adapters).
            stream: Enable streaming responses.
            reasoning: Enable extended thinking.
            output: Structured output schema.
            verbose: Observability level.
            resilience: Retry/fallback configuration.
            interrupt_on: HITL tool approval rules.
            observer: Custom observer.
            taskboard: Enable task tracking.
        """
        # Create RAG config
        config = RAGConfig(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            top_k=top_k,
            backend=backend,
            show_progress=show_progress,
        )
        
        # Create RAG capability
        self._rag_capability = RAG(
            embeddings=embeddings,
            config=config,
            pipelines=pipelines,
            vectorstore=vectorstore,
            retriever=retriever,
            reranker=reranker,
        )
        
        # Combine capabilities
        all_capabilities = [self._rag_capability]
        if capabilities:
            all_capabilities.extend(capabilities)
        
        # Default instructions if not provided
        if instructions is None:
            instructions = RAG.DEFAULT_INSTRUCTIONS
        
        # Initialize Agent
        super().__init__(
            name=name,
            model=model,
            instructions=instructions,
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
    
    # ================================================================
    # Document Loading (delegates to RAG capability)
    # ================================================================
    
    async def _ensure_initialized(self) -> None:
        """Ensure RAG capability is initialized with agent reference."""
        if self._rag_capability._agent is None:
            await self._rag_capability.initialize(self)
    
    async def load(
        self,
        *sources: str | Path,
        glob: str | None = None,
    ) -> None:
        """Load documents from files, directories, or URLs.
        
        Supports all file types via the extensible loader system:
        - Text: .txt, .md, .rst
        - Documents: .pdf, .docx
        - Data: .csv, .json, .jsonl, .xlsx
        - Web: .html, .htm
        - Code: .py, .js, .ts, .java, and many more
        
        Args:
            *sources: File paths, directory paths, or URLs.
            glob: Glob pattern when loading from directories.
            
        Example:
            >>> await rag.load("report.pdf", "notes/", glob="**/*.md")
        """
        await self._ensure_initialized()
        await self._rag_capability.load(*sources, glob=glob)
    
    async def load_text(
        self,
        text: str,
        source: str = "text",
    ) -> None:
        """Load text content directly.
        
        Args:
            text: Text content to load.
            source: Source name for citations.
        """
        await self._ensure_initialized()
        await self._rag_capability.load_text(text, source)
    
    # Legacy aliases for backward compatibility
    async def load_documents(
        self,
        paths: list[str | Path],
        *,
        glob: str | None = None,
        show_progress: bool = True,
    ) -> None:
        """Load documents from files (legacy API).
        
        Deprecated: Use load() instead for cleaner API.
        """
        await self._ensure_initialized()
        await self._rag_capability.load(*paths, glob=glob)
    
    async def load_url(
        self,
        url: str,
        *,
        show_progress: bool = True,
    ) -> None:
        """Load content from a URL (legacy API).
        
        Deprecated: Use load(url) instead.
        """
        await self._ensure_initialized()
        await self._rag_capability.load(url)
    
    # ================================================================
    # Query & Search (delegates to RAG capability)
    # ================================================================
    
    async def query(
        self,
        question: str,
        *,
        k: int | None = None,
        context: dict[str, Any] | RunContext | None = None,
    ) -> RAGResponse:
        """Ask a question about loaded documents.
        
        Uses the agent's LLM with retrieved context to generate
        an answer with citations.
        
        Args:
            question: Your question.
            k: Number of passages to retrieve (default: config.top_k).
            context: Optional context for tools/interceptors.
            
        Returns:
            RAGResponse with answer, citations, and metadata.
            
        Example:
            >>> response = await rag.query("What are the key findings?")
            >>> print(response.answer)
            >>> print(response.format_bibliography())
        """
        await self._ensure_initialized()
        return await self._rag_capability.query(question, k=k)
    
    async def search(
        self,
        query: str,
        *,
        k: int | None = None,
    ) -> list[CitedPassage]:
        """Direct semantic search over documents.
        
        Returns the most relevant passages without LLM generation.
        Useful for exploring document content.
        
        Args:
            query: Search query.
            k: Number of results (default: config.top_k).
            
        Returns:
            List of CitedPassage objects with source info and scores.
            
        Example:
            >>> passages = await rag.search("machine learning")
            >>> for p in passages:
            ...     print(f"{p.format_reference()}: {p.text[:100]}...")
        """
        await self._ensure_initialized()
        return await self._rag_capability.search(query, k=k)
    
    # ================================================================
    # Pipeline Registration (delegates to RAG capability)
    # ================================================================
    
    def register_pipeline(
        self,
        extension: str,
        pipeline: DocumentPipeline,
    ) -> None:
        """Register a processing pipeline for a file extension.
        
        Args:
            extension: File extension (e.g., ".pdf", ".custom").
            pipeline: DocumentPipeline with loader and splitter.
        """
        self._rag_capability.register_pipeline(extension, pipeline)
    
    # ================================================================
    # Properties (delegates to RAG capability)
    # ================================================================
    
    @property
    def rag(self) -> RAG:
        """Direct access to the RAG capability."""
        return self._rag_capability
    
    @property
    def document_count(self) -> int:
        """Number of document chunks loaded."""
        return self._rag_capability.document_count
    
    @property
    def sources(self) -> list[str]:
        """List of loaded document sources."""
        return self._rag_capability.sources
    
    @property
    def citations(self) -> list[CitedPassage]:
        """Citations from the last query/search."""
        return self._rag_capability.citations
    
    @property
    def is_ready(self) -> bool:
        """Whether documents have been loaded."""
        return self._rag_capability.is_ready


def create_rag_agent(
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
) -> RAGAgent:
    """
    Create a RAG agent for document Q&A (factory function).
    
    This is equivalent to directly instantiating RAGAgent, provided
    for API consistency with other create_* functions.
    
    Args:
        model: Chat model for answer generation.
        embeddings: Embedding provider for vector search.
        
        **RAG Configuration:**
        pipelines: PipelineRegistry for per-file-type processing.
        vectorstore: Pre-configured VectorStore.
        retriever: Pre-configured retriever.
        reranker: Pre-configured reranker.
        chunk_size: Chunk size for text splitting.
        chunk_overlap: Chunk overlap.
        top_k: Number of passages to retrieve.
        backend: Vector store backend.
        show_progress: Print progress messages.
        
        **All Agent parameters are supported.**
        
    Returns:
        Configured RAGAgent instance.
        
    Example:
        >>> rag = create_rag_agent(
        ...     model=ChatModel(model="gpt-4o-mini"),
        ...     embeddings=OpenAIEmbeddings(),
        ... )
        >>> await rag.load("report.pdf")
        >>> response = await rag.query("What are the key findings?")
    """
    return RAGAgent(
        model=model,
        embeddings=embeddings,
        pipelines=pipelines,
        vectorstore=vectorstore,
        retriever=retriever,
        reranker=reranker,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        top_k=top_k,
        backend=backend,
        show_progress=show_progress,
        name=name,
        instructions=instructions,
        tools=tools,
        capabilities=capabilities,
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


# Re-export types from capability for convenience
__all__ = [
    "RAGAgent",
    "RAGResponse",
    "CitedPassage",
    "DocumentPipeline",
    "PipelineRegistry",
    "create_rag_agent",
]
