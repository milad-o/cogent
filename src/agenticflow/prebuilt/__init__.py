"""
Prebuilt agents for common use cases.

This module provides ready-to-use agents that inherit from Agent:
- Chatbot: Conversational agent with memory (default enabled)
- RAGAgent: Document Q&A with unified run() API

Both inherit from Agent, so you get full access to all capabilities:
- tools, streaming, reasoning, structured output
- memory, store, interceptors, resilience, observability
- run(), chat(), think(), and all other Agent methods

Example:
    ```python
    from agenticflow.prebuilt import create_rag_agent, create_chatbot
    
    # RAG agent with unified run() API
    rag = create_rag_agent(
        model=ChatModel(model="gpt-4o-mini"),
        embeddings=OpenAIEmbeddings(),
    )
    await rag.load("report.pdf", "code.py", "notes.md")
    
    # Agent uses tools intelligently
    answer = await rag.run("What is the main topic?")
    
    # Get structured citations
    response = await rag.run("Key findings?", citations=True)
    print(response.format_full())
    
    # Direct vectorstore search
    results = await rag.vectorstore.search("query", k=10)
    
    # Chatbot with memory + custom tools
    bot = create_chatbot(
        model=ChatModel(model="gpt-4o-mini"),
        personality="helpful assistant",
        tools=[my_tool],
    )
    response = await bot.chat("Hello!", thread_id="user-123")
    ```

For supervisor/delegation patterns, see examples/12_supervisor_chat.py.
It's just an Agent with tools that wrap other agents - simple enough
that a prebuilt isn't needed.
"""

from agenticflow.prebuilt.chatbot import Chatbot, create_chatbot
from agenticflow.prebuilt.rag import (
    CitedPassage,
    DocumentPipeline,
    PipelineRegistry,
    RAGAgent,
    RAGResponse,
    create_rag_agent,
)

__all__ = [
    # Chatbot
    "Chatbot",
    "create_chatbot",
    # RAG
    "RAGAgent",
    "create_rag_agent",
    "DocumentPipeline",
    "PipelineRegistry",
    # Citation types
    "CitedPassage",
    "RAGResponse",
]
