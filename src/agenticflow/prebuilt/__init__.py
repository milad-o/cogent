"""
Prebuilt agents for common use cases.

This module provides ready-to-use agents that inherit from Agent:
- Chatbot: Conversational agent with memory (default enabled)
- RAGAgent: Document Q&A with per-file-type processing pipelines

Both inherit from Agent, so you get full access to all capabilities:
- tools, streaming, reasoning, structured output
- memory, store, interceptors, resilience, observability
- run(), chat(), think(), and all other Agent methods

Example:
    ```python
    from agenticflow.prebuilt import create_rag_agent, create_chatbot
    
    # RAG agent with per-file-type processing
    rag = create_rag_agent(
        model=ChatModel(model="gpt-4o-mini"),
        embeddings=OpenAIEmbeddings(),
    )
    await rag.load_documents(["report.pdf", "code.py", "notes.md"])
    # Each file type uses its optimal splitter!
    answer = await rag.query("What is the main topic?")
    
    # Chatbot with memory + custom tools
    bot = create_chatbot(
        model=ChatModel(model="gpt-4o-mini"),
        personality="helpful assistant",
        tools=[my_tool],  # Full tools support!
    )
    response = await bot.chat("Hello!", thread_id="user-123")
    ```

For supervisor/delegation patterns, see examples/12_supervisor_chat.py.
It's just an Agent with tools that wrap other agents - simple enough
that a prebuilt isn't needed.
"""

from agenticflow.prebuilt.chatbot import Chatbot, create_chatbot
from agenticflow.prebuilt.rag import (
    DocumentPipeline,
    PipelineRegistry,
    RAGAgent,
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
]
