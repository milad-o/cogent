"""
Prebuilt agents for common use cases.

This module provides ready-to-use agents:
- Chatbot: Conversational agent with memory
- RAGAgent: Document Q&A with retrieval

Example:
    ```python
    from agenticflow.prebuilt import create_rag_agent, create_chatbot
    
    # Quick RAG agent
    rag = create_rag_agent(
        documents=["doc1.pdf", "doc2.txt"],
        model=ChatOpenAI(model="gpt-4o-mini"),
    )
    answer = await rag.query("What is the main topic?")
    
    # Quick chatbot with memory
    bot = create_chatbot(
        model=ChatOpenAI(model="gpt-4o-mini"),
        personality="helpful assistant",
    )
    response = await bot.chat("Hello!", thread_id="user-123")
    ```

For supervisor/delegation patterns, see examples/12_supervisor_chat.py.
It's just an Agent with tools that wrap other agents - simple enough
that a prebuilt isn't needed.
"""

from agenticflow.prebuilt.chatbot import Chatbot, create_chatbot
from agenticflow.prebuilt.rag import RAGAgent, create_rag_agent

__all__ = [
    "Chatbot",
    "RAGAgent",
    "create_chatbot",
    "create_rag_agent",
]
