"""
Core module - foundational types and utilities for AgenticFlow.

For models and embeddings, use LangChain directly:
    from langchain_openai import ChatOpenAI, OpenAIEmbeddings
    from langchain_anthropic import ChatAnthropic
    from langchain_google_genai import ChatGoogleGenerativeAI
"""

from agenticflow.core.enums import (
    AgentRole,
    AgentStatus,
    EventType,
    Priority,
    TaskStatus,
)
from agenticflow.core.utils import generate_id, now_utc

__all__ = [
    # Enums
    "TaskStatus",
    "AgentStatus",
    "EventType",
    "Priority",
    "AgentRole",
    # Utilities
    "generate_id",
    "now_utc",
]
