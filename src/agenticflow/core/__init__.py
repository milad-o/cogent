"""
Core module - foundational types and utilities for AgenticFlow.

For models and embeddings, use native AgenticFlow models:
    from agenticflow.models import ChatModel, create_chat
    from agenticflow.models.anthropic import AnthropicChat
    from agenticflow.models.gemini import GeminiChat
"""

from agenticflow.core.enums import (
    AgentRole,
    AgentStatus,
    EventType,
    Priority,
    TaskStatus,
)
from agenticflow.core.utils import (
    format_timestamp,
    generate_id,
    now_local,
    now_utc,
    to_local,
)

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
    "now_local",
    "to_local",
    "format_timestamp",
]
