"""
Core module - foundational types and utilities for AgenticFlow.

For models and embeddings, use native AgenticFlow models:
    from agenticflow.models import ChatModel, create_chat
    from agenticflow.models.anthropic import AnthropicChat
    from agenticflow.models.gemini import GeminiChat
"""

from agenticflow.core.context import EMPTY_CONTEXT, RunContext
from agenticflow.core.document import Document, DocumentMetadata
from agenticflow.core.enums import (
    AgentRole,
    AgentStatus,
    Priority,
    TaskStatus,
    TraceType,
)
from agenticflow.core.response import (
    ErrorInfo,
    Response,
    ResponseError,
    ResponseMetadata,
    TokenUsage,
    ToolCall,
)
from agenticflow.core.utils import (
    IdempotencyGuard,
    RetryBudget,
    Stopwatch,
    emit_later,
    format_timestamp,
    generate_id,
    jittered_delay,
    now_local,
    now_utc,
    to_local,
)

__all__ = [
    # Enums
    "TaskStatus",
    "AgentStatus",
    "TraceType",
    "Priority",
    "AgentRole",
    # Context
    "RunContext",
    "EMPTY_CONTEXT",
    # Documents
    "Document",
    "DocumentMetadata",
    # Response Protocol
    "Response",
    "ResponseMetadata",
    "TokenUsage",
    "ToolCall",
    "ErrorInfo",
    "ResponseError",
    # Utilities
    "generate_id",
    "now_utc",
    "now_local",
    "to_local",
    "format_timestamp",
    # Reactive/Event-driven utilities
    "IdempotencyGuard",
    "RetryBudget",
    "emit_later",
    "jittered_delay",
    "Stopwatch",
]
