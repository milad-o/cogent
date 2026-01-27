"""
Core module - foundational types and utilities for AgenticFlow.

For models and embeddings, use native AgenticFlow models:
    from cogent.models import ChatModel, create_chat
    from cogent.models.anthropic import AnthropicChat
    from cogent.models.gemini import GeminiChat
"""

from cogent.core.context import EMPTY_CONTEXT, RunContext
from cogent.core.document import Document, DocumentMetadata
from cogent.core.enums import (
    AgentRole,
    AgentStatus,
    Priority,
    TaskStatus,
    TraceType,
)
from cogent.core.response import (
    ErrorInfo,
    Response,
    ResponseError,
    ResponseMetadata,
    TokenUsage,
    ToolCall,
)
from cogent.core.utils import (
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
