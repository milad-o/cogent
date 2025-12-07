"""
Interceptors - composable units that intercept agent execution.

Interceptors enable cross-cutting concerns like cost control, context management,
security, and observability without cluttering core agent logic.

Example:
    from agenticflow import Agent
    from agenticflow.interceptors import BudgetGuard, PIIShield, ToolGate
    
    agent = Agent(
        name="assistant",
        model=model,
        intercept=[
            BudgetGuard(max_model_calls=10, max_tool_calls=50),
            PIIShield(patterns=["email", "ssn"], action=PIIAction.MASK),
        ],
    )
"""

from agenticflow.interceptors.base import (
    Interceptor,
    InterceptContext,
    InterceptResult,
    Phase,
    StopExecution,
    run_interceptors,
)
from agenticflow.interceptors.budget import BudgetGuard
from agenticflow.interceptors.context import ContextCompressor, TokenLimiter
from agenticflow.interceptors.security import PIIAction, PIIShield, ContentFilter
from agenticflow.interceptors.ratelimit import RateLimiter, ThrottleInterceptor
from agenticflow.interceptors.audit import Auditor, AuditEvent, AuditEventType
from agenticflow.interceptors.gates import ToolGate, PermissionGate, ConversationGate
from agenticflow.interceptors.failover import Failover, FailoverTrigger, FailoverState
from agenticflow.interceptors.guards import ToolGuard, ToolRetryState, CircuitBreaker
from agenticflow.interceptors.prompt import (
    PromptAdapter,
    ContextPrompt,
    ConversationPrompt,
    LambdaPrompt,
)
from agenticflow.interceptors.rag import RAGInterceptor, RAGPostProcessor

__all__ = [
    # Core
    "Interceptor",
    "InterceptContext",
    "InterceptResult",
    "Phase",
    "StopExecution",
    "run_interceptors",
    # Budget
    "BudgetGuard",
    # Context
    "ContextCompressor",
    "TokenLimiter",
    # Security
    "PIIAction",
    "PIIShield",
    "ContentFilter",
    # Rate Limiting
    "RateLimiter",
    "ThrottleInterceptor",
    # Audit
    "Auditor",
    "AuditEvent",
    "AuditEventType",
    # Gates (tool filtering)
    "ToolGate",
    "PermissionGate",
    "ConversationGate",
    # Failover
    "Failover",
    "FailoverTrigger",
    "FailoverState",
    # Guards (tool retry/circuit breaker)
    "ToolGuard",
    "ToolRetryState",
    "CircuitBreaker",
    # Prompt adapters
    "PromptAdapter",
    "ContextPrompt",
    "ConversationPrompt",
    "LambdaPrompt",
    # RAG
    "RAGInterceptor",
    "RAGPostProcessor",
]
