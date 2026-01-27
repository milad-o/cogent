"""
Interceptors - composable units that intercept agent execution.

Interceptors enable cross-cutting concerns like cost control, context management,
security, and observability without cluttering core agent logic.

Example:
    from cogent import Agent
    from cogent.interceptors import BudgetGuard, PIIShield, ToolGate

    agent = Agent(
        name="assistant",
        model=model,
        intercept=[
            BudgetGuard(max_model_calls=10, max_tool_calls=50),
            PIIShield(patterns=["email", "ssn"], action=PIIAction.MASK),
        ],
    )
"""

from cogent.interceptors.audit import AuditEvent, Auditor, AuditTraceType
from cogent.interceptors.base import (
    InterceptContext,
    Interceptor,
    InterceptResult,
    Phase,
    StopExecution,
    run_interceptors,
)
from cogent.interceptors.budget import BudgetGuard
from cogent.interceptors.context import ContextCompressor, TokenLimiter
from cogent.interceptors.failover import Failover, FailoverState, FailoverTrigger
from cogent.interceptors.gates import ConversationGate, PermissionGate, ToolGate
from cogent.interceptors.guards import CircuitBreaker, ToolGuard, ToolRetryState
from cogent.interceptors.prompt import (
    ContextPrompt,
    ConversationPrompt,
    LambdaPrompt,
    PromptAdapter,
)
from cogent.interceptors.ratelimit import RateLimiter, ThrottleInterceptor
from cogent.interceptors.security import ContentFilter, PIIAction, PIIShield

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
    "AuditTraceType",
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
]
