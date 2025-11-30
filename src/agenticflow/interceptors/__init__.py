"""
Interceptors - composable units that intercept agent execution.

Interceptors enable cross-cutting concerns like cost control, context management,
security, and observability without cluttering core agent logic.

Example:
    from agenticflow import Agent
    from agenticflow.interceptors import BudgetGuard, PIIShield
    
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
]
