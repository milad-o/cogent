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
            BudgetGuard(model_calls=10, tool_calls=50),
            PIIShield(["email", "ssn"], action="mask"),
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

__all__ = [
    # Core
    "Interceptor",
    "InterceptContext",
    "InterceptResult",
    "Phase",
    "StopExecution",
    "run_interceptors",
    # Built-in
    "BudgetGuard",
]
