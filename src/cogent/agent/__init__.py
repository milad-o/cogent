"""
Agent module - Agent class and related components.

This module defines WHO does the work:
- Agent: The base agent class
- AgentConfig: Configuration for agents
- AgentState: Runtime state management
- Memory: Short-term and long-term memory
- Resilience: Retry, circuit breaker, fallback patterns

Example:
    # Create an agent
    agent = Agent(name="Helper", model=model, tools=[search_tool])

    # With memory
    from cogent.agent.memory import InMemorySaver

    agent = Agent(
        name="Assistant",
        model=model,
        memory=InMemorySaver(),  # Enables conversation memory
    )

    # Run with thread-based memory
    response = await agent.run("Hi!", thread_id="conv-1")
"""

from cogent.agent.base import Agent
from cogent.agent.config import AgentConfig
from cogent.agent.hitl import (
    AbortedException,
    DecisionRequiredException,
    DecisionType,
    GuidanceResult,
    HumanDecision,
    HumanResponse,
    InterruptedException,
    InterruptedState,
    InterruptReason,
    PendingAction,
    should_interrupt,
)
# Memory components moved to cogent.memory
from cogent.memory import (
    InMemorySaver,
    MemorySnapshot,
    ThreadConfig,
)
from cogent.agent.output import (
    OutputMethod,
    OutputValidationError,
    ResponseSchema,
    StructuredResult,
)
from cogent.agent.reasoning import (
    ReasoningConfig,
    ReasoningResult,
    ReasoningStyle,
    ThinkingStep,
)
from cogent.agent.resilience import (
    CircuitBreaker,
    CircuitState,
    ExecutionResult,
    FailureMemory,
    FailureRecord,
    FallbackConfig,
    FallbackRegistry,
    ModelExecutionResult,
    ModelResilience,
    RecoveryAction,
    ResilienceConfig,
    RetryPolicy,
    RetryStrategy,
    ToolResilience,
)
from cogent.agent.state import AgentState

__all__ = [
    # Core
    "Agent",
    "AgentConfig",
    "AgentState",
    # Memory (now from cogent.memory)
    "MemorySnapshot",
    "ThreadConfig",
    "InMemorySaver",
    # Resilience
    "RetryStrategy",
    "RetryPolicy",
    "CircuitState",
    "CircuitBreaker",
    "RecoveryAction",
    "FallbackRegistry",
    "FallbackConfig",
    "FailureRecord",
    "FailureMemory",
    "ResilienceConfig",
    "ExecutionResult",
    "ToolResilience",
    "ModelResilience",
    "ModelExecutionResult",
    # Human-in-the-Loop
    "InterruptReason",
    "DecisionType",
    "PendingAction",
    "HumanDecision",
    "InterruptedState",
    "InterruptedException",
    "DecisionRequiredException",
    "AbortedException",
    "GuidanceResult",
    "HumanResponse",
    "should_interrupt",
    # Reasoning
    "ReasoningConfig",
    "ReasoningStyle",
    "ThinkingStep",
    "ReasoningResult",
    # Structured Output
    "ResponseSchema",
    "OutputMethod",
    "StructuredResult",
    "OutputValidationError",
]
