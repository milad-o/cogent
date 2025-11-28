"""
Agent module - Agent class and related components.

This module defines WHO does the work:
- Agent: The base agent class with role-specific factory methods
- AgentConfig: Configuration for agents
- AgentState: Runtime state management
- Roles: Role-specific behaviors and prompts
- Memory: Short-term and long-term memory with LangGraph compatibility
- Resilience: Retry, circuit breaker, fallback patterns

For HOW agents execute tasks (execution strategies), see:
    agenticflow.executors - NativeExecutor, SequentialExecutor, TreeSearchExecutor

Example:
    # Create role-specific agents
    supervisor = Agent.as_supervisor(name="Manager", model=model, workers=["A", "B"])
    worker = Agent.as_worker(name="Analyst", model=model, tools=[search_tool])
    
    # Or use the generic constructor
    agent = Agent(name="Helper", model=model, role="worker")
    
    # With memory
    from agenticflow.agent.memory import InMemorySaver
    
    agent = Agent(
        name="Assistant",
        model=model,
        memory=InMemorySaver(),  # Enables conversation memory
    )
    
    # Chat with thread-based memory
    response = await agent.chat("Hi!", thread_id="conv-1")
"""

from agenticflow.agent.base import Agent
from agenticflow.agent.config import AgentConfig
from agenticflow.agent.state import AgentState
from agenticflow.agent.memory import (
    AgentMemory,
    MemorySnapshot,
    MemoryCheckpoint,  # Backward compat alias
    ThreadConfig,
    InMemorySaver,
    InMemoryCheckpointer,  # Backward compat alias
)
from agenticflow.agent.roles import (
    RoleBehavior,
    DelegationCommand,
    get_role_prompt,
    get_role_behavior,
    parse_delegation,
    has_final_answer,
    extract_final_answer,
    ROLE_PROMPTS,
    ROLE_BEHAVIORS,
)
from agenticflow.agent.resilience import (
    RetryStrategy,
    RetryPolicy,
    CircuitState,
    CircuitBreaker,
    RecoveryAction,
    FallbackRegistry,
    FallbackConfig,
    FailureRecord,
    FailureMemory,
    ResilienceConfig,
    ExecutionResult,
    ToolResilience,
)
from agenticflow.agent.hitl import (
    InterruptReason,
    DecisionType,
    PendingAction,
    HumanDecision,
    InterruptedState,
    InterruptedException,
    DecisionRequiredException,
    AbortedException,
    GuidanceResult,
    HumanResponse,
    should_interrupt,
)

__all__ = [
    # Core
    "Agent",
    "AgentConfig",
    "AgentState",
    # Memory
    "AgentMemory",
    "MemorySnapshot",
    "MemoryCheckpoint",  # Backward compat alias
    "ThreadConfig",
    "InMemorySaver",
    "InMemoryCheckpointer",  # Backward compat alias
    # Roles
    "RoleBehavior",
    "DelegationCommand",
    "get_role_prompt",
    "get_role_behavior",
    "parse_delegation",
    "has_final_answer",
    "extract_final_answer",
    "ROLE_PROMPTS",
    "ROLE_BEHAVIORS",
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
]
