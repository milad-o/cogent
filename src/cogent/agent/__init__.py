"""
Agent module - Agent class and related components.

This module defines WHO does the work:
- Agent: The base agent class with role-specific factory methods
- AgentConfig: Configuration for agents
- AgentState: Runtime state management
- Roles: Role-specific behaviors and prompts
- Memory: Short-term and long-term memory
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
from cogent.agent.memory import (
    AgentMemory,
    InMemoryCheckpointer,  # Backward compat alias
    InMemorySaver,
    MemoryCheckpoint,  # Backward compat alias
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
from cogent.agent.roles import (
    ROLE_BEHAVIORS,
    ROLE_PROMPTS,
    DelegationCommand,
    RoleBehavior,
    extract_final_answer,
    get_role_behavior,
    get_role_prompt,
    has_final_answer,
    parse_delegation,
)
from cogent.agent.spawning import (
    AgentSpec,
    SpawnedAgentInfo,
    SpawningConfig,
    SpawnManager,
)
from cogent.agent.state import AgentState
from cogent.agent.taskboard import (
    Task,
    TaskBoard,
    TaskBoardConfig,
    TaskStatus,
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
    # TaskBoard
    "TaskBoard",
    "TaskBoardConfig",
    "Task",
    "TaskStatus",
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
    # Spawning
    "AgentSpec",
    "SpawningConfig",
    "SpawnedAgentInfo",
    "SpawnManager",
]
