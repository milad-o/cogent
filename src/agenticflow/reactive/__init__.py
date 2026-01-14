"""Reactive multi-agent orchestration.

This module provides a reactive/event-driven architecture for agent coordination
where agents react to events rather than being called imperatively.

Core Concepts:
- **Trigger**: Condition that activates an agent (event type + filter)
- **EventFlow**: Orchestrator that routes events to triggered agents
- **Reaction**: What an agent does when triggered (run, emit, delegate)

Benefits over imperative orchestration:
- **Loose coupling**: Agents don't know about each other
- **Dynamic workflows**: Add/remove agents without changing flow logic
- **Reactive patterns**: Agents respond to conditions, not commands
- **Event sourcing**: Full audit trail of what happened and why

Example:
    ```python
    from agenticflow import Agent
    from agenticflow.reactive import EventFlow, react_to, AgentTriggerConfig

    # Define agents
    researcher = Agent(
        name="researcher",
        model=model,
        system_prompt="You are a research assistant.",
    )

    writer = Agent(
        name="writer",
        model=model,
        system_prompt="You write engaging content.",
    )

    # Create reactive flow
    flow = EventFlow()

    # Register with triggers
    flow.register(
        researcher,
        [react_to("task.created").when(lambda e: "research" in e.data.get("type", ""))]
    )
    flow.register(writer, [react_to("researcher.completed")])

    # Start with an event - agents react automatically
    result = await flow.run(
        "Write a blog post about quantum computing",
        initial_event="task.created",
        initial_data={"type": "research"},
    )
    ```

Patterns:
    - **Chain**: A → B → C (each agent emits event for next)
    - **Fan-out**: A → [B, C, D] (multiple agents react to same event)
    - **Fan-in**: [A, B, C] → D (agent waits for multiple events)
    - **Conditional**: Different agents react based on event data
    - **Saga**: Long-running workflow with compensation on failure
"""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING

from agenticflow.reactive.core import (
    AgentTriggerConfig,
    Reaction,
    ReactionType,
    Trigger,
    TriggerBuilder,
    TriggerCondition,
    on,  # Backward compat alias
    react_to,
    when,
)
from agenticflow.reactive.skills import Skill, SkillBuilder, skill

if TYPE_CHECKING:
    # Re-exported from flow package (canonical location). Imported lazily at runtime
    # to avoid circular imports:
    # - flow.reactive imports agenticflow.reactive.core
    # - importing agenticflow.reactive.core executes this __init__.py first
    from agenticflow.flow.reactive import (  # noqa: TC004
        EventFlow,
        EventFlowConfig,
        EventFlowResult,
        ReactiveFlow,
        ReactiveFlowConfig,
        ReactiveFlowResult,
    )
    from agenticflow.observability.observer import Observer  # noqa: TC004
    from agenticflow.reactive.agent import (  # noqa: TC004
        ReactiveAgent,
        build_reactive_system_prompt,
    )
    # Re-export checkpointer from flow module for backward compatibility
    from agenticflow.flow.checkpointer import (  # noqa: TC004
        Checkpointer,
        FileCheckpointer,
        FlowState,
        MemoryCheckpointer,
        generate_checkpoint_id,
        generate_flow_id,
    )
    from agenticflow.reactive.kit import (  # noqa: TC004
        IdempotencyGuard,
        RetryBudget,
        emit_later,
    )

    from agenticflow.reactive.patterns import (  # noqa: TC004
        Chain,
        ChainPattern,
        FanIn,
        FanInPattern,
        FanOut,
        FanOutPattern,
        Route,
        Router,
        RouterPattern,
        Saga,
        SagaPattern,
        SagaStep,
        chain,
        fanout,
        route,
    )
    from agenticflow.reactive.threading import thread_id_from_data  # noqa: TC004

__all__ = [
    # Core
    "AgentTriggerConfig",
    "Reaction",
    "ReactionType",
    "Trigger",
    "TriggerBuilder",
    "TriggerCondition",
    # Builders
    "react_to",
    "on",
    "when",
    # Skills (event-triggered specializations)
    "Skill",
    "SkillBuilder",
    "skill",
    # Flow (new names)
    "ReactiveFlow",
    "ReactiveFlowConfig",
    "ReactiveFlowResult",
    # Flow (legacy aliases)
    "EventFlow",
    "EventFlowConfig",
    "EventFlowResult",
    # Checkpointing
    "Checkpointer",
    "FlowState",
    "MemoryCheckpointer",
    "FileCheckpointer",
    "generate_checkpoint_id",
    "generate_flow_id",
    # Reactive agent helper
    "ReactiveAgent",
    "build_reactive_system_prompt",
    # Threading helpers
    "thread_id_from_data",
    # Reactive kit helpers
    "IdempotencyGuard",
    "RetryBudget",
    "emit_later",
    # High-level API (recommended)
    "chain",
    "fanout",
    "route",
    # Mid-level API
    "Chain",
    "FanIn",
    "FanOut",
    "Router",
    "Saga",
    # Observability
    "Observer",
    # Legacy compatibility
    "ChainPattern",
    "FanInPattern",
    "FanOutPattern",
    "Route",
    "RouterPattern",
    "SagaPattern",
    "SagaStep",
]



_LAZY_FLOW_EXPORTS: frozenset[str] = frozenset(
    {
        "ReactiveFlow",
        "ReactiveFlowConfig",
        "ReactiveFlowResult",
        "EventFlow",
        "EventFlowConfig",
        "EventFlowResult",
    }
)

_LAZY_PATTERNS_EXPORTS: frozenset[str] = frozenset(
    {
        # High-level functions
        "chain",
        "fanout",
        "route",
        # Mid-level classes
        "Chain",
        "FanIn",
        "FanOut",
        "Router",
        "Saga",
        # Legacy compatibility
        "ChainPattern",
        "FanInPattern",
        "FanOutPattern",
        "Route",
        "RouterPattern",
        "SagaPattern",
        "SagaStep",
    }
)

_LAZY_AGENT_EXPORTS: frozenset[str] = frozenset(
    {
        "ReactiveAgent",
        "build_reactive_system_prompt",
    }
)

_LAZY_THREADING_EXPORTS: frozenset[str] = frozenset({"thread_id_from_data"})

_LAZY_KIT_EXPORTS: frozenset[str] = frozenset(
    {
        "IdempotencyGuard",
        "RetryBudget",
        "emit_later",
    }
)

_LAZY_CHECKPOINTER_EXPORTS: frozenset[str] = frozenset(
    {
        "Checkpointer",
        "FlowState",
        "MemoryCheckpointer",
        "FileCheckpointer",
        "generate_checkpoint_id",
        "generate_flow_id",
    }
)

_LAZY_OBSERVABILITY_EXPORTS: frozenset[str] = frozenset({"Observer"})



def __getattr__(name: str) -> object:
    if name in _LAZY_FLOW_EXPORTS:
        module = import_module("agenticflow.flow.reactive")
        return getattr(module, name)
    if name in _LAZY_PATTERNS_EXPORTS:
        module = import_module("agenticflow.reactive.patterns")
        return getattr(module, name)
    if name in _LAZY_AGENT_EXPORTS:
        module = import_module("agenticflow.reactive.agent")
        return getattr(module, name)
    if name in _LAZY_KIT_EXPORTS:
        module = import_module("agenticflow.reactive.kit")
        return getattr(module, name)
    if name in _LAZY_CHECKPOINTER_EXPORTS:
        module = import_module("agenticflow.reactive.checkpointer")
        return getattr(module, name)
    if name in _LAZY_THREADING_EXPORTS:
        module = import_module("agenticflow.reactive.threading")
        return getattr(module, name)
    if name in _LAZY_OBSERVABILITY_EXPORTS:
        module = import_module("agenticflow.observability.observer")
        return getattr(module, name)
    raise AttributeError(name)


def __dir__() -> list[str]:
    return sorted(
        {
            *globals().keys(),
            *_LAZY_FLOW_EXPORTS,
            *_LAZY_PATTERNS_EXPORTS,
            *_LAZY_AGENT_EXPORTS,
            *_LAZY_KIT_EXPORTS,
            *_LAZY_CHECKPOINTER_EXPORTS,
            *_LAZY_THREADING_EXPORTS,
            *_LAZY_OBSERVABILITY_EXPORTS,
        }
    )
