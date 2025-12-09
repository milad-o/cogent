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


from agenticflow.reactive.core import (
    AgentTriggerConfig,
    Reaction,
    ReactionType,
    Trigger,
    TriggerBuilder,
    TriggerCondition,
    react_to,
    on,  # Backward compat alias
    when,
)
# Re-export from flow package (canonical location)
from agenticflow.flow.reactive import (
    ReactiveFlow,
    ReactiveFlowConfig,
    ReactiveFlowResult,
    # Backward compatibility aliases
    EventFlow,
    EventFlowConfig,
    EventFlowResult,
)
from agenticflow.reactive.patterns import (
    # High-level functions (recommended)
    chain,
    fanout,
    route,
    # Mid-level classes
    Chain,
    FanIn,
    FanOut,
    Router,
    Saga,
    # Legacy compatibility
    ChainPattern,
    FanInPattern,
    FanOutPattern,
    Route,
    RouterPattern,
    SagaPattern,
    SagaStep,
)
# Re-export Observer for convenience
from agenticflow.observability.observer import Observer

__all__ = [
    # Core
    "AgentTriggerConfig",
    "Reaction",
    "ReactionType",
    "Trigger",
    "TriggerBuilder",
    "TriggerCondition",
    # Builders
    "on",
    "when",
    # Flow (new names)
    "ReactiveFlow",
    "ReactiveFlowConfig",
    "ReactiveFlowResult",
    # Flow (legacy aliases)
    "EventFlow",
    "EventFlowConfig",
    "EventFlowResult",
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

