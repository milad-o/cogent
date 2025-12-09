"""Flow orchestration module.

This module provides the core flow classes for multi-agent orchestration:

- **Flow**: Imperative/topology-based orchestration (supervisor, pipeline, mesh, hierarchical)
- **ReactiveFlow**: Event-driven reactive orchestration (chain, fanout, route, saga)

Both inherit from BaseFlow and implement the FlowProtocol.

Example - Imperative Flow (topology-based):
    ```python
    from agenticflow import Agent, Flow

    researcher = Agent(name="researcher", model=model)
    writer = Agent(name="writer", model=model)

    # Pipeline topology: research → write
    flow = Flow(
        name="content-team",
        agents=[researcher, writer],
        topology="pipeline",
        verbose=True,
    )

    result = await flow.run("Create a blog post about AI")
    ```

Example - Reactive Flow (event-driven):
    ```python
    from agenticflow import Agent, ReactiveFlow, on

    researcher = Agent(name="researcher", model=model)
    writer = Agent(name="writer", model=model)

    # Event-driven: agents react to events
    flow = ReactiveFlow(observer=Observer.trace())
    flow.register(researcher, [on("task.created")])
    flow.register(writer, [on("researcher.completed")])

    result = await flow.run("Write about quantum computing")
    ```
"""

# Base classes and protocols
from agenticflow.flow.base import (
    BaseFlow,
    FlowProtocol,
    FlowResult,
)

# Imperative flow (topology-based)
from agenticflow.flow.orchestrated import (
    Flow,
    FlowConfig,
    create_flow,
    supervisor_flow,
    pipeline_flow,
    mesh_flow,
)

# Reactive flow (event-driven)
from agenticflow.flow.reactive import (
    ReactiveFlow,
    ReactiveFlowConfig,
    ReactiveFlowResult,
    # Backward compatibility aliases
    EventFlow,
    EventFlowConfig,
    EventFlowResult,
)

__all__ = [
    # Base
    "BaseFlow",
    "FlowProtocol",
    "FlowResult",
    # Imperative Flow
    "Flow",
    "FlowConfig",
    "create_flow",
    "supervisor_flow",
    "pipeline_flow",
    "mesh_flow",
    # Reactive Flow
    "ReactiveFlow",
    "ReactiveFlowConfig",
    "ReactiveFlowResult",
    # Backward compatibility
    "EventFlow",
    "EventFlowConfig",
    "EventFlowResult",
]
