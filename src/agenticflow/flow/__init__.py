"""Flow orchestration module.

This module provides the unified Flow class for event-driven multi-agent orchestration.

The Flow is the core orchestration engine that coordinates reactors (event handlers)
through an event bus. All orchestration patterns (pipeline, supervisor, mesh) are
implemented as functions that configure a Flow instance.

Example - Basic Flow:
    ```python
    from agenticflow import Agent, Flow

    researcher = Agent(name="researcher", model=model)
    writer = Agent(name="writer", model=model)

    # Event-driven orchestration
    flow = Flow()
    flow.register(researcher, on="task.created", emits="research.done")
    flow.register(writer, on="research.done", emits="flow.done")

    result = await flow.run("Create a blog post about AI")
    ```

Example - Using Patterns:
    ```python
    from agenticflow.flow import pipeline, supervisor

    # Pipeline pattern: sequential processing
    flow = pipeline([researcher, writer, editor])
    result = await flow.run("Write about quantum computing")

    # Supervisor pattern: one coordinator, many workers
    flow = supervisor(
        coordinator=manager,
        workers=[analyst, writer, reviewer],
    )
    result = await flow.run("Complete this task")
    ```
"""

# Core Flow class
# Base classes and protocols
from agenticflow.flow.base import (
    BaseFlow,
    FlowProtocol,
)
from agenticflow.flow.base import (
    FlowResult as BaseFlowResult,
)

# Configuration and results
from agenticflow.flow.config import (
    FlowConfig,
    FlowResult,
    ReactorBinding,
)

# Execution context
from agenticflow.flow.context import (
    Context,  # Preferred alias
    ExecutionContext,
    FlowContext,
    ReactiveContext,  # Backward compatibility alias
)
from agenticflow.flow.core import (
    # Backward compatibility
    EventFlow,
    Flow,
    ReactiveFlow,
)

# Pattern helpers
from agenticflow.flow.patterns import (
    brainstorm,
    chain,
    collaborative,
    coordinator,
    mesh,
    pipeline,
    supervisor,
)

__all__ = [
    # Core
    "Flow",
    # Configuration
    "FlowConfig",
    "FlowResult",
    "ReactorBinding",
    # Base
    "BaseFlow",
    "FlowProtocol",
    # Context
    "FlowContext",
    "ExecutionContext",
    "Context",
    # Patterns
    "pipeline",
    "chain",
    "supervisor",
    "coordinator",
    "mesh",
    "collaborative",
    "brainstorm",
    # Backward compatibility
    "EventFlow",
    "ReactiveFlow",
    "ReactiveContext",
    "BaseFlowResult",
]

