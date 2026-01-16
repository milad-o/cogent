"""Reactors - event handlers for event-driven flows.

Reactors are the building blocks of event-driven orchestration.
They subscribe to events and optionally emit new events.

Built-in Reactors:
- **AgentReactor**: Wraps an Agent for event-driven execution
- **FunctionReactor**: Wraps a plain function
- **Aggregator**: Collects multiple events (fan-in patterns)
- **Router**: Routes events based on conditions
- **Transform**: Transforms event data
- **Gateway**: Bridges external systems

Example:
    ```python
    from agenticflow import Flow
    from agenticflow.reactors import Aggregator, Router, Transform

    flow = Flow()

    # Agents are auto-wrapped as AgentReactor
    flow.register(agent, on="task.created")

    # Functions are auto-wrapped as FunctionReactor
    flow.register(lambda e: e.data["x"] * 2, on="data.ready")

    # Use built-in reactors for complex patterns
    flow.register(
        Aggregator(collect=3, emit="all.done"),
        on="worker.done",
    )

    flow.register(
        Router({"high": "priority.high", "low": "priority.low"}, key="level"),
        on="task.classified",
    )
    ```
"""

# AgentReactor is in agent/ to avoid circular imports
# Import it here for convenience
from agenticflow.agent.reactor import AgentReactor, wrap_agent
from agenticflow.reactors.aggregator import (
    Aggregator,
    FirstWins,
    WaitAll,
)
from agenticflow.reactors.base import (
    BaseReactor,
    ErrorPolicy,
    FanInMode,
    HandoverStrategy,
    Reactor,
    ReactorConfig,
)
from agenticflow.reactors.function import (
    FunctionReactor,
    function_reactor,
)
from agenticflow.reactors.gateway import (
    CallbackGateway,
    Gateway,
    HttpGateway,
    LogGateway,
)
from agenticflow.reactors.router import (
    ConditionalRouter,
    Router,
)
from agenticflow.reactors.transform import (
    MapTransform,
    Transform,
)

__all__ = [
    # Base
    "Reactor",
    "ReactorConfig",
    "BaseReactor",
    "FanInMode",
    "HandoverStrategy",
    "ErrorPolicy",
    # Function
    "FunctionReactor",
    "function_reactor",
    # Aggregator
    "Aggregator",
    "FirstWins",
    "WaitAll",
    # Router
    "Router",
    "ConditionalRouter",
    # Transform
    "Transform",
    "MapTransform",
    # Gateway
    "Gateway",
    "HttpGateway",
    "LogGateway",
    "CallbackGateway",
    # Agent
    "AgentReactor",
    "wrap_agent",
]
