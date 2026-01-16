"""Supervisor pattern - coordinator delegates to workers.

The supervisor pattern has one coordinator agent that delegates tasks
to worker agents and aggregates their results.

```
        ┌→ Worker A ─┐
Task → Supervisor ──┼→ Worker B ─┼→ Result
        └→ Worker C ─┘
```

Example:
    ```python
    from agenticflow.flow import supervisor

    flow = supervisor(
        coordinator=manager,
        workers=[analyst, writer, reviewer],
    )
    result = await flow.run("Complete this project")
    ```
"""

from __future__ import annotations

from dataclasses import asdict
from typing import TYPE_CHECKING, Any

from agenticflow.flow.config import FlowConfig
from agenticflow.flow.core import Flow

if TYPE_CHECKING:
    from agenticflow.agent.base import Agent
    from agenticflow.reactors.base import Reactor

def supervisor(
    coordinator: Agent | Reactor,
    workers: list[Agent | Reactor],
    *,
    config: FlowConfig | None = None,
    start_event: str = "task.created",
    done_event: str = "flow.done",
    parallel: bool = True,
    aggregate: bool = True,
) -> Flow:
    """Create a supervisor flow with coordinator and workers.

    The coordinator receives the initial task and can delegate to workers.
    Workers process their assigned tasks and report back.

    Event flow:
    1. `task.created` → coordinator
    2. Coordinator emits `delegate.*` events
    3. Workers react to delegation events
    4. Workers emit `worker.*.done` events
    5. Coordinator aggregates results (if aggregate=True)
    6. Coordinator emits `flow.done`

    Args:
        coordinator: The coordinating agent/reactor
        workers: List of worker agents/reactors
        config: Optional flow configuration
        start_event: Event that starts the flow
        done_event: Event that signals completion
        parallel: Whether workers can run in parallel
        aggregate: Whether coordinator aggregates worker results

    Returns:
        Configured Flow instance

    Example:
        ```python
        # Manager coordinates analysts and writers
        flow = supervisor(
            coordinator=manager,
            workers=[data_analyst, research_analyst, writer],
        )

        result = await flow.run("Analyze market trends and write report")
        ```

    Example with sequential workers:
        ```python
        flow = supervisor(
            coordinator=manager,
            workers=[researcher, writer, editor],
            parallel=False,  # Workers process sequentially
        )
        ```
    """
    if not workers:
        raise ValueError("Supervisor requires at least one worker")

    # Create flow
    flow_config = config or FlowConfig()
    if done_event not in flow_config.stop_events:
        config_dict = asdict(flow_config)
        config_dict["stop_events"] = frozenset([*flow_config.stop_events, done_event])
        flow_config = FlowConfig(**config_dict)

    flow = Flow(config=flow_config)

    coordinator_name = _get_name(coordinator, "coordinator")

    # Register coordinator
    # Coordinator handles initial task and worker completions
    flow.register(
        coordinator,
        on=[start_event, "worker.*.done"],
        name=coordinator_name,
        emits=done_event,
    )

    # Register workers
    for i, worker in enumerate(workers):
        worker_name = _get_name(worker, f"worker_{i}")

        # Workers listen for delegation from coordinator
        flow.register(
            worker,
            on=[f"delegate.{worker_name}", f"{coordinator_name}.delegate"],
            name=worker_name,
            emits=f"worker.{worker_name}.done",
        )

    return flow


def _get_name(reactor: Any, default: str) -> str:
    """Get reactor name or default."""
    if hasattr(reactor, "name") and reactor.name:
        return reactor.name
    return default


def coordinator(
    coordinator: Agent | Reactor,
    workers: list[Agent | Reactor],
    **kwargs: Any,
) -> Flow:
    """Alias for supervisor()."""
    return supervisor(coordinator, workers, **kwargs)
