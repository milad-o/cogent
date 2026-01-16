"""Pipeline pattern - sequential agent processing.

The pipeline pattern connects agents in a chain where each agent's output
becomes the next agent's input. Events flow linearly through the chain.

```
A → B → C → done
```

Example:
    ```python
    from agenticflow.flow import pipeline

    flow = pipeline([researcher, writer, editor])
    result = await flow.run("Write a blog post")
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


def pipeline(
    stages: list[Agent | Reactor],
    *,
    config: FlowConfig | None = None,
    start_event: str = "task.created",
    done_event: str = "flow.done",
) -> Flow:
    """Create a pipeline flow with sequential agent processing.

    Each agent processes in order, with output passed to the next stage.
    The pipeline automatically wires events between stages.

    Args:
        stages: List of agents/reactors to chain
        config: Optional flow configuration
        start_event: Event type that starts the pipeline (default: "task.created")
        done_event: Event type emitted when pipeline completes (default: "flow.done")

    Returns:
        Configured Flow instance

    Example:
        ```python
        # Create pipeline: research → write → edit
        flow = pipeline([researcher, writer, editor])

        # Run the pipeline
        result = await flow.run("Write about AI")
        print(result.output)
        ```

    Example with custom events:
        ```python
        flow = pipeline(
            [stage1, stage2, stage3],
            start_event="pipeline.start",
            done_event="pipeline.complete",
        )

        result = await flow.run(
            "Process this",
            initial_event="pipeline.start",
        )
        ```
    """
    if not stages:
        raise ValueError("Pipeline requires at least one stage")

    # Create flow with stop event
    flow_config = config or FlowConfig()
    if done_event not in flow_config.stop_events:
        config_dict = asdict(flow_config)
        config_dict["stop_events"] = frozenset([*flow_config.stop_events, done_event])
        flow_config = FlowConfig(**config_dict)

    flow = Flow(config=flow_config)

    # Wire stages in sequence
    for i, stage in enumerate(stages):
        # Determine event names
        if i == 0:
            on_event = start_event
        else:
            prev_name = _get_stage_name(stages[i - 1], i - 1)
            on_event = f"{prev_name}.done"

        # Determine emit event
        if i == len(stages) - 1:
            emits = done_event
        else:
            stage_name = _get_stage_name(stage, i)
            emits = f"{stage_name}.done"

        # Register with auto-wiring
        flow.register(stage, on=on_event, emits=emits)

    return flow


def _get_stage_name(stage: Any, index: int) -> str:
    """Get a name for a stage."""
    if hasattr(stage, "name") and stage.name:
        return stage.name
    return f"stage_{index}"


def chain(
    stages: list[Agent | Reactor],
    **kwargs: Any,
) -> Flow:
    """Alias for pipeline()."""
    return pipeline(stages, **kwargs)
