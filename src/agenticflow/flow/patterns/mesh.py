"""Mesh pattern - collaborative multi-agent discussion.

The mesh pattern enables all agents to collaborate through multiple
rounds of discussion until consensus or max rounds.

```
    A ←→ B
    ↕   ↕  × N rounds
    C ←→ D
```

Example:
    ```python
    from agenticflow.flow import mesh

    flow = mesh(
        agents=[expert1, expert2, expert3],
        max_rounds=3,
    )
    result = await flow.run("Debate this topic")
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


def mesh(
    agents: list[Agent | Reactor],
    *,
    config: FlowConfig | None = None,
    start_event: str = "task.created",
    done_event: str = "flow.done",
    max_rounds: int = 3,
    consensus_event: str | None = "consensus.reached",
) -> Flow:
    """Create a mesh flow for multi-agent collaboration.

    All agents participate in discussion rounds. Each agent can see
    outputs from previous agents in the current round.

    Event flow:
    1. `task.created` triggers round 1
    2. All agents process in round N
    3. At end of round, `round.N.done` is emitted
    4. If not converged and rounds < max, start round N+1
    5. `flow.done` when complete

    Args:
        agents: List of collaborating agents
        config: Optional flow configuration
        start_event: Event that starts the collaboration
        done_event: Event that signals completion
        max_rounds: Maximum collaboration rounds
        consensus_event: Optional event type for early termination

    Returns:
        Configured Flow instance

    Example:
        ```python
        # Three experts debate a topic
        flow = mesh(
            agents=[economist, sociologist, technologist],
            max_rounds=3,
        )

        result = await flow.run("Discuss the future of work")
        ```

    Example with consensus detection:
        ```python
        flow = mesh(
            agents=[expert1, expert2, expert3],
            consensus_event="agreement.reached",  # Early exit
        )
        ```
    """
    if len(agents) < 2:
        raise ValueError("Mesh requires at least 2 agents")

    # Create flow
    flow_config = config or FlowConfig(max_rounds=max_rounds * len(agents) + 10)
    stop_events = set(flow_config.stop_events)
    stop_events.add(done_event)
    if consensus_event:
        stop_events.add(consensus_event)

    # Create new config with updated stop_events
    config_dict = asdict(flow_config)
    config_dict["stop_events"] = frozenset(stop_events)
    flow_config = FlowConfig(**config_dict)

    flow = Flow(config=flow_config)

    # In mesh, all agents listen to:
    # 1. Initial task
    # 2. Other agents' outputs
    agent_names = [_get_name(agent, f"agent_{i}") for i, agent in enumerate(agents)]

    for i, agent in enumerate(agents):
        name = agent_names[i]

        # Each agent reacts to task and other agents' contributions
        other_events = [f"{n}.contributed" for n in agent_names if n != name]

        flow.register(
            agent,
            on=[start_event, "round.*.start", *other_events],
            name=name,
            emits=f"{name}.contributed",
        )

    # Add round controller (simple function reactor)
    round_tracker: dict[str, int] = {"current": 1, "contributions": 0}
    total_agents = len(agents)

    async def round_controller(event: Any) -> Any:
        """Track contributions and manage rounds."""
        from agenticflow.events import Event

        if event.type.endswith(".contributed"):
            round_tracker["contributions"] += 1

            if round_tracker["contributions"] >= total_agents:
                current_round = round_tracker["current"]
                round_tracker["contributions"] = 0

                if current_round >= max_rounds:
                    # Final round complete
                    return Event(
                        type=done_event,
                        source="round_controller",
                        data={
                            "rounds_completed": current_round,
                            "output": event.data,
                        },
                    )
                else:
                    # Start next round
                    round_tracker["current"] += 1
                    return Event(
                        type=f"round.{round_tracker['current']}.start",
                        source="round_controller",
                        data={"round": round_tracker["current"]},
                    )

        return None

    flow.register(
        round_controller,
        on="*.contributed",
        name="round_controller",
    )

    return flow


def _get_name(reactor: Any, default: str) -> str:
    """Get reactor name or default."""
    if hasattr(reactor, "name") and reactor.name:
        return reactor.name
    return default


def collaborative(
    agents: list[Agent | Reactor],
    **kwargs: Any,
) -> Flow:
    """Alias for mesh()."""
    return mesh(agents, **kwargs)


def brainstorm(
    agents: list[Agent | Reactor],
    **kwargs: Any,
) -> Flow:
    """Alias for mesh()."""
    return mesh(agents, **kwargs)
