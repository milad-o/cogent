"""
Flow - the main entry point for AgenticFlow.

A Flow orchestrates multiple agents using simple coordination patterns:
- Supervisor: One agent coordinates and delegates to workers
- Pipeline: Sequential processing A → B → C
- Mesh: All agents collaborate in rounds
- Hierarchical: Tree structure with delegation levels

Example:
    ```python
    from agenticflow import Agent, Flow
    from agenticflow.models import ChatModel

    model = ChatModel(model="gpt-4o")
    
    researcher = Agent(name="researcher", model=model)
    writer = Agent(name="writer", model=model)
    editor = Agent(name="editor", model=model)

    # Pipeline: research → write → edit
    flow = Flow(
        name="content-team",
        agents=[researcher, writer, editor],
        topology="pipeline",
        verbose=True,  # Simple observability
    )

    result = await flow.run("Create a blog post about AI")
    print(result.output)
    ```
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

from agenticflow.observability.bus import EventBus
from agenticflow.observability.handlers import ConsoleEventHandler
from agenticflow.tools.base import BaseTool
from agenticflow.tools.registry import ToolRegistry
from agenticflow.topologies import (
    AgentConfig,
    BaseTopology,
    Hierarchical,
    Mesh,
    Pipeline,
    Supervisor,
    TopologyResult,
    TopologyType,
)

if TYPE_CHECKING:
    from agenticflow.observability.observer import Observer

# Import Agent for runtime isinstance checks
from agenticflow.agent import Agent

# Type for simple verbosity levels
VerbosityLevel = Literal[False, True, "minimal", "verbose", "debug", "trace"]


def _create_observer_from_verbose(verbose: VerbosityLevel) -> Observer:
    """Create a Observer from a simple verbosity level.
    
    Args:
        verbose: Verbosity level setting
        
    Returns:
        Configured Observer instance
    """
    from agenticflow.observability.observer import Observer
    
    if verbose is True or verbose == "minimal":
        # Basic progress - agent start/complete with timing
        return Observer.progress()
    elif verbose == "verbose":
        # Show agent outputs/thoughts
        return Observer.verbose()
    elif verbose == "debug":
        # Show everything including tool calls
        return Observer.debug()
    elif verbose == "trace":
        # Maximum detail + execution graph
        return Observer.trace()
    else:
        # Default to progress for any truthy value
        return Observer.progress()


@dataclass
class FlowConfig:
    """Configuration options for a Flow."""

    max_rounds: int = 3
    """Maximum rounds for mesh collaboration."""

    parallel: bool = True
    """Run workers in parallel (for supervisor pattern)."""

    verbose: bool = False
    """Enable verbose logging."""

    metadata: dict[str, Any] = field(default_factory=dict)
    """Additional metadata."""


class Flow:
    """
    Orchestrate multiple agents using coordination patterns.

    A Flow wraps agents in a topology (supervisor, pipeline, mesh, hierarchical)
    and provides a simple run() interface.

    Patterns:
        - **supervisor**: One agent coordinates, others do the work
        - **pipeline**: Sequential processing through stages
        - **mesh**: All agents collaborate with rounds of feedback
        - **hierarchical**: Tree structure with managers delegating to teams

    Example - Pipeline:
        ```python
        from agenticflow import Agent, Flow
        from agenticflow.models import ChatModel

        model = ChatModel(model="gpt-4o")

        flow = Flow(
            name="content",
            agents=[
                Agent(name="researcher", model=model),
                Agent(name="writer", model=model),
            ],
            topology="pipeline",
        )

        result = await flow.run("Write about quantum computing")
        print(result.output)
        ```

    Example - Supervisor:
        ```python
        flow = Flow(
            name="team",
            agents=[manager, analyst, writer],
            topology="supervisor",
            supervisor="manager",  # Who coordinates
        )
        ```

    Example - Mesh:
        ```python
        flow = Flow(
            name="brainstorm",
            agents=[analyst1, analyst2, analyst3],
            topology="mesh",
            max_rounds=2,  # Collaboration rounds
        )
        ```
    
    Observability Levels:
        verbose=False      - No output (silent)
        verbose=True       - Progress updates (agent start/complete with timing)
        verbose="verbose"  - Show agent outputs/thoughts
        verbose="debug"    - Show everything including tool calls
        verbose="trace"    - Maximum detail + execution graph
        
        Or pass a Observer for full customization.
    """

    def __init__(
        self,
        name: str,
        agents: Sequence[Agent],
        topology: str = "pipeline",
        *,
        # Supervisor options
        supervisor: Agent | str | None = None,
        parallel: bool = True,
        # Mesh options
        max_rounds: int = 3,
        synthesizer: Agent | None = None,
        # Hierarchical options
        structure: dict[str, list[Agent]] | None = None,
        # General options
        config: FlowConfig | None = None,
        # Observability - simple API
        verbose: VerbosityLevel = False,
        # Observability - advanced API
        observer: Observer | None = None,
    ) -> None:
        """
        Create a Flow.

        Args:
            name: Name for this flow.
            agents: List of agents to coordinate.
            topology: Pattern - "supervisor", "pipeline", "mesh", "hierarchical".
            supervisor: For supervisor pattern - which agent coordinates.
            parallel: For supervisor - run workers in parallel.
            max_rounds: For mesh - number of collaboration rounds.
            synthesizer: For mesh - dedicated agent to synthesize results.
            structure: For hierarchical - {manager_name: [subordinate_agents]}.
            config: Advanced configuration.
            verbose: Simple observability control:
                - False: Silent (no output)
                - True: Progress (agent transitions with timing)
                - "verbose": Show agent outputs/thoughts
                - "debug": Show everything including tool calls
                - "trace": Maximum detail + execution graph
            observer: Observer for full customization (overrides verbose).
            
        Example:
            ```python
            # Simple - just see progress
            flow = Flow(..., verbose=True)
            
            # See agent thoughts
            flow = Flow(..., verbose="verbose")
            
            # See everything including tools
            flow = Flow(..., verbose="debug")
            
            # Full customization
            from agenticflow import Observer
            observer = Observer(
                level=ObservabilityLevel.DEBUG,
                channels=[Channel.AGENTS, Channel.TOOLS],
                show_timestamps=True,
                on_error=my_error_handler,
            )
            flow = Flow(..., observer=observer)
            ```
        """
        self.name = name
        self._agents = list(agents)
        self._config = config or FlowConfig()
        
        # Handle observability
        if observer is not None:
            self._observer = observer
        elif verbose:
            self._observer = _create_observer_from_verbose(verbose)
        else:
            self._observer = None

        # Override config with explicit params
        if max_rounds != 3:
            self._config.max_rounds = max_rounds
        if not parallel:
            self._config.parallel = parallel

        # Store topology-specific params
        self._supervisor_param = supervisor
        self._synthesizer_param = synthesizer
        self._structure_param = structure

        # Normalize topology name
        self._topology_type = TopologyType(topology.lower())

        # Internal infrastructure
        self._event_bus = EventBus()
        self._tool_registry = ToolRegistry()

        # Setup
        self._setup_tools()
        self._setup_agents()
        self._setup_event_handlers()

        # Build topology
        self._topology = self._build_topology()

    def _setup_tools(self) -> None:
        """Register tools from all agents."""
        seen: set[str] = set()
        for agent in self._agents:
            for tool in agent.config.tools:
                if isinstance(tool, BaseTool) and tool.name not in seen:
                    self._tool_registry.register(tool)
                    seen.add(tool.name)

    def _setup_agents(self) -> None:
        """Wire agents to infrastructure."""
        for agent in self._agents:
            agent.event_bus = self._event_bus
            agent.tool_registry = self._tool_registry

    def _setup_event_handlers(self) -> None:
        """Setup event handlers."""
        # Observer handles all observability now
        if self._observer is not None:
            self._observer.attach(self._event_bus)

    def _build_topology(self) -> BaseTopology:
        """Build the coordination topology."""
        # Convert agents to AgentConfig
        agent_configs = [
            AgentConfig(
                agent=a,
                name=a.name,
                role=getattr(a.config, "role", None),
            )
            for a in self._agents
        ]

        if self._topology_type == TopologyType.SUPERVISOR:
            return self._build_supervisor(agent_configs)
        elif self._topology_type == TopologyType.PIPELINE:
            return self._build_pipeline(agent_configs)
        elif self._topology_type == TopologyType.MESH:
            return self._build_mesh(agent_configs)
        elif self._topology_type == TopologyType.HIERARCHICAL:
            return self._build_hierarchical(agent_configs)
        else:
            raise ValueError(f"Unknown topology: {self._topology_type}")

    def _build_supervisor(self, configs: list[AgentConfig]) -> Supervisor:
        """Build supervisor topology."""
        # Determine coordinator
        coord_name = self._supervisor_param
        if isinstance(coord_name, Agent):
            coord_name = coord_name.name
        elif coord_name is None:
            # Use first agent as coordinator
            coord_name = configs[0].name

        # Find coordinator config
        coordinator = None
        workers = []
        for cfg in configs:
            if cfg.name == coord_name:
                coordinator = cfg
            else:
                workers.append(cfg)

        if coordinator is None:
            raise ValueError(f"Supervisor '{coord_name}' not found in agents")

        return Supervisor(
            coordinator=coordinator,
            workers=workers,
            parallel=self._config.parallel,
        )

    def _build_pipeline(self, configs: list[AgentConfig]) -> Pipeline:
        """Build pipeline topology."""
        return Pipeline(stages=configs)

    def _build_mesh(self, configs: list[AgentConfig]) -> Mesh:
        """Build mesh topology."""
        synthesizer_cfg = None
        if self._synthesizer_param:
            for cfg in configs:
                if cfg.agent is self._synthesizer_param:
                    synthesizer_cfg = cfg
                    break

        return Mesh(
            agents=configs,
            max_rounds=self._config.max_rounds,
            synthesizer=synthesizer_cfg,
        )

    def _build_hierarchical(self, configs: list[AgentConfig]) -> Hierarchical:
        """Build hierarchical topology."""
        # Map agents to configs
        agent_to_config = {cfg.agent: cfg for cfg in configs}

        # Root is first agent
        root = configs[0]

        # Convert structure (Agent -> AgentConfig)
        structure: dict[str, list[AgentConfig]] = {}
        if self._structure_param:
            for manager_name, subordinates in self._structure_param.items():
                structure[manager_name] = [
                    agent_to_config[s] for s in subordinates if s in agent_to_config
                ]

        return Hierarchical(root=root, structure=structure)

    # ==================== Public API ====================

    @property
    def agents(self) -> dict[str, Agent]:
        """Get agents by name."""
        return {a.name: a for a in self._agents}

    @property
    def topology(self) -> BaseTopology:
        """Access the underlying topology."""
        return self._topology

    @property
    def event_bus(self) -> EventBus:
        """Access the event bus."""
        return self._event_bus

    async def run(self, task: str) -> TopologyResult:
        """
        Run the flow on a task.

        Args:
            task: The task description.

        Returns:
            TopologyResult with output and agent outputs.

        Example:
            ```python
            result = await flow.run("Analyze sales data and write a report")
            print(result.output)

            for name, output in result.agent_outputs.items():
                print(f"[{name}]: {output[:100]}...")
            ```
        """
        from agenticflow.core.enums import EventType
        
        # Emit user input event
        await self._event_bus.publish(EventType.USER_INPUT.value, {
            "content": task,
            "flow_name": self.name,
        })
        
        # Run the topology
        result = await self._topology.run(task)
        
        # Emit output generated event
        await self._event_bus.publish(EventType.OUTPUT_GENERATED.value, {
            "content": result.output,
            "flow_name": self.name,
            "agent_count": len(result.agent_outputs),
        })
        
        return result

    async def stream(self, task: str):
        """
        Stream flow execution.

        Args:
            task: The task description.

        Yields:
            Status updates and results.

        Example:
            ```python
            async for event in flow.stream("Do complex task"):
                print(event["type"], event.get("data", ""))
            ```
        """
        async for event in self._topology.stream(task):
            yield event

    def get_agent(self, name: str) -> Agent | None:
        """Get an agent by name."""
        return self.agents.get(name)

    def __repr__(self) -> str:
        return (
            f"Flow(name={self.name!r}, "
            f"topology={self._topology_type.value!r}, "
            f"agents={[a.name for a in self._agents]})"
        )


# ==================== Convenience Functions ====================


def create_flow(
    name: str,
    agents: Sequence[Agent],
    topology: str = "pipeline",
    **kwargs: Any,
) -> Flow:
    """Create a flow with the given agents and topology."""
    return Flow(name=name, agents=agents, topology=topology, **kwargs)


def supervisor_flow(
    name: str,
    supervisor: Agent,
    workers: Sequence[Agent],
    **kwargs: Any,
) -> Flow:
    """Create a supervisor flow."""
    return Flow(
        name=name,
        agents=[supervisor, *workers],
        topology="supervisor",
        supervisor=supervisor,
        **kwargs,
    )


def pipeline_flow(
    name: str,
    stages: Sequence[Agent],
    **kwargs: Any,
) -> Flow:
    """Create a pipeline flow."""
    return Flow(name=name, agents=stages, topology="pipeline", **kwargs)


def mesh_flow(
    name: str,
    agents: Sequence[Agent],
    max_rounds: int = 3,
    **kwargs: Any,
) -> Flow:
    """Create a mesh flow."""
    return Flow(
        name=name,
        agents=agents,
        topology="mesh",
        max_rounds=max_rounds,
        **kwargs,
    )
