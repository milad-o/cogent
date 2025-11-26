"""
Flow - the main entry point for AgenticFlow.

A Flow is a complete multi-agent system that handles:
- Agent coordination via topologies
- Tool registration (automatic)
- Event management (internal)
- Observability (configurable)
- Resilience (configurable)

Users only need to define agents and choose a topology pattern.
Everything else is wired automatically.

Example:
    ```python
    from langchain_openai import ChatOpenAI
    from langchain.tools import tool
    from agenticflow import Flow, Agent

    # Create model
    model = ChatOpenAI(model="gpt-4o")

    # Define tools
    @tool
    def search(query: str) -> str:
        '''Search for information.'''
        return f"Results for {query}"

    @tool
    def write(content: str) -> str:
        '''Write content.'''
        return f"Written: {content}"

    # Create agents (simplified API)
    researcher = Agent(
        name="Researcher",
        model=model,
        tools=[search],
    )

    writer = Agent(
        name="Writer",
        model=model,
        tools=[write],
    )

    # Create flow
    flow = Flow(
        name="content-team",
        agents=[researcher, writer],
        topology="pipeline",
    )

    # Run
    result = await flow.run("Research AI and write a summary")
    ```
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Callable, Sequence
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Literal

from langchain_core.tools import BaseTool

from agenticflow.agents.base import Agent
from agenticflow.core.enums import AgentRole
from agenticflow.events.bus import EventBus
from agenticflow.events.handlers import ConsoleEventHandler
from agenticflow.observability.progress import OutputConfig, ProgressTracker, Verbosity
from agenticflow.tasks.manager import TaskManager
from agenticflow.tools.registry import ToolRegistry
from agenticflow.topologies.base import (
    BaseTopology,
    HierarchicalTopology,
    MeshTopology,
    PipelineTopology,
    SupervisorTopology,
    TopologyConfig,
    TopologyState,
)
from agenticflow.topologies.policies import TopologyPolicy


class TopologyPattern(str, Enum):
    """Available topology patterns.
    
    Each pattern has specific requirements and behaviors:
    
    - SUPERVISOR: One coordinator delegates to workers. Requires a supervisor agent.
    - PIPELINE: Sequential processing through stages.
    - MESH: All agents can communicate with each other.
    - HIERARCHICAL: Agents organized in levels (org chart style).
    - CUSTOM: User-defined routing via policy.
    """
    
    SUPERVISOR = "supervisor"
    PIPELINE = "pipeline"
    MESH = "mesh"
    HIERARCHICAL = "hierarchical"
    CUSTOM = "custom"


@dataclass
class FlowConfig:
    """Configuration options for a Flow.
    
    Most users won't need this - Flow has sensible defaults.
    Use this for advanced customization.
    """
    
    # Execution
    max_iterations: int = 100
    recursion_limit: int = 50
    
    # Observability
    verbose: bool = False
    log_events: bool = False
    
    # Resilience (applied to all agents)
    enable_resilience: bool = True
    
    # Memory
    enable_checkpointing: bool = True
    
    # Metadata
    metadata: dict[str, Any] = field(default_factory=dict)


class Flow:
    """
    The main entry point for AgenticFlow.
    
    A Flow orchestrates multiple agents to accomplish complex tasks.
    It automatically handles:
    - Event bus setup and agent subscriptions
    - Tool registration from agent tool lists
    - Topology construction based on pattern
    - Progress tracking and observability
    
    Topology Patterns:
        - "supervisor": One agent coordinates others (requires supervisor role)
        - "pipeline": Sequential processing through agents
        - "mesh": All agents can communicate freely
        - "hierarchical": Org-chart style levels
        - "custom": User-defined routing
    
    Example - Basic Pipeline:
        ```python
        from langchain_openai import ChatOpenAI
        from agenticflow import Flow, Agent
        
        model = ChatOpenAI(model="gpt-4o")
        
        researcher = Agent(name="Researcher", model=model, tools=[search])
        writer = Agent(name="Writer", model=model, tools=[write])
        
        flow = Flow(
            name="content-team",
            agents=[researcher, writer],
            topology="pipeline",
        )
        
        result = await flow.run("Research AI trends and write a summary")
        ```
    
    Example - Supervisor with Workers:
        ```python
        supervisor = Agent(
            name="Manager", 
            role="supervisor",  # Required for supervisor topology
            model=model,
        )
        worker1 = Agent(name="Analyst", model=model, tools=[analyze])
        worker2 = Agent(name="Writer", model=model, tools=[write])
        
        flow = Flow(
            name="managed-team",
            agents=[supervisor, worker1, worker2],
            topology="supervisor",
        )
        ```
    
    Example - Streaming Progress:
        ```python
        async for state in flow.stream("Do complex task"):
            agent = state.current_agent
            if state.results:
                print(f"[{agent}]: {state.results[-1]['thought'][:100]}...")
        ```
    """
    
    def __init__(
        self,
        name: str,
        agents: Sequence[Agent],
        topology: TopologyPattern | str = "mesh",
        *,
        # Supervisor-specific
        supervisor: Agent | str | None = None,
        # Pipeline-specific
        stages: Sequence[str] | None = None,
        # Hierarchical-specific
        levels: list[list[str]] | None = None,
        # Custom policy
        policy: TopologyPolicy | None = None,
        # Configuration
        config: FlowConfig | None = None,
        # Observability shortcuts
        verbose: bool = False,
    ) -> None:
        """
        Create a Flow.
        
        Args:
            name: Name for this flow (for logging/debugging).
            agents: List of agents to include.
            topology: Coordination pattern ("supervisor", "pipeline", "mesh", etc).
            supervisor: For supervisor topology - the supervisor agent or its name.
            stages: For pipeline topology - ordered list of agent names.
            levels: For hierarchical topology - list of levels (each a list of names).
            policy: For custom topology - explicit routing policy.
            config: Advanced configuration options.
            verbose: Shortcut to enable verbose output.
        """
        self.name = name
        self._agents = list(agents)
        self._config = config or FlowConfig()
        
        if verbose:
            self._config.verbose = True
        
        # Parse topology pattern
        if isinstance(topology, str):
            topology = TopologyPattern(topology.lower())
        self._topology_pattern = topology
        
        # Store topology-specific params
        self._supervisor_param = supervisor
        self._stages_param = stages
        self._levels_param = levels
        self._policy_param = policy
        
        # Internal infrastructure (hidden from users)
        self._event_bus = EventBus()
        self._task_manager = TaskManager(self._event_bus)
        self._tool_registry = ToolRegistry()
        
        # Setup
        self._setup_tools()
        self._setup_agents()
        self._setup_event_handlers()
        
        # Build topology
        self._topology = self._build_topology()
    
    def _setup_tools(self) -> None:
        """Collect and register tools from all agents."""
        seen_tools: set[str] = set()
        
        for agent in self._agents:
            # Get tools from agent config
            for tool in agent.config.tools:
                if isinstance(tool, str):
                    # Tool name reference - skip, will be resolved later
                    continue
                elif isinstance(tool, BaseTool):
                    if tool.name not in seen_tools:
                        self._tool_registry.register(tool)
                        seen_tools.add(tool.name)
    
    def _setup_agents(self) -> None:
        """Wire agents to internal infrastructure."""
        for agent in self._agents:
            # Connect to event bus (agents auto-subscribe internally)
            agent.event_bus = self._event_bus
            
            # Connect to shared tool registry
            agent.tool_registry = self._tool_registry
    
    def _setup_event_handlers(self) -> None:
        """Setup observability based on config."""
        if self._config.verbose or self._config.log_events:
            handler = ConsoleEventHandler(verbose=self._config.verbose)
            self._event_bus.subscribe_all(handler)
    
    def _build_topology(self) -> BaseTopology:
        """Build the topology based on pattern and agents."""
        topology_config = TopologyConfig(
            name=self.name,
            max_iterations=self._config.max_iterations,
            enable_checkpointing=self._config.enable_checkpointing,
            recursion_limit=self._config.recursion_limit,
        )
        
        if self._topology_pattern == TopologyPattern.SUPERVISOR:
            return self._build_supervisor_topology(topology_config)
        elif self._topology_pattern == TopologyPattern.PIPELINE:
            return self._build_pipeline_topology(topology_config)
        elif self._topology_pattern == TopologyPattern.MESH:
            return self._build_mesh_topology(topology_config)
        elif self._topology_pattern == TopologyPattern.HIERARCHICAL:
            return self._build_hierarchical_topology(topology_config)
        elif self._topology_pattern == TopologyPattern.CUSTOM:
            return self._build_custom_topology(topology_config)
        else:
            raise ValueError(f"Unknown topology pattern: {self._topology_pattern}")
    
    def _build_supervisor_topology(self, config: TopologyConfig) -> SupervisorTopology:
        """Build supervisor topology, auto-detecting supervisor if needed."""
        supervisor_name: str | None = None
        
        # Check explicit supervisor param
        if self._supervisor_param:
            if isinstance(self._supervisor_param, Agent):
                supervisor_name = self._supervisor_param.name
            else:
                supervisor_name = self._supervisor_param
        else:
            # Auto-detect: look for agent with SUPERVISOR or COORDINATOR role
            for agent in self._agents:
                if agent.role in (AgentRole.SUPERVISOR, AgentRole.COORDINATOR):
                    supervisor_name = agent.name
                    break
        
        if not supervisor_name:
            raise ValueError(
                "Supervisor topology requires a supervisor agent. Either:\n"
                "1. Pass supervisor=<agent> to Flow()\n"
                "2. Create an agent with role='supervisor' or role='coordinator'\n"
                "3. Use a different topology pattern"
            )
        
        return SupervisorTopology(
            config=config,
            agents=self._agents,
            supervisor_name=supervisor_name,
            event_bus=self._event_bus,
        )
    
    def _build_pipeline_topology(self, config: TopologyConfig) -> PipelineTopology:
        """Build pipeline topology."""
        stages = self._stages_param
        if stages is None:
            # Default: use agent order
            stages = [a.name for a in self._agents]
        
        return PipelineTopology(
            config=config,
            agents=self._agents,
            stages=list(stages),
            event_bus=self._event_bus,
        )
    
    def _build_mesh_topology(self, config: TopologyConfig) -> MeshTopology:
        """Build mesh topology."""
        return MeshTopology(
            config=config,
            agents=self._agents,
            event_bus=self._event_bus,
        )
    
    def _build_hierarchical_topology(self, config: TopologyConfig) -> HierarchicalTopology:
        """Build hierarchical topology."""
        levels = self._levels_param
        if levels is None:
            raise ValueError(
                "Hierarchical topology requires levels parameter.\n"
                "Example: levels=[['ceo'], ['manager1', 'manager2'], ['worker1', 'worker2']]"
            )
        
        return HierarchicalTopology(
            config=config,
            agents=self._agents,
            levels=levels,
            event_bus=self._event_bus,
        )
    
    def _build_custom_topology(self, config: TopologyConfig) -> BaseTopology:
        """Build custom topology from policy."""
        if self._policy_param is None:
            raise ValueError(
                "Custom topology requires a policy parameter.\n"
                "Example: policy=TopologyPolicy.pipeline(['a', 'b', 'c'])"
            )
        
        return BaseTopology(
            config=config,
            agents=self._agents,
            policy=self._policy_param,
            event_bus=self._event_bus,
        )
    
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
        """Access the event bus (for advanced use)."""
        return self._event_bus
    
    @property
    def tool_registry(self) -> ToolRegistry:
        """Access the tool registry (for advanced use)."""
        return self._tool_registry
    
    async def run(
        self,
        task: str,
        context: dict[str, Any] | None = None,
        *,
        thread_id: str | None = None,
        on_step: Callable[[dict[str, Any]], None] | None = None,
    ) -> TopologyState:
        """
        Run the flow on a task.
        
        Args:
            task: The task description.
            context: Optional context dict.
            thread_id: Optional thread ID for memory/checkpointing.
            on_step: Optional callback after each agent step.
        
        Returns:
            Final topology state with results.
        
        Example:
            ```python
            result = await flow.run("Analyze sales data and write a report")
            
            print(f"Completed in {result.iteration} steps")
            for r in result.results:
                print(f"[{r['agent']}]: {r['thought'][:100]}...")
            ```
        """
        return await self._topology.run(
            task=task,
            context=context,
            thread_id=thread_id,
            on_step=on_step,
        )
    
    async def stream(
        self,
        task: str,
        context: dict[str, Any] | None = None,
        *,
        thread_id: str | None = None,
    ) -> AsyncIterator[dict[str, Any]]:
        """
        Stream flow execution, yielding state after each step.
        
        Args:
            task: The task description.
            context: Optional context dict.
            thread_id: Optional thread ID for memory.
        
        Yields:
            State dict after each agent step.
        
        Example:
            ```python
            async for state in flow.stream("Do complex task"):
                if state.get("results"):
                    latest = state["results"][-1]
                    print(f"[{latest['agent']}]: {latest['thought'][:100]}...")
            ```
        """
        async for state in self._topology.stream(
            task=task,
            context=context,
            thread_id=thread_id,
        ):
            yield state
    
    async def run_parallel(
        self,
        task: str,
        context: dict[str, Any] | None = None,
        *,
        agent_names: list[str] | None = None,
        merge_strategy: Literal["combine", "first", "vote"] = "combine",
    ) -> dict[str, Any]:
        """
        Run multiple agents in parallel on the same task.
        
        Useful for getting diverse perspectives or fan-out patterns.
        
        Args:
            task: Task for all agents.
            context: Optional shared context.
            agent_names: Which agents to run (default: all).
            merge_strategy: How to combine results.
        
        Returns:
            Dict with results, timing, errors, and merged result.
        
        Example:
            ```python
            results = await flow.run_parallel(
                "Analyze this data",
                agent_names=["analyst1", "analyst2", "analyst3"],
            )
            
            for name, thought in results["results"].items():
                print(f"[{name}]: {thought[:100]}...")
            ```
        """
        return await self._topology.run_parallel(
            task=task,
            context=context,
            agent_names=agent_names,
            merge_strategy=merge_strategy,
        )
    
    def add_agent(self, agent: Agent) -> None:
        """
        Add an agent to the flow.
        
        Note: This rebuilds the topology graph.
        
        Args:
            agent: Agent to add.
        """
        # Wire agent
        agent.event_bus = self._event_bus
        agent.tool_registry = self._tool_registry
        
        # Add to list
        self._agents.append(agent)
        
        # Register any new tools
        for tool in agent.config.tools:
            if isinstance(tool, BaseTool):
                if tool.name not in self._tool_registry:
                    self._tool_registry.register(tool)
        
        # Rebuild topology
        self._topology = self._build_topology()
    
    def remove_agent(self, name: str) -> bool:
        """
        Remove an agent from the flow.
        
        Note: This rebuilds the topology graph.
        
        Args:
            name: Name of agent to remove.
        
        Returns:
            True if agent was removed.
        """
        for i, agent in enumerate(self._agents):
            if agent.name == name:
                self._agents.pop(i)
                self._topology = self._build_topology()
                return True
        return False
    
    def draw_mermaid(
        self,
        *,
        theme: str = "default",
        direction: str = "TB",
        show_tools: bool = True,
        show_roles: bool = True,
    ) -> str:
        """
        Generate a Mermaid diagram of the flow.
        
        Args:
            theme: Mermaid theme.
            direction: Graph direction (TB, LR, etc).
            show_tools: Show agent tools.
            show_roles: Show agent roles.
        
        Returns:
            Mermaid diagram code.
        """
        return self._topology.draw_mermaid(
            theme=theme,
            direction=direction,
            title=self.name,
            show_tools=show_tools,
            show_roles=show_roles,
        )
    
    def __repr__(self) -> str:
        return (
            f"Flow(name={self.name!r}, "
            f"topology={self._topology_pattern.value!r}, "
            f"agents={[a.name for a in self._agents]})"
        )


# ==================== Convenience Functions ====================


def create_flow(
    name: str,
    agents: Sequence[Agent],
    topology: str = "mesh",
    **kwargs: Any,
) -> Flow:
    """
    Create a flow with the given agents and topology.
    
    This is a convenience function equivalent to Flow(...).
    
    Args:
        name: Flow name.
        agents: List of agents.
        topology: Topology pattern.
        **kwargs: Additional Flow arguments.
    
    Returns:
        Configured Flow instance.
    """
    return Flow(name=name, agents=agents, topology=topology, **kwargs)


def supervisor_flow(
    name: str,
    supervisor: Agent,
    workers: Sequence[Agent],
    **kwargs: Any,
) -> Flow:
    """
    Create a supervisor flow quickly.
    
    Args:
        name: Flow name.
        supervisor: The supervisor agent.
        workers: Worker agents.
        **kwargs: Additional Flow arguments.
    
    Returns:
        Configured supervisor Flow.
    """
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
    """
    Create a pipeline flow quickly.
    
    Args:
        name: Flow name.
        stages: Agents in pipeline order.
        **kwargs: Additional Flow arguments.
    
    Returns:
        Configured pipeline Flow.
    """
    return Flow(
        name=name,
        agents=stages,
        topology="pipeline",
        **kwargs,
    )


def mesh_flow(
    name: str,
    agents: Sequence[Agent],
    **kwargs: Any,
) -> Flow:
    """
    Create a mesh flow quickly.
    
    Args:
        name: Flow name.
        agents: Peer agents.
        **kwargs: Additional Flow arguments.
    
    Returns:
        Configured mesh Flow.
    """
    return Flow(
        name=name,
        agents=agents,
        topology="mesh",
        **kwargs,
    )
