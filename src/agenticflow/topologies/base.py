"""Base topology classes and interfaces.

Provides abstract base for multi-agent coordination patterns.
Users define topologies by specifying policies (handoff rules),
not by implementing graph internals.
"""

from dataclasses import dataclass, field
from typing import Any, Callable, Sequence
from enum import Enum

from langgraph.graph import StateGraph, START, END
from langgraph.graph.state import CompiledStateGraph
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.store.base import BaseStore
from langgraph.types import Command, interrupt

from agenticflow.agents import Agent
from agenticflow.events import EventBus
from agenticflow.topologies.policies import (
    TopologyPolicy,
    AgentPolicy,
    HandoffRule,
    HandoffCondition,
    AcceptancePolicy,
)


class HandoffStrategy(Enum):
    """How agents hand off work to each other."""

    COMMAND = "command"  # Use LangGraph Command for explicit handoffs
    INTERRUPT = "interrupt"  # Use interrupt for human-in-the-loop
    AUTOMATIC = "automatic"  # Let the topology decide routing
    BROADCAST = "broadcast"  # Send to all agents


@dataclass
class TopologyConfig:
    """Configuration for a multi-agent topology."""

    name: str
    description: str = ""
    max_iterations: int = 100
    handoff_strategy: HandoffStrategy = HandoffStrategy.AUTOMATIC
    enable_memory: bool = True
    enable_checkpointing: bool = True
    recursion_limit: int = 50
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class TopologyState:
    """Shared state across all agents in a topology.

    This is the state type for LangGraph's StateGraph.
    """

    messages: list[dict[str, Any]] = field(default_factory=list)
    current_agent: str | None = None
    task: str = ""
    context: dict[str, Any] = field(default_factory=dict)
    iteration: int = 0
    completed: bool = False
    error: str | None = None
    results: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convert state to dictionary."""
        return {
            "messages": self.messages,
            "current_agent": self.current_agent,
            "task": self.task,
            "context": self.context,
            "iteration": self.iteration,
            "completed": self.completed,
            "error": self.error,
            "results": self.results,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TopologyState":
        """Create state from dictionary."""
        return cls(
            messages=data.get("messages", []),
            current_agent=data.get("current_agent"),
            task=data.get("task", ""),
            context=data.get("context", {}),
            iteration=data.get("iteration", 0),
            completed=data.get("completed", False),
            error=data.get("error"),
            results=data.get("results", []),
        )


class BaseTopology:
    """Base class for multi-agent topologies.

    Define your topology by providing a policy that specifies
    handoff rules between agents. The graph is built automatically.

    Simple Example:
        >>> # Define agents
        >>> agents = [researcher, writer, reviewer]
        >>>
        >>> # Define policy - who talks to whom
        >>> policy = TopologyPolicy.pipeline(["researcher", "writer", "reviewer"])
        >>>
        >>> # Create topology
        >>> topology = BaseTopology(
        ...     config=TopologyConfig(name="content-pipeline"),
        ...     agents=agents,
        ...     policy=policy,
        ... )

    Custom Policy Example:
        >>> policy = TopologyPolicy(entry_point="gateway")
        >>> policy.add_rule("gateway", "validator")
        >>> policy.add_rule("validator", "processor", label="valid")
        >>> policy.add_rule("validator", "gateway", label="invalid")
        >>> policy.add_rule("processor", "storage")
        >>>
        >>> topology = BaseTopology(
        ...     config=TopologyConfig(name="custom-flow"),
        ...     agents=agents,
        ...     policy=policy,
        ... )
    """

    def __init__(
        self,
        config: TopologyConfig,
        agents: Sequence[Agent],
        policy: TopologyPolicy | None = None,
        event_bus: EventBus | None = None,
        checkpointer: BaseCheckpointSaver | None = None,
        store: BaseStore | None = None,
    ) -> None:
        """Initialize topology.

        Args:
            config: Topology configuration.
            agents: List of agents in the topology.
            policy: Handoff policy defining agent relationships.
                   If None, a default mesh policy is created.
            event_bus: Optional event bus for publishing events.
            checkpointer: Optional checkpointer for short-term memory.
            store: Optional store for long-term memory.
        """
        self.config = config
        self._agents = list(agents)
        self.agents = {agent.config.name: agent for agent in agents}
        self.event_bus = event_bus
        self.checkpointer = checkpointer
        self.store = store

        # Set up policy
        if policy is None:
            # Default to mesh (everyone can talk to everyone)
            policy = TopologyPolicy.mesh(list(self.agents.keys()))
        self._policy = policy

        # Ensure entry point is valid
        if self._policy.entry_point and self._policy.entry_point not in self.agents:
            raise ValueError(
                f"Entry point '{self._policy.entry_point}' not in agents"
            )

        self._compiled_graph: CompiledStateGraph | None = None

    @property
    def policy(self) -> TopologyPolicy:
        """Get the topology's handoff policy."""
        return self._policy

    @property
    def graph(self) -> CompiledStateGraph:
        """Get or build the compiled graph."""
        if self._compiled_graph is None:
            self._compiled_graph = self._build_graph()
        return self._compiled_graph

    def _build_graph(self) -> CompiledStateGraph:
        """Build the LangGraph StateGraph from the policy.

        The graph structure is determined by the policy's rules.
        Override this method only for very custom graph structures.

        Returns:
            Compiled StateGraph ready for execution.
        """
        builder = StateGraph(dict)

        # Add all agent nodes
        for name, agent in self.agents.items():
            builder.add_node(name, self._create_agent_node(agent))

        # Set entry point
        entry = self._policy.entry_point or next(iter(self.agents.keys()))
        builder.add_edge(START, entry)

        # Add edges based on policy
        agent_names = list(self.agents.keys())
        for agent_name in agent_names:
            # Get all possible targets for this agent
            targets = self._policy.get_allowed_targets(
                agent_name, {}, agent_names
            )

            agent_policy = self._policy.get_agent_policy(agent_name)

            if not targets and agent_policy.can_finish:
                # No targets, go to END
                builder.add_edge(agent_name, END)
            elif len(targets) == 1 and not agent_policy.can_finish:
                # Single target, no finish option
                builder.add_edge(agent_name, targets[0])
            else:
                # Multiple targets or can finish - use routing
                edge_map = {t: t for t in targets}
                if agent_policy.can_finish:
                    edge_map["__end__"] = END

                builder.add_conditional_edges(
                    agent_name,
                    self._create_router(agent_name),
                    edge_map,
                )

        return builder.compile(checkpointer=self.checkpointer)

    def _create_router(self, agent_name: str) -> Callable[[dict[str, Any]], str]:
        """Create a routing function for an agent.

        Args:
            agent_name: The agent to create router for.

        Returns:
            Router function.
        """

        def router(state: dict[str, Any]) -> str:
            """Route to next agent based on state and policy."""
            # Check iteration limit
            if state.get("iteration", 0) >= self.config.max_iterations:
                return "__end__"

            # Check if completed
            if state.get("completed"):
                return "__end__"

            # Check for explicit next agent request
            next_agent = state.get("next_agent")
            if next_agent:
                if next_agent == "__end__" or next_agent == "FINISH":
                    return "__end__"
                if self._policy.can_handoff(agent_name, next_agent, state):
                    return next_agent

            # Get allowed targets and pick first
            targets = self._policy.get_allowed_targets(
                agent_name, state, list(self.agents.keys())
            )

            if targets:
                # Use custom routing if provided
                return self._route(agent_name, state, targets)

            return "__end__"

        return router

    def _route(
        self,
        current_agent: str,
        state: dict[str, Any],
        allowed_targets: list[str],
    ) -> str:
        """Route to next agent from allowed targets.

        Override this method to implement custom routing logic.

        Args:
            current_agent: Current agent name.
            state: Current state.
            allowed_targets: List of allowed target agents.

        Returns:
            Name of next agent or "__end__".
        """
        # Default: return first allowed target
        if allowed_targets:
            return allowed_targets[0]
        return "__end__"

    def _create_agent_node(
        self, agent: Agent
    ) -> Callable[[dict[str, Any]], dict[str, Any]]:
        """Create a graph node function for an agent.

        Override this to customize agent processing.

        Args:
            agent: The agent to wrap.

        Returns:
            Node function that processes state and returns updates.
        """

        async def node(state: dict[str, Any]) -> dict[str, Any]:
            """Process state through agent."""
            # Publish event
            if self.event_bus:
                await self.event_bus.publish(
                    "topology.agent.start",
                    {
                        "topology": self.config.name,
                        "agent": agent.config.name,
                        "iteration": state.get("iteration", 0),
                    },
                )

            # Build prompt for agent including task and conversation context
            task = state.get("task", "")
            results = state.get("results", [])
            
            # Build prompt with context from previous agents
            prompt_parts = [f"Task: {task}"]
            
            if results:
                prompt_parts.append("\nPrevious contributions:")
                for r in results:
                    prompt_parts.append(f"\n[{r['agent']}]: {r['thought']}")
                prompt_parts.append("\n\nYour turn to contribute:")
            
            prompt = "\n".join(prompt_parts)

            # Let agent think about the task
            thought = await agent.think(prompt)

            # Build updated messages list
            messages = state.get("messages", [])
            new_messages = messages + [
                {
                    "role": "assistant",
                    "content": thought,
                    "agent": agent.config.name,
                }
            ]

            # Determine result
            result = {
                "messages": new_messages,
                "current_agent": agent.config.name,
                "iteration": state.get("iteration", 0) + 1,
                "results": state.get("results", [])
                + [
                    {
                        "agent": agent.config.name,
                        "thought": thought,
                    }
                ],
            }

            # Parse for next agent hint
            next_agent = self._parse_agent_output(thought, agent.config.name)
            if next_agent:
                result["next_agent"] = next_agent

            # Check for completion
            if self._is_task_complete(thought):
                result["completed"] = True

            # Publish completion event
            if self.event_bus:
                await self.event_bus.publish(
                    "topology.agent.complete",
                    {
                        "topology": self.config.name,
                        "agent": agent.config.name,
                        "thought_length": len(thought),
                    },
                )

            return result

        return node

    def _parse_agent_output(self, output: str, current_agent: str) -> str | None:
        """Parse agent output to determine next agent.

        Override this for custom output parsing.

        Args:
            output: Agent's output text.
            current_agent: Current agent name.

        Returns:
            Target agent name, "__end__", or None.
        """
        output_lower = output.lower()

        # Check for handoff mentions FIRST (higher priority than finish)
        handoff_keywords = ["hand off to", "pass to", "delegate to", "send to", "assigned to"]
        for keyword in handoff_keywords:
            if keyword in output_lower:
                for name in self.agents:
                    if name != current_agent and name.lower() in output_lower:
                        return name

        # Check for any agent mention (before finish signals)
        for name in self.agents:
            if name != current_agent and name.lower() in output_lower:
                return name

        # Check for explicit finish signals (only if no agent mentioned)
        finish_patterns = [
            "task is complete",
            "task is now complete",
            "task complete",
            "job is complete",
            "we can finish",
            "all done",
            "final answer:",
        ]
        for pattern in finish_patterns:
            if pattern in output_lower:
                return "__end__"

        return None

    def _is_task_complete(self, output: str) -> bool:
        """Check if task is complete based on output.

        Override this for custom completion detection.

        Args:
            output: Agent's output text.

        Returns:
            True if task appears complete.
        """
        # More precise completion detection
        completion_signals = [
            "task complete",
            "task is complete",
            "job is done",
            "all tasks complete",
            "final answer:",
            "here is the final",
        ]
        output_lower = output.lower()
        return any(signal in output_lower for signal in completion_signals)

    def handoff(
        self,
        target: str,
        state_update: dict[str, Any] | None = None,
        *,
        resume_value: Any = None,
    ) -> Command:
        """Create a handoff command to another agent.

        Uses LangGraph's Command for explicit routing.

        Args:
            target: Name of target agent or node.
            state_update: Optional state updates to apply.
            resume_value: Optional value to resume with after interrupt.

        Returns:
            LangGraph Command for handoff.
        """
        update = state_update or {}
        update["current_agent"] = target
        update["next_agent"] = target

        return Command(
            goto=target,
            update=update,
            resume=resume_value,
        )

    def request_human_input(
        self,
        question: str,
        state: dict[str, Any],
    ) -> Any:
        """Request human input using LangGraph interrupt.

        Args:
            question: Question to ask human.
            state: Current state for context.

        Returns:
            Human's response after resumption.
        """
        return interrupt(
            {
                "question": question,
                "context": state.get("context", {}),
                "current_agent": state.get("current_agent"),
            }
        )

    async def run(
        self,
        task: str,
        context: dict[str, Any] | None = None,
        thread_id: str | None = None,
    ) -> TopologyState:
        """Run the topology on a task.

        Args:
            task: The task to process.
            context: Optional initial context.
            thread_id: Optional thread ID for checkpointing.

        Returns:
            Final topology state.
        """
        from agenticflow.core import generate_id

        thread_id = thread_id or generate_id("thread")

        # Publish start event
        if self.event_bus:
            await self.event_bus.publish(
                "topology.run.start",
                {
                    "topology": self.config.name,
                    "task": task,
                    "thread_id": thread_id,
                },
            )

        # Initial state
        initial_state = {
            "messages": [],
            "current_agent": None,
            "task": task,
            "context": context or {},
            "iteration": 0,
            "completed": False,
            "error": None,
            "results": [],
        }

        # Run graph
        config = {
            "configurable": {"thread_id": thread_id},
            "recursion_limit": self.config.recursion_limit,
        }

        final_state = None
        async for state in self.graph.astream(initial_state, config, stream_mode="values"):
            final_state = state
            if self.event_bus:
                await self.event_bus.publish(
                    "topology.run.step",
                    {
                        "topology": self.config.name,
                        "state": state,
                    },
                )

        # Publish completion
        if self.event_bus:
            await self.event_bus.publish(
                "topology.run.complete",
                {
                    "topology": self.config.name,
                    "thread_id": thread_id,
                    "iterations": final_state.get("iteration", 0) if final_state else 0,
                },
            )

        return TopologyState.from_dict(final_state or initial_state)

    async def resume(
        self,
        thread_id: str,
        human_input: Any,
    ) -> TopologyState:
        """Resume execution after an interrupt.

        Args:
            thread_id: Thread ID to resume.
            human_input: Human's response to the interrupt.

        Returns:
            Final topology state after resumption.
        """
        config = {
            "configurable": {"thread_id": thread_id},
            "recursion_limit": self.config.recursion_limit,
        }

        # Resume with Command
        command = Command(resume=human_input)

        final_state = None
        async for state in self.graph.astream(command, config):
            final_state = state

        return TopologyState.from_dict(final_state or {})

    def draw_mermaid(
        self,
        *,
        theme: str = "default",
        direction: str = "TB",
        title: str | None = None,
        show_tools: bool = True,
        show_roles: bool = True,
    ) -> str:
        """Generate a Mermaid diagram showing this topology.

        Args:
            theme: Mermaid theme (default, forest, dark, neutral, base)
            direction: Graph direction (TB, TD, BT, LR, RL)
            title: Optional diagram title
            show_tools: Whether to show agent tools
            show_roles: Whether to show agent roles

        Returns:
            Mermaid diagram code as string
        """
        from agenticflow.visualization.mermaid import (
            MermaidConfig,
            MermaidDirection,
            MermaidTheme,
            TopologyDiagram,
        )

        theme_enum = MermaidTheme(theme)
        direction_enum = MermaidDirection(direction)

        config = MermaidConfig(
            title=title or "",
            theme=theme_enum,
            direction=direction_enum,
            show_tools=show_tools,
            show_roles=show_roles,
        )
        diagram = TopologyDiagram(self, config=config)
        return diagram.to_mermaid()

    def draw_mermaid_png(
        self,
        *,
        theme: str = "default",
        direction: str = "TB",
        title: str | None = None,
        show_tools: bool = True,
        show_roles: bool = True,
    ) -> bytes:
        """Generate a PNG image of this topology's Mermaid diagram.

        Requires httpx to be installed for mermaid.ink API.

        Args:
            theme: Mermaid theme (default, forest, dark, neutral, base)
            direction: Graph direction (TB, TD, BT, LR, RL)
            title: Optional diagram title
            show_tools: Whether to show agent tools
            show_roles: Whether to show agent roles

        Returns:
            PNG image as bytes
        """
        from agenticflow.visualization.mermaid import (
            MermaidConfig,
            MermaidDirection,
            MermaidTheme,
            TopologyDiagram,
        )

        theme_enum = MermaidTheme(theme)
        direction_enum = MermaidDirection(direction)

        config = MermaidConfig(
            title=title or "",
            theme=theme_enum,
            direction=direction_enum,
            show_tools=show_tools,
            show_roles=show_roles,
        )
        diagram = TopologyDiagram(self, config=config)
        return diagram.draw_png()


# ==================== Prebuilt Topology Classes ====================
# These are convenience subclasses that set up common patterns


class SupervisorTopology(BaseTopology):
    """Supervisor pattern: one coordinator, multiple workers.

    Example:
        >>> topology = SupervisorTopology(
        ...     config=TopologyConfig(name="team"),
        ...     agents=[supervisor, worker1, worker2],
        ...     supervisor_name="supervisor",
        ... )
    """

    def __init__(
        self,
        config: TopologyConfig,
        agents: Sequence[Agent],
        supervisor_name: str | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize supervisor topology.

        Args:
            config: Topology configuration.
            agents: List of agents (supervisor + workers).
            supervisor_name: Name of supervisor agent.
                           Defaults to first agent.
            **kwargs: Additional arguments for BaseTopology.
        """
        agent_names = [a.config.name for a in agents]

        if supervisor_name is None:
            supervisor_name = agent_names[0]

        if supervisor_name not in agent_names:
            raise ValueError(f"Supervisor '{supervisor_name}' not in agents")

        workers = [n for n in agent_names if n != supervisor_name]

        policy = TopologyPolicy.supervisor(supervisor_name, workers)

        super().__init__(config=config, agents=agents, policy=policy, **kwargs)

        self.supervisor_name = supervisor_name
        self.worker_names = workers


class PipelineTopology(BaseTopology):
    """Pipeline pattern: sequential agent processing.

    Example:
        >>> topology = PipelineTopology(
        ...     config=TopologyConfig(name="content-pipeline"),
        ...     agents=[researcher, writer, editor],
        ...     stages=["researcher", "writer", "editor"],
        ... )
    """

    def __init__(
        self,
        config: TopologyConfig,
        agents: Sequence[Agent],
        stages: Sequence[str] | None = None,
        allow_skip: bool = False,
        allow_repeat: bool = False,
        **kwargs: Any,
    ) -> None:
        """Initialize pipeline topology.

        Args:
            config: Topology configuration.
            agents: List of agents.
            stages: Ordered list of agent names.
                   Defaults to agent order.
            allow_skip: Allow skipping stages.
            allow_repeat: Allow going back.
            **kwargs: Additional arguments for BaseTopology.
        """
        agent_names = [a.config.name for a in agents]

        if stages is None:
            stages = agent_names
        else:
            stages = list(stages)
            for s in stages:
                if s not in agent_names:
                    raise ValueError(f"Stage '{s}' not in agents")

        policy = TopologyPolicy.pipeline(stages, allow_skip, allow_repeat)

        super().__init__(config=config, agents=agents, policy=policy, **kwargs)

        self.stages = stages
        self.allow_skip = allow_skip
        self.allow_repeat = allow_repeat


class MeshTopology(BaseTopology):
    """Mesh pattern: all agents can communicate with each other.

    Example:
        >>> topology = MeshTopology(
        ...     config=TopologyConfig(name="collaborative-team"),
        ...     agents=[analyst, reviewer, editor],
        ... )
    """

    def __init__(
        self,
        config: TopologyConfig,
        agents: Sequence[Agent],
        **kwargs: Any,
    ) -> None:
        """Initialize mesh topology.

        Args:
            config: Topology configuration.
            agents: List of agents.
            **kwargs: Additional arguments for BaseTopology.
        """
        agent_names = [a.config.name for a in agents]
        policy = TopologyPolicy.mesh(agent_names)

        super().__init__(config=config, agents=agents, policy=policy, **kwargs)


class HierarchicalTopology(BaseTopology):
    """Hierarchical pattern: agents organized in levels.

    Example:
        >>> topology = HierarchicalTopology(
        ...     config=TopologyConfig(name="org"),
        ...     agents=[ceo, manager1, manager2, worker1, worker2, worker3],
        ...     levels=[
        ...         ["ceo"],
        ...         ["manager1", "manager2"],
        ...         ["worker1", "worker2", "worker3"],
        ...     ],
        ... )
    """

    def __init__(
        self,
        config: TopologyConfig,
        agents: Sequence[Agent],
        levels: list[list[str]],
        **kwargs: Any,
    ) -> None:
        """Initialize hierarchical topology.

        Args:
            config: Topology configuration.
            agents: List of agents.
            levels: List of levels (each a list of agent names).
            **kwargs: Additional arguments for BaseTopology.
        """
        agent_names = [a.config.name for a in agents]

        for level in levels:
            for name in level:
                if name not in agent_names:
                    raise ValueError(f"Agent '{name}' not in agents")

        policy = TopologyPolicy.hierarchical(levels)

        super().__init__(config=config, agents=agents, policy=policy, **kwargs)

        self.levels = levels
