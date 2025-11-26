"""
Topology configuration with type-safe construction.

Provides dataclass-based topology configuration with enums for type safety,
clear required/optional parameters, and IDE-friendly construction.

Example:
    ```python
    from agenticflow import Agent, TopologySpec, TopologyPattern, DelegationPolicy
    
    # Type-safe, explicit construction
    spec = TopologySpec(
        name="research-team",
        pattern=TopologyPattern.SUPERVISOR,
        supervisor=manager,
        workers=[analyst, writer],
        delegation=DelegationPolicy(
            strategy=DelegationStrategy.ROUND_ROBIN,
            max_concurrent=3,
        ),
    )
    
    # Or use factory functions for common patterns
    spec = supervisor_topology("team", manager, [worker1, worker2])
    spec = pipeline_topology("flow", [researcher, writer, editor])
    spec = mesh_topology("brainstorm", [agent1, agent2, agent3])
    ```
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from agenticflow.agent.base import Agent
from agenticflow.core.enums import AgentRole


# ==================== Enums ====================


class TopologyPattern(str, Enum):
    """Available topology patterns.
    
    Each pattern has different communication and authority structures:
    
    - SUPERVISOR: Hub-and-spoke with authority. One agent directs others.
    - COORDINATOR: Hub-and-spoke without authority. One agent facilitates.
    - PIPELINE: Sequential processing. Each stage passes to next.
    - HIERARCHICAL: Multi-level tree. Authority flows down.
    - MESH: Peer-to-peer. All agents can communicate.
    - CUSTOM: User-defined communication rules.
    """
    
    SUPERVISOR = "supervisor"    # One boss, many workers (has authority)
    COORDINATOR = "coordinator"  # One facilitator, many peers (no authority)
    PIPELINE = "pipeline"        # Sequential A → B → C
    HIERARCHICAL = "hierarchical"  # Tree structure with levels
    MESH = "mesh"               # Everyone can talk to everyone
    CUSTOM = "custom"           # User-defined
    
    @property
    def requires_leader(self) -> bool:
        """Whether this pattern requires a leader agent."""
        return self in (TopologyPattern.SUPERVISOR, TopologyPattern.COORDINATOR)
    
    @property
    def requires_stages(self) -> bool:
        """Whether this pattern requires ordered stages."""
        return self == TopologyPattern.PIPELINE
    
    @property
    def requires_levels(self) -> bool:
        """Whether this pattern requires hierarchical levels."""
        return self == TopologyPattern.HIERARCHICAL


class DelegationStrategy(str, Enum):
    """How a leader delegates tasks to workers.
    
    Strategies for choosing which agent receives a task:
    
    - ROUND_ROBIN: Cycle through workers in order.
    - LOAD_BALANCE: Pick the least busy worker.
    - CAPABILITY: Match task requirements to worker skills.
    - RANDOM: Random selection.
    - BROADCAST: Send to all workers simultaneously.
    - FIRST_AVAILABLE: Pick first idle worker.
    """
    
    ROUND_ROBIN = "round_robin"
    LOAD_BALANCE = "load_balance"
    CAPABILITY = "capability"
    RANDOM = "random"
    BROADCAST = "broadcast"
    FIRST_AVAILABLE = "first_available"


class CompletionCondition(str, Enum):
    """When a topology run is considered complete.
    
    - ALL_AGENTS: All agents must contribute before completion.
    - ANY_FINISH: Any agent can signal completion.
    - LEADER: Only the leader (supervisor/coordinator) can complete.
    - LAST_STAGE: Pipeline completes when last stage finishes.
    - CONSENSUS: Majority of agents must agree.
    """
    
    ALL_AGENTS = "all_agents"
    ANY_FINISH = "any_finish"
    LEADER = "leader"
    LAST_STAGE = "last_stage"
    CONSENSUS = "consensus"


# ==================== Policies ====================


@dataclass(frozen=True)
class DelegationPolicy:
    """Immutable policy for task delegation.
    
    Args:
        strategy: How to choose which agent gets a task.
        max_concurrent: Maximum concurrent tasks across all agents.
        timeout_seconds: Task timeout in seconds.
        retry_on_failure: Whether to retry failed tasks.
        max_retries: Maximum retry attempts.
        fallback_agent: Agent name to use if primary fails.
    
    Example:
        ```python
        # Explicit construction
        policy = DelegationPolicy(
            strategy=DelegationStrategy.ROUND_ROBIN,
            max_concurrent=3,
            timeout_seconds=60.0,
        )
        
        # Or use presets
        policy = DelegationPolicy.fast()
        policy = DelegationPolicy.reliable()
        ```
    """
    
    strategy: DelegationStrategy = DelegationStrategy.CAPABILITY
    max_concurrent: int = 5
    timeout_seconds: float = 300.0
    retry_on_failure: bool = True
    max_retries: int = 3
    fallback_agent: str | None = None
    
    @classmethod
    def fast(cls) -> DelegationPolicy:
        """Fast policy: no retries, short timeout, first available."""
        return cls(
            strategy=DelegationStrategy.FIRST_AVAILABLE,
            timeout_seconds=60.0,
            retry_on_failure=False,
        )
    
    @classmethod
    def reliable(cls) -> DelegationPolicy:
        """Reliable policy: retries enabled, capability matching."""
        return cls(
            strategy=DelegationStrategy.CAPABILITY,
            retry_on_failure=True,
            max_retries=5,
        )
    
    @classmethod
    def broadcast(cls) -> DelegationPolicy:
        """Broadcast policy: send to all workers."""
        return cls(strategy=DelegationStrategy.BROADCAST)


@dataclass(frozen=True)
class EventHooks:
    """Event callbacks for topology lifecycle.
    
    Args:
        on_handoff: Called when task passes between agents.
        on_complete: Called when an agent completes work.
        on_error: Called when an agent encounters an error.
    
    Example:
        ```python
        hooks = EventHooks(
            on_handoff=lambda from_, to, data: print(f"{from_} → {to}"),
            on_complete=lambda agent, result: log(f"{agent} done"),
            on_error=lambda agent, err: alert(f"{agent} failed: {err}"),
        )
        ```
    """
    
    on_handoff: Callable[[str, str, dict[str, Any]], None] | None = None
    on_complete: Callable[[str, Any], None] | None = None
    on_error: Callable[[str, Exception], None] | None = None


# ==================== Main Spec ====================


@dataclass
class TopologySpec:
    """Complete specification for a multi-agent topology.
    
    This is the primary API for topology configuration. All parameters
    are explicit with clear types and defaults.
    
    Args:
        name: Unique identifier for the topology.
        pattern: The topology pattern (SUPERVISOR, PIPELINE, etc.).
        agents: All agents in the topology.
        
        # Pattern-specific (use the one matching your pattern):
        supervisor: The supervisor/coordinator agent (for SUPERVISOR/COORDINATOR).
        workers: Worker agents (for SUPERVISOR/COORDINATOR).
        stages: Ordered stage agents (for PIPELINE).
        levels: Hierarchical levels (for HIERARCHICAL).
        
        # Policies:
        delegation: How tasks are delegated.
        completion: When the topology is considered complete.
        
        # Options:
        hooks: Event callbacks.
        auto_assign_roles: Whether to auto-assign agent roles.
    
    Example:
        ```python
        # Supervisor pattern
        spec = TopologySpec(
            name="research-team",
            pattern=TopologyPattern.SUPERVISOR,
            supervisor=manager,
            workers=[analyst, writer],
            delegation=DelegationPolicy(strategy=DelegationStrategy.ROUND_ROBIN),
        )
        
        # Pipeline pattern
        spec = TopologySpec(
            name="content-flow",
            pattern=TopologyPattern.PIPELINE,
            stages=[researcher, writer, editor],
        )
        
        # Mesh pattern
        spec = TopologySpec(
            name="brainstorm",
            pattern=TopologyPattern.MESH,
            agents=[agent1, agent2, agent3],
        )
        ```
    """
    
    # Required
    name: str
    pattern: TopologyPattern
    
    # Agents - provide based on pattern
    agents: list[Agent] = field(default_factory=list)
    supervisor: Agent | None = None  # For SUPERVISOR/COORDINATOR
    workers: list[Agent] = field(default_factory=list)  # For SUPERVISOR/COORDINATOR
    stages: list[Agent] = field(default_factory=list)  # For PIPELINE
    levels: list[list[Agent]] = field(default_factory=list)  # For HIERARCHICAL
    
    # Policies
    delegation: DelegationPolicy = field(default_factory=DelegationPolicy)
    completion: CompletionCondition | None = None  # Auto-set based on pattern
    
    # Options
    hooks: EventHooks = field(default_factory=EventHooks)
    auto_assign_roles: bool = True
    
    # Internal - populated during validation
    _all_agents: list[Agent] = field(default_factory=list, repr=False)
    _roles: dict[str, AgentRole] = field(default_factory=dict, repr=False)
    _validated: bool = field(default=False, repr=False)
    
    def __post_init__(self) -> None:
        """Validate and normalize after construction."""
        self._normalize()
        errors = self.validate()
        if errors:
            raise ValueError(
                f"Invalid TopologySpec:\n" + "\n".join(f"  - {e}" for e in errors)
            )
        if self.auto_assign_roles:
            self._assign_roles()
        self._validated = True
    
    def _normalize(self) -> None:
        """Normalize agent lists and set defaults."""
        # Collect all agents from pattern-specific fields
        all_agents: list[Agent] = []
        
        if self.supervisor:
            all_agents.append(self.supervisor)
        all_agents.extend(self.workers)
        all_agents.extend(self.stages)
        for level in self.levels:
            all_agents.extend(level)
        all_agents.extend(self.agents)
        
        # Deduplicate while preserving order
        seen: set[str] = set()
        unique: list[Agent] = []
        for agent in all_agents:
            if agent.name not in seen:
                seen.add(agent.name)
                unique.append(agent)
        
        self._all_agents = unique
        
        # Set default completion based on pattern
        if self.completion is None:
            self.completion = _default_completion(self.pattern)
    
    def validate(self) -> list[str]:
        """Validate the specification.
        
        Returns:
            List of validation errors (empty if valid).
        """
        errors: list[str] = []
        
        if not self.name:
            errors.append("name is required")
        
        if not self._all_agents:
            errors.append("At least one agent is required")
        
        # Pattern-specific validation
        if self.pattern == TopologyPattern.SUPERVISOR:
            if not self.supervisor:
                errors.append("SUPERVISOR pattern requires 'supervisor' agent")
            if not self.workers:
                errors.append("SUPERVISOR pattern requires 'workers' list")
        
        elif self.pattern == TopologyPattern.COORDINATOR:
            if not self.supervisor:  # Reusing supervisor field for coordinator
                errors.append("COORDINATOR pattern requires 'supervisor' (coordinator) agent")
            if not self.workers:
                errors.append("COORDINATOR pattern requires 'workers' (peers) list")
        
        elif self.pattern == TopologyPattern.PIPELINE:
            if not self.stages:
                errors.append("PIPELINE pattern requires 'stages' list")
            if len(self.stages) < 2:
                errors.append("PIPELINE requires at least 2 stages")
        
        elif self.pattern == TopologyPattern.HIERARCHICAL:
            if not self.levels:
                errors.append("HIERARCHICAL pattern requires 'levels' list")
            if len(self.levels) < 2:
                errors.append("HIERARCHICAL requires at least 2 levels")
        
        elif self.pattern == TopologyPattern.MESH:
            if not self._all_agents:
                errors.append("MESH pattern requires 'agents' list")
            if len(self._all_agents) < 2:
                errors.append("MESH requires at least 2 agents")
        
        return errors
    
    def _assign_roles(self) -> None:
        """Auto-assign roles based on pattern."""
        if self.pattern == TopologyPattern.SUPERVISOR:
            if self.supervisor:
                self._roles[self.supervisor.name] = AgentRole.SUPERVISOR
            for worker in self.workers:
                self._roles[worker.name] = AgentRole.WORKER
        
        elif self.pattern == TopologyPattern.COORDINATOR:
            if self.supervisor:
                self._roles[self.supervisor.name] = AgentRole.COORDINATOR
            for worker in self.workers:
                self._roles[worker.name] = AgentRole.WORKER
        
        elif self.pattern == TopologyPattern.PIPELINE:
            for i, agent in enumerate(self.stages):
                if i == 0:
                    self._roles[agent.name] = AgentRole.RESEARCHER
                elif i == len(self.stages) - 1:
                    self._roles[agent.name] = AgentRole.VALIDATOR
                else:
                    self._roles[agent.name] = AgentRole.WORKER
        
        elif self.pattern == TopologyPattern.HIERARCHICAL:
            for i, level in enumerate(self.levels):
                for agent in level:
                    if i == 0:
                        self._roles[agent.name] = AgentRole.SUPERVISOR
                    elif i == len(self.levels) - 1:
                        self._roles[agent.name] = AgentRole.WORKER
                    else:
                        self._roles[agent.name] = AgentRole.COORDINATOR
        
        elif self.pattern == TopologyPattern.MESH:
            for agent in self._all_agents:
                self._roles[agent.name] = AgentRole.WORKER
    
    @property
    def all_agents(self) -> list[Agent]:
        """All agents in the topology."""
        return self._all_agents
    
    @property
    def roles(self) -> dict[str, AgentRole]:
        """Role assignments for each agent."""
        return self._roles
    
    @property
    def leader(self) -> Agent | None:
        """The leader agent (supervisor/coordinator), if any."""
        return self.supervisor


def _default_completion(pattern: TopologyPattern) -> CompletionCondition:
    """Get default completion condition for a pattern."""
    return {
        TopologyPattern.SUPERVISOR: CompletionCondition.LEADER,
        TopologyPattern.COORDINATOR: CompletionCondition.CONSENSUS,
        TopologyPattern.PIPELINE: CompletionCondition.LAST_STAGE,
        TopologyPattern.HIERARCHICAL: CompletionCondition.LEADER,
        TopologyPattern.MESH: CompletionCondition.ANY_FINISH,
        TopologyPattern.CUSTOM: CompletionCondition.ANY_FINISH,
    }.get(pattern, CompletionCondition.ANY_FINISH)


# ==================== Factory Functions ====================


def supervisor_topology(
    name: str,
    supervisor: Agent,
    workers: Sequence[Agent],
    *,
    delegation: DelegationPolicy | None = None,
    hooks: EventHooks | None = None,
) -> TopologySpec:
    """Create a supervisor topology.
    
    One supervisor with authority directs multiple workers.
    
    Args:
        name: Topology name.
        supervisor: The supervisor agent (will be assigned SUPERVISOR role).
        workers: Worker agents (will be assigned WORKER role).
        delegation: Optional delegation policy.
        hooks: Optional event hooks.
    
    Returns:
        Configured TopologySpec.
    
    Example:
        ```python
        spec = supervisor_topology(
            "research-team",
            supervisor=manager,
            workers=[analyst, writer],
            delegation=DelegationPolicy(strategy=DelegationStrategy.ROUND_ROBIN),
        )
        ```
    """
    return TopologySpec(
        name=name,
        pattern=TopologyPattern.SUPERVISOR,
        supervisor=supervisor,
        workers=list(workers),
        delegation=delegation or DelegationPolicy(),
        hooks=hooks or EventHooks(),
    )


def coordinator_topology(
    name: str,
    coordinator: Agent,
    peers: Sequence[Agent],
    *,
    delegation: DelegationPolicy | None = None,
    hooks: EventHooks | None = None,
) -> TopologySpec:
    """Create a coordinator topology.
    
    One coordinator facilitates collaboration between peers (no authority).
    
    Args:
        name: Topology name.
        coordinator: The coordinator agent (will be assigned COORDINATOR role).
        peers: Peer agents (will be assigned WORKER role).
        delegation: Optional delegation policy.
        hooks: Optional event hooks.
    
    Returns:
        Configured TopologySpec.
    
    Example:
        ```python
        spec = coordinator_topology(
            "project-team",
            coordinator=facilitator,
            peers=[frontend_dev, backend_dev, designer],
        )
        ```
    """
    return TopologySpec(
        name=name,
        pattern=TopologyPattern.COORDINATOR,
        supervisor=coordinator,  # Reusing field
        workers=list(peers),
        delegation=delegation or DelegationPolicy(),
        hooks=hooks or EventHooks(),
    )


def pipeline_topology(
    name: str,
    stages: Sequence[Agent],
    *,
    delegation: DelegationPolicy | None = None,
    hooks: EventHooks | None = None,
) -> TopologySpec:
    """Create a pipeline topology.
    
    Sequential processing where each stage passes to the next.
    
    Args:
        name: Topology name.
        stages: Agents in pipeline order (first to last).
        delegation: Optional delegation policy.
        hooks: Optional event hooks.
    
    Returns:
        Configured TopologySpec.
    
    Example:
        ```python
        spec = pipeline_topology(
            "content-flow",
            stages=[researcher, writer, editor],
        )
        ```
    """
    return TopologySpec(
        name=name,
        pattern=TopologyPattern.PIPELINE,
        stages=list(stages),
        delegation=delegation or DelegationPolicy(),
        hooks=hooks or EventHooks(),
    )


def mesh_topology(
    name: str,
    agents: Sequence[Agent],
    *,
    delegation: DelegationPolicy | None = None,
    hooks: EventHooks | None = None,
) -> TopologySpec:
    """Create a mesh topology.
    
    All agents can communicate with all others (peer-to-peer).
    
    Args:
        name: Topology name.
        agents: All peer agents.
        delegation: Optional delegation policy.
        hooks: Optional event hooks.
    
    Returns:
        Configured TopologySpec.
    
    Example:
        ```python
        spec = mesh_topology(
            "brainstorm",
            agents=[creative1, creative2, creative3],
        )
        ```
    """
    return TopologySpec(
        name=name,
        pattern=TopologyPattern.MESH,
        agents=list(agents),
        delegation=delegation or DelegationPolicy(),
        hooks=hooks or EventHooks(),
    )


def hierarchical_topology(
    name: str,
    levels: Sequence[Sequence[Agent]],
    *,
    delegation: DelegationPolicy | None = None,
    hooks: EventHooks | None = None,
) -> TopologySpec:
    """Create a hierarchical topology.
    
    Multi-level tree structure. Authority flows from top to bottom.
    
    Args:
        name: Topology name.
        levels: Agents organized by level (levels[0] is top/executives).
        delegation: Optional delegation policy.
        hooks: Optional event hooks.
    
    Returns:
        Configured TopologySpec.
    
    Example:
        ```python
        spec = hierarchical_topology(
            "org",
            levels=[
                [ceo],                    # Level 0: top
                [vp_eng, vp_sales],       # Level 1: middle
                [dev1, dev2, sales1],     # Level 2: bottom
            ],
        )
        ```
    """
    return TopologySpec(
        name=name,
        pattern=TopologyPattern.HIERARCHICAL,
        levels=[list(level) for level in levels],
        delegation=delegation or DelegationPolicy(),
        hooks=hooks or EventHooks(),
    )


# ==================== Role Requirements ====================


TOPOLOGY_ROLE_REQUIREMENTS: dict[TopologyPattern, dict[str, list[AgentRole]]] = {
    TopologyPattern.SUPERVISOR: {
        "leader": [AgentRole.SUPERVISOR],
        "workers": [AgentRole.WORKER, AgentRole.SPECIALIST, AgentRole.RESEARCHER],
    },
    TopologyPattern.COORDINATOR: {
        "leader": [AgentRole.COORDINATOR],
        "peers": [AgentRole.WORKER, AgentRole.SPECIALIST, AgentRole.RESEARCHER, AgentRole.PLANNER],
    },
    TopologyPattern.PIPELINE: {
        "first": [AgentRole.RESEARCHER, AgentRole.WORKER],
        "middle": [AgentRole.WORKER, AgentRole.SPECIALIST, AgentRole.PLANNER],
        "last": [AgentRole.VALIDATOR, AgentRole.WORKER],
    },
    TopologyPattern.HIERARCHICAL: {
        "top": [AgentRole.SUPERVISOR, AgentRole.ORCHESTRATOR],
        "middle": [AgentRole.COORDINATOR, AgentRole.PLANNER],
        "bottom": [AgentRole.WORKER, AgentRole.SPECIALIST],
    },
    TopologyPattern.MESH: {
        "any": list(AgentRole),
    },
}


def validate_roles(spec: TopologySpec) -> list[str]:
    """Validate agent roles match topology requirements.
    
    Args:
        spec: Topology specification.
    
    Returns:
        List of warnings (not errors - roles can be auto-assigned).
    """
    warnings: list[str] = []
    requirements = TOPOLOGY_ROLE_REQUIREMENTS.get(spec.pattern, {})
    
    if spec.pattern in (TopologyPattern.SUPERVISOR, TopologyPattern.COORDINATOR):
        if spec.supervisor:
            leader_roles = requirements.get("leader", [])
            actual_role = spec.roles.get(spec.supervisor.name)
            if actual_role and actual_role not in leader_roles:
                warnings.append(
                    f"Leader '{spec.supervisor.name}' has role '{actual_role.value}', "
                    f"recommended: {[r.value for r in leader_roles]}"
                )
    
    return warnings
