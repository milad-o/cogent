"""Handoff policies for multi-agent topologies.

Policies define the rules for agent communication:
- Who can send tasks to whom
- Who accepts tasks from whom  
- Under what conditions handoffs occur
- What data flows between agents
- Execution mode (sequential vs parallel)
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable


class HandoffCondition(Enum):
    """Built-in conditions for handoff decisions."""

    ALWAYS = "always"  # Always allow handoff
    ON_SUCCESS = "on_success"  # Only on successful completion
    ON_FAILURE = "on_failure"  # Only on failure/error
    ON_REQUEST = "on_request"  # Only when explicitly requested
    CONDITIONAL = "conditional"  # Custom condition function


class AcceptancePolicy(Enum):
    """How an agent decides to accept incoming tasks."""

    ACCEPT_ALL = "accept_all"  # Accept from anyone
    ACCEPT_LISTED = "accept_listed"  # Only accept from specific agents
    REJECT_LISTED = "reject_listed"  # Accept from all except specific agents
    CONDITIONAL = "conditional"  # Custom acceptance function


class ExecutionMode(Enum):
    """How agents execute within a topology.
    
    SEQUENTIAL: Agents run one at a time (default, safe)
    PARALLEL: Independent agents can run concurrently
    FAN_OUT: Supervisor fans out to multiple workers simultaneously
    FAN_IN: Multiple workers converge results to coordinator
    """
    
    SEQUENTIAL = "sequential"
    PARALLEL = "parallel"
    FAN_OUT = "fan_out"
    FAN_IN = "fan_in"


@dataclass
class HandoffRule:
    """A single handoff rule between two agents.

    Defines when and how one agent can hand off to another.

    Attributes:
        source: Name of source agent (or "*" for any).
        target: Name of target agent (or "*" for any).
        condition: When this handoff is allowed.
        label: Human-readable label for diagrams.
        priority: Priority when multiple rules match (higher = preferred).
        transform: Optional function to transform state during handoff.
        condition_fn: Custom condition function if condition is CONDITIONAL.
    """

    source: str
    target: str
    condition: HandoffCondition = HandoffCondition.ALWAYS
    label: str = ""
    priority: int = 0
    transform: Callable[[dict[str, Any]], dict[str, Any]] | None = None
    condition_fn: Callable[[dict[str, Any]], bool] | None = None

    def matches(self, from_agent: str, to_agent: str) -> bool:
        """Check if this rule matches a handoff request.

        Args:
            from_agent: Source agent name.
            to_agent: Target agent name.

        Returns:
            True if rule applies to this handoff.
        """
        source_match = self.source == "*" or self.source == from_agent
        target_match = self.target == "*" or self.target == to_agent
        return source_match and target_match

    def evaluate(self, state: dict[str, Any]) -> bool:
        """Evaluate if handoff is allowed given current state.

        Args:
            state: Current topology state.

        Returns:
            True if handoff should proceed.
        """
        if self.condition == HandoffCondition.ALWAYS:
            return True

        if self.condition == HandoffCondition.ON_SUCCESS:
            return not state.get("error") and not state.get("failed")

        if self.condition == HandoffCondition.ON_FAILURE:
            return bool(state.get("error") or state.get("failed"))

        if self.condition == HandoffCondition.ON_REQUEST:
            requested = state.get("handoff_requested")
            return requested == self.target

        if self.condition == HandoffCondition.CONDITIONAL:
            if self.condition_fn:
                return self.condition_fn(state)
            return False

        return False

    def apply_transform(self, state: dict[str, Any]) -> dict[str, Any]:
        """Apply state transformation for this handoff.

        Args:
            state: Current state.

        Returns:
            Transformed state.
        """
        if self.transform:
            return self.transform(state)
        return state


@dataclass
class AgentPolicy:
    """Policy for a single agent's handoff behavior.

    Defines what an agent accepts and where it can send.

    Attributes:
        agent_name: Name of the agent this policy applies to.
        acceptance: How the agent decides to accept tasks.
        accept_from: List of agents to accept from (for ACCEPT_LISTED).
        reject_from: List of agents to reject (for REJECT_LISTED).
        can_send_to: List of agents this agent can hand off to.
                     None means can send to anyone, [] means can't send to anyone.
        can_finish: Whether this agent can end the workflow.
        acceptance_fn: Custom acceptance function if acceptance is CONDITIONAL.
    """

    agent_name: str
    acceptance: AcceptancePolicy = AcceptancePolicy.ACCEPT_ALL
    accept_from: list[str] = field(default_factory=list)
    reject_from: list[str] = field(default_factory=list)
    can_send_to: list[str] | None = None  # None = anyone, [] = nobody
    can_finish: bool = True
    acceptance_fn: Callable[[str, dict[str, Any]], bool] | None = None

    def accepts_from(self, source: str, state: dict[str, Any]) -> bool:
        """Check if this agent accepts tasks from a source.

        Args:
            source: Source agent name.
            state: Current state for context.

        Returns:
            True if task should be accepted.
        """
        if self.acceptance == AcceptancePolicy.ACCEPT_ALL:
            return True

        if self.acceptance == AcceptancePolicy.ACCEPT_LISTED:
            return source in self.accept_from

        if self.acceptance == AcceptancePolicy.REJECT_LISTED:
            return source not in self.reject_from

        if self.acceptance == AcceptancePolicy.CONDITIONAL:
            if self.acceptance_fn:
                return self.acceptance_fn(source, state)
            return False

        return True

    def can_handoff_to(self, target: str) -> bool:
        """Check if this agent can hand off to a target.

        Args:
            target: Target agent name.

        Returns:
            True if handoff is allowed.
        """
        if self.can_send_to is None:
            # None means can send to anyone
            return True
        # Explicit list (including empty) - check membership
        return target in self.can_send_to


@dataclass
class TopologyPolicy:
    """Complete handoff policy for a topology.

    Combines agent policies and handoff rules to define
    the full communication pattern.

    Example:
        >>> policy = TopologyPolicy(
        ...     rules=[
        ...         HandoffRule("supervisor", "*", label="delegate"),
        ...         HandoffRule("*", "supervisor", label="report"),
        ...     ],
        ...     agent_policies={
        ...         "supervisor": AgentPolicy("supervisor", can_finish=True),
        ...         "worker1": AgentPolicy("worker1", can_finish=False),
        ...     },
        ...     entry_point="supervisor",
        ...     execution_mode=ExecutionMode.FAN_OUT,  # Parallel workers
        ... )
    """

    rules: list[HandoffRule] = field(default_factory=list)
    agent_policies: dict[str, AgentPolicy] = field(default_factory=dict)
    entry_point: str | None = None
    default_acceptance: AcceptancePolicy = AcceptancePolicy.ACCEPT_ALL
    allow_self_handoff: bool = False
    execution_mode: ExecutionMode = ExecutionMode.SEQUENTIAL
    parallel_groups: list[list[str]] = field(default_factory=list)  # Groups of agents that can run in parallel

    def add_rule(
        self,
        source: str,
        target: str,
        condition: HandoffCondition = HandoffCondition.ALWAYS,
        label: str = "",
        priority: int = 0,
    ) -> "TopologyPolicy":
        """Add a handoff rule.

        Args:
            source: Source agent ("*" for any).
            target: Target agent ("*" for any).
            condition: When handoff is allowed.
            label: Label for diagrams.
            priority: Rule priority.

        Returns:
            Self for chaining.
        """
        self.rules.append(
            HandoffRule(
                source=source,
                target=target,
                condition=condition,
                label=label,
                priority=priority,
            )
        )
        return self

    def add_agent_policy(self, policy: AgentPolicy) -> "TopologyPolicy":
        """Add an agent policy.

        Args:
            policy: The agent policy.

        Returns:
            Self for chaining.
        """
        self.agent_policies[policy.agent_name] = policy
        return self

    def get_agent_policy(self, agent_name: str) -> AgentPolicy:
        """Get policy for an agent, creating default if needed.

        Args:
            agent_name: Agent name.

        Returns:
            Agent's policy.
        """
        if agent_name not in self.agent_policies:
            self.agent_policies[agent_name] = AgentPolicy(
                agent_name=agent_name,
                acceptance=self.default_acceptance,
            )
        return self.agent_policies[agent_name]

    def can_handoff(
        self,
        from_agent: str,
        to_agent: str,
        state: dict[str, Any],
    ) -> bool:
        """Check if a handoff is allowed.

        Args:
            from_agent: Source agent.
            to_agent: Target agent.
            state: Current state.

        Returns:
            True if handoff is allowed.
        """
        # Check self-handoff
        if from_agent == to_agent and not self.allow_self_handoff:
            return False

        # Check agent policies
        source_policy = self.get_agent_policy(from_agent)
        if not source_policy.can_handoff_to(to_agent):
            return False

        target_policy = self.get_agent_policy(to_agent)
        if not target_policy.accepts_from(from_agent, state):
            return False

        # Check rules (at least one must match and evaluate true)
        matching_rules = [
            rule for rule in self.rules if rule.matches(from_agent, to_agent)
        ]

        if not matching_rules:
            # No rules = check if there's a catch-all
            catch_all = [r for r in self.rules if r.source == "*" or r.target == "*"]
            if not catch_all:
                # No rules at all means allow by default
                return True
            matching_rules = catch_all

        # Sort by priority and evaluate
        matching_rules.sort(key=lambda r: r.priority, reverse=True)
        return any(rule.evaluate(state) for rule in matching_rules)

    def get_allowed_targets(
        self,
        from_agent: str,
        state: dict[str, Any],
        all_agents: list[str],
    ) -> list[str]:
        """Get all agents that from_agent can hand off to.

        Args:
            from_agent: Source agent.
            state: Current state.
            all_agents: List of all agent names.

        Returns:
            List of valid target agent names.
        """
        return [
            agent
            for agent in all_agents
            if self.can_handoff(from_agent, agent, state)
        ]

    def get_rule_for_handoff(
        self,
        from_agent: str,
        to_agent: str,
    ) -> HandoffRule | None:
        """Get the rule that applies to a handoff (for labels, etc).

        Args:
            from_agent: Source agent.
            to_agent: Target agent.

        Returns:
            Matching rule or None.
        """
        matching = [r for r in self.rules if r.matches(from_agent, to_agent)]
        if matching:
            matching.sort(key=lambda r: r.priority, reverse=True)
            return matching[0]
        return None

    def get_edges_for_diagram(
        self,
        agents: list[str],
    ) -> list[tuple[str, str, str]]:
        """Get all edges for Mermaid diagram generation.

        Args:
            agents: List of agent names.

        Returns:
            List of (source, target, label) tuples.
        """
        edges: list[tuple[str, str, str]] = []
        seen: set[tuple[str, str]] = set()

        # First, explicit rules
        for rule in self.rules:
            if rule.source == "*":
                sources = agents
            else:
                sources = [rule.source] if rule.source in agents else []

            if rule.target == "*":
                targets = agents
            else:
                targets = [rule.target] if rule.target in agents else []

            for src in sources:
                for tgt in targets:
                    if src != tgt and (src, tgt) not in seen:
                        edges.append((src, tgt, rule.label))
                        seen.add((src, tgt))

        return edges

    # ==================== Factory Methods ====================

    @classmethod
    def supervisor(
        cls,
        supervisor: str,
        workers: list[str],
        parallel_workers: bool = False,
    ) -> "TopologyPolicy":
        """Create a supervisor topology policy.

        Supervisor delegates to workers, workers report back.

        Args:
            supervisor: Supervisor agent name.
            workers: List of worker agent names.
            parallel_workers: If True, workers can execute in parallel.

        Returns:
            Configured policy.
        """
        policy = cls(
            entry_point=supervisor,
            execution_mode=ExecutionMode.FAN_OUT if parallel_workers else ExecutionMode.SEQUENTIAL,
            parallel_groups=[workers] if parallel_workers else [],
        )

        # Supervisor can delegate to any worker
        for worker in workers:
            policy.add_rule(supervisor, worker, label="delegate")
            policy.add_rule(worker, supervisor, label="report")

        # Supervisor policy
        policy.add_agent_policy(
            AgentPolicy(
                agent_name=supervisor,
                can_send_to=workers,
                can_finish=True,
            )
        )

        # Worker policies - can only send to supervisor
        for worker in workers:
            policy.add_agent_policy(
                AgentPolicy(
                    agent_name=worker,
                    can_send_to=[supervisor],
                    can_finish=False,
                )
            )

        return policy

    @classmethod
    def pipeline(
        cls,
        stages: list[str],
        allow_skip: bool = False,
        allow_repeat: bool = False,
    ) -> "TopologyPolicy":
        """Create a pipeline topology policy.

        Sequential flow through stages.

        Args:
            stages: Ordered list of stage/agent names.
            allow_skip: Allow skipping stages.
            allow_repeat: Allow going back to previous stages.

        Returns:
            Configured policy.
        """
        policy = cls(entry_point=stages[0] if stages else None)

        for i, stage in enumerate(stages):
            targets = []

            # Can go to next stage
            if i < len(stages) - 1:
                next_stage = stages[i + 1]
                policy.add_rule(stage, next_stage, label="next")
                targets.append(next_stage)

            # Can skip forward
            if allow_skip and i < len(stages) - 2:
                for skip_stage in stages[i + 2:]:
                    policy.add_rule(stage, skip_stage, label="skip")
                    targets.append(skip_stage)

            # Can go back
            if allow_repeat and i > 0:
                for prev_stage in stages[:i]:
                    policy.add_rule(stage, prev_stage, label="retry")
                    targets.append(prev_stage)

            # Agent policy
            policy.add_agent_policy(
                AgentPolicy(
                    agent_name=stage,
                    can_send_to=targets,
                    can_finish=(i == len(stages) - 1),  # Only last can finish
                )
            )

        return policy

    @classmethod
    def mesh(
        cls,
        agents: list[str],
        parallel: bool = False,
    ) -> "TopologyPolicy":
        """Create a mesh topology policy.

        All agents can communicate with all others.

        Args:
            agents: List of agent names.
            parallel: If True, agents can execute in parallel.

        Returns:
            Configured policy.
        """
        policy = cls(
            entry_point=agents[0] if agents else None,
            execution_mode=ExecutionMode.PARALLEL if parallel else ExecutionMode.SEQUENTIAL,
            parallel_groups=[agents] if parallel else [],
        )

        # Everyone can talk to everyone
        policy.add_rule("*", "*", label="")

        # All agents can finish and send to anyone
        for agent in agents:
            others = [a for a in agents if a != agent]
            policy.add_agent_policy(
                AgentPolicy(
                    agent_name=agent,
                    can_send_to=others,
                    can_finish=True,
                )
            )

        return policy

    @classmethod
    def hierarchical(
        cls,
        levels: list[list[str]],
    ) -> "TopologyPolicy":
        """Create a hierarchical topology policy.

        Agents organized in levels, higher levels coordinate lower.

        Args:
            levels: List of levels, each a list of agent names.
                   levels[0] is top (e.g., CEO), levels[-1] is bottom.

        Returns:
            Configured policy.
        """
        policy = cls()

        if levels:
            # Entry point is first agent of top level
            policy.entry_point = levels[0][0] if levels[0] else None

        for level_idx, level_agents in enumerate(levels):
            for agent in level_agents:
                targets = []

                # Can send to same level (peer coordination)
                peers = [a for a in level_agents if a != agent]
                for peer in peers:
                    policy.add_rule(agent, peer, label="coordinate")
                targets.extend(peers)

                # Can send to level below (delegate)
                if level_idx < len(levels) - 1:
                    subordinates = levels[level_idx + 1]
                    for sub in subordinates:
                        policy.add_rule(agent, sub, label="delegate")
                    targets.extend(subordinates)

                # Can send to level above (report)
                if level_idx > 0:
                    superiors = levels[level_idx - 1]
                    for sup in superiors:
                        policy.add_rule(agent, sup, label="report")
                    targets.extend(superiors)

                policy.add_agent_policy(
                    AgentPolicy(
                        agent_name=agent,
                        can_send_to=targets,
                        can_finish=(level_idx == 0),  # Only top level finishes
                    )
                )

        return policy
