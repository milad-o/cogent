"""Delegation configuration mixin for all flow types.

This module provides unified A2A delegation configuration that works across
all flow types (reactive, supervisor, pipeline, mesh, etc.).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from agenticflow.agent.base import Agent
    from agenticflow.flow.triggers import AgentTriggerConfig


class DelegationMixin:
    """Mixin for configuring A2A delegation in any flow type.

    This provides a unified API for delegation configuration that works
    across imperative topologies (Supervisor, Pipeline, Mesh) and
    reactive flows.

    Design Philosophy:
    - Declarative: Specify delegation policy, framework handles implementation
    - Automatic: Tools and prompts are injected automatically
    - Policy-driven: Explicit allow-lists prevent unauthorized delegation
    - Topology-agnostic: Same API regardless of flow type
    """

    def configure_delegation(
        self,
        agent: Agent,
        *,
        can_delegate: list[str] | bool | None = None,
        can_reply: bool = False,
        trigger_config: AgentTriggerConfig | None = None,
    ) -> None:
        """Configure A2A delegation for an agent.

        This method:
        1. Injects delegate_to tool if can_delegate is specified
        2. Injects reply_with_result tool if can_reply is True
        3. Auto-enhances agent's system prompt with delegation instructions
        4. Enforces delegation policy (who can delegate to whom)

        Args:
            agent: The agent to configure
            can_delegate: Delegation policy:
                         - list[str]: Agent can delegate to these specific agents
                         - True: Agent can delegate to any available specialist
                         - None: Agent cannot delegate
            can_reply: Whether agent can reply to delegated requests
            trigger_config: Optional trigger config (for reactive flows, to find specialists)

        Examples:
            # Coordinator that can delegate to specific specialists
            flow.configure_delegation(
                coordinator,
                can_delegate=["data_analyst", "writer", "researcher"]
            )

            # Specialist that handles delegated work
            flow.configure_delegation(data_analyst, can_reply=True)

            # In a topology
            supervisor_config.configure_delegation(
                coordinator_agent,
                can_delegate=[w.name for w in workers]
            )
        """
        from agenticflow.flow.a2a_tools import (
            create_delegate_tool,
            create_reply_tool,
        )

        # === Inject reply tool for specialists ===
        if can_reply:
            reply_tool = create_reply_tool(self, agent.name)  # type: ignore[arg-type]

            # Add to agent's tool list
            if not hasattr(agent, '_direct_tools'):
                agent._direct_tools = []  # type: ignore[attr-defined]
            agent._direct_tools.append(reply_tool)  # type: ignore[attr-defined]
            if reply_tool.name not in agent.config.tools:
                agent.config.tools.append(reply_tool.name)

            # Auto-enhance prompt
            reply_instructions = (
                "\n\nDELEGATION PROTOCOL:\n"
                "When you receive a delegated task, complete it and use the "
                "reply_with_result tool to send your results back to the coordinator."
            )
            agent.config.system_prompt = (agent.config.system_prompt or "") + reply_instructions

        # === Inject delegation tool for coordinators ===
        if can_delegate:
            # Resolve available specialists
            specialists: list[str] = []

            if can_delegate is True:
                # Auto-discover specialists
                # For reactive flows: agents that handle agent.request
                if trigger_config and hasattr(self, '_agents_registry'):
                    registry = self._agents_registry
                    specialists = [
                        name for name, (_, cfg) in registry.items()
                        if any(t.on == "agent.request" for t in cfg.triggers)
                    ]
                # For topologies: agents with can_reply=True or explicit delegation config
                elif hasattr(self, '_agents') or hasattr(self, 'workers'):
                    # Agents are registered dynamically; delegation list populated at registration time
                    specialists = []
            else:
                specialists = can_delegate

            if specialists:
                delegate_tool = create_delegate_tool(self, agent.name, specialists)  # type: ignore[arg-type]

                # Add to agent's tool list
                if not hasattr(agent, '_direct_tools'):
                    agent._direct_tools = []  # type: ignore[attr-defined]
                agent._direct_tools.append(delegate_tool)  # type: ignore[attr-defined]
                if delegate_tool.name not in agent.config.tools:
                    agent.config.tools.append(delegate_tool.name)

                # Auto-enhance prompt
                specialists_list = ", ".join(specialists)
                delegate_instructions = (
                    f"\n\nDELEGATION PROTOCOL:\n"
                    f"You can delegate specialized work using the delegate_to tool.\n"
                    f"Available specialists: {specialists_list}\n"
                    f"Use delegate_to(specialist='name', task='description') to hand off work."
                )
                agent.config.system_prompt = (agent.config.system_prompt or "") + delegate_instructions

    def _resolve_specialists(
        self,
        agent: Agent,
        policy: list[str] | bool,
    ) -> list[str]:
        """Resolve the list of specialists an agent can delegate to.

        Args:
            agent: The agent that wants to delegate
            policy: Delegation policy (list of names or True for auto-discovery)

        Returns:
            List of agent names this agent can delegate to
        """
        if isinstance(policy, list):
            return policy

        # Auto-discovery based on flow type
        specialists: list[str] = []

        # Reactive flows: look for agents handling agent.request
        if hasattr(self, '_agents_registry'):
            registry = self._agents_registry
            specialists = [
                name for name, (_, cfg) in registry.items()
                if any(t.on == "agent.request" for t in getattr(cfg, 'triggers', []))
            ]

        # Topologies: look for workers or other agents
        elif hasattr(self, 'workers'):
            workers = self.workers
            specialists = [w.name for w in workers if w.name != agent.name]

        elif hasattr(self, '_agents'):
            agents_list = self._agents
            specialists = [a.name for a in agents_list if a.name != agent.name]

        return specialists
