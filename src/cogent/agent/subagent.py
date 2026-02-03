"""Subagent registry and management.

Enables native subagent support where agents can delegate to other agents
while preserving full Response[T] metadata for aggregation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from cogent.agent.base import Agent
    from cogent.core.context import RunContext
    from cogent.core.response import Response


@dataclass(slots=True)
class SubagentRegistry:
    """Manages subagent registration, execution, and response tracking.
    
    The registry allows agents to delegate to other agents while preserving
    full Response[T] objects for metadata aggregation (tokens, duration, etc.).
    
    Subagents are registered as tools from the LLM's perspective, but the
    executor recognizes them and handles them specially to preserve metadata.
    
    Example:
        ```python
        registry = SubagentRegistry()
        registry.register("analyst", analyst_agent)
        
        # In executor
        if registry.has_subagent(tool_name):
            response = await registry.execute(tool_name, task, context)
            # Full Response[T] preserved, not just string
        ```
    """
    
    _agents: dict[str, Agent] = field(default_factory=dict)
    _responses: list[Response] = field(default_factory=list)
    
    def register(self, name: str, agent: Agent) -> None:
        """Register a subagent.
        
        Args:
            name: The name to register the agent under (used in tool calls).
            agent: The Agent instance to register.
            
        Example:
            ```python
            registry.register("analyst", analyst_agent)
            registry.register("writer", writer_agent)
            ```
        """
        self._agents[name] = agent
    
    def has_subagent(self, name: str) -> bool:
        """Check if a name corresponds to a registered subagent.
        
        Args:
            name: The name to check.
            
        Returns:
            True if name is a registered subagent, False otherwise.
            
        Example:
            ```python
            if registry.has_subagent("analyst"):
                # Handle as subagent
            else:
                # Handle as regular tool
            ```
        """
        return name in self._agents
    
    async def execute(
        self,
        name: str,
        task: str,
        context: RunContext | None = None,
    ) -> Response:
        """Execute a subagent and cache its Response.
        
        Args:
            name: Name of the subagent to execute.
            task: Task to delegate to the subagent.
            context: Optional RunContext to propagate to subagent.
            
        Returns:
            Full Response[T] object from subagent execution.
            
        Raises:
            KeyError: If subagent name is not registered.
            
        Example:
            ```python
            response = await registry.execute(
                "analyst",
                "Analyze Q4 sales data",
                context=run_context
            )
            # response.metadata.tokens available for aggregation
            ```
        """
        if name not in self._agents:
            raise KeyError(f"Subagent '{name}' not registered")
        
        agent = self._agents[name]
        
        # Execute subagent with context propagation
        response = await agent.run(task, context=context)
        
        # Cache response for metadata aggregation
        self._responses.append(response)
        
        return response
    
    def get_responses(self) -> list[Response]:
        """Get all cached subagent responses for metadata aggregation.
        
        Returns:
            List of Response objects from all subagent executions.
            
        Example:
            ```python
            # After execution
            responses = registry.get_responses()
            total_tokens = sum(
                r.metadata.tokens.total_tokens
                for r in responses
                if r.metadata and r.metadata.tokens
            )
            ```
        """
        return list(self._responses)
    
    def clear(self) -> None:
        """Clear cached responses.
        
        Should be called before each new run to avoid accumulating
        responses from previous executions.
        
        Example:
            ```python
            # Before new run
            registry.clear()
            result = await agent.run(task)
            ```
        """
        self._responses.clear()
    
    def generate_documentation(self) -> str:
        """Generate system prompt documentation for registered subagents.
        
        Creates human-readable documentation that gets added to the agent's
        system prompt, explaining what specialist agents are available.
        
        Returns:
            Formatted documentation string, or empty if no subagents.
            
        Example:
            ```python
            docs = registry.generate_documentation()
            # "# Specialist Agents
            #
            # You have access to specialist agents:
            # - analyst: Financial data analysis specialist
            # - writer: Report writing specialist
            # ..."
            ```
        """
        if not self._agents:
            return ""
        
        lines = ["# Specialist Agents", ""]
        lines.append("You have access to specialist agents for complex tasks:")
        lines.append("")
        
        for name, agent in self._agents.items():
            description = agent.config.description or "No description available"
            lines.append(f"- **{name}**: {description}")
        
        lines.append("")
        lines.append("To use a specialist, call them like a tool with a 'task' parameter describing what you want them to do.")
        
        return "\n".join(lines)
    
    @property
    def agent_names(self) -> list[str]:
        """Get list of registered subagent names.
        
        Returns:
            List of subagent names.
        """
        return list(self._agents.keys())
    
    @property
    def count(self) -> int:
        """Get count of registered subagents.
        
        Returns:
            Number of registered subagents.
        """
        return len(self._agents)
    
    def __repr__(self) -> str:
        """String representation of registry."""
        agents = ", ".join(self._agents.keys()) if self._agents else "none"
        return f"SubagentRegistry(agents=[{agents}], cached_responses={len(self._responses)})"
