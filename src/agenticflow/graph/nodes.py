"""Graph node implementations.

Pre-built node types for common agent graph patterns.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, TYPE_CHECKING

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from agenticflow.graph.state import AgentGraphState

if TYPE_CHECKING:
    from agenticflow.agents import Agent
    from agenticflow.tools import ToolRegistry


@dataclass
class NodeResult:
    """Result from a node execution.

    Attributes:
        updates: State updates to apply.
        next_node: Explicit next node (for routing).
        interrupt: Whether to interrupt for human input.
    """

    updates: dict[str, Any] = field(default_factory=dict)
    next_node: str | None = None
    interrupt: bool = False


class BaseNode(ABC):
    """Abstract base for graph nodes."""

    def __init__(self, name: str) -> None:
        """Initialize node.

        Args:
            name: Node name.
        """
        self.name = name

    @abstractmethod
    async def __call__(self, state: dict[str, Any]) -> dict[str, Any]:
        """Execute node.

        Args:
            state: Current graph state.

        Returns:
            State updates.
        """
        ...


class AgentNode(BaseNode):
    """Node that wraps an Agent.

    Executes the agent's think method and returns state updates.

    Example:
        >>> node = AgentNode("researcher", researcher_agent)
        >>> result = await node(state)
    """

    def __init__(
        self,
        name: str,
        agent: "Agent",
        include_history: bool = True,
        max_history: int = 10,
    ) -> None:
        """Initialize agent node.

        Args:
            name: Node name.
            agent: Agent to wrap.
            include_history: Whether to include message history.
            max_history: Max messages to include.
        """
        super().__init__(name)
        self.agent = agent
        self.include_history = include_history
        self.max_history = max_history

    async def __call__(self, state: dict[str, Any]) -> dict[str, Any]:
        """Execute agent."""
        task = state.get("task", "")
        context = state.get("context", {})

        # Include message history in context
        if self.include_history:
            messages = state.get("messages", [])
            if messages:
                history = messages[-self.max_history:]
                context["message_history"] = [
                    {"role": "assistant" if isinstance(m, AIMessage) else "user", "content": m.content}
                    for m in history
                ]

        # Agent thinks
        thought = await self.agent.think(task, context)

        return {
            "messages": [AIMessage(content=thought)],
            "current_agent": self.name,
            "results": [{"agent": self.name, "thought": thought}],
            "iteration": 1,
        }


class ToolNode(BaseNode):
    """Node that executes tools based on state.

    Looks for tool calls in the state and executes them.

    Example:
        >>> node = ToolNode("tools", tool_registry)
        >>> result = await node(state)
    """

    def __init__(
        self,
        name: str,
        tool_registry: "ToolRegistry",
    ) -> None:
        """Initialize tool node.

        Args:
            name: Node name.
            tool_registry: Registry of available tools.
        """
        super().__init__(name)
        self.tool_registry = tool_registry

    async def __call__(self, state: dict[str, Any]) -> dict[str, Any]:
        """Execute tools from state."""
        tool_calls = state.get("tool_calls", [])
        results = []
        messages = []

        for call in tool_calls:
            tool_name = call.get("name")
            tool_args = call.get("args", {})

            tool = self.tool_registry.get(tool_name)
            if tool:
                try:
                    result = await tool.ainvoke(tool_args)
                    results.append({
                        "tool": tool_name,
                        "result": result,
                        "success": True,
                    })
                    messages.append(ToolMessage(
                        content=str(result),
                        tool_call_id=call.get("id", tool_name),
                    ))
                except Exception as e:
                    results.append({
                        "tool": tool_name,
                        "error": str(e),
                        "success": False,
                    })
            else:
                results.append({
                    "tool": tool_name,
                    "error": f"Tool '{tool_name}' not found",
                    "success": False,
                })

        return {
            "messages": messages,
            "tool_results": results,
            "tool_calls": [],  # Clear processed calls
        }


class RouterNode(BaseNode):
    """Node that routes to other nodes based on state.

    Uses a routing function to determine the next node.

    Example:
        >>> def route(state):
        ...     if state.get("needs_research"):
        ...         return "researcher"
        ...     return "writer"
        >>> node = RouterNode("router", route)
    """

    def __init__(
        self,
        name: str,
        route_fn: Callable[[dict[str, Any]], str],
    ) -> None:
        """Initialize router node.

        Args:
            name: Node name.
            route_fn: Function that returns next node name.
        """
        super().__init__(name)
        self.route_fn = route_fn

    async def __call__(self, state: dict[str, Any]) -> dict[str, Any]:
        """Determine routing."""
        next_node = self.route_fn(state)
        return {
            "next_agent": next_node,
        }


class HumanNode(BaseNode):
    """Node that requests human input.

    Uses LangGraph's interrupt for human-in-the-loop.

    Example:
        >>> node = HumanNode("human_review", "Please review this output")
        >>> result = await node(state)  # Will interrupt
    """

    def __init__(
        self,
        name: str,
        prompt: str | Callable[[dict[str, Any]], str] | None = None,
    ) -> None:
        """Initialize human node.

        Args:
            name: Node name.
            prompt: Prompt for human or function to generate prompt.
        """
        super().__init__(name)
        self.prompt = prompt

    async def __call__(self, state: dict[str, Any]) -> dict[str, Any]:
        """Request human input."""
        from langgraph.types import interrupt

        # Generate prompt
        if callable(self.prompt):
            prompt_text = self.prompt(state)
        elif self.prompt:
            prompt_text = self.prompt
        else:
            prompt_text = "Please provide input:"

        # Include context
        context = {
            "prompt": prompt_text,
            "current_state": {
                "task": state.get("task"),
                "iteration": state.get("iteration"),
                "current_agent": state.get("current_agent"),
            },
        }

        # Interrupt for human input
        human_input = interrupt(context)

        # Return with human message
        return {
            "messages": [HumanMessage(content=str(human_input))],
            "human_input": human_input,
            "context": {"last_human_input": human_input},
        }


class ConditionalNode(BaseNode):
    """Node that conditionally executes different logic.

    Example:
        >>> node = ConditionalNode(
        ...     "check",
        ...     condition=lambda s: s.get("score", 0) > 0.8,
        ...     if_true=lambda s: {"status": "approved"},
        ...     if_false=lambda s: {"status": "needs_review"},
        ... )
    """

    def __init__(
        self,
        name: str,
        condition: Callable[[dict[str, Any]], bool],
        if_true: Callable[[dict[str, Any]], dict[str, Any]],
        if_false: Callable[[dict[str, Any]], dict[str, Any]],
    ) -> None:
        """Initialize conditional node.

        Args:
            name: Node name.
            condition: Function that returns True/False.
            if_true: Function to call if condition is True.
            if_false: Function to call if condition is False.
        """
        super().__init__(name)
        self.condition = condition
        self.if_true = if_true
        self.if_false = if_false

    async def __call__(self, state: dict[str, Any]) -> dict[str, Any]:
        """Evaluate condition and execute."""
        if self.condition(state):
            result = self.if_true(state)
        else:
            result = self.if_false(state)

        # Handle async functions
        if hasattr(result, "__await__"):
            result = await result

        return result
