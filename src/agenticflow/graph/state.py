"""Graph state definitions.

Defines state schemas for LangGraph workflows
with proper typing and merge behaviors.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Annotated, TypeVar, get_type_hints
from operator import add

from langchain_core.messages import BaseMessage, HumanMessage, AIMessage


def _merge_messages(left: list[BaseMessage], right: list[BaseMessage]) -> list[BaseMessage]:
    """Merge message lists, appending new messages."""
    return left + right


def _merge_dicts(left: dict[str, Any], right: dict[str, Any]) -> dict[str, Any]:
    """Merge dictionaries, with right overwriting left."""
    return {**left, **right}


@dataclass
class AgentGraphState:
    """Standard state schema for agent graphs.

    This state can be used directly or extended for custom needs.
    Uses LangGraph's annotation system for merge behavior.

    Attributes:
        messages: Conversation history (appends).
        context: Shared context dictionary (merges).
        current_agent: Currently active agent name.
        next_agent: Next agent to route to.
        task: Current task description.
        iteration: Current iteration count.
        completed: Whether execution is complete.
        error: Error message if any.
        results: Accumulated results from agents.
        metadata: Additional metadata.
    """

    messages: Annotated[list[BaseMessage], _merge_messages] = field(default_factory=list)
    context: Annotated[dict[str, Any], _merge_dicts] = field(default_factory=dict)
    current_agent: str | None = None
    next_agent: str | None = None
    task: str = ""
    iteration: Annotated[int, add] = 0
    completed: bool = False
    error: str | None = None
    results: Annotated[list[dict[str, Any]], add] = field(default_factory=list)
    metadata: Annotated[dict[str, Any], _merge_dicts] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "messages": [
                {"type": type(m).__name__, "content": m.content}
                for m in self.messages
            ],
            "context": self.context,
            "current_agent": self.current_agent,
            "next_agent": self.next_agent,
            "task": self.task,
            "iteration": self.iteration,
            "completed": self.completed,
            "error": self.error,
            "results": self.results,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> AgentGraphState:
        """Create from dictionary."""
        messages = []
        for m in data.get("messages", []):
            if isinstance(m, BaseMessage):
                messages.append(m)
            elif isinstance(m, dict):
                msg_type = m.get("type", "HumanMessage")
                content = m.get("content", "")
                if msg_type == "AIMessage":
                    messages.append(AIMessage(content=content))
                else:
                    messages.append(HumanMessage(content=content))

        return cls(
            messages=messages,
            context=data.get("context", {}),
            current_agent=data.get("current_agent"),
            next_agent=data.get("next_agent"),
            task=data.get("task", ""),
            iteration=data.get("iteration", 0),
            completed=data.get("completed", False),
            error=data.get("error"),
            results=data.get("results", []),
            metadata=data.get("metadata", {}),
        )


def create_state_schema(
    base_class: type = AgentGraphState,
    extra_fields: dict[str, tuple[type, Any]] | None = None,
) -> type:
    """Create a custom state schema class.

    Args:
        base_class: Base state class to extend.
        extra_fields: Additional fields as {name: (type, default)}.

    Returns:
        New state class with extra fields.

    Example:
        >>> State = create_state_schema(
        ...     extra_fields={
        ...         "sentiment": (str, "neutral"),
        ...         "confidence": (float, 0.0),
        ...     }
        ... )
    """
    if not extra_fields:
        return base_class

    # Create namespace for new class
    namespace: dict[str, Any] = {"__annotations__": {}}

    # Add extra fields with defaults wrapped in field()
    for name, (field_type, default) in extra_fields.items():
        namespace["__annotations__"][name] = field_type
        # Use field() with default_factory for mutable defaults
        if isinstance(default, (list, dict, set)):
            namespace[name] = field(default_factory=lambda d=default: type(d)(d))
        else:
            namespace[name] = field(default=default)

    # Create new class extending base
    new_class = type(
        f"Custom{base_class.__name__}",
        (base_class,),
        namespace,
    )

    return dataclass(new_class)


def merge_states(
    base: dict[str, Any],
    update: dict[str, Any],
    state_class: type = AgentGraphState,
) -> dict[str, Any]:
    """Merge state updates according to state schema.

    Respects annotated merge behaviors for each field.

    Args:
        base: Current state dictionary.
        update: State updates to apply.
        state_class: State class for merge behavior hints.

    Returns:
        Merged state dictionary.
    """
    result = base.copy()

    hints = get_type_hints(state_class, include_extras=True)

    for key, value in update.items():
        if key not in result:
            result[key] = value
            continue

        # Check for annotated merge behavior
        if key in hints:
            hint = hints[key]
            if hasattr(hint, "__metadata__"):
                # Has merge function
                merge_fn = hint.__metadata__[0]
                if callable(merge_fn):
                    result[key] = merge_fn(result[key], value)
                    continue

        # Default: overwrite
        result[key] = value

    return result
