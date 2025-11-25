"""Handoff mechanisms for agent coordination.

Provides utilities for agent-to-agent handoffs using
LangGraph's Command and interrupt primitives.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from langgraph.types import Command, interrupt


class HandoffType(Enum):
    """Types of agent handoffs."""

    DIRECT = "direct"  # Direct transfer to specific agent
    BROADCAST = "broadcast"  # Send to multiple agents
    CONDITIONAL = "conditional"  # Route based on condition
    HUMAN = "human"  # Hand off to human


@dataclass
class Handoff:
    """Represents an agent handoff.

    Attributes:
        target: Target agent or agents.
        handoff_type: Type of handoff.
        state_update: State changes to apply.
        message: Message to include with handoff.
        metadata: Additional handoff metadata.
    """

    target: str | list[str]
    handoff_type: HandoffType = HandoffType.DIRECT
    state_update: dict[str, Any] = field(default_factory=dict)
    message: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_command(self) -> Command:
        """Convert to LangGraph Command.

        Returns:
            Command for graph execution.
        """
        update = self.state_update.copy()

        if self.message:
            from langchain_core.messages import AIMessage

            update["messages"] = [AIMessage(content=self.message)]

        update["metadata"] = {
            **update.get("metadata", {}),
            "handoff_type": self.handoff_type.value,
            "handoff_target": self.target,
        }

        if isinstance(self.target, list):
            # For broadcast, go to first target
            # (parallel execution requires different graph structure)
            goto = self.target[0] if self.target else None
        else:
            goto = self.target

        return Command(
            goto=goto,
            update=update,
        )


def create_handoff(
    target: str,
    *,
    message: str = "",
    state_update: dict[str, Any] | None = None,
    **metadata: Any,
) -> Command:
    """Create a direct handoff to another agent.

    Args:
        target: Target agent name.
        message: Message to pass to target.
        state_update: State changes to apply.
        **metadata: Additional metadata.

    Returns:
        LangGraph Command for the handoff.

    Example:
        >>> command = create_handoff(
        ...     "reviewer",
        ...     message="Please review this draft",
        ...     draft=draft_content,
        ... )
    """
    handoff = Handoff(
        target=target,
        handoff_type=HandoffType.DIRECT,
        state_update=state_update or {},
        message=message,
        metadata=metadata,
    )
    return handoff.to_command()


def create_interrupt(
    question: str,
    *,
    context: dict[str, Any] | None = None,
    options: list[str] | None = None,
) -> Any:
    """Create an interrupt for human input.

    Uses LangGraph's interrupt for human-in-the-loop.

    Args:
        question: Question to ask human.
        context: Additional context for the human.
        options: Optional list of choices.

    Returns:
        Human's response after resumption.

    Example:
        >>> response = create_interrupt(
        ...     "Should we proceed with this plan?",
        ...     options=["yes", "no", "modify"],
        ... )
    """
    payload = {
        "question": question,
        "context": context or {},
    }

    if options:
        payload["options"] = options

    return interrupt(payload)


def resume_with_command(
    response: Any,
    *,
    goto: str | None = None,
    state_update: dict[str, Any] | None = None,
) -> Command:
    """Create a resume command after interrupt.

    Args:
        response: Human's response to include.
        goto: Optional node to route to.
        state_update: State updates to apply.

    Returns:
        Command to resume execution.
    """
    update = state_update or {}
    update["human_response"] = response

    return Command(
        resume=response,
        goto=goto,
        update=update,
    )


class HandoffBuilder:
    """Fluent builder for complex handoffs.

    Example:
        >>> handoff = (
        ...     HandoffBuilder("reviewer")
        ...     .with_message("Please review")
        ...     .with_context(draft=content)
        ...     .with_priority("high")
        ...     .build()
        ... )
    """

    def __init__(self, target: str | list[str]) -> None:
        """Initialize builder.

        Args:
            target: Target agent(s).
        """
        self._target = target
        self._type = HandoffType.DIRECT
        self._message = ""
        self._state_update: dict[str, Any] = {}
        self._metadata: dict[str, Any] = {}

    def with_message(self, message: str) -> "HandoffBuilder":
        """Add handoff message.

        Args:
            message: Message content.

        Returns:
            Self for chaining.
        """
        self._message = message
        return self

    def with_context(self, **context: Any) -> "HandoffBuilder":
        """Add context to state update.

        Args:
            **context: Context key-value pairs.

        Returns:
            Self for chaining.
        """
        if "context" not in self._state_update:
            self._state_update["context"] = {}
        self._state_update["context"].update(context)
        return self

    def with_state(self, **updates: Any) -> "HandoffBuilder":
        """Add state updates.

        Args:
            **updates: State updates.

        Returns:
            Self for chaining.
        """
        self._state_update.update(updates)
        return self

    def with_metadata(self, **metadata: Any) -> "HandoffBuilder":
        """Add metadata.

        Args:
            **metadata: Metadata key-value pairs.

        Returns:
            Self for chaining.
        """
        self._metadata.update(metadata)
        return self

    def with_priority(self, priority: str) -> "HandoffBuilder":
        """Set handoff priority.

        Args:
            priority: Priority level.

        Returns:
            Self for chaining.
        """
        self._metadata["priority"] = priority
        return self

    def as_broadcast(self) -> "HandoffBuilder":
        """Set handoff type to broadcast.

        Returns:
            Self for chaining.
        """
        self._type = HandoffType.BROADCAST
        return self

    def build(self) -> Handoff:
        """Build the handoff.

        Returns:
            Configured Handoff instance.
        """
        return Handoff(
            target=self._target,
            handoff_type=self._type,
            state_update=self._state_update,
            message=self._message,
            metadata=self._metadata,
        )

    def to_command(self) -> Command:
        """Build and convert to Command.

        Returns:
            LangGraph Command.
        """
        return self.build().to_command()
