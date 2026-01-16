"""Reactive Skills - Event-triggered behavioral specializations.

Skills are prompt-driven specializations that dynamically modify agent behavior
when matching events occur. Unlike tools (code-based functions), skills inject
context, prompts, and temporarily available tools based on event patterns.

Example:
    ```python
    from agenticflow.reactive import ReactiveFlow, skill

    # Define skills with the clean kwargs API
    python_skill = skill(
        "python_expert",
        on="code.write",
        when=lambda e: e.data.get("language") == "python",
        prompt="You are a Python expert. Write PEP 8 compliant, typed code...",
        tools=[run_python, lint_code],
    )

    debug_skill = skill(
        "debugger",
        on="error.*",
        prompt="You are debugging. Be systematic: reproduce → hypothesize → verify.",
        tools=[read_logs, inspect_vars],
        priority=10,
    )

    # Register skills on the flow
    flow = ReactiveFlow()
    flow.register_skill(python_skill)
    flow.register_skill(debug_skill)

    # When code.write event with language=python occurs, any triggered agent
    # receives the python_expert prompt and tools
    ```
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from agenticflow.reactive.core import EventPattern, Trigger, TriggerBuilder

if TYPE_CHECKING:
    from agenticflow.events.event import Event


@dataclass(frozen=True, slots=True, kw_only=True)
class Skill:
    """Event-triggered behavioral specialization.

    When an event matches the skill's trigger, the skill's context is injected
    into the executing agent's prompt and its tools become temporarily available.

    Attributes:
        name: Unique identifier for the skill.
        trigger: Trigger pattern that activates this skill.
        prompt: Prompt content injected into agent context when skill activates.
        tools: Optional list of tools available only when this skill is active.
        context_enricher: Optional function to modify context when skill activates.
        priority: Higher priority skills are applied first (default 0).

    Example:
        ```python
        skill = Skill(
            name="python_expert",
            trigger=Trigger(
                on="code.write",
                condition=lambda e: e.data.get("language") == "python",
            ),
            prompt="You are a Python expert. Write PEP 8 compliant, typed code...",
            tools=[run_python, lint_code],
        )
        ```
    """

    name: str
    """Unique identifier for the skill."""

    trigger: Trigger
    """Trigger pattern that activates this skill."""

    prompt: str
    """Prompt content injected into agent context when skill activates."""

    tools: tuple[Callable[..., Any], ...] = field(default_factory=tuple)
    """Tools available only when this skill is active."""

    context_enricher: Callable[[Event, dict[str, Any]], dict[str, Any]] | None = None
    """Optional function to modify context when skill activates.

    Receives the triggering event and current context, returns enriched context.
    """

    priority: int = 0
    """Higher priority skills are applied first."""

    def matches(self, event: Event) -> bool:
        """Check if this skill should activate for an event.

        Args:
            event: The event to check against.

        Returns:
            True if the skill's trigger matches the event.
        """
        return self.trigger.matches(event)

    def enrich_context(
        self,
        event: Event,
        context: dict[str, Any],
    ) -> dict[str, Any]:
        """Apply context enrichment if configured.

        Args:
            event: The triggering event.
            context: Current execution context.

        Returns:
            Enriched context dict (may be the same or modified).
        """
        if self.context_enricher is not None:
            return self.context_enricher(event, context)
        return context


def skill(
    name: str,
    *,
    on: EventPattern,
    prompt: str,
    when: Callable[[Event], bool] | None = None,
    tools: list[Callable[..., Any]] | None = None,
    context_enricher: Callable[[Event, dict[str, Any]], dict[str, Any]] | None = None,
    priority: int = 0,
) -> Skill:
    """Create a skill with a clean kwargs API.

    This is the recommended way to define skills. All configuration is passed
    as keyword arguments directly—no builder pattern or chained calls needed.

    Args:
        name: Unique skill identifier.
        on: Event pattern to match (string with optional wildcards, regex, or TraceType).
        prompt: Prompt content injected into agent context when skill activates.
        when: Optional condition function to filter events.
        tools: Optional list of tools available only when skill is active.
        context_enricher: Optional function to modify context when skill activates.
        priority: Higher priority skills are applied first (default 0).

    Returns:
        Configured Skill instance.

    Example:
        ```python
        from agenticflow.reactive import skill

        # Simple skill
        python_skill = skill(
            "python_expert",
            on="code.write",
            when=lambda e: e.data.get("language") == "python",
            prompt="You are a Python expert. Write PEP 8 compliant, typed code.",
        )

        # Skill with tools and priority
        debug_skill = skill(
            "debugger",
            on="error.*",
            prompt="You are debugging. Be systematic: reproduce → hypothesize → verify.",
            tools=[read_logs, inspect_vars],
            priority=10,
        )

        # Register on flow
        flow.register_skill(python_skill)
        flow.register_skill(debug_skill)
        ```
    """
    from agenticflow.reactive.core import Trigger

    trigger = Trigger(on=on, condition=when)

    return Skill(
        name=name,
        trigger=trigger,
        prompt=prompt,
        tools=tuple(tools) if tools else (),
        context_enricher=context_enricher,
        priority=priority,
    )


# Keep SkillBuilder for backward compatibility but mark as legacy
class SkillBuilder:
    """Fluent builder for creating skills.

    Note: The `skill()` function with kwargs is the recommended API.
    This builder is kept for backward compatibility.
    """

    def __init__(self, name: str, pattern: EventPattern) -> None:
        self._name = name
        self._trigger_builder = TriggerBuilder(pattern)
        self._prompt: str = ""
        self._tools: list[Callable[..., Any]] = []
        self._context_enricher: Callable[[Event, dict[str, Any]], dict[str, Any]] | None = None
        self._priority: int = 0

    def when(self, condition: Callable[[Event], bool]) -> SkillBuilder:
        self._trigger_builder.when(condition)
        return self

    def with_prompt(self, prompt: str) -> SkillBuilder:
        self._prompt = prompt
        return self

    def with_tools(self, tools: list[Callable[..., Any]]) -> SkillBuilder:
        self._tools = tools
        return self

    def with_context_enricher(
        self,
        enricher: Callable[[Event, dict[str, Any]], dict[str, Any]],
    ) -> SkillBuilder:
        self._context_enricher = enricher
        return self

    def with_priority(self, priority: int) -> SkillBuilder:
        self._priority = priority
        return self

    def build(self) -> Skill:
        if not self._prompt:
            raise ValueError(f"Skill '{self._name}' requires a prompt.")
        return Skill(
            name=self._name,
            trigger=self._trigger_builder.build(),
            prompt=self._prompt,
            tools=tuple(self._tools),
            context_enricher=self._context_enricher,
            priority=self._priority,
        )

    def __iter__(self):
        yield self.build()

