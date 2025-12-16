"""Reactive agent helpers.

The regular `Agent` API is optimized for imperative prompts.
Reactive flows are easier to author when you can describe event handling
without writing brittle "MUST call tool" prompts.

This module intentionally keeps things explicit:
- you still create an `Agent`
- you still register triggers on an `EventFlow`

We only provide a tiny prompt builder to reduce prompt boilerplate.
"""

from __future__ import annotations

import json
from typing import Any, Mapping

from agenticflow.agent.base import Agent
from agenticflow.core.messages import HumanMessage, SystemMessage
from agenticflow.executors.native import NativeExecutor
from agenticflow.models.base import BaseChatModel
from agenticflow.reactive.core import TriggerBuilder, TriggerCondition, react_to
from agenticflow.tools.base import BaseTool


def _tool_name(tool: BaseTool | str) -> str:
    if isinstance(tool, BaseTool):
        return tool.name
    return tool


class ReactiveAgent(Agent):
    """An `Agent` optimized for reactive/event-driven flows.

    This is intentionally a thin layer on top of the base `Agent`:
    - You configure event handlers via `.on(event, tool=...)`
    - You register it into an `EventFlow`/`ReactiveFlow` via `.register(flow)`

    The goal is to provide a clean, future-friendly API for reactive systems
    without forcing users to write brittle system prompts.
    """

    def __init__(
        self,
        *,
        name: str,
        model: BaseChatModel,
        description: str = "",
        instructions: str | None = None,
        tools: list[BaseTool] | None = None,
        memory: Any = None,
        store: Any = None,
    ) -> None:
        self._reactive_handlers: dict[str, BaseTool] = {}
        self._reactive_triggers: list[TriggerBuilder] = []
        self._reactive_instructions: str | None = instructions

        super().__init__(
            name=name,
            model=model,
            description=description,
            system_prompt=build_reactive_system_prompt({}, instructions=instructions),
            tools=tools or [],
            memory=memory,
            store=store,
        )

    def _ensure_tool(self, tool: BaseTool) -> None:
        """Ensure a tool is available to the agent."""
        # Add to direct tool objects
        if tool not in getattr(self, "_direct_tools", []):
            self._direct_tools.append(tool)  # type: ignore[attr-defined]

        # Add tool name to config.tools for registry lookups
        if tool.name not in self.config.tools:
            self.config.tools.append(tool.name)

    def _refresh_system_prompt(self) -> None:
        self.config.system_prompt = build_reactive_system_prompt(
            self._reactive_handlers,
            instructions=self._reactive_instructions,
        )

    def on(
        self,
        event_name: str,
        *,
        tool: BaseTool,
        condition: TriggerCondition | None = None,
        emits: str | None = None,
        priority: int = 0,
    ) -> "ReactiveAgent":
        """Handle an event by making the tool available and adding a trigger.

        Args:
            event_name: Event name (e.g. "webhook.video_complete").
            tool: Tool the agent should use for this event.
            condition: Optional filter for event data.
            emits: Optional event to emit after completion.
            priority: Trigger priority.
        """

        self._reactive_handlers[event_name] = tool
        self._ensure_tool(tool)

        trigger = react_to(event_name)
        if condition is not None:
            trigger = trigger.when(condition)
        if emits is not None:
            trigger = trigger.emits(emits)
        if priority:
            trigger = trigger.with_priority(priority)
        self._reactive_triggers.append(trigger)

        self._refresh_system_prompt()
        return self

    @property
    def triggers(self) -> list[TriggerBuilder]:
        """Triggers for registering into an EventFlow/ReactiveFlow."""
        return list(self._reactive_triggers)

    @property
    def handlers(self) -> dict[str, BaseTool]:
        """Event -> tool mapping."""
        return dict(self._reactive_handlers)

    def register(self, flow: Any) -> None:
        """Register this agent into a reactive flow."""
        flow.register(self, self.triggers)

    async def react(
        self,
        event: Any,
        *,
        task: str,
        context: dict[str, Any] | None = None,
        thread_id: str | None = None,
        max_iterations: int = 10,
    ) -> Any:
        """React to an event.

        This is the reactive counterpart to `run()`. Reactive flows should call
        this method when available.

        Args:
            event: Core event (typically agenticflow.events.event.Event).
            task: The overarching flow task.
            context: Shared flow context.
            thread_id: Optional thread id for memory-backed conversations.
            max_iterations: Max agent loop iterations.
        """

        try:
            event_name = getattr(event, "name", None) or "(unknown)"
            event_data = getattr(event, "data", None) or {}
        except Exception:
            event_name = "(unknown)"
            event_data = {}

        envelope: dict[str, Any] = {
            "task": task,
            "event": {"name": event_name, "data": event_data},
            "context": context or {},
        }

        user_content = json.dumps(envelope, default=str, ensure_ascii=False)

        messages = []
        if self.config.system_prompt:
            messages.append(SystemMessage(content=self.config.system_prompt))

        # Thread history (if enabled)
        if thread_id:
            if getattr(self, "_memory_manager", None):
                history = await self._memory_manager.get_thread_messages(thread_id)  # type: ignore[union-attr]
            else:
                history = await self._memory.get_messages(thread_id)
            messages.extend(history)

        messages.append(HumanMessage(content=user_content))

        executor = NativeExecutor(self)
        executor.max_iterations = max_iterations
        result = await executor.execute_messages(
            task=task,
            messages=messages,
            context=context,
        )

        # Persist to thread history (if enabled)
        if thread_id:
            assistant_text = result.output if hasattr(result, "output") else str(result)
            await self._save_to_thread(thread_id, user_content, assistant_text)

        return result

def build_reactive_system_prompt(
    handlers: Mapping[str, BaseTool],
    *,
    instructions: str | None = None,
) -> str:
    """Build a system prompt for an agent that reacts to events.

    This avoids over-prescriptive prompts while still being clear.

    Args:
        handlers: Mapping of `event_name -> tool` that is appropriate for that event.
        instructions: Optional additional instructions appended at the end.

    Returns:
        A system prompt string.
    """
    lines: list[str] = [
        "You are a reactive agent.",
        "You will be invoked with a JSON event envelope in the latest user message.",
        "The envelope schema is:",
        '{"task": str, "event": {"name": str, "data": object}, "context": object}',
        "Use the event.name and event.data to decide what to do, and use tools when needed.",
        "",
        "Supported events:",
    ]

    if handlers:
        for event_name, tool in handlers.items():
            lines.append(f"- {event_name} -> call tool '{_tool_name(tool)}'")
    else:
        lines.append("- (no handlers configured yet)")

    lines += [
        "",
        "Guidelines:",
        "- Use tools when needed to produce the result.",
        "- If required fields are missing, respond with a short error describing what's missing.",
        "- Keep responses concise.",
    ]

    if instructions:
        lines += ["", "Extra instructions:", instructions]

    return "\n".join(lines)
