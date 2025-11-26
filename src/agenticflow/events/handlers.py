"""
Event handlers - built-in handlers for common use cases.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import TextIO

from agenticflow.core.enums import EventType
from agenticflow.models.event import Event


class ConsoleEventHandler:
    """
    Logs events to console with formatting and icons.
    
    Provides a human-readable view of events for development
    and debugging purposes.
    
    Attributes:
        verbose: Whether to show detailed event data
        show_timestamp: Whether to show timestamps
        
    Example:
        ```python
        handler = ConsoleEventHandler(verbose=True)
        event_bus.subscribe_all(handler)
        ```
    """

    # Icons for different event types
    ICONS: dict[EventType, str] = {
        # System
        EventType.SYSTEM_STARTED: "🚀",
        EventType.SYSTEM_STOPPED: "🏁",
        EventType.SYSTEM_ERROR: "💥",
        # Tasks
        EventType.TASK_CREATED: "📝",
        EventType.TASK_SCHEDULED: "📋",
        EventType.TASK_STARTED: "▶️",
        EventType.TASK_COMPLETED: "✅",
        EventType.TASK_FAILED: "❌",
        EventType.TASK_CANCELLED: "🚫",
        EventType.TASK_BLOCKED: "⏸️",
        EventType.TASK_UNBLOCKED: "⏯️",
        EventType.TASK_RETRYING: "🔄",
        # Subtasks
        EventType.SUBTASK_SPAWNED: "🔀",
        EventType.SUBTASK_COMPLETED: "✔️",
        EventType.SUBTASKS_AGGREGATED: "📦",
        # Agents
        EventType.AGENT_REGISTERED: "🤖",
        EventType.AGENT_UNREGISTERED: "👋",
        EventType.AGENT_INVOKED: "📞",
        EventType.AGENT_THINKING: "🧠",
        EventType.AGENT_ACTING: "⚡",
        EventType.AGENT_RESPONDED: "💬",
        EventType.AGENT_ERROR: "🔥",
        EventType.AGENT_STATUS_CHANGED: "🔄",
        # Tools
        EventType.TOOL_REGISTERED: "🔧",
        EventType.TOOL_CALLED: "🛠️",
        EventType.TOOL_RESULT: "📤",
        EventType.TOOL_ERROR: "⚠️",
        # Planning
        EventType.PLAN_CREATED: "📊",
        EventType.PLAN_STEP_STARTED: "⚡",
        EventType.PLAN_STEP_COMPLETED: "✔️",
        EventType.PLAN_FAILED: "❌",
        # Messages
        EventType.MESSAGE_SENT: "📨",
        EventType.MESSAGE_RECEIVED: "📩",
        EventType.MESSAGE_BROADCAST: "📢",
        # Clients
        EventType.CLIENT_CONNECTED: "🔌",
        EventType.CLIENT_DISCONNECTED: "🔌",
        EventType.CLIENT_MESSAGE: "💬",
    }

    def __init__(
        self,
        verbose: bool = False,
        show_timestamp: bool = True,
        show_source: bool = True,
    ) -> None:
        """
        Initialize the console handler.
        
        Args:
            verbose: Show detailed event data
            show_timestamp: Show timestamps
            show_source: Show event source
        """
        self.verbose = verbose
        self.show_timestamp = show_timestamp
        self.show_source = show_source

    def __call__(self, event: Event) -> None:
        """Handle an event by logging to console."""
        icon = self.ICONS.get(event.type, "•")
        timestamp = ""
        if self.show_timestamp:
            timestamp = f"[{event.timestamp.strftime('%H:%M:%S.%f')[:-3]}] "

        source = ""
        if self.show_source and event.source != "system":
            source = f" ({event.source})"

        # Format based on event type
        message = self._format_event(event)
        print(f"  {icon} {timestamp}{message}{source}")

        if self.verbose and event.data:
            print(f"      Data: {json.dumps(event.data, default=str)[:200]}")

    def _format_event(self, event: Event) -> str:
        """Format event for display."""
        data = event.data

        match event.type:
            # System events
            case EventType.SYSTEM_STARTED:
                return "System started"
            case EventType.SYSTEM_STOPPED:
                return "System stopped"
            case EventType.SYSTEM_ERROR:
                return f"System error: {data.get('error', 'unknown')}"

            # Task events
            case EventType.TASK_CREATED:
                name = data.get("name", data.get("task", {}).get("name", "unknown"))
                deps = data.get("depends_on", [])
                dep_str = f" → depends on {deps}" if deps else ""
                return f"Created: {name}{dep_str}"
            case EventType.TASK_STARTED:
                task = data.get("task", {})
                return f"{task.get('name', 'unknown')} STARTED"
            case EventType.TASK_COMPLETED:
                task = data.get("task", {})
                duration = task.get("duration_ms", 0)
                return f"{task.get('name', 'unknown')} COMPLETED ({duration:.0f}ms)"
            case EventType.TASK_FAILED:
                task = data.get("task", {})
                error = task.get("error", "unknown error")
                return f"{task.get('name', 'unknown')} FAILED: {error}"

            # Agent events
            case EventType.AGENT_REGISTERED:
                return f"Agent registered: {data.get('agent_name', 'unknown')}"
            case EventType.AGENT_THINKING:
                return f"{data.get('agent_name', 'Agent')} thinking..."
            case EventType.AGENT_ACTING:
                return f"{data.get('agent_name', 'Agent')} acting..."
            case EventType.AGENT_RESPONDED:
                preview = data.get("result_preview", "")[:50]
                return f"{data.get('agent_name', 'Agent')}: {preview}..."
            case EventType.AGENT_ERROR:
                return f"{data.get('agent_name', 'Agent')} error: {data.get('error', 'unknown')}"

            # Tool events
            case EventType.TOOL_CALLED:
                return f"Calling {data.get('tool', 'unknown')}"
            case EventType.TOOL_RESULT:
                return f"Tool result received"
            case EventType.TOOL_ERROR:
                return f"Tool error: {data.get('error', 'unknown')}"

            # Plan events
            case EventType.PLAN_CREATED:
                return f"Plan created: {data.get('step_count', '?')} steps"
            case EventType.PLAN_STEP_STARTED:
                step = data.get("step", "?")
                count = data.get("task_count", 1)
                parallel = " ⚡PARALLEL" if count > 1 else ""
                return f"Step {step}: {count} task(s){parallel}"
            case EventType.PLAN_STEP_COMPLETED:
                return f"Step {data.get('step', '?')} completed"

            # Default
            case _:
                return event.type.value


class FileEventHandler:
    """
    Logs events to a file in JSON Lines format.
    
    Each event is written as a single JSON line, making it easy
    to parse and analyze later.
    
    Attributes:
        file_path: Path to the log file
        
    Example:
        ```python
        handler = FileEventHandler("events.jsonl")
        event_bus.subscribe_all(handler)
        ```
    """

    def __init__(
        self,
        file_path: str | Path,
        append: bool = True,
    ) -> None:
        """
        Initialize the file handler.
        
        Args:
            file_path: Path to the log file
            append: Whether to append to existing file
        """
        self.file_path = Path(file_path)
        self._file: TextIO | None = None
        self._append = append

    def _ensure_file(self) -> TextIO:
        """Ensure file is open for writing."""
        if self._file is None or self._file.closed:
            mode = "a" if self._append else "w"
            self._file = open(self.file_path, mode)
        return self._file

    def __call__(self, event: Event) -> None:
        """Handle an event by writing to file."""
        f = self._ensure_file()
        f.write(event.to_json() + "\n")
        f.flush()

    def close(self) -> None:
        """Close the file."""
        if self._file and not self._file.closed:
            self._file.close()

    def __del__(self) -> None:
        """Cleanup on deletion."""
        self.close()


class FilteringEventHandler:
    """
    Wraps another handler with event filtering.
    
    Example:
        ```python
        # Only handle task events
        handler = FilteringEventHandler(
            ConsoleEventHandler(),
            categories=["task"],
        )
        event_bus.subscribe_all(handler)
        ```
    """

    def __init__(
        self,
        inner_handler: ConsoleEventHandler | FileEventHandler,
        event_types: list[EventType] | None = None,
        categories: list[str] | None = None,
        sources: list[str] | None = None,
    ) -> None:
        """
        Initialize the filtering handler.
        
        Args:
            inner_handler: Handler to delegate to
            event_types: Only handle these event types
            categories: Only handle events in these categories
            sources: Only handle events from these sources
        """
        self.inner = inner_handler
        self.event_types = set(event_types) if event_types else None
        self.categories = set(categories) if categories else None
        self.sources = set(sources) if sources else None

    def __call__(self, event: Event) -> None:
        """Handle event if it passes filters."""
        # Check event type filter
        if self.event_types and event.type not in self.event_types:
            return

        # Check category filter
        if self.categories and event.category not in self.categories:
            return

        # Check source filter
        if self.sources and event.source not in self.sources:
            return

        # Delegate to inner handler
        self.inner(event)


class MetricsEventHandler:
    """
    Collects metrics from events.
    
    Tracks counts, durations, and error rates for analysis.
    
    Example:
        ```python
        metrics = MetricsEventHandler()
        event_bus.subscribe_all(metrics)
        
        # Later...
        print(metrics.get_metrics())
        ```
    """

    def __init__(self) -> None:
        """Initialize the metrics handler."""
        self.event_counts: dict[str, int] = {}
        self.task_durations: list[float] = []
        self.error_count = 0
        self.start_time = datetime.now(timezone.utc)

    def __call__(self, event: Event) -> None:
        """Collect metrics from event."""
        # Count events by type
        key = event.type.value
        self.event_counts[key] = self.event_counts.get(key, 0) + 1

        # Track task durations
        if event.type == EventType.TASK_COMPLETED:
            task = event.data.get("task", {})
            duration = task.get("duration_ms")
            if duration:
                self.task_durations.append(duration)

        # Count errors
        if event.type in (
            EventType.TASK_FAILED,
            EventType.AGENT_ERROR,
            EventType.TOOL_ERROR,
            EventType.SYSTEM_ERROR,
        ):
            self.error_count += 1

    def get_metrics(self) -> dict:
        """
        Get collected metrics.
        
        Returns:
            Dictionary with metrics
        """
        total_events = sum(self.event_counts.values())
        avg_duration = (
            sum(self.task_durations) / len(self.task_durations)
            if self.task_durations
            else 0
        )

        return {
            "total_events": total_events,
            "event_counts": self.event_counts,
            "error_count": self.error_count,
            "error_rate": self.error_count / total_events if total_events > 0 else 0,
            "tasks_completed": len(self.task_durations),
            "avg_task_duration_ms": avg_duration,
            "min_task_duration_ms": min(self.task_durations) if self.task_durations else 0,
            "max_task_duration_ms": max(self.task_durations) if self.task_durations else 0,
            "uptime_seconds": (datetime.now(timezone.utc) - self.start_time).total_seconds(),
        }

    def reset(self) -> None:
        """Reset all metrics."""
        self.event_counts.clear()
        self.task_durations.clear()
        self.error_count = 0
        self.start_time = datetime.now(timezone.utc)
