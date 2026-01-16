"""
AgentState - runtime state of an Agent.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING

from agenticflow.core.enums import AgentStatus
from agenticflow.core.utils import now_utc

if TYPE_CHECKING:
    from agenticflow.core.messages import BaseMessage


@dataclass
class AgentState:
    """
    Runtime state of an Agent.

    Tracks the agent's current status, active tasks, conversation history,
    and performance metrics. State is mutable and updated during execution.

    Attributes:
        status: Current agent status
        current_task_id: ID of the task currently being executed
        active_task_ids: IDs of all active tasks assigned to this agent
        message_history: Conversation history for context
        last_activity: Timestamp of last activity
        error_count: Total errors encountered
        tasks_completed: Total tasks successfully completed
        tasks_failed: Total tasks that failed
        total_thinking_time_ms: Cumulative LLM processing time
        total_acting_time_ms: Cumulative tool execution time

    Example:
        ```python
        state = AgentState()
        state.status = AgentStatus.THINKING
        state.current_task_id = "task-123"
        ```
    """

    status: AgentStatus = AgentStatus.IDLE
    current_task_id: str | None = None
    active_task_ids: list[str] = field(default_factory=list)
    message_history: list[BaseMessage] = field(default_factory=list)
    last_activity: datetime = field(default_factory=now_utc)

    # Error tracking
    error_count: int = 0
    last_error: str | None = None
    last_error_at: datetime | None = None

    # Performance metrics
    tasks_completed: int = 0
    tasks_failed: int = 0
    total_thinking_time_ms: float = 0.0
    total_acting_time_ms: float = 0.0

    def is_available(self) -> bool:
        """Check if agent can accept new work."""
        return self.status.is_available()

    def is_working(self) -> bool:
        """Check if agent is currently working."""
        return self.status.is_working()

    def has_capacity(self, max_concurrent: int) -> bool:
        """
        Check if agent has capacity for more tasks.

        Args:
            max_concurrent: Maximum concurrent tasks allowed

        Returns:
            True if agent can accept more tasks
        """
        return len(self.active_task_ids) < max_concurrent

    def record_activity(self) -> None:
        """Update last activity timestamp."""
        self.last_activity = now_utc()

    def record_error(self, error: str) -> None:
        """
        Record an error occurrence.

        Args:
            error: Error message
        """
        self.error_count += 1
        self.last_error = error
        self.last_error_at = now_utc()
        self.record_activity()

    def record_task_completed(self) -> None:
        """Record successful task completion."""
        self.tasks_completed += 1
        self.record_activity()

    def record_task_failed(self) -> None:
        """Record task failure."""
        self.tasks_failed += 1
        self.record_activity()

    def add_thinking_time(self, duration_ms: float) -> None:
        """Add to cumulative thinking time."""
        self.total_thinking_time_ms += duration_ms

    def add_acting_time(self, duration_ms: float) -> None:
        """Add to cumulative acting time."""
        self.total_acting_time_ms += duration_ms

    def start_task(self, task_id: str) -> None:
        """
        Mark a task as started.

        Args:
            task_id: ID of the task being started
        """
        self.current_task_id = task_id
        if task_id not in self.active_task_ids:
            self.active_task_ids.append(task_id)
        self.record_activity()

    def finish_task(self, task_id: str, success: bool = True) -> None:
        """
        Mark a task as finished.

        Args:
            task_id: ID of the finished task
            success: Whether task completed successfully
        """
        if self.current_task_id == task_id:
            self.current_task_id = None
        if task_id in self.active_task_ids:
            self.active_task_ids.remove(task_id)

        if success:
            self.record_task_completed()
        else:
            self.record_task_failed()

    def add_message(self, message: BaseMessage) -> None:
        """
        Add a message to history.

        Args:
            message: Message to add
        """
        self.message_history.append(message)
        self.record_activity()

    def get_recent_history(self, count: int = 10) -> list[BaseMessage]:
        """
        Get recent message history.

        Args:
            count: Number of recent messages to return

        Returns:
            List of recent messages
        """
        return self.message_history[-count:]

    def clear_history(self) -> None:
        """Clear message history."""
        self.message_history.clear()

    def reset(self) -> None:
        """Reset state to initial values (except metrics)."""
        self.status = AgentStatus.IDLE
        self.current_task_id = None
        self.active_task_ids.clear()
        self.message_history.clear()
        self.last_error = None
        self.last_error_at = None

    @property
    def success_rate(self) -> float:
        """Calculate task success rate."""
        total = self.tasks_completed + self.tasks_failed
        if total == 0:
            return 1.0
        return self.tasks_completed / total

    @property
    def active_task_count(self) -> int:
        """Get number of active tasks."""
        return len(self.active_task_ids)

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "status": self.status.value,
            "current_task_id": self.current_task_id,
            "active_task_ids": self.active_task_ids,
            "message_history_length": len(self.message_history),
            "last_activity": self.last_activity.isoformat(),
            "error_count": self.error_count,
            "last_error": self.last_error,
            "tasks_completed": self.tasks_completed,
            "tasks_failed": self.tasks_failed,
            "success_rate": self.success_rate,
            "total_thinking_time_ms": self.total_thinking_time_ms,
            "total_acting_time_ms": self.total_acting_time_ms,
        }
