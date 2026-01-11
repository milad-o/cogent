"""System inspectors for deep observability.

Provides inspection capabilities for agents, tasks,
events, and the overall system.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, TYPE_CHECKING

from agenticflow.core import now_utc
from agenticflow.core.enums import TaskStatus, AgentStatus

if TYPE_CHECKING:
    from agenticflow.agent import Agent
    from agenticflow.tasks import TaskManager
    from agenticflow.events import TraceBus


@dataclass
class InspectionResult:
    """Result of an inspection.

    Attributes:
        component: What was inspected.
        timestamp: When inspection occurred.
        data: Inspection data.
        issues: Any issues found.
        recommendations: Suggestions for improvement.
    """

    component: str
    timestamp: datetime = field(default_factory=now_utc)
    data: dict[str, Any] = field(default_factory=dict)
    issues: list[str] = field(default_factory=list)
    recommendations: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "component": self.component,
            "timestamp": self.timestamp.isoformat(),
            "data": self.data,
            "issues": self.issues,
            "recommendations": self.recommendations,
        }


class AgentInspector:
    """Inspect agent state and behavior.

    Example:
        >>> inspector = AgentInspector()
        >>> result = inspector.inspect(agent)
        >>> print(result.issues)
    """

    def inspect(self, agent: "Agent") -> InspectionResult:
        """Inspect an agent.

        Args:
            agent: Agent to inspect.

        Returns:
            Inspection result with state and issues.
        """
        issues: list[str] = []
        recommendations: list[str] = []

        # Gather state
        state = agent.state
        config = agent.config

        data = {
            "name": config.name,
            "role": config.role.value if config.role else None,
            "status": state.status.value,
            "tasks_completed": state.tasks_completed,
            "tasks_failed": state.tasks_failed,
            "current_task": state.current_task,
            "memory_size": len(state.memory),
            "has_llm": agent.llm is not None,
            "tool_count": len(agent.tools) if hasattr(agent, "tools") else 0,
        }

        # Check for issues
        if state.status == AgentStatus.ERROR:
            issues.append(f"Agent is in ERROR state: {state.error}")

        if state.tasks_failed > state.tasks_completed:
            issues.append("More tasks failed than completed")
            recommendations.append("Review task complexity or agent capabilities")

        if not agent.llm:
            issues.append("No LLM configured - agent cannot think")
            recommendations.append("Configure an LLM for the agent")

        if len(state.memory) > 1000:
            issues.append("Large memory accumulation")
            recommendations.append("Consider clearing old memory entries")

        return InspectionResult(
            component=f"agent:{config.name}",
            data=data,
            issues=issues,
            recommendations=recommendations,
        )

    def inspect_all(self, agents: list["Agent"]) -> list[InspectionResult]:
        """Inspect multiple agents.

        Args:
            agents: List of agents.

        Returns:
            List of inspection results.
        """
        return [self.inspect(agent) for agent in agents]


class TaskInspector:
    """Inspect task state and execution.

    Example:
        >>> inspector = TaskInspector()
        >>> result = inspector.inspect_manager(task_manager)
    """

    def inspect_manager(self, manager: "TaskManager") -> InspectionResult:
        """Inspect a task manager.

        Args:
            manager: Task manager to inspect.

        Returns:
            Inspection result.
        """
        issues: list[str] = []
        recommendations: list[str] = []

        tasks = manager.tasks
        pending = [t for t in tasks.values() if t.status == TaskStatus.PENDING]
        running = [t for t in tasks.values() if t.status == TaskStatus.RUNNING]
        completed = [t for t in tasks.values() if t.status == TaskStatus.COMPLETED]
        failed = [t for t in tasks.values() if t.status == TaskStatus.FAILED]

        data = {
            "total_tasks": len(tasks),
            "pending": len(pending),
            "running": len(running),
            "completed": len(completed),
            "failed": len(failed),
            "completion_rate": (
                len(completed) / len(tasks) * 100 if tasks else 0
            ),
        }

        # Check for issues
        if len(pending) > 50:
            issues.append(f"Large task backlog: {len(pending)} pending")
            recommendations.append("Consider adding more agents or reducing task creation rate")

        if len(failed) > len(completed) and len(tasks) > 5:
            issues.append("High failure rate")
            recommendations.append("Review failed tasks for patterns")

        # Check for stuck tasks
        now = now_utc()
        for task in running:
            if task.started_at:
                runtime = (now - task.started_at).total_seconds()
                if runtime > 300:  # 5 minutes
                    issues.append(f"Task {task.id} running for {runtime:.0f}s")

        return InspectionResult(
            component="task_manager",
            data=data,
            issues=issues,
            recommendations=recommendations,
        )


class EventInspector:
    """Inspect event bus and event flow.

    Example:
        >>> inspector = EventInspector()
        >>> result = inspector.inspect(event_bus)
    """

    def inspect(self, event_bus: "TraceBus") -> InspectionResult:
        """Inspect an event bus.

        Args:
            event_bus: Event bus to inspect.

        Returns:
            Inspection result.
        """
        issues: list[str] = []
        recommendations: list[str] = []

        # Get event history
        history = event_bus.get_history()

        # Analyze event types
        event_types: dict[str, int] = {}
        for event in history:
            event_type = event.event_type
            event_types[event_type] = event_types.get(event_type, 0) + 1

        # Get handler counts
        handler_counts = {}
        for event_type, handlers in event_bus._handlers.items():
            handler_counts[event_type] = len(handlers)

        data = {
            "total_events": len(history),
            "event_types": event_types,
            "handler_counts": handler_counts,
            "total_handlers": sum(handler_counts.values()),
        }

        # Check for issues
        if len(history) > 9000:
            issues.append("Event history approaching limit")
            recommendations.append("Consider clearing old events")

        # Check for unhandled event types
        for event_type in event_types:
            if event_type not in handler_counts or handler_counts[event_type] == 0:
                issues.append(f"Event type '{event_type}' has no handlers")

        return InspectionResult(
            component="event_bus",
            data=data,
            issues=issues,
            recommendations=recommendations,
        )


class SystemInspector:
    """Full system inspection.

    Combines all inspectors for comprehensive system health check.

    Example:
        >>> inspector = SystemInspector()
        >>> inspector.register_agents(agents)
        >>> inspector.register_task_manager(task_manager)
        >>> inspector.register_event_bus(event_bus)
        >>> report = inspector.full_inspection()
    """

    def __init__(self) -> None:
        """Initialize system inspector."""
        self._agents: list["Agent"] = []
        self._task_manager: "TaskManager | None" = None
        self._event_bus: "TraceBus | None" = None

        self.agent_inspector = AgentInspector()
        self.task_inspector = TaskInspector()
        self.event_inspector = EventInspector()

    def register_agents(self, agents: list["Agent"]) -> None:
        """Register agents for inspection.

        Args:
            agents: List of agents.
        """
        self._agents = agents

    def register_task_manager(self, manager: "TaskManager") -> None:
        """Register task manager for inspection.

        Args:
            manager: Task manager.
        """
        self._task_manager = manager

    def register_event_bus(self, event_bus: "TraceBus") -> None:
        """Register event bus for inspection.

        Args:
            event_bus: Event bus.
        """
        self._event_bus = event_bus

    def full_inspection(self) -> dict[str, Any]:
        """Perform full system inspection.

        Returns:
            Complete inspection report.
        """
        results: list[InspectionResult] = []

        # Inspect agents
        for agent in self._agents:
            results.append(self.agent_inspector.inspect(agent))

        # Inspect task manager
        if self._task_manager:
            results.append(self.task_inspector.inspect_manager(self._task_manager))

        # Inspect event bus
        if self._event_bus:
            results.append(self.event_inspector.inspect(self._event_bus))

        # Aggregate issues and recommendations
        all_issues: list[str] = []
        all_recommendations: list[str] = []
        for result in results:
            for issue in result.issues:
                all_issues.append(f"[{result.component}] {issue}")
            for rec in result.recommendations:
                all_recommendations.append(f"[{result.component}] {rec}")

        # Calculate health score
        total_checks = len(results) * 3  # Rough estimate
        issue_penalty = len(all_issues) * 10
        health_score = max(0, min(100, 100 - issue_penalty))

        return {
            "timestamp": now_utc().isoformat(),
            "health_score": health_score,
            "health_status": self._health_status(health_score),
            "components_inspected": len(results),
            "total_issues": len(all_issues),
            "total_recommendations": len(all_recommendations),
            "issues": all_issues,
            "recommendations": all_recommendations,
            "component_results": [r.to_dict() for r in results],
        }

    @staticmethod
    def _health_status(score: int) -> str:
        """Convert health score to status string."""
        if score >= 90:
            return "healthy"
        if score >= 70:
            return "degraded"
        if score >= 50:
            return "unhealthy"
        return "critical"

    def quick_check(self) -> dict[str, Any]:
        """Quick health check without full inspection.

        Returns:
            Quick health summary.
        """
        issues_count = 0
        components_ok = 0
        total_components = 0

        # Quick agent check
        for agent in self._agents:
            total_components += 1
            if agent.state.status != AgentStatus.ERROR:
                components_ok += 1
            else:
                issues_count += 1

        # Quick task check
        if self._task_manager:
            total_components += 1
            failed = sum(
                1 for t in self._task_manager.tasks.values()
                if t.status == TaskStatus.FAILED
            )
            if failed == 0:
                components_ok += 1
            else:
                issues_count += failed

        # Quick event bus check
        if self._event_bus:
            total_components += 1
            components_ok += 1  # Event bus is usually ok

        return {
            "status": "ok" if issues_count == 0 else "issues_detected",
            "components_ok": components_ok,
            "total_components": total_components,
            "issues_count": issues_count,
            "timestamp": now_utc().isoformat(),
        }
