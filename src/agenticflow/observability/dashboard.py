"""Dashboard for system visualization.

Provides a unified view of system state, metrics,
and activity for monitoring and debugging.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any
from collections.abc import Callable

from agenticflow.core import now_utc
from agenticflow.observability.tracer import Tracer, Span
from agenticflow.observability.metrics import MetricsCollector
from agenticflow.observability.logger import ObservabilityLogger, LogLevel


@dataclass
class DashboardConfig:
    """Dashboard configuration.

    Attributes:
        name: Dashboard name.
        refresh_interval_ms: Auto-refresh interval.
        history_window_minutes: How much history to show.
        enable_live_updates: Enable real-time updates.
    """

    name: str = "AgenticFlow Dashboard"
    refresh_interval_ms: int = 1000
    history_window_minutes: int = 60
    enable_live_updates: bool = True
    max_spans_display: int = 100
    max_logs_display: int = 500


@dataclass
class DashboardSnapshot:
    """Point-in-time dashboard snapshot.

    Contains all the data needed to render the dashboard.
    """

    timestamp: datetime
    agents: dict[str, dict[str, Any]]
    tasks: dict[str, dict[str, Any]]
    events: list[dict[str, Any]]
    spans: list[dict[str, Any]]
    logs: list[dict[str, Any]]
    metrics: dict[str, Any]
    topology_state: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "timestamp": self.timestamp.isoformat(),
            "agents": self.agents,
            "tasks": self.tasks,
            "events": self.events,
            "spans": self.spans,
            "logs": self.logs,
            "metrics": self.metrics,
            "topology_state": self.topology_state,
        }


class Dashboard:
    """System observability dashboard.

    Aggregates data from tracers, metrics, and loggers
    to provide a unified view of system activity.

    Example:
        >>> dashboard = Dashboard(DashboardConfig())
        >>> dashboard.register_tracer(tracer)
        >>> dashboard.register_metrics(metrics)
        >>> snapshot = dashboard.snapshot()
        >>> print(snapshot.metrics)
    """

    def __init__(self, config: DashboardConfig | None = None) -> None:
        """Initialize dashboard.

        Args:
            config: Dashboard configuration.
        """
        self.config = config or DashboardConfig()
        self._tracers: list[Tracer] = []
        self._metrics_collectors: list[MetricsCollector] = []
        self._loggers: list[ObservabilityLogger] = []
        self._agents: dict[str, dict[str, Any]] = {}
        self._tasks: dict[str, dict[str, Any]] = {}
        self._events: list[dict[str, Any]] = []
        self._topology_state: dict[str, Any] | None = None
        self._update_callbacks: list[Callable[[DashboardSnapshot], None]] = []

    def register_tracer(self, tracer: Tracer) -> None:
        """Register a tracer for span collection.

        Args:
            tracer: Tracer to register.
        """
        self._tracers.append(tracer)

    def register_metrics(self, metrics: MetricsCollector) -> None:
        """Register a metrics collector.

        Args:
            metrics: Metrics collector to register.
        """
        self._metrics_collectors.append(metrics)

    def register_logger(self, logger: ObservabilityLogger) -> None:
        """Register a logger for log collection.

        Args:
            logger: Logger to register.
        """
        self._loggers.append(logger)

    def update_agent(self, agent_id: str, data: dict[str, Any]) -> None:
        """Update agent state in dashboard.

        Args:
            agent_id: Agent identifier.
            data: Agent state data.
        """
        self._agents[agent_id] = {
            **data,
            "last_updated": now_utc().isoformat(),
        }

    def update_task(self, task_id: str, data: dict[str, Any]) -> None:
        """Update task state in dashboard.

        Args:
            task_id: Task identifier.
            data: Task state data.
        """
        self._tasks[task_id] = {
            **data,
            "last_updated": now_utc().isoformat(),
        }

    def add_event(self, event: dict[str, Any]) -> None:
        """Add event to dashboard.

        Args:
            event: Event data.
        """
        self._events.append({
            **event,
            "dashboard_received": now_utc().isoformat(),
        })

        # Trim old events
        max_events = 1000
        if len(self._events) > max_events:
            self._events = self._events[-max_events:]

    def update_topology(self, state: dict[str, Any]) -> None:
        """Update topology visualization state.

        Args:
            state: Topology state data.
        """
        self._topology_state = state

    def on_update(self, callback: Callable[[DashboardSnapshot], None]) -> None:
        """Register callback for dashboard updates.

        Args:
            callback: Function to call on updates.
        """
        self._update_callbacks.append(callback)

    def snapshot(self) -> DashboardSnapshot:
        """Get current dashboard snapshot.

        Returns:
            Current state of all observability data.
        """
        now = now_utc()
        history_start = now - timedelta(minutes=self.config.history_window_minutes)

        # Collect spans from all tracers
        all_spans: list[dict[str, Any]] = []
        for tracer in self._tracers:
            for span in tracer.finished_spans[-self.config.max_spans_display:]:
                all_spans.append(span.to_dict())

        # Collect logs from all loggers
        all_logs: list[dict[str, Any]] = []
        for logger in self._loggers:
            for entry in logger.get_entries(since=history_start)[-self.config.max_logs_display:]:
                all_logs.append(entry.to_dict())

        # Aggregate metrics
        aggregated_metrics: dict[str, Any] = {}
        for collector in self._metrics_collectors:
            snapshot = collector.snapshot()
            for category, values in snapshot.items():
                if category == "timestamp":
                    continue
                if category not in aggregated_metrics:
                    aggregated_metrics[category] = {}
                aggregated_metrics[category].update(values)

        # Build snapshot
        return DashboardSnapshot(
            timestamp=now,
            agents=self._agents.copy(),
            tasks=self._tasks.copy(),
            events=self._events[-100:],  # Last 100 events
            spans=all_spans,
            logs=all_logs,
            metrics=aggregated_metrics,
            topology_state=self._topology_state,
        )

    def notify_update(self) -> None:
        """Trigger update callbacks with current snapshot."""
        snapshot = self.snapshot()
        for callback in self._update_callbacks:
            try:
                callback(snapshot)
            except Exception:
                pass  # Don't let callback errors break the dashboard

    def get_trace_view(self, trace_id: str) -> dict[str, Any]:
        """Get detailed view of a specific trace.

        Args:
            trace_id: Trace ID to view.

        Returns:
            Trace details including all spans.
        """
        spans: list[dict[str, Any]] = []
        for tracer in self._tracers:
            for span in tracer.get_trace(trace_id):
                spans.append(span.to_dict())

        # Sort by start time
        spans.sort(key=lambda s: s.get("start_time", 0))

        # Calculate trace metrics
        if spans:
            start = min(s.get("start_time", float("inf")) for s in spans)
            end = max(s.get("end_time", 0) or 0 for s in spans)
            duration_ms = (end - start) * 1000 if start and end else None
        else:
            duration_ms = None

        return {
            "trace_id": trace_id,
            "span_count": len(spans),
            "duration_ms": duration_ms,
            "spans": spans,
        }

    def get_agent_timeline(self, agent_id: str) -> list[dict[str, Any]]:
        """Get activity timeline for an agent.

        Args:
            agent_id: Agent identifier.

        Returns:
            List of agent activities.
        """
        activities: list[dict[str, Any]] = []

        # Find spans for this agent
        for tracer in self._tracers:
            for span in tracer.finished_spans:
                if span.attributes.get("agent_id") == agent_id:
                    activities.append({
                        "type": "span",
                        "time": span.start_time,
                        "data": span.to_dict(),
                    })

        # Find logs for this agent
        for logger in self._loggers:
            for entry in logger.entries:
                if entry.context.get("agent_id") == agent_id:
                    activities.append({
                        "type": "log",
                        "time": entry.timestamp.timestamp(),
                        "data": entry.to_dict(),
                    })

        # Sort by time
        activities.sort(key=lambda a: a["time"])
        return activities

    def summary(self) -> dict[str, Any]:
        """Get high-level summary of system state.

        Returns:
            Summary statistics.
        """
        snapshot = self.snapshot()

        return {
            "agents": {
                "total": len(snapshot.agents),
                "active": sum(
                    1 for a in snapshot.agents.values()
                    if a.get("status") == "active"
                ),
            },
            "tasks": {
                "total": len(snapshot.tasks),
                "by_status": self._count_by_key(snapshot.tasks, "status"),
            },
            "spans": {
                "total": len(snapshot.spans),
                "by_kind": self._count_by_key(
                    {str(i): s for i, s in enumerate(snapshot.spans)},
                    "kind",
                ),
            },
            "logs": {
                "total": len(snapshot.logs),
                "by_level": self._count_by_key(
                    {str(i): l for i, l in enumerate(snapshot.logs)},
                    "level",
                ),
            },
            "events_count": len(snapshot.events),
            "timestamp": snapshot.timestamp.isoformat(),
        }

    @staticmethod
    def _count_by_key(items: dict[str, dict[str, Any]], key: str) -> dict[str, int]:
        """Count items by a specific key value."""
        counts: dict[str, int] = {}
        for item in items.values():
            value = str(item.get(key, "unknown"))
            counts[value] = counts.get(value, 0) + 1
        return counts
