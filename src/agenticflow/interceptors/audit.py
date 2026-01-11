"""
Audit and logging interceptors.

Interceptors for compliance logging, audit trails, and observability.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Callable

from agenticflow.interceptors.base import (
    Interceptor,
    InterceptContext,
    InterceptResult,
    Phase,
)


class AuditTraceType(Enum):
    """Types of audit events."""
    RUN_START = "run_start"
    RUN_END = "run_end"
    MODEL_REQUEST = "model_request"
    MODEL_RESPONSE = "model_response"
    TOOL_REQUEST = "tool_request"
    TOOL_RESPONSE = "tool_response"
    ERROR = "error"


@dataclass
class AuditEvent:
    """A single audit log entry.
    
    Attributes:
        timestamp: When the event occurred (UTC).
        event_type: Type of event.
        agent_name: Name of the agent.
        task: The original task/prompt.
        phase: Execution phase.
        data: Additional event data.
        duration_ms: Duration in milliseconds (for response events).
    """
    timestamp: str
    event_type: AuditTraceType
    agent_name: str
    task: str
    phase: Phase
    data: dict[str, Any] = field(default_factory=dict)
    duration_ms: float | None = None
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "timestamp": self.timestamp,
            "event_type": self.event_type.value,
            "agent_name": self.agent_name,
            "task": self.task[:100] + "..." if len(self.task) > 100 else self.task,
            "phase": self.phase.value,
            "data": self.data,
            "duration_ms": self.duration_ms,
        }
    
    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict())


@dataclass
class Auditor(Interceptor):
    """Audit logger for compliance and observability.
    
    Records all agent execution events to an audit log. Can log to:
    - In-memory list (default)
    - File (JSON lines format)
    - Custom callback
    
    Attributes:
        log_to_file: Path to log file (optional).
        callback: Custom callback for each event (optional).
        include_content: Whether to include message/result content.
        max_content_length: Max length of content to log.
        redact_patterns: Patterns to redact from logs.
        
    Example:
        ```python
        from agenticflow import Agent
        from agenticflow.interceptors import Auditor
        
        # Log to file
        auditor = Auditor(log_to_file="audit.jsonl")
        
        agent = Agent(
            name="assistant",
            model=model,
            intercept=[auditor],
        )
        
        # Run agent
        await agent.run("Help me")
        
        # Access logs
        for event in auditor.events:
            print(event.to_json())
        ```
    """
    
    log_to_file: str | Path | None = None
    callback: Callable[[AuditEvent], None] | None = None
    include_content: bool = True
    max_content_length: int = 500
    redact_patterns: list[str] = field(default_factory=list)
    
    def __post_init__(self) -> None:
        """Initialize event storage."""
        self._events: list[AuditEvent] = []
        self._run_start_time: float | None = None
        self._think_start_time: float | None = None
        self._act_start_time: float | None = None
        
        # Compile redaction patterns
        import re
        self._redact_compiled = [
            re.compile(p, re.IGNORECASE) for p in self.redact_patterns
        ]
    
    def _now_iso(self) -> str:
        """Get current UTC timestamp in ISO format."""
        return datetime.now(timezone.utc).isoformat()
    
    def _now_mono(self) -> float:
        """Get monotonic time for duration calculations."""
        import time
        return time.monotonic()
    
    def _redact(self, text: str) -> str:
        """Apply redaction patterns to text."""
        result = text
        for pattern in self._redact_compiled:
            result = pattern.sub("[REDACTED]", result)
        return result
    
    def _truncate(self, text: str) -> str:
        """Truncate text to max length."""
        if len(text) > self.max_content_length:
            return text[:self.max_content_length] + "..."
        return text
    
    def _log_event(self, event: AuditEvent) -> None:
        """Log an event to all configured destinations."""
        self._events.append(event)
        
        if self.callback:
            self.callback(event)
        
        if self.log_to_file:
            path = Path(self.log_to_file)
            with path.open("a", encoding="utf-8") as f:
                f.write(event.to_json() + "\n")
    
    async def pre_run(self, ctx: InterceptContext) -> InterceptResult:
        """Log run start."""
        self._run_start_time = self._now_mono()
        
        event = AuditEvent(
            timestamp=self._now_iso(),
            event_type=AuditTraceType.RUN_START,
            agent_name=ctx.agent.name,
            task=self._redact(ctx.task),
            phase=ctx.phase,
            data={
                "message_count": len(ctx.messages),
            },
        )
        self._log_event(event)
        return InterceptResult.ok()
    
    async def pre_think(self, ctx: InterceptContext) -> InterceptResult:
        """Log model request."""
        self._think_start_time = self._now_mono()
        
        data: dict[str, Any] = {
            "model_call_number": ctx.model_calls + 1,
            "message_count": len(ctx.messages),
        }
        
        if self.include_content and ctx.messages:
            last_msg = ctx.messages[-1]
            content = last_msg.get("content", "")
            if isinstance(content, str):
                data["last_message"] = self._truncate(self._redact(content))
        
        event = AuditEvent(
            timestamp=self._now_iso(),
            event_type=AuditTraceType.MODEL_REQUEST,
            agent_name=ctx.agent.name,
            task=self._redact(ctx.task),
            phase=ctx.phase,
            data=data,
        )
        self._log_event(event)
        return InterceptResult.ok()
    
    async def post_think(self, ctx: InterceptContext) -> InterceptResult:
        """Log model response."""
        duration_ms = None
        if self._think_start_time:
            duration_ms = (self._now_mono() - self._think_start_time) * 1000
        
        data: dict[str, Any] = {
            "model_call_number": ctx.model_calls,
        }
        
        if self.include_content and ctx.model_response:
            content = getattr(ctx.model_response, "content", "")
            if content:
                data["response_preview"] = self._truncate(self._redact(str(content)))
            
            # Log tool calls if any
            tool_calls = getattr(ctx.model_response, "tool_calls", None)
            if tool_calls:
                data["tool_calls"] = [
                    {
                        "name": tc.get("name", "unknown") if isinstance(tc, dict) else getattr(tc, "name", "unknown"),
                        "id": tc.get("id", "") if isinstance(tc, dict) else getattr(tc, "id", ""),
                    }
                    for tc in tool_calls
                ]
        
        event = AuditEvent(
            timestamp=self._now_iso(),
            event_type=AuditTraceType.MODEL_RESPONSE,
            agent_name=ctx.agent.name,
            task=self._redact(ctx.task),
            phase=ctx.phase,
            data=data,
            duration_ms=duration_ms,
        )
        self._log_event(event)
        return InterceptResult.ok()
    
    async def pre_act(self, ctx: InterceptContext) -> InterceptResult:
        """Log tool request."""
        self._act_start_time = self._now_mono()
        
        data: dict[str, Any] = {
            "tool_name": ctx.tool_name,
            "tool_call_number": ctx.tool_calls + 1,
        }
        
        if self.include_content and ctx.tool_args:
            args_str = json.dumps(ctx.tool_args)
            data["arguments"] = self._truncate(self._redact(args_str))
        
        event = AuditEvent(
            timestamp=self._now_iso(),
            event_type=AuditTraceType.TOOL_REQUEST,
            agent_name=ctx.agent.name,
            task=self._redact(ctx.task),
            phase=ctx.phase,
            data=data,
        )
        self._log_event(event)
        return InterceptResult.ok()
    
    async def post_act(self, ctx: InterceptContext) -> InterceptResult:
        """Log tool response."""
        duration_ms = None
        if self._act_start_time:
            duration_ms = (self._now_mono() - self._act_start_time) * 1000
        
        data: dict[str, Any] = {
            "tool_name": ctx.tool_name,
            "tool_call_number": ctx.tool_calls,
        }
        
        if self.include_content and ctx.tool_result is not None:
            result_str = str(ctx.tool_result)
            data["result_preview"] = self._truncate(self._redact(result_str))
        
        event = AuditEvent(
            timestamp=self._now_iso(),
            event_type=AuditTraceType.TOOL_RESPONSE,
            agent_name=ctx.agent.name,
            task=self._redact(ctx.task),
            phase=ctx.phase,
            data=data,
            duration_ms=duration_ms,
        )
        self._log_event(event)
        return InterceptResult.ok()
    
    async def post_run(self, ctx: InterceptContext) -> InterceptResult:
        """Log run end."""
        duration_ms = None
        if self._run_start_time:
            duration_ms = (self._now_mono() - self._run_start_time) * 1000
        
        event = AuditEvent(
            timestamp=self._now_iso(),
            event_type=AuditTraceType.RUN_END,
            agent_name=ctx.agent.name,
            task=self._redact(ctx.task),
            phase=ctx.phase,
            data={
                "total_model_calls": ctx.model_calls,
                "total_tool_calls": ctx.tool_calls,
            },
            duration_ms=duration_ms,
        )
        self._log_event(event)
        return InterceptResult.ok()
    
    async def on_error(self, ctx: InterceptContext) -> InterceptResult:
        """Log errors."""
        event = AuditEvent(
            timestamp=self._now_iso(),
            event_type=AuditTraceType.ERROR,
            agent_name=ctx.agent.name,
            task=self._redact(ctx.task),
            phase=ctx.phase,
            data={
                "error_type": type(ctx.error).__name__ if ctx.error else "Unknown",
                "error_message": str(ctx.error) if ctx.error else "",
            },
        )
        self._log_event(event)
        return InterceptResult.ok()
    
    @property
    def events(self) -> list[AuditEvent]:
        """Get all recorded events."""
        return self._events.copy()
    
    def clear(self) -> None:
        """Clear recorded events."""
        self._events.clear()
    
    def export_json(self, path: str | Path) -> None:
        """Export all events to a JSON file."""
        with Path(path).open("w", encoding="utf-8") as f:
            json.dump([e.to_dict() for e in self._events], f, indent=2)
    
    def summary(self) -> dict[str, Any]:
        """Get summary statistics of recorded events."""
        if not self._events:
            return {"total_events": 0}
        
        by_type = {}
        for event in self._events:
            key = event.event_type.value
            by_type[key] = by_type.get(key, 0) + 1
        
        durations = [e.duration_ms for e in self._events if e.duration_ms]
        
        return {
            "total_events": len(self._events),
            "by_type": by_type,
            "avg_duration_ms": sum(durations) / len(durations) if durations else None,
            "first_event": self._events[0].timestamp,
            "last_event": self._events[-1].timestamp,
        }


__all__ = [
    "AuditTraceType",
    "AuditEvent",
    "Auditor",
]
