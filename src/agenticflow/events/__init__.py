"""Core eventing (non-observability).

This module is the foundation for orchestration features (e.g., reactive flows).
It is intentionally separate from `agenticflow.observability`, which is reserved
for tracing/telemetry and developer-facing instrumentation.
"""

from agenticflow.events.bus import EventBus
from agenticflow.events.event import Event

__all__ = ["Event", "EventBus"]
