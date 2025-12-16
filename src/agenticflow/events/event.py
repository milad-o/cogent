"""Core event types for orchestration.

These events drive application logic (reactive flows, external triggers, etc.).
They are *not* observability events.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from agenticflow.core.utils import generate_id, now_utc


@dataclass(frozen=True, slots=True, kw_only=True)
class Event:
    """A core event used for orchestration."""

    name: str
    """Event name/type (e.g., 'webhook.video_complete')."""

    data: dict[str, Any] = field(default_factory=dict)
    """Event payload."""

    id: str = field(default_factory=generate_id)
    """Unique event id."""

    timestamp: datetime = field(default_factory=now_utc)
    """Event timestamp."""

    source: str | None = None
    """Optional source identifier."""

    correlation_id: str | None = None
    """Optional correlation id for tracing across components."""
