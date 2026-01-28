"""
JSON Formatter - Structured JSON output for events.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

from cogent.observability.formatters.base import BaseFormatter

if TYPE_CHECKING:
    from cogent.observability.core.config import FormatConfig
    from cogent.observability.core.event import Event


class JSONFormatter(BaseFormatter):
    """
    Formats events as JSON lines.

    Produces one JSON object per event, suitable for log aggregation
    and analysis tools.
    """

    patterns = ["*"]  # Handles all events

    def __init__(self, pretty: bool = False) -> None:
        """
        Initialize JSON formatter.

        Args:
            pretty: If True, output indented JSON (one event per multiple lines)
        """
        self.pretty = pretty

    def format(self, event: Event, config: FormatConfig) -> str | None:
        """Format event as JSON."""
        obj = {
            "timestamp": event.timestamp.isoformat(),
            "type": event.type,
            "source": event.source,
            "data": event.data,
        }

        if event.correlation_id:
            obj["correlation_id"] = event.correlation_id

        if self.pretty:
            return json.dumps(obj, indent=2, default=str)
        return json.dumps(obj, default=str)
