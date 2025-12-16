"""Threading helpers for reactive flows.

Reactive systems often need *per-entity* conversational memory (e.g. per job_id,
per ticket_id, per user_id). The flow can derive a `thread_id` from each event
and pass it into `agent.react(..., thread_id=...)`.

These helpers keep that wiring explicit, but avoid inline lambdas everywhere.
"""

from __future__ import annotations

from typing import Any, Callable


def thread_id_from_data(
    key: str,
    *,
    prefix: str | None = None,
) -> Callable[[Any, dict[str, Any]], str | None]:
    """Build a thread_id resolver from `event.data[key]`.

    Args:
        key: Key to look up in `event.data`.
        prefix: Optional prefix to namespace the thread id.

    Returns:
        A callable `(event, context) -> thread_id | None`.
    """

    def _resolver(event: Any, _context: dict[str, Any]) -> str | None:
        data = getattr(event, "data", None) or {}
        value = data.get(key)
        if value is None:
            return None
        thread_id = str(value)
        return f"{prefix}{thread_id}" if prefix else thread_id

    return _resolver
