"""
Utility functions for AgenticFlow.
"""

from __future__ import annotations

import asyncio
import time
import uuid
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any


def generate_id(prefix: str | int = "", length: int = 8) -> str:
    """
    Generate a short unique identifier.

    Args:
        prefix: Optional prefix string, or length for backwards compatibility
        length: Number of characters (default 8)

    Returns:
        A unique ID string, optionally with prefix
    """
    # Handle backwards compatibility where first arg was length
    if isinstance(prefix, int):
        length = prefix
        prefix = ""

    uid = str(uuid.uuid4()).replace("-", "")[:length]
    if prefix:
        return f"{prefix}_{uid}"
    return uid


def now_utc() -> datetime:
    """
    Get current UTC timestamp (timezone-aware).

    Returns:
        Current datetime in UTC timezone
    """
    return datetime.now(UTC)


def now_local() -> datetime:
    """
    Get current local timestamp (timezone-aware).

    Returns:
        Current datetime in local timezone
    """
    return datetime.now(UTC).astimezone()


def to_local(dt: datetime) -> datetime:
    """
    Convert a datetime to local timezone.

    Args:
        dt: Datetime to convert (can be naive or timezone-aware)

    Returns:
        Datetime in local timezone
    """
    if dt.tzinfo is None:
        # Assume naive datetime is UTC
        dt = dt.replace(tzinfo=UTC)
    return dt.astimezone()


def format_timestamp(dt: datetime | None, local: bool = True) -> str | None:
    """
    Format a datetime for display.

    Args:
        dt: Datetime to format (can be None)
        local: If True, convert to local timezone first (default: True)

    Returns:
        Formatted string or None
    """
    if dt is None:
        return None
    if local:
        dt = to_local(dt)
    return dt.strftime("%H:%M:%S.%f")[:-3]


def format_duration_ms(start: datetime | None, end: datetime | None) -> float | None:
    """
    Calculate duration in milliseconds between two timestamps.

    Args:
        start: Start datetime
        end: End datetime

    Returns:
        Duration in milliseconds or None if either is None
    """
    if start is None or end is None:
        return None
    return (end - start).total_seconds() * 1000


def truncate_string(s: str, max_length: int = 200, suffix: str = "...") -> str:
    """
    Truncate a string to a maximum length.

    Args:
        s: String to truncate
        max_length: Maximum length (default 200)
        suffix: Suffix to add if truncated (default "...")

    Returns:
        Truncated string
    """
    if len(s) <= max_length:
        return s
    return s[: max_length - len(suffix)] + suffix


def safe_json_value(value: object) -> object:
    """
    Convert a value to a JSON-safe representation.

    Args:
        value: Any value

    Returns:
        JSON-serializable value
    """
    if value is None:
        return None
    if isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, (list, tuple)):
        return [safe_json_value(v) for v in value]
    if isinstance(value, dict):
        return {str(k): safe_json_value(v) for k, v in value.items()}
    if hasattr(value, "to_dict"):
        return value.to_dict()
    return str(value)


def model_identifier(model: object) -> str:
    """Return a safe, human-friendly identifier for a model.

    Observability should never log full model reprs because they may contain
    secrets (e.g., API keys). This helper prefers common name fields and falls
    back to the class name.
    """
    if model is None:
        return "unknown"

    for attr in ("model_name", "model", "name", "deployment_name"):
        value = getattr(model, attr, None)
        if isinstance(value, str) and value:
            return value

    # As a last resort, use the type name (safe) rather than str(model).
    return type(model).__name__


# ============================================================================
# Reactive/Event-Driven Utilities
# ============================================================================


class IdempotencyGuard:
    """In-memory idempotency guard.

    Use this to ensure you only perform a side-effect once per key.

    Notes:
    - This is process-local memory (not persistent) and is best suited for demos,
      single-process workers, or where the upstream guarantees at-least-once but
      duplicates are rare.
    - For multi-process / distributed systems, you typically back this with a DB
      or cache (Redis) keyed by event_id / business_id.
    """

    def __init__(self) -> None:
        self._seen: set[str] = set()
        self._lock = asyncio.Lock()

    async def claim(self, key: str) -> bool:
        """Atomically claim a key.

        Returns True if the key was not previously claimed.
        """
        async with self._lock:
            if key in self._seen:
                return False
            self._seen.add(key)
            return True

    async def run_once(self, key: str, fn: Callable[[], Awaitable[Any]]) -> tuple[bool, Any]:
        """Run a coroutine exactly once per key.

        Returns (ran, result). If ran is False, result is None.
        """
        if not await self.claim(key):
            return (False, None)
        return (True, await fn())


@dataclass(slots=True)
class RetryBudget:
    """Bounded retry tracker for a key.

    This doesn't perform retries itself; it tracks attempt counts so your
    reactive workflow can decide whether to reschedule/retry/escalate.
    """

    max_attempts: int
    attempts: dict[str, int]

    @classmethod
    def in_memory(cls, *, max_attempts: int) -> RetryBudget:
        return cls(max_attempts=max_attempts, attempts={})

    def next_attempt(self, key: str) -> int:
        """Increment and return the current attempt number (0-based)."""
        current = int(self.attempts.get(key, -1)) + 1
        self.attempts[key] = current
        return current

    def can_retry(self, key: str) -> bool:
        return int(self.attempts.get(key, -1)) + 1 < int(self.max_attempts)


async def emit_later(
    *,
    flow: Any,
    delay_seconds: float,
    event_name: str,
    data: dict[str, Any] | None = None,
) -> None:
    """Emit an event after a delay.

    This is a thin helper around asyncio.sleep + flow.emit.

    The caller typically schedules this using flow.spawn(emit_later(...)).
    """
    await asyncio.sleep(max(0.0, float(delay_seconds)))
    await flow.emit(event_name, data or {})


def jittered_delay(
    base_seconds: float,
    *,
    jitter_seconds: float = 0.0,
    min_seconds: float = 0.0,
    max_seconds: float | None = None,
) -> float:
    """Compute a jittered delay value.

    This avoids importing random here; callers can supply their own randomness
    and simply use this helper for clamping.
    """
    delay = float(base_seconds) + float(jitter_seconds)
    if max_seconds is not None:
        delay = min(delay, float(max_seconds))
    return max(float(min_seconds), delay)


class Stopwatch:
    """Tiny stopwatch helper (useful for timeouts/metrics in reactive demos)."""

    def __init__(self) -> None:
        self._t0 = time.perf_counter()

    @property
    def elapsed_s(self) -> float:
        return time.perf_counter() - self._t0

