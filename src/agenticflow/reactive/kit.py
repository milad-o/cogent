"""Reusable helper utilities for reactive/event-driven apps.

This module intentionally avoids domain-specific concepts. It provides small,
composable building blocks that show up in most real reactive systems:

- idempotency / de-duplication (per event or per business key)
- scheduling (emit an event after a delay)
- bounded retries (track how many attempts were made)

These utilities are safe to use from both examples and production code.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from typing import Any, Awaitable, Callable


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
    def in_memory(cls, *, max_attempts: int) -> "RetryBudget":
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
