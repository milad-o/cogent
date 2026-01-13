"""Metrics collection for system monitoring.

Provides counters, gauges, histograms, and timers
for tracking system performance and behavior.
"""

from __future__ import annotations

import time
import statistics
from dataclasses import dataclass, field
from typing import Any
from contextlib import contextmanager
from collections import defaultdict

from agenticflow.core import now_utc


@dataclass
class MetricValue:
    """A single metric data point."""

    value: float
    timestamp: float = field(default_factory=time.time)
    labels: dict[str, str] = field(default_factory=dict)


class Counter:
    """Monotonically increasing counter.

    Use for counting events like task completions,
    errors, or tool invocations.

    Example:
        >>> counter = Counter("tasks_completed")
        >>> counter.inc()  # increment by 1
        >>> counter.inc(5)  # increment by 5
        >>> print(counter.value)  # 6
    """

    def __init__(self, name: str, description: str = "") -> None:
        """Initialize counter.

        Args:
            name: Metric name.
            description: Human-readable description.
        """
        self.name = name
        self.description = description
        self._value: float = 0
        self._labels_values: dict[tuple[tuple[str, str], ...], float] = defaultdict(float)

    def inc(self, amount: float = 1, labels: dict[str, str] | None = None) -> None:
        """Increment counter.

        Args:
            amount: Amount to increment by.
            labels: Optional labels for the metric.
        """
        if amount < 0:
            raise ValueError("Counter can only be incremented")

        if labels:
            key = tuple(sorted(labels.items()))
            self._labels_values[key] += amount
        else:
            self._value += amount

    @property
    def value(self) -> float:
        """Get current counter value (without labels)."""
        return self._value

    def get(self, labels: dict[str, str] | None = None) -> float:
        """Get counter value for specific labels.

        Args:
            labels: Labels to filter by.

        Returns:
            Counter value for labels.
        """
        if labels:
            key = tuple(sorted(labels.items()))
            return self._labels_values.get(key, 0)
        return self._value

    def reset(self) -> None:
        """Reset counter to zero."""
        self._value = 0
        self._labels_values.clear()


class Gauge:
    """Value that can go up or down.

    Use for values like current queue size, active agents,
    or memory usage.

    Example:
        >>> gauge = Gauge("active_agents")
        >>> gauge.set(5)
        >>> gauge.inc()  # 6
        >>> gauge.dec(2)  # 4
    """

    def __init__(self, name: str, description: str = "") -> None:
        """Initialize gauge.

        Args:
            name: Metric name.
            description: Human-readable description.
        """
        self.name = name
        self.description = description
        self._value: float = 0
        self._labels_values: dict[tuple[tuple[str, str], ...], float] = defaultdict(float)

    def set(self, value: float, labels: dict[str, str] | None = None) -> None:
        """Set gauge value.

        Args:
            value: New value.
            labels: Optional labels.
        """
        if labels:
            key = tuple(sorted(labels.items()))
            self._labels_values[key] = value
        else:
            self._value = value

    def inc(self, amount: float = 1, labels: dict[str, str] | None = None) -> None:
        """Increment gauge.

        Args:
            amount: Amount to increment.
            labels: Optional labels.
        """
        if labels:
            key = tuple(sorted(labels.items()))
            self._labels_values[key] += amount
        else:
            self._value += amount

    def dec(self, amount: float = 1, labels: dict[str, str] | None = None) -> None:
        """Decrement gauge.

        Args:
            amount: Amount to decrement.
            labels: Optional labels.
        """
        if labels:
            key = tuple(sorted(labels.items()))
            self._labels_values[key] -= amount
        else:
            self._value -= amount

    @property
    def value(self) -> float:
        """Get current gauge value."""
        return self._value

    def get(self, labels: dict[str, str] | None = None) -> float:
        """Get gauge value for specific labels.

        Args:
            labels: Labels to filter by.

        Returns:
            Gauge value.
        """
        if labels:
            key = tuple(sorted(labels.items()))
            return self._labels_values.get(key, 0)
        return self._value


class Histogram:
    """Distribution of values.

    Use for tracking distributions like response times,
    task durations, or message sizes.

    Example:
        >>> hist = Histogram("task_duration_ms", buckets=[10, 50, 100, 500])
        >>> hist.observe(45)
        >>> hist.observe(120)
        >>> print(hist.mean)
    """

    def __init__(
        self,
        name: str,
        description: str = "",
        buckets: list[float] | None = None,
    ) -> None:
        """Initialize histogram.

        Args:
            name: Metric name.
            description: Human-readable description.
            buckets: Bucket boundaries for distribution.
        """
        self.name = name
        self.description = description
        self.buckets = sorted(buckets or [10, 25, 50, 100, 250, 500, 1000])
        self._values: list[float] = []
        self._bucket_counts: dict[float, int] = {b: 0 for b in self.buckets}
        self._bucket_counts[float("inf")] = 0

    def observe(self, value: float) -> None:
        """Record an observation.

        Args:
            value: Observed value.
        """
        self._values.append(value)

        # Update bucket counts
        for bucket in self.buckets:
            if value <= bucket:
                self._bucket_counts[bucket] += 1
                return
        self._bucket_counts[float("inf")] += 1

    @property
    def count(self) -> int:
        """Get total number of observations."""
        return len(self._values)

    @property
    def sum(self) -> float:
        """Get sum of all observations."""
        return sum(self._values)

    @property
    def mean(self) -> float | None:
        """Get mean of observations."""
        if not self._values:
            return None
        return statistics.mean(self._values)

    @property
    def median(self) -> float | None:
        """Get median of observations."""
        if not self._values:
            return None
        return statistics.median(self._values)

    @property
    def stddev(self) -> float | None:
        """Get standard deviation."""
        if len(self._values) < 2:
            return None
        return statistics.stdev(self._values)

    def percentile(self, p: float) -> float | None:
        """Get percentile value.

        Args:
            p: Percentile (0-100).

        Returns:
            Value at percentile.
        """
        if not self._values:
            return None
        sorted_values = sorted(self._values)
        k = (len(sorted_values) - 1) * (p / 100)
        f = int(k)
        c = f + 1 if f + 1 < len(sorted_values) else f
        return sorted_values[f] + (k - f) * (sorted_values[c] - sorted_values[f])

    def get_buckets(self) -> dict[float, int]:
        """Get bucket counts."""
        return self._bucket_counts.copy()

    def reset(self) -> None:
        """Reset histogram."""
        self._values.clear()
        self._bucket_counts = {b: 0 for b in self.buckets}
        self._bucket_counts[float("inf")] = 0


class Timer:
    """Convenience wrapper for timing operations.

    Example:
        >>> timer = Timer("operation_duration")
        >>> with timer.time():
        ...     do_operation()
        >>> print(timer.histogram.mean)
    """

    def __init__(
        self,
        name: str,
        description: str = "",
        buckets: list[float] | None = None,
    ) -> None:
        """Initialize timer.

        Args:
            name: Metric name.
            description: Human-readable description.
            buckets: Histogram buckets in milliseconds.
        """
        self.name = name
        self.histogram = Histogram(
            f"{name}_ms",
            description,
            buckets or [1, 5, 10, 25, 50, 100, 250, 500, 1000, 5000],
        )

    @contextmanager
    def time(self):
        """Time a block of code.

        Yields:
            None - use as context manager.
        """
        start = time.perf_counter()
        try:
            yield
        finally:
            duration_ms = (time.perf_counter() - start) * 1000
            self.histogram.observe(duration_ms)

    def record(self, duration_ms: float) -> None:
        """Record a duration directly.

        Args:
            duration_ms: Duration in milliseconds.
        """
        self.histogram.observe(duration_ms)


class MetricsCollector:
    """Central metrics registry and collector.

    Example:
        >>> metrics = MetricsCollector()
        >>> task_counter = metrics.counter("tasks_total", "Total tasks")
        >>> task_counter.inc()
        >>> metrics.snapshot()
    """

    def __init__(self) -> None:
        """Initialize metrics collector."""
        self._counters: dict[str, Counter] = {}
        self._gauges: dict[str, Gauge] = {}
        self._histograms: dict[str, Histogram] = {}
        self._timers: dict[str, Timer] = {}

    def counter(self, name: str, description: str = "") -> Counter:
        """Get or create a counter.

        Args:
            name: Counter name.
            description: Description.

        Returns:
            Counter instance.
        """
        if name not in self._counters:
            self._counters[name] = Counter(name, description)
        return self._counters[name]

    def gauge(self, name: str, description: str = "") -> Gauge:
        """Get or create a gauge.

        Args:
            name: Gauge name.
            description: Description.

        Returns:
            Gauge instance.
        """
        if name not in self._gauges:
            self._gauges[name] = Gauge(name, description)
        return self._gauges[name]

    def histogram(
        self,
        name: str,
        description: str = "",
        buckets: list[float] | None = None,
    ) -> Histogram:
        """Get or create a histogram.

        Args:
            name: Histogram name.
            description: Description.
            buckets: Bucket boundaries.

        Returns:
            Histogram instance.
        """
        if name not in self._histograms:
            self._histograms[name] = Histogram(name, description, buckets)
        return self._histograms[name]

    def timer(
        self,
        name: str,
        description: str = "",
        buckets: list[float] | None = None,
    ) -> Timer:
        """Get or create a timer.

        Args:
            name: Timer name.
            description: Description.
            buckets: Histogram buckets.

        Returns:
            Timer instance.
        """
        if name not in self._timers:
            self._timers[name] = Timer(name, description, buckets)
        return self._timers[name]

    def snapshot(self) -> dict[str, Any]:
        """Get snapshot of all metrics.

        Returns:
            Dictionary with all metric values.
        """
        return {
            "counters": {
                name: c.value for name, c in self._counters.items()
            },
            "gauges": {
                name: g.value for name, g in self._gauges.items()
            },
            "histograms": {
                name: {
                    "count": h.count,
                    "sum": h.sum,
                    "mean": h.mean,
                    "median": h.median,
                    "p95": h.percentile(95),
                    "p99": h.percentile(99),
                }
                for name, h in self._histograms.items()
            },
            "timers": {
                name: {
                    "count": t.histogram.count,
                    "mean_ms": t.histogram.mean,
                    "p95_ms": t.histogram.percentile(95),
                    "p99_ms": t.histogram.percentile(99),
                }
                for name, t in self._timers.items()
            },
            "timestamp": now_utc().isoformat(),
        }

    def reset_all(self) -> None:
        """Reset all metrics."""
        for c in self._counters.values():
            c.reset()
        for h in self._histograms.values():
            h.reset()
        # Gauges typically aren't reset
