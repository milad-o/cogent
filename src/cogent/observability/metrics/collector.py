"""
Metrics Collector - Simple metrics collection.

Provides basic counter, gauge, and histogram metrics
without external dependencies.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass


@dataclass
class Counter:
    """
    A monotonically increasing counter.

    Example:
        ```python
        requests = Counter("requests_total")
        requests.inc()
        requests.inc(5)
        print(requests.value)  # 6
        ```
    """

    name: str
    """Metric name."""

    description: str = ""
    """Human-readable description."""

    labels: dict[str, str] = field(default_factory=dict)
    """Labels for this counter."""

    _value: float = 0.0
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def inc(self, amount: float = 1.0) -> None:
        """Increment the counter."""
        with self._lock:
            self._value += amount

    @property
    def value(self) -> float:
        """Current counter value."""
        with self._lock:
            return self._value

    def reset(self) -> None:
        """Reset counter to 0."""
        with self._lock:
            self._value = 0.0


@dataclass
class Gauge:
    """
    A value that can go up or down.

    Example:
        ```python
        active = Gauge("active_requests")
        active.inc()
        active.dec()
        active.set(10)
        print(active.value)  # 10
        ```
    """

    name: str
    """Metric name."""

    description: str = ""
    """Human-readable description."""

    labels: dict[str, str] = field(default_factory=dict)
    """Labels for this gauge."""

    _value: float = 0.0
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def set(self, value: float) -> None:
        """Set the gauge value."""
        with self._lock:
            self._value = value

    def inc(self, amount: float = 1.0) -> None:
        """Increment the gauge."""
        with self._lock:
            self._value += amount

    def dec(self, amount: float = 1.0) -> None:
        """Decrement the gauge."""
        with self._lock:
            self._value -= amount

    @property
    def value(self) -> float:
        """Current gauge value."""
        with self._lock:
            return self._value


@dataclass
class Histogram:
    """
    Tracks distribution of values.

    Example:
        ```python
        latency = Histogram("request_latency_ms")
        latency.observe(45.2)
        latency.observe(123.5)
        print(latency.count)  # 2
        print(latency.sum)    # 168.7
        print(latency.avg)    # 84.35
        ```
    """

    name: str
    """Metric name."""

    description: str = ""
    """Human-readable description."""

    labels: dict[str, str] = field(default_factory=dict)
    """Labels for this histogram."""

    _count: int = 0
    _sum: float = 0.0
    _min: float = float("inf")
    _max: float = float("-inf")
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def observe(self, value: float) -> None:
        """Record a value."""
        with self._lock:
            self._count += 1
            self._sum += value
            if value < self._min:
                self._min = value
            if value > self._max:
                self._max = value

    @property
    def count(self) -> int:
        """Number of observations."""
        with self._lock:
            return self._count

    @property
    def sum(self) -> float:
        """Sum of all observed values."""
        with self._lock:
            return self._sum

    @property
    def min(self) -> float:
        """Minimum observed value."""
        with self._lock:
            return self._min if self._count > 0 else 0.0

    @property
    def max(self) -> float:
        """Maximum observed value."""
        with self._lock:
            return self._max if self._count > 0 else 0.0

    @property
    def avg(self) -> float:
        """Average observed value."""
        with self._lock:
            if self._count == 0:
                return 0.0
            return self._sum / self._count

    def reset(self) -> None:
        """Reset all values."""
        with self._lock:
            self._count = 0
            self._sum = 0.0
            self._min = float("inf")
            self._max = float("-inf")


class MetricsCollector:
    """
    Collects and manages metrics.

    Example:
        ```python
        metrics = MetricsCollector()

        requests = metrics.counter("requests_total", "Total requests")
        latency = metrics.histogram("latency_ms", "Request latency")

        requests.inc()
        latency.observe(45.2)

        # Get all metrics
        for name, metric in metrics.all():
            print(f"{name}: {metric.value if hasattr(metric, 'value') else metric.count}")
        ```
    """

    def __init__(self) -> None:
        """Initialize collector."""
        self._counters: dict[str, Counter] = {}
        self._gauges: dict[str, Gauge] = {}
        self._histograms: dict[str, Histogram] = {}
        self._lock = threading.Lock()

    def counter(
        self,
        name: str,
        description: str = "",
        labels: dict[str, str] | None = None,
    ) -> Counter:
        """
        Get or create a counter.

        Args:
            name: Metric name
            description: Human-readable description
            labels: Labels for the counter

        Returns:
            Counter instance
        """
        with self._lock:
            if name not in self._counters:
                self._counters[name] = Counter(
                    name=name,
                    description=description,
                    labels=labels or {},
                )
            return self._counters[name]

    def gauge(
        self,
        name: str,
        description: str = "",
        labels: dict[str, str] | None = None,
    ) -> Gauge:
        """
        Get or create a gauge.

        Args:
            name: Metric name
            description: Human-readable description
            labels: Labels for the gauge

        Returns:
            Gauge instance
        """
        with self._lock:
            if name not in self._gauges:
                self._gauges[name] = Gauge(
                    name=name,
                    description=description,
                    labels=labels or {},
                )
            return self._gauges[name]

    def histogram(
        self,
        name: str,
        description: str = "",
        labels: dict[str, str] | None = None,
    ) -> Histogram:
        """
        Get or create a histogram.

        Args:
            name: Metric name
            description: Human-readable description
            labels: Labels for the histogram

        Returns:
            Histogram instance
        """
        with self._lock:
            if name not in self._histograms:
                self._histograms[name] = Histogram(
                    name=name,
                    description=description,
                    labels=labels or {},
                )
            return self._histograms[name]

    def all(self) -> list[tuple[str, Counter | Gauge | Histogram]]:
        """
        Get all registered metrics.

        Returns:
            List of (name, metric) tuples
        """
        with self._lock:
            result: list[tuple[str, Counter | Gauge | Histogram]] = []
            result.extend(self._counters.items())
            result.extend(self._gauges.items())
            result.extend(self._histograms.items())
            return result

    def reset_all(self) -> None:
        """Reset all metrics."""
        with self._lock:
            for counter in self._counters.values():
                counter.reset()
            for gauge in self._gauges.values():
                gauge.set(0.0)
            for histogram in self._histograms.values():
                histogram.reset()

    def to_dict(self) -> dict[str, dict[str, float | int]]:
        """
        Export all metrics as a dictionary.

        Returns:
            Dictionary of metric values
        """
        with self._lock:
            result: dict[str, dict[str, float | int]] = {}

            for name, counter in self._counters.items():
                result[name] = {"value": counter.value}

            for name, gauge in self._gauges.items():
                result[name] = {"value": gauge.value}

            for name, histogram in self._histograms.items():
                result[name] = {
                    "count": histogram.count,
                    "sum": histogram.sum,
                    "min": histogram.min,
                    "max": histogram.max,
                    "avg": histogram.avg,
                }

            return result
