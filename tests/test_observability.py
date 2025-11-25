"""Tests for the observability module."""

import pytest
import time
from io import StringIO

from agenticflow.observability import (
    Tracer,
    Span,
    SpanContext,
    SpanKind,
    MetricsCollector,
    Counter,
    Gauge,
    Histogram,
    Timer,
    ObservabilityLogger,
    LogLevel,
    LogEntry,
    Dashboard,
    DashboardConfig,
)


class TestTracer:
    """Tests for the Tracer class."""

    def test_create_span_context_manager(self):
        """Test creating span with context manager."""
        tracer = Tracer("test-service")

        with tracer.span("test-operation", SpanKind.TASK) as span:
            span.set_attribute("task_id", "123")
            assert not span.is_finished

        assert span.is_finished
        assert span.status == "ok"
        assert span.attributes["task_id"] == "123"

    def test_span_captures_errors(self):
        """Test span captures errors."""
        tracer = Tracer("test-service")

        with pytest.raises(ValueError):
            with tracer.span("failing-op") as span:
                raise ValueError("test error")

        assert span.status == "error"
        assert "test error" in span.error

    def test_span_hierarchy(self):
        """Test parent-child span relationships."""
        tracer = Tracer("test-service")

        with tracer.span("parent", SpanKind.TASK) as parent:
            with tracer.span("child", SpanKind.INTERNAL) as child:
                assert child.context.parent_span_id == parent.context.span_id
                assert child.context.trace_id == parent.context.trace_id

    def test_span_events(self):
        """Test adding events to spans."""
        tracer = Tracer("test-service")

        with tracer.span("operation") as span:
            span.add_event("checkpoint", {"step": 1})
            span.add_event("checkpoint", {"step": 2})

        assert len(span.events) == 2

    def test_span_duration(self):
        """Test span duration calculation."""
        tracer = Tracer("test-service")

        with tracer.span("timed-op") as span:
            time.sleep(0.01)  # 10ms

        assert span.duration_ms is not None
        assert span.duration_ms >= 10

    def test_get_trace(self):
        """Test retrieving all spans for a trace."""
        tracer = Tracer("test-service")

        with tracer.span("op1") as span1:
            with tracer.span("op2"):
                pass

        trace_spans = tracer.get_trace(span1.context.trace_id)
        assert len(trace_spans) == 2

    def test_export_spans(self):
        """Test exporting spans."""
        tracer = Tracer("test-service")

        with tracer.span("op1"):
            pass

        exported = tracer.export()
        assert len(exported) == 1
        assert exported[0]["name"] == "op1"


class TestMetrics:
    """Tests for metrics classes."""

    def test_counter_increment(self):
        """Test counter increment."""
        counter = Counter("requests_total")
        counter.inc()
        counter.inc(5)
        assert counter.value == 6

    def test_counter_with_labels(self):
        """Test counter with labels."""
        counter = Counter("requests_total")
        counter.inc(1, {"method": "GET"})
        counter.inc(2, {"method": "POST"})

        assert counter.get({"method": "GET"}) == 1
        assert counter.get({"method": "POST"}) == 2

    def test_counter_no_negative(self):
        """Test counter rejects negative increments."""
        counter = Counter("count")
        with pytest.raises(ValueError):
            counter.inc(-1)

    def test_gauge_set_inc_dec(self):
        """Test gauge operations."""
        gauge = Gauge("active_connections")
        gauge.set(10)
        assert gauge.value == 10

        gauge.inc(5)
        assert gauge.value == 15

        gauge.dec(3)
        assert gauge.value == 12

    def test_histogram_observe(self):
        """Test histogram observations."""
        hist = Histogram("request_duration", buckets=[10, 50, 100])
        hist.observe(25)
        hist.observe(75)
        hist.observe(150)

        assert hist.count == 3
        assert hist.mean == pytest.approx(83.33, rel=0.1)

    def test_histogram_percentiles(self):
        """Test histogram percentiles."""
        hist = Histogram("latency")
        for i in range(100):
            hist.observe(i)

        p50 = hist.percentile(50)
        p99 = hist.percentile(99)

        assert p50 is not None
        assert 45 <= p50 <= 55
        assert p99 is not None
        assert p99 >= 95

    def test_timer_context_manager(self):
        """Test timer as context manager."""
        timer = Timer("operation_time")

        with timer.time():
            time.sleep(0.01)

        assert timer.histogram.count == 1
        assert timer.histogram.mean >= 10


class TestMetricsCollector:
    """Tests for MetricsCollector."""

    def test_get_or_create_metrics(self):
        """Test getting or creating metrics."""
        collector = MetricsCollector()

        counter1 = collector.counter("requests")
        counter2 = collector.counter("requests")
        assert counter1 is counter2

    def test_snapshot(self):
        """Test metrics snapshot."""
        collector = MetricsCollector()

        collector.counter("requests").inc(100)
        collector.gauge("connections").set(50)
        collector.histogram("latency").observe(25)

        snapshot = collector.snapshot()

        assert snapshot["counters"]["requests"] == 100
        assert snapshot["gauges"]["connections"] == 50
        assert snapshot["histograms"]["latency"]["count"] == 1


class TestObservabilityLogger:
    """Tests for ObservabilityLogger."""

    def test_log_levels(self):
        """Test different log levels."""
        output = StringIO()
        logger = ObservabilityLogger("test", level=LogLevel.DEBUG, output=output)

        logger.debug("debug message")
        logger.info("info message")
        logger.warning("warning message")
        logger.error("error message")

        log_output = output.getvalue()
        assert "debug message" in log_output
        assert "info message" in log_output
        assert "warning message" in log_output
        assert "error message" in log_output

    def test_log_filtering(self):
        """Test log level filtering."""
        output = StringIO()
        logger = ObservabilityLogger("test", level=LogLevel.WARNING, output=output)

        logger.debug("debug")
        logger.info("info")
        logger.warning("warning")

        log_output = output.getvalue()
        assert "debug" not in log_output
        assert "info" not in log_output
        assert "warning" in log_output

    def test_structured_logging(self):
        """Test structured log context."""
        output = StringIO()
        logger = ObservabilityLogger("test", level=LogLevel.INFO, output=output)

        logger.info("task started", task_id="123", agent="worker")

        log_output = output.getvalue()
        assert "task started" in log_output
        assert "task_id=123" in log_output

    def test_json_format(self):
        """Test JSON log format."""
        output = StringIO()
        logger = ObservabilityLogger("test", level=LogLevel.INFO, output=output, format="json")

        logger.info("test message")

        import json
        log_output = output.getvalue().strip()
        parsed = json.loads(log_output)
        assert parsed["message"] == "test message"

    def test_child_logger(self):
        """Test creating child logger."""
        parent = ObservabilityLogger("parent")
        child = parent.child("child")

        assert child.name == "parent.child"


class TestDashboard:
    """Tests for Dashboard."""

    def test_dashboard_snapshot(self):
        """Test getting dashboard snapshot."""
        dashboard = Dashboard(DashboardConfig())

        dashboard.update_agent("agent-1", {"status": "active"})
        dashboard.update_task("task-1", {"status": "running"})
        dashboard.add_event({"type": "test"})

        snapshot = dashboard.snapshot()

        assert "agent-1" in snapshot.agents
        assert "task-1" in snapshot.tasks
        assert len(snapshot.events) == 1

    def test_dashboard_with_tracer(self):
        """Test dashboard with tracer integration."""
        dashboard = Dashboard()
        tracer = Tracer("test")

        dashboard.register_tracer(tracer)

        with tracer.span("test-op"):
            pass

        snapshot = dashboard.snapshot()
        assert len(snapshot.spans) == 1

    def test_dashboard_summary(self):
        """Test dashboard summary."""
        dashboard = Dashboard()

        dashboard.update_agent("a1", {"status": "active"})
        dashboard.update_agent("a2", {"status": "idle"})
        dashboard.update_task("t1", {"status": "running"})

        summary = dashboard.summary()

        assert summary["agents"]["total"] == 2
        assert summary["tasks"]["total"] == 1
