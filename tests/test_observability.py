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


class TestObserver:
    """Tests for the Observer class."""

    def test_factory_methods(self):
        """Test observer factory methods."""
        from agenticflow.observability import Observer, ObservabilityLevel, Channel
        
        obs = Observer.off()
        assert obs.config.level == ObservabilityLevel.OFF
        
        obs = Observer.minimal()
        assert obs.config.level == ObservabilityLevel.RESULT
        
        obs = Observer.normal()
        assert obs.config.level == ObservabilityLevel.PROGRESS
        
        obs = Observer.detailed()
        assert obs.config.level == ObservabilityLevel.DETAILED
        
        obs = Observer.debug()
        assert obs.config.level == ObservabilityLevel.DEBUG
        assert Channel.ALL in obs.config.channels
        
        obs = Observer.trace()
        assert obs.config.level == ObservabilityLevel.TRACE
        
        obs = Observer.agents_only()
        assert Channel.AGENTS in obs.config.channels
        
        obs = Observer.tools_only()
        assert Channel.TOOLS in obs.config.channels

    def test_custom_channels(self):
        """Test observer with custom channels."""
        from agenticflow.observability import Observer, ObservabilityLevel, Channel
        
        obs = Observer(
            level=ObservabilityLevel.DETAILED,
            channels={Channel.AGENTS, Channel.TOOLS},
        )
        
        assert Channel.AGENTS in obs.config.channels
        assert Channel.TOOLS in obs.config.channels
        assert Channel.MESSAGES not in obs.config.channels
        assert Channel.ALL not in obs.config.channels

    @pytest.mark.asyncio
    async def test_attach_to_event_bus(self):
        """Test attaching observer to event bus."""
        from agenticflow.observability import Observer, ObservabilityLevel
        from agenticflow.events.bus import EventBus
        from agenticflow.schemas.event import Event
        from agenticflow.core.enums import EventType
        
        bus = EventBus()
        obs = Observer(level=ObservabilityLevel.DEBUG)
        
        obs.attach(bus)
        
        # Publish an event
        event = Event(type=EventType.AGENT_INVOKED, data={"agent_name": "Test"})
        await bus.publish(event)
        
        # Check metrics
        metrics = obs.metrics()
        assert metrics.get("total_events", 0) >= 1

    @pytest.mark.asyncio
    async def test_callbacks(self):
        """Test observer callbacks."""
        from agenticflow.observability import Observer, ObservabilityLevel, Channel
        from agenticflow.events.bus import EventBus
        from agenticflow.schemas.event import Event
        from agenticflow.core.enums import EventType
        
        agent_calls = []
        tool_calls = []
        
        obs = Observer(
            level=ObservabilityLevel.DEBUG,
            channels={Channel.AGENTS, Channel.TOOLS},
            on_agent=lambda name, action, data: agent_calls.append(f"{name}:{action}"),
            on_tool=lambda name, action, data: tool_calls.append(f"{name}:{action}"),
        )
        
        bus = EventBus()
        obs.attach(bus)
        
        # Publish events
        await bus.publish(Event(type=EventType.AGENT_INVOKED, data={"agent_name": "Test"}))
        await bus.publish(Event(type=EventType.TOOL_CALLED, data={"tool": "search"}))
        
        assert len(agent_calls) == 1
        assert "Test:invoked" in agent_calls[0]
        assert len(tool_calls) == 1
        assert "search:called" in tool_calls[0]

    @pytest.mark.asyncio
    async def test_events_query(self):
        """Test querying observed events."""
        from agenticflow.observability import Observer, ObservabilityLevel, Channel
        from agenticflow.events.bus import EventBus
        from agenticflow.schemas.event import Event
        from agenticflow.core.enums import EventType
        
        obs = Observer(level=ObservabilityLevel.DEBUG)
        bus = EventBus()
        obs.attach(bus)
        
        # Publish multiple events
        await bus.publish(Event(type=EventType.AGENT_INVOKED, data={"agent_name": "A1"}))
        await bus.publish(Event(type=EventType.TOOL_CALLED, data={"tool": "search"}))
        await bus.publish(Event(type=EventType.AGENT_RESPONDED, data={"agent_name": "A1"}))
        
        # Query all
        events = obs.events()
        assert len(events) >= 3
        
        # Query with limit
        events = obs.events(limit=2)
        assert len(events) == 2
        
        # Query by channel
        agent_events = obs.events(channel=Channel.AGENTS)
        assert all(e.type in {EventType.AGENT_INVOKED, EventType.AGENT_RESPONDED} for e in agent_events)

    @pytest.mark.asyncio
    async def test_timeline(self):
        """Test timeline generation."""
        from agenticflow.observability import Observer, ObservabilityLevel
        from agenticflow.events.bus import EventBus
        from agenticflow.schemas.event import Event
        from agenticflow.core.enums import EventType
        
        obs = Observer(level=ObservabilityLevel.DEBUG)
        bus = EventBus()
        obs.attach(bus)
        
        await bus.publish(Event(type=EventType.TASK_STARTED, data={"task": "test"}))
        await bus.publish(Event(type=EventType.TASK_COMPLETED, data={}))
        
        timeline = obs.timeline()
        assert "Timeline:" in timeline
        assert "s" in timeline  # Time markers

    @pytest.mark.asyncio
    async def test_summary(self):
        """Test summary generation."""
        from agenticflow.observability import Observer, ObservabilityLevel
        from agenticflow.events.bus import EventBus
        from agenticflow.schemas.event import Event
        from agenticflow.core.enums import EventType
        
        obs = Observer(level=ObservabilityLevel.DEBUG)
        bus = EventBus()
        obs.attach(bus)
        
        await bus.publish(Event(type=EventType.AGENT_INVOKED, data={"agent_name": "Test"}))
        
        summary = obs.summary()
        assert "Execution Summary" in summary
        assert "Total events:" in summary


class TestObserverStreaming:
    """Tests for Observer streaming integration."""
    
    def test_streaming_channel_exists(self):
        """Test that STREAMING channel is available."""
        from agenticflow.observability import Channel
        
        assert hasattr(Channel, "STREAMING")
        assert Channel.STREAMING.value == "streaming"
    
    def test_streaming_factory_method(self):
        """Test Observer.streaming() factory method."""
        from agenticflow.observability import Observer, ObservabilityLevel, Channel
        
        obs = Observer.streaming()
        assert obs.config.level == ObservabilityLevel.DEBUG
        assert Channel.STREAMING in obs.config.channels
        assert Channel.AGENTS in obs.config.channels
        
        # With show_tokens=False
        obs2 = Observer.streaming(show_tokens=False)
        assert obs2.config.level == ObservabilityLevel.DETAILED
    
    def test_streaming_only_factory_method(self):
        """Test Observer.streaming_only() factory method."""
        from agenticflow.observability import Observer, ObservabilityLevel, Channel
        
        obs = Observer.streaming_only()
        assert obs.config.level == ObservabilityLevel.DEBUG
        assert Channel.STREAMING in obs.config.channels
    
    def test_on_stream_callback_config(self):
        """Test on_stream callback is properly configured."""
        from agenticflow.observability import Observer
        
        stream_calls = []
        
        obs = Observer(
            on_stream=lambda agent, token, data: stream_calls.append((agent, token)),
        )
        
        assert obs.config.on_stream is not None
    
    @pytest.mark.asyncio
    async def test_streaming_events_dispatched(self):
        """Test streaming events are dispatched to callback."""
        from agenticflow.observability import Observer, ObservabilityLevel, Channel
        from agenticflow.events.bus import EventBus
        from agenticflow.schemas.event import Event
        from agenticflow.core.enums import EventType
        
        stream_calls = []
        
        obs = Observer(
            level=ObservabilityLevel.DEBUG,
            channels={Channel.STREAMING},
            on_stream=lambda agent, token, data: stream_calls.append((agent, token)),
        )
        
        bus = EventBus()
        obs.attach(bus)
        
        # Publish streaming events
        await bus.publish(Event(
            type=EventType.STREAM_START,
            data={"agent_name": "TestAgent", "model": "gpt-4"}
        ))
        await bus.publish(Event(
            type=EventType.TOKEN_STREAMED,
            data={"agent_name": "TestAgent", "token": "Hello"}
        ))
        await bus.publish(Event(
            type=EventType.TOKEN_STREAMED,
            data={"agent_name": "TestAgent", "token": " world"}
        ))
        await bus.publish(Event(
            type=EventType.STREAM_END,
            data={"agent_name": "TestAgent"}
        ))
        
        # Check callbacks were called
        assert len(stream_calls) >= 4
        assert ("TestAgent", "start") in stream_calls
        assert ("TestAgent", "Hello") in stream_calls
        assert ("TestAgent", " world") in stream_calls
        assert ("TestAgent", "end") in stream_calls
    
    @pytest.mark.asyncio
    async def test_streaming_events_in_metrics(self):
        """Test streaming events are counted in metrics."""
        from agenticflow.observability import Observer, ObservabilityLevel, Channel
        from agenticflow.events.bus import EventBus
        from agenticflow.schemas.event import Event
        from agenticflow.core.enums import EventType
        
        obs = Observer(level=ObservabilityLevel.DEBUG)
        bus = EventBus()
        obs.attach(bus)
        
        await bus.publish(Event(type=EventType.STREAM_START, data={"agent_name": "Test"}))
        await bus.publish(Event(type=EventType.TOKEN_STREAMED, data={"agent_name": "Test", "token": "x"}))
        await bus.publish(Event(type=EventType.STREAM_END, data={"agent_name": "Test"}))
        
        metrics = obs.metrics()
        assert metrics.get("events.stream.start", 0) == 1
        assert metrics.get("events.stream.token", 0) == 1
        assert metrics.get("events.stream.end", 0) == 1
    
    @pytest.mark.asyncio
    async def test_streaming_events_formatted(self):
        """Test streaming events are properly formatted."""
        from agenticflow.observability import Observer, ObservabilityLevel, Channel
        from agenticflow.events.bus import EventBus
        from agenticflow.schemas.event import Event
        from agenticflow.core.enums import EventType
        from io import StringIO
        
        output = StringIO()
        obs = Observer(
            level=ObservabilityLevel.DETAILED,
            channels={Channel.STREAMING},
            stream=output,
        )
        
        bus = EventBus()
        obs.attach(bus)
        
        await bus.publish(Event(
            type=EventType.STREAM_START,
            data={"agent_name": "TestAgent", "model": "gpt-4"}
        ))
        await bus.publish(Event(
            type=EventType.STREAM_END,
            data={"agent_name": "TestAgent"}
        ))
        
        result = output.getvalue()
        assert "TestAgent" in result
        assert "streaming" in result
        assert "complete" in result
    
    @pytest.mark.asyncio
    async def test_stream_error_triggers_on_error(self):
        """Test STREAM_ERROR triggers on_error callback."""
        from agenticflow.observability import Observer, ObservabilityLevel, Channel
        from agenticflow.events.bus import EventBus
        from agenticflow.schemas.event import Event
        from agenticflow.core.enums import EventType
        
        errors = []
        
        obs = Observer(
            level=ObservabilityLevel.DEBUG,
            channels={Channel.STREAMING},
            on_error=lambda source, err: errors.append((source, err)),
        )
        
        bus = EventBus()
        obs.attach(bus)
        
        await bus.publish(Event(
            type=EventType.STREAM_ERROR,
            data={"agent_name": "TestAgent", "error": "Connection failed"}
        ))
        
        assert len(errors) == 1
        assert errors[0][0] == "TestAgent"
        assert "Connection failed" in errors[0][1]
