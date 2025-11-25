"""
Observability Example - demonstrates tracing, metrics, and logging.

This example shows:
1. Distributed tracing with spans
2. Metrics collection (counters, gauges, histograms)
3. Structured logging
4. Dashboard for monitoring

Usage:
    uv run python examples/observability_example.py
"""

import asyncio
import time
import random

from agenticflow.observability import (
    # Tracing
    Tracer,
    Span,
    SpanKind,
    SpanContext,
    # Metrics
    MetricsCollector,
    Counter,
    Gauge,
    Histogram,
    Timer,
    # Logging
    ObservabilityLogger,
    LogLevel,
    LogEntry,
    # Dashboard
    Dashboard,
    DashboardConfig,
)


# =============================================================================
# Example 1: Distributed Tracing
# =============================================================================

async def tracing_example():
    """Demonstrate distributed tracing with spans."""
    print("\n" + "=" * 60)
    print("Example 1: Distributed Tracing")
    print("=" * 60)
    
    # Create a tracer
    tracer = Tracer("agent-service")
    
    # Create a root span
    with tracer.span("process_request", SpanKind.AGENT) as root_span:
        root_span.set_attribute("request_id", "req-123")
        root_span.set_attribute("user_id", "user-456")
        
        print(f"\n🔍 Started trace: {root_span.context.trace_id[:8]}...")
        
        # Simulate work
        await asyncio.sleep(0.1)
        root_span.add_event("request_parsed")
        
        # Create a child span for research
        with tracer.span("research_task", SpanKind.TASK) as research_span:
            research_span.set_attribute("topic", "AI trends")
            await asyncio.sleep(0.05)
            research_span.add_event("research_complete", {"sources": 5})
        
        # Create a child span for tool call
        with tracer.span("search_tool", SpanKind.TOOL) as tool_span:
            tool_span.set_attribute("tool_name", "web_search")
            tool_span.set_attribute("query", "AI trends 2024")
            await asyncio.sleep(0.03)
        
        # Create a child span for LLM call
        with tracer.span("llm_call", SpanKind.LLM) as llm_span:
            llm_span.set_attribute("model", "gpt-4o")
            llm_span.set_attribute("tokens", 150)
            await asyncio.sleep(0.08)
    
    # Print trace information
    print(f"\n📊 Trace Summary:")
    print(f"   Spans collected: {len(tracer.finished_spans)}")
    
    for span in tracer.finished_spans:
        duration = f"{span.duration_ms:.1f}ms" if span.duration_ms else "N/A"
        print(f"   - {span.name} ({span.kind.value}): {duration}")
    
    # Export spans
    exported = tracer.export()
    print(f"\n📤 Exported {len(exported)} spans as JSON")
    
    print("\n✅ Tracing complete - use traces to debug and optimize")


# =============================================================================
# Example 2: Metrics Collection
# =============================================================================

async def metrics_example():
    """Demonstrate metrics collection."""
    print("\n" + "=" * 60)
    print("Example 2: Metrics Collection")
    print("=" * 60)
    
    # Create metrics collector
    collector = MetricsCollector()
    
    # Counter: things that only go up
    requests_counter = collector.counter(
        "requests_total",
        description="Total number of requests"
    )
    errors_counter = collector.counter(
        "errors_total",
        description="Total number of errors"
    )
    
    # Gauge: things that go up and down
    active_agents = collector.gauge(
        "active_agents",
        description="Number of currently active agents"
    )
    
    # Histogram: distribution of values
    latency_hist = collector.histogram(
        "request_latency_ms",
        description="Request latency in milliseconds",
        buckets=[10, 50, 100, 250, 500, 1000]
    )
    
    print("\n📈 Simulating metrics...")
    
    # Simulate some activity
    for i in range(10):
        # Increment request counter
        requests_counter.inc()
        
        # Randomly increment error counter
        if random.random() < 0.2:
            errors_counter.inc(labels={"type": "timeout"})
        
        # Update active agents gauge
        active_agents.set(random.randint(1, 5))
        
        # Record latency
        latency = random.uniform(20, 200)
        latency_hist.observe(latency)
        
        await asyncio.sleep(0.01)
    
    # Get snapshot
    snapshot = collector.snapshot()
    
    print(f"\n📊 Metrics Snapshot:")
    for name, metric in snapshot.items():
        print(f"\n   {name}:")
        if isinstance(metric, dict):
            for key, value in metric.items():
                if isinstance(value, float):
                    print(f"      {key}: {value:.2f}")
                else:
                    print(f"      {key}: {value}")
        else:
            print(f"      value: {metric}")
    
    print("\n✅ Metrics collection complete")


# =============================================================================
# Example 3: Timer Context Manager
# =============================================================================

async def timer_example():
    """Demonstrate timer for measuring durations."""
    print("\n" + "=" * 60)
    print("Example 3: Timer Context Manager")
    print("=" * 60)
    
    # Create a timer (wraps a histogram)
    operation_timer = Timer(
        "operation_duration",
        description="Duration of operations"
    )
    
    print("\n⏱️ Timing operations...")
    
    # Time sync operation
    print("   Timing sync operation...")
    with operation_timer.time():
        time.sleep(0.1)
    
    # Time another sync operation
    print("   Timing another sync operation...")
    with operation_timer.time():
        time.sleep(0.15)
    
    # Record a duration directly
    print("   Recording manual duration...")
    operation_timer.record(50.0)  # 50ms
    
    print(f"\n📊 Timer Results:")
    print(f"   Observations: {operation_timer.histogram.count}")
    print(f"   Total time: {operation_timer.histogram.sum:.1f}ms")
    print(f"   Mean time: {operation_timer.histogram.mean:.1f}ms")
    
    print("\n✅ Timer example complete")


# =============================================================================
# Example 4: Structured Logging
# =============================================================================

async def logging_example():
    """Demonstrate structured logging."""
    print("\n" + "=" * 60)
    print("Example 4: Structured Logging")
    print("=" * 60)
    
    # Create logger (output to /dev/null for demo, but we'll capture entries)
    import io
    output = io.StringIO()
    
    logger = ObservabilityLogger(
        name="agent.executor",
        level=LogLevel.DEBUG,
        output=output,
        format="text",
    )
    
    # Set persistent context
    logger.set_context(agent_id="agent-123", session="sess-456")
    
    # Set trace correlation
    logger.set_trace("trace-abc", "span-xyz")
    
    print("\n📝 Logging at different levels...")
    
    # Log at different levels
    logger.debug("Starting task processing", task_id="task-1")
    logger.info("Agent initialized", model="gpt-4o")
    logger.warning("Rate limit approaching", remaining=10)
    logger.error("Tool execution failed", tool="search", error="timeout")
    
    # Log an exception
    try:
        raise ValueError("Example error")
    except Exception as e:
        logger.exception("Caught exception", e, context="demo")
    
    # Create child logger
    child = logger.child("tools")
    child.info("Tool registry loaded", tool_count=5)
    
    # Get logged entries
    entries = logger.entries
    
    print(f"\n📊 Log Summary:")
    print(f"   Total entries: {len(entries)}")
    
    level_counts = {}
    for entry in entries:
        level = entry.level.name
        level_counts[level] = level_counts.get(level, 0) + 1
    
    for level, count in level_counts.items():
        print(f"   {level}: {count}")
    
    # Show formatted output
    print(f"\n📄 Sample log entries:")
    for entry in entries[:3]:
        print(f"   {entry.format('text')[:80]}...")
    
    print("\n✅ Structured logging complete")


# =============================================================================
# Example 5: Dashboard Integration
# =============================================================================

async def dashboard_example():
    """Demonstrate dashboard for unified observability."""
    print("\n" + "=" * 60)
    print("Example 5: Dashboard Integration")
    print("=" * 60)
    
    # Create components
    tracer = Tracer("my-service")
    collector = MetricsCollector()
    
    import io
    logger = ObservabilityLogger(
        name="dashboard-demo",
        level=LogLevel.INFO,
        output=io.StringIO(),
    )
    
    # Create dashboard with config
    dashboard = Dashboard(
        config=DashboardConfig(
            name="AgenticFlow Monitor",
            refresh_interval_ms=5000,  # 5 seconds
        ),
    )
    
    # Register components
    dashboard.register_tracer(tracer)
    dashboard.register_metrics(collector)
    dashboard.register_logger(logger)
    
    print(f"\n🖥️ Dashboard: {dashboard.config.name}")
    
    # Simulate some activity
    counter = collector.counter("dashboard_events")
    gauge = collector.gauge("active_tasks")
    
    for i in range(5):
        with tracer.span(f"task_{i}", SpanKind.TASK):
            counter.inc()
            gauge.set(i + 1)
            logger.info(f"Processing task {i}")
            await asyncio.sleep(0.02)
    
    # Get dashboard snapshot (DashboardSnapshot dataclass)
    snapshot = dashboard.snapshot()
    
    print(f"\n📊 Dashboard Snapshot:")
    print(f"   Timestamp: {snapshot.timestamp.isoformat()}")
    print(f"   Spans: {len(snapshot.spans)}")
    print(f"   Metrics: {list(snapshot.metrics.keys()) if snapshot.metrics else 'empty'}")
    print(f"   Log entries: {len(snapshot.logs)}")
    
    # Get summary (returns dict)
    summary = dashboard.summary()
    
    print(f"\n📈 Dashboard Summary:")
    print(f"   Total spans: {summary['spans']['total']}")
    print(f"   By span kind: {summary['spans']['by_kind']}")
    print(f"   Total logs: {summary['logs']['total']}")
    print(f"   Events count: {summary['events_count']}")
    
    print("\n✅ Dashboard ready for monitoring")


# =============================================================================
# Example 6: End-to-End Observability
# =============================================================================

async def end_to_end_example():
    """Demonstrate complete observability stack."""
    print("\n" + "=" * 60)
    print("Example 6: End-to-End Observability")
    print("=" * 60)
    
    # Setup observability stack
    tracer = Tracer("agent-system")
    collector = MetricsCollector()
    
    import io
    logger = ObservabilityLogger(
        name="system",
        level=LogLevel.DEBUG,
        output=io.StringIO(),
    )
    
    # Create metrics
    request_counter = collector.counter("agent_requests_total")
    latency_hist = collector.histogram("agent_latency_ms")
    active_gauge = collector.gauge("active_agents")
    
    print("\n🚀 Simulating agent workflow with full observability...")
    
    # Simulate a multi-agent workflow
    with tracer.span("workflow", SpanKind.TOPOLOGY) as workflow_span:
        workflow_span.set_attribute("workflow_type", "content_creation")
        logger.info("Workflow started", workflow_id=workflow_span.context.trace_id[:8])
        
        # Supervisor agent
        with tracer.span("supervisor", SpanKind.AGENT) as sup_span:
            active_gauge.inc()
            request_counter.inc()
            logger.debug("Supervisor analyzing task")
            await asyncio.sleep(0.05)
            sup_span.add_event("task_delegated", {"workers": ["researcher", "writer"]})
        
        # Research phase
        with tracer.span("research_phase", SpanKind.TASK) as research_span:
            with tracer.span("researcher", SpanKind.AGENT):
                active_gauge.set(2)
                request_counter.inc()
                
                # Tool call
                with tracer.span("web_search", SpanKind.TOOL):
                    await asyncio.sleep(0.03)
                    latency_hist.observe(30)
                
                # LLM call
                with tracer.span("analyze_results", SpanKind.LLM):
                    await asyncio.sleep(0.08)
                    latency_hist.observe(80)
                
                logger.info("Research complete", sources=5)
        
        # Writing phase
        with tracer.span("writing_phase", SpanKind.TASK) as write_span:
            with tracer.span("writer", SpanKind.AGENT):
                request_counter.inc()
                
                with tracer.span("generate_content", SpanKind.LLM):
                    await asyncio.sleep(0.1)
                    latency_hist.observe(100)
                
                logger.info("Content generated", word_count=500)
        
        active_gauge.set(0)
        logger.info("Workflow complete", 
                   total_spans=len(tracer.finished_spans),
                   duration_ms=workflow_span.duration_ms)
    
    # Print summary
    print(f"\n📊 Observability Summary:")
    print(f"\n   Traces:")
    print(f"      Total spans: {len(tracer.finished_spans)}")
    print(f"      Span types: {set(s.kind.value for s in tracer.finished_spans)}")
    
    print(f"\n   Metrics:")
    snapshot = collector.snapshot()
    print(f"      Requests: {snapshot.get('agent_requests_total', 0)}")
    print(f"      Latency (mean): {snapshot.get('agent_latency_ms', {}).get('mean', 0):.1f}ms")
    
    print(f"\n   Logs:")
    print(f"      Entries: {len(logger.entries)}")
    
    print("\n✅ Full observability stack demonstrated")


# =============================================================================
# Main
# =============================================================================

async def main():
    """Run all observability examples."""
    print("\n" + "👁️ " * 20)
    print("AgenticFlow Observability Examples")
    print("👁️ " * 20)
    
    await tracing_example()
    await metrics_example()
    await timer_example()
    await logging_example()
    await dashboard_example()
    await end_to_end_example()
    
    print("\n" + "=" * 60)
    print("All observability examples complete!")
    print("=" * 60)
    
    print("\n💡 Integration tips:")
    print("   - Use tracer.span() to wrap agent operations")
    print("   - Use metrics for monitoring and alerting")
    print("   - Use structured logging for debugging")
    print("   - Use dashboard for unified visibility")


if __name__ == "__main__":
    asyncio.run(main())
