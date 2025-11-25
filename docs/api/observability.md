# Observability API Reference

AgenticFlow provides comprehensive observability features for monitoring, debugging, and tracking multi-agent systems.

## Overview

```python
from agenticflow import (
    # Progress Tracking
    ProgressTracker,
    ProgressEvent,
    OutputConfig,
    Verbosity,
    OutputFormat,
    ProgressStyle,
    Theme,
    
    # Metrics
    MetricsCollector,
    
    # Tracing
    Tracer,
    
    # Dashboard
    Dashboard,
    
    # Logging
    Logger,
    
    # Inspector
    Inspector,
)
```

---

## ProgressTracker

The main progress tracking class providing real-time visibility into agent execution.

### Quick Start

```python
from agenticflow import ProgressTracker, OutputConfig

# Create tracker with default settings
tracker = ProgressTracker()

# Use verbose config
tracker = ProgressTracker(OutputConfig.verbose())

# Basic usage
with tracker.task("Processing data"):
    tracker.update("Loading files...")
    tracker.progress(50, "Halfway done")
    tracker.update("Analyzing...")
```

### Constructor

```python
ProgressTracker(
    config: OutputConfig | None = None,
    name: str = "progress",
)
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `config` | `OutputConfig \| None` | Output configuration |
| `name` | `str` | Tracker name for logging |

### Methods

#### `start(title, **kwargs) -> str`

Start a tracked operation.

```python
trace_id = tracker.start("Processing batch")
# ... do work ...
tracker.complete(result="Success")
```

#### `complete(result=None, **kwargs)`

Mark operation as complete.

```python
tracker.complete(result="Processed 100 items", items=100)
```

#### `error(message, exception=None, **kwargs)`

Report an error.

```python
try:
    risky_operation()
except Exception as e:
    tracker.error("Operation failed", exception=e)
```

#### `update(message, **kwargs)`

Send a progress update.

```python
tracker.update("Step 1 complete")
tracker.update("Loading data", bytes_loaded=1024)
```

#### `step(step, total, name="", **kwargs)`

Report step progress.

```python
for i, item in enumerate(items):
    tracker.step(i + 1, len(items), f"Processing {item}")
    process(item)
```

#### `progress(percent, message="", **kwargs)`

Report percentage progress.

```python
tracker.progress(25, "Quarter done")
tracker.progress(50, "Halfway")
tracker.progress(100, "Complete!")
```

#### Agent Events

```python
# Agent starts processing
tracker.agent_start("Researcher")

# Agent completes with result
tracker.agent_complete("Researcher", "Found 5 relevant articles")
```

#### Tool Events

```python
# Tool being called
tracker.tool_call("search", {"query": "Python tutorials"})

# Tool returned result
tracker.tool_result("search", "Found 10 results")

# Tool error
tracker.tool_error("search", "Rate limit exceeded")
```

#### Resilience Events

```python
# Retry attempt
tracker.retry(
    tool="api_call",
    attempt=2,
    max_retries=5,
    delay=2.5,
    error="Connection timeout",
)

# Circuit breaker opened
tracker.circuit_open(
    tool="external_api",
    failure_count=3,
    reset_timeout=30.0,
)

# Circuit breaker recovered
tracker.circuit_close(tool="external_api")

# Fallback being used
tracker.fallback(
    from_tool="primary_api",
    to_tool="backup_api",
    reason="primary_unavailable",
)

# Successful recovery
tracker.recovery(
    tool="primary_api",
    method="retry",       # or "fallback"
    attempts=3,
    fallback_tool=None,   # Set if recovered via fallback
)
```

#### DAG Execution Events

```python
# Report wave starting in DAG execution
tracker.wave_start(
    wave=1,
    total_waves=3,
    parallel_calls=4,
)
```

#### Custom Events

```python
# Emit any custom event
tracker.custom("my_event", data="value", count=42)
```

### Context Managers

#### `task(title, **kwargs)` - Sync

```python
with tracker.task("Processing batch"):
    tracker.update("Step 1")
    process_data()
    tracker.update("Step 2")
# Automatically calls complete() or error()
```

#### `async_task(title, **kwargs)` - Async

```python
async with tracker.async_task("Async operation"):
    await async_operation()
```

#### `spinner(message)` - Animated Spinner

```python
with tracker.spinner("Loading..."):
    time.sleep(2)
# Shows animated spinner, then checkmark on completion
```

### Streaming

#### `stream_events(async_generator) -> AsyncIterator[ProgressEvent]`

Stream progress events from an async generator.

```python
async for event in tracker.stream_events(topology.stream(task)):
    if event.event_type == "agent_complete":
        print(f"Agent done: {event.data['agent']}")
```

### Analysis

#### `get_events() -> list[ProgressEvent]`

Get all recorded events.

```python
events = tracker.get_events()
for event in events:
    print(f"{event.timestamp}: {event.event_type}")
```

#### `get_timeline() -> str`

Get a timeline visualization.

```python
print(tracker.get_timeline())
# Output:
# Timeline:
# ────────────────────────────────
#    +0.00s 🚀 Starting
#    +0.12s 👤 Researcher
#    +0.45s 🔧 search
#    +0.89s ✓ Completed
```

#### `get_summary() -> dict`

Get execution summary.

```python
summary = tracker.get_summary()
print(f"Duration: {summary['duration_seconds']:.2f}s")
print(f"Events: {summary['event_counts']}")
```

### Using with Context Manager

```python
from agenticflow import ProgressTracker, OutputConfig

tracker = ProgressTracker(OutputConfig.verbose())

# Use __enter__ and __exit__ directly (alternative to task())
with tracker:
    tracker.start("My workflow")
    do_work()
    tracker.complete()
```

---

## OutputConfig

Configuration for the output system.

### Presets

```python
from agenticflow import OutputConfig

# Minimal output (errors only)
config = OutputConfig.minimal()

# Verbose output with timestamps
config = OutputConfig.verbose()

# Debug mode (everything)
config = OutputConfig.debug()

# JSON format for programmatic use
config = OutputConfig.json()

# Silent (no output)
config = OutputConfig.silent()
```

### Full Configuration

```python
from agenticflow import OutputConfig, Verbosity, OutputFormat, ProgressStyle, Theme

config = OutputConfig(
    verbosity=Verbosity.VERBOSE,        # How much to show
    format=OutputFormat.RICH,           # Output format
    progress_style=ProgressStyle.SPINNER,  # Progress indicator
    theme=Theme.DEFAULT,                # Color theme
    show_timestamps=True,               # Include timestamps
    show_duration=True,                 # Show operation duration
    show_agent_names=True,              # Show agent names
    show_tool_names=True,               # Show tool names
    show_dag=True,                      # Visualize DAG
    show_trace_ids=False,               # Show correlation IDs
    truncate_results=200,               # Max result chars (0=unlimited)
    use_unicode=True,                   # Use Unicode symbols
    use_colors=True,                    # Use ANSI colors
)
```

### Verbosity Levels

```python
from agenticflow import Verbosity

class Verbosity(IntEnum):
    SILENT = 0    # No output
    MINIMAL = 1   # Only final results
    NORMAL = 2    # Key milestones (default)
    VERBOSE = 3   # Detailed progress
    DEBUG = 4     # Everything including internals
    TRACE = 5     # Maximum detail
```

### Output Formats

```python
from agenticflow import OutputFormat

class OutputFormat(Enum):
    TEXT = "text"           # Plain text
    RICH = "rich"           # Rich formatting (colors, Unicode)
    JSON = "json"           # JSON lines
    STRUCTURED = "structured"  # For programmatic use
    MINIMAL = "minimal"     # Minimal output
```

### Progress Styles

```python
from agenticflow import ProgressStyle

class ProgressStyle(Enum):
    SPINNER = "spinner"   # Spinning indicator
    BAR = "bar"           # Progress bar
    DOTS = "dots"         # Dot sequence
    STEPS = "steps"       # Step counter [1/5]
    PERCENT = "percent"   # Percentage
    NONE = "none"         # No indicator
```

### Themes

```python
from agenticflow import Theme

class Theme(Enum):
    DEFAULT = "default"
    DARK = "dark"
    LIGHT = "light"
    MINIMAL = "minimal"
    COLORFUL = "colorful"
```

---

## ProgressEvent

A progress event for tracking execution.

```python
from agenticflow import ProgressEvent

@dataclass
class ProgressEvent:
    event_type: str                              # Event type
    data: dict[str, Any]                         # Event data
    timestamp: datetime                          # When it occurred
    trace_id: str | None                         # Correlation ID
    parent_id: str | None                        # Parent event ID
    event_id: str                                # Unique event ID

# Methods
event.to_dict()  # Convert to dictionary
event.to_json()  # Convert to JSON string
```

---

## Callback Helpers

### `create_on_step_callback(tracker)`

Create a callback for topology execution.

```python
from agenticflow import ProgressTracker, create_on_step_callback

tracker = ProgressTracker(OutputConfig.verbose())
callback = create_on_step_callback(tracker)

result = await topology.run(
    "Analyze data",
    on_step=callback,
)
```

### `create_executor_callback(tracker)`

Create a callback for agent executor strategies.

```python
from agenticflow import ProgressTracker, create_executor_callback

tracker = ProgressTracker(OutputConfig.verbose())
callback = create_executor_callback(tracker)

result = await agent.run(
    "Complex task",
    strategy="dag",
    on_step=callback,
)
```

---

## DAG Visualization

### `render_dag_ascii(nodes, edges, node_status=None) -> str`

Render a DAG as ASCII art.

```python
from agenticflow import render_dag_ascii

diagram = render_dag_ascii(
    nodes=["search", "analyze", "summarize", "report"],
    edges=[
        ("search", "analyze"),
        ("search", "summarize"),
        ("analyze", "report"),
        ("summarize", "report"),
    ],
    node_status={
        "search": "completed",
        "analyze": "running",
        "summarize": "running",
        "report": "pending",
    },
)
print(diagram)
# Output:
#   Level 1: ● search
#       │
#       ▼
#   Level 2: ◐ analyze | ◐ summarize
#       │
#       ▼
#   Level 3: ○ report
```

---

## MetricsCollector

Collect and aggregate metrics from agent execution.

```python
from agenticflow import MetricsCollector

collector = MetricsCollector()

# Record metrics
collector.record("tool_calls", 1)
collector.record("thinking_time_ms", 150.5)
collector.record("tokens_used", 1250)

# Get aggregates
stats = collector.get_stats("thinking_time_ms")
print(f"Avg thinking time: {stats['mean']:.2f}ms")
print(f"Total calls: {stats['count']}")

# Export all metrics
all_metrics = collector.export()
```

---

## Tracer

Distributed tracing for complex workflows.

```python
from agenticflow import Tracer

tracer = Tracer()

# Start a trace
with tracer.span("workflow") as span:
    span.set_tag("task_type", "research")
    
    # Nested spans
    with tracer.span("search") as search_span:
        search_span.set_tag("query", "Python async")
        result = await search(query)
        search_span.log("results_count", len(result))
    
    with tracer.span("analyze"):
        analysis = analyze(result)

# Export trace
trace_data = tracer.export()
```

---

## Complete Example

```python
import asyncio
from agenticflow import (
    Agent,
    AgentConfig,
    AgentRole,
    EventBus,
    ToolRegistry,
    SupervisorTopology,
    TopologyConfig,
    ProgressTracker,
    OutputConfig,
    Verbosity,
    create_on_step_callback,
)
from langchain_core.tools import tool

@tool
def search(query: str) -> str:
    """Search for information."""
    return f"Results for: {query}"

async def main():
    # Setup
    event_bus = EventBus()
    tool_registry = ToolRegistry()
    tool_registry.register(search)
    
    # Create agents
    supervisor = Agent(
        AgentConfig(
            name="Supervisor",
            role=AgentRole.SUPERVISOR,
            model="gpt-4o",
            tools=["search"],
        ),
        event_bus=event_bus,
        tool_registry=tool_registry,
    )
    
    researcher = Agent(
        AgentConfig(
            name="Researcher",
            role=AgentRole.WORKER,
            model="gpt-4o",
            tools=["search"],
        ),
        event_bus=event_bus,
        tool_registry=tool_registry,
    )
    
    # Create topology
    topology = SupervisorTopology(
        config=TopologyConfig(name="research-team"),
        agents=[supervisor, researcher],
        supervisor_name="Supervisor",
    )
    
    # Setup progress tracking
    tracker = ProgressTracker(
        OutputConfig(
            verbosity=Verbosity.VERBOSE,
            show_timestamps=True,
            show_duration=True,
            show_dag=True,
        )
    )
    
    # Create callback
    callback = create_on_step_callback(tracker)
    
    # Run with tracking
    with tracker.task("Research Workflow"):
        result = await topology.run(
            "Research the latest developments in AI agents",
            on_step=callback,
        )
        
        tracker.update(f"Completed with {len(result.results)} results")
    
    # Show analysis
    print("\n" + "="*50)
    print("EXECUTION ANALYSIS")
    print("="*50)
    
    # Timeline
    print("\n" + tracker.get_timeline())
    
    # Summary
    summary = tracker.get_summary()
    print(f"\nTotal events: {summary['total_events']}")
    print(f"Duration: {summary['duration_seconds']:.2f}s")
    print(f"Event types: {summary['event_counts']}")

if __name__ == "__main__":
    asyncio.run(main())
```

---

## Integration with Agent Resilience

The progress tracker integrates with the resilience system to show retry, circuit breaker, and fallback events:

```python
from agenticflow import (
    Agent,
    AgentConfig,
    ResilienceConfig,
    ProgressTracker,
    OutputConfig,
)

# Create agent with resilience
agent = Agent(
    AgentConfig(
        name="ResilientAgent",
        model="gpt-4o",
        tools=["flaky_api", "backup_api"],
    ).with_resilience(
        ResilienceConfig.aggressive()
    ).with_fallbacks({
        "flaky_api": ["backup_api"],
    }),
    event_bus=event_bus,
    tool_registry=tool_registry,
)

# Track with full visibility
tracker = ProgressTracker(OutputConfig.verbose())

# Execute with tracking - will show retry/fallback events
result = await agent.act(
    "flaky_api",
    {"data": "test"},
    tracker=tracker,
)

# Output shows:
# → flaky_api ({"data": "test"})
# 🔄 flaky_api retry 1/5 in 0.5s
# 🔄 flaky_api retry 2/5 in 1.0s
# ↩️ flaky_api → backup_api (retry_exhausted)
# ✅ backup_api recovered via fallback
```

---

## Next Steps

- [Agents](agents.md) - Agent configuration and resilience
- [Topologies](topologies.md) - Multi-agent patterns
- [Events](events.md) - Event-driven communication
