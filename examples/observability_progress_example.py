"""
Observability Progress System Example
=====================================

Demonstrates the comprehensive progress/output system for AgenticFlow.
This system provides a unified API for all output needs across sync/async
operations, with customizable verbosity, formats, themes, and styles.

Features demonstrated:
- OutputConfig presets (verbose, minimal, debug, json, silent)
- Verbosity levels (SILENT, MINIMAL, NORMAL, VERBOSE, DEBUG, TRACE)
- Output formats (TEXT, RICH, JSON, STRUCTURED, MINIMAL)
- Progress styles (SPINNER, BAR, DOTS, STEPS, PERCENT, NONE)
- Themes (DEFAULT, DARK, LIGHT, MINIMAL, COLORFUL)
- Task tracking with context managers (sync and async)
- Custom callbacks for agent/executor integration
- DAG visualization in ASCII
- Timestamps and duration tracking
"""

import asyncio
import time
from datetime import datetime

from agenticflow import (
    # Progress System
    OutputConfig,
    Verbosity,
    OutputFormat,
    ProgressStyle,
    ProgressTracker,
    ProgressEvent,
    Styler,
    Colors,
    Symbols,
    configure_output,
    render_dag_ascii,
    # Execution
    ExecutionStrategy,
)
from agenticflow.observability.progress import Theme


def section(title: str) -> None:
    """Print a section header."""
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}\n")


def example_basic_tracking() -> None:
    """Basic progress tracking with default settings."""
    section("1. Basic Progress Tracking")
    
    tracker = ProgressTracker()
    
    # Simple task tracking
    with tracker.task("Loading configuration"):
        time.sleep(0.1)  # Simulate work
        
    with tracker.task("Connecting to database"):
        time.sleep(0.1)
        
    with tracker.task("Initializing agents"):
        time.sleep(0.1)


def example_output_presets() -> None:
    """Demonstrate different output presets."""
    section("2. Output Configuration Presets")
    
    presets = [
        ("Verbose", OutputConfig.verbose()),
        ("Minimal", OutputConfig.minimal()),
        ("Debug", OutputConfig.debug()),
        ("JSON", OutputConfig.json()),
        ("Silent", OutputConfig.silent()),
    ]
    
    for name, config in presets:
        print(f"\n--- {name} Preset ---")
        print(f"  Verbosity: {config.verbosity.name}")
        print(f"  Format: {config.format.name}")
        print(f"  Show timestamps: {config.show_timestamps}")
        print(f"  Show duration: {config.show_duration}")
        print(f"  Progress style: {config.progress_style.name}")
        
        tracker = ProgressTracker(config=config)
        with tracker.task(f"Task with {name} config"):
            time.sleep(0.05)


def example_verbosity_levels() -> None:
    """Demonstrate verbosity levels."""
    section("3. Verbosity Levels")
    
    levels = [
        (Verbosity.SILENT, "No output at all"),
        (Verbosity.MINIMAL, "Only final results"),
        (Verbosity.NORMAL, "Key milestones"),
        (Verbosity.VERBOSE, "Detailed progress"),
        (Verbosity.DEBUG, "Everything including internals"),
        (Verbosity.TRACE, "Maximum detail"),
    ]
    
    for level, description in levels:
        print(f"\n--- {level.name}: {description} ---")
        config = OutputConfig(verbosity=level, show_timestamps=False)
        tracker = ProgressTracker(config=config)
        
        # Use update() which respects verbosity (shows at NORMAL+)
        tracker.update(f"Update message at {level.name}")
        # Use custom() for any event type
        tracker.custom("debug_event", message=f"Debug event at {level.name}")
        tracker.custom("trace_event", message=f"Trace event at {level.name}")


def example_progress_styles() -> None:
    """Demonstrate different progress styles."""
    section("4. Progress Styles")
    
    styles = [
        (ProgressStyle.SPINNER, "Spinning indicator"),
        (ProgressStyle.BAR, "Progress bar"),
        (ProgressStyle.DOTS, "Dot sequence"),
        (ProgressStyle.STEPS, "Step counter"),
        (ProgressStyle.PERCENT, "Percentage"),
        (ProgressStyle.NONE, "No indicator"),
    ]
    
    for style, description in styles:
        print(f"\n--- {style.name}: {description} ---")
        config = OutputConfig(
            progress_style=style,
            show_timestamps=False,
            verbosity=Verbosity.VERBOSE,
        )
        tracker = ProgressTracker(config=config)
        
        with tracker.task(f"Task with {style.name}"):
            time.sleep(0.05)


def example_themes() -> None:
    """Demonstrate output themes with styled text."""
    section("5. Output Themes & Styling")
    
    # Show available themes
    print("Available Themes:")
    for theme in Theme:
        print(f"  - {theme.name}: {theme.value}")
    
    # Show available symbols
    print("Available Symbols:")
    print(f"  Check: {Symbols.CHECK}")
    print(f"  Cross: {Symbols.CROSS}")
    print(f"  Arrow: {Symbols.ARROW}")
    print(f"  Bullet: {Symbols.BULLET}")
    print(f"  Star: {Symbols.STAR}")
    print(f"  Circle: {Symbols.CIRCLE}")
    print(f"  Filled Circle: {Symbols.FILLED_CIRCLE}")
    print(f"  Spinner frames: {Symbols.SPINNER}")
    print(f"  Progress Bar: {Symbols.BAR_FILLED} {Symbols.BAR_EMPTY}")
    
    # Show colors with styler
    print("\nStyled Output (ANSI colors):")
    config = OutputConfig(use_colors=True, format=OutputFormat.RICH)
    styler = Styler(config)
    print(f"  {styler.success('Success text')}")
    print(f"  {styler.error('Error text')}")
    print(f"  {styler.warning('Warning text')}")
    print(f"  {styler.info('Info text')}")
    print(f"  {styler.agent('Agent name')}")
    print(f"  {styler.tool('Tool name')}")
    print(f"  {styler.bold('Bold text')}")
    print(f"  {styler.dim('Dimmed text')}")


def example_nested_tasks() -> None:
    """Demonstrate nested task tracking."""
    section("6. Nested Task Tracking")
    
    config = OutputConfig(
        verbosity=Verbosity.DEBUG,
        show_timestamps=True,
        show_duration=True,
    )
    tracker = ProgressTracker(config=config)
    
    with tracker.task("Orchestrator Run"):
        time.sleep(0.05)
        
        with tracker.task("Agent 1: Research"):
            tracker.update("Searching documents...")
            time.sleep(0.1)
            tracker.update("Found 5 relevant documents")
            
        with tracker.task("Agent 2: Analysis"):
            tracker.update("Analyzing findings...")
            time.sleep(0.1)
            tracker.update("Analysis complete")
            
        with tracker.task("Agent 3: Synthesis"):
            tracker.update("Synthesizing results...")
            time.sleep(0.1)
            tracker.update("Final report generated")


async def example_async_tracking() -> None:
    """Demonstrate async task tracking."""
    section("7. Async Task Tracking")
    
    config = OutputConfig(
        verbosity=Verbosity.VERBOSE,
        show_timestamps=True,
        show_duration=True,
    )
    tracker = ProgressTracker(config=config)
    
    async def simulate_agent(name: str, duration: float) -> str:
        async with tracker.async_task(f"{name} processing"):
            await asyncio.sleep(duration)
            return f"{name} result"
    
    # Run agents in parallel
    tracker.update("Starting parallel agent execution...")
    results = await asyncio.gather(
        simulate_agent("Agent-A", 0.2),
        simulate_agent("Agent-B", 0.15),
        simulate_agent("Agent-C", 0.1),
    )
    
    tracker.update(f"All agents complete: {results}")


def example_callbacks() -> None:
    """Demonstrate callback integration for agents and executors."""
    section("8. Callback Integration")
    
    config = OutputConfig(
        verbosity=Verbosity.DEBUG,
        show_timestamps=True,
    )
    tracker = ProgressTracker(config=config)
    
    # Simulate agent steps using direct tracker methods
    print("Simulating agent execution with tool tracking:\n")
    
    with tracker.task("Agent: Researcher"):
        # Thinking phase
        tracker.update("Analyzing task requirements...")
        
        # Tool calls with detailed tracking
        tracker.tool_call("web_search", {"query": "Python asyncio tutorial"}, agent="Researcher")
        time.sleep(0.1)  # Simulate work
        tracker.tool_result("web_search", "Found 10 relevant articles", duration_ms=100)
        
        tracker.tool_call("read_document", {"url": "https://docs.python.org/asyncio"}, agent="Researcher")
        time.sleep(0.1)
        tracker.tool_result("read_document", "Document content extracted", duration_ms=85)
        
        # Simulate a tool error
        tracker.tool_call("analyze_sentiment", {"text": "async is powerful"}, agent="Researcher")
        time.sleep(0.05)
        tracker.tool_error("analyze_sentiment", "API rate limit exceeded")
        
        tracker.agent_complete("Researcher", "Completed research with 2/3 tools successful")
    
    print("\nSimulating DAG executor with parallel tool calls:\n")
    
    with tracker.task("DAG Executor"):
        tracker.update("Building execution DAG...")
        time.sleep(0.05)
        
        # Wave 1: Parallel research
        tracker.wave_start(wave=1, total_waves=2, parallel_calls=2)
        tracker.tool_call("search_docs", {"topic": "Python"}, agent="DAGExecutor")
        tracker.tool_call("search_code", {"repo": "langchain"}, agent="DAGExecutor")
        time.sleep(0.1)  # Simulate parallel execution
        tracker.tool_result("search_docs", "Found 15 documentation pages", duration_ms=95)
        tracker.tool_result("search_code", "Found 8 code examples", duration_ms=88)
        
        # Wave 2: Synthesis (depends on Wave 1)
        tracker.wave_start(wave=2, total_waves=2, parallel_calls=1)
        tracker.tool_call("synthesize", {"docs": "$call_0", "code": "$call_1"}, agent="DAGExecutor")
        time.sleep(0.05)
        tracker.tool_result("synthesize", "Combined analysis complete", duration_ms=45)


def example_dag_visualization() -> None:
    """Demonstrate DAG ASCII visualization."""
    section("9. DAG Visualization")
    
    # Example DAG structure - nodes and edges
    nodes = ["research_1", "research_2", "analyze_1", "analyze_2", "synthesize", "finalize"]
    edges = [
        ("research_1", "analyze_1"),  # analyze_1 depends on research_1
        ("research_2", "analyze_2"),  # analyze_2 depends on research_2
        ("analyze_1", "synthesize"),  # synthesize depends on analyze_1
        ("analyze_2", "synthesize"),  # synthesize depends on analyze_2
        ("synthesize", "finalize"),   # finalize depends on synthesize
    ]
    
    print("DAG Structure:")
    print(f"  Nodes: {nodes}")
    print(f"  Edges: {edges}")
    
    # Optional status for each node
    node_status = {
        "research_1": "completed",
        "research_2": "completed",
        "analyze_1": "running",
        "analyze_2": "running",
        "synthesize": "pending",
        "finalize": "pending",
    }
    
    print("\nASCII DAG Visualization:")
    ascii_dag = render_dag_ascii(nodes, edges, node_status)
    print(ascii_dag)


def example_json_output() -> None:
    """Demonstrate JSON output format for programmatic parsing."""
    section("10. JSON Output Format")
    
    config = OutputConfig.json()
    tracker = ProgressTracker(config=config)
    
    print("JSON output for programmatic parsing:\n")
    
    with tracker.task("API Request"):
        tracker.update("Processing request...")
        tracker.custom("request_details", method="POST", url="/api/v1/run")
        time.sleep(0.05)
        tracker.custom("request_complete", status=200, latency_ms=50)


def example_custom_config() -> None:
    """Demonstrate fully custom configuration."""
    section("11. Custom Configuration")
    
    config = OutputConfig(
        verbosity=Verbosity.DEBUG,
        format=OutputFormat.RICH,
        progress_style=ProgressStyle.STEPS,
        theme=Theme.COLORFUL,
        show_timestamps=True,
        show_duration=True,
        show_agent_names=True,
        show_tool_names=True,
        show_dag=False,
        show_trace_ids=True,
        truncate_results=100,
        indent="    ",  # 4 spaces
        use_unicode=True,
        use_colors=True,
    )
    
    print("Custom config created:")
    print(f"  Verbosity: {config.verbosity.name}")
    print(f"  Format: {config.format.name}")
    print(f"  Progress style: {config.progress_style.name}")
    print(f"  Theme: {config.theme.name}")
    print(f"  Show timestamps: {config.show_timestamps}")
    print(f"  Show duration: {config.show_duration}")
    print(f"  Truncate results: {config.truncate_results}")
    print(f"  Indent: '{config.indent}'")
    
    tracker = ProgressTracker(config=config)
    
    with tracker.task("Custom configured task"):
        tracker.update("This uses custom settings")
        tracker.custom("debug_info", message="With debug level enabled")


def example_progress_events() -> None:
    """Demonstrate progress events and custom event handling."""
    section("12. Progress Events")
    
    # Create custom events using the actual ProgressEvent structure
    events = [
        ProgressEvent(
            event_type="agent_start",
            data={"message": "Agent starting execution", "agent": "Researcher", "task_count": 5},
        ),
        ProgressEvent(
            event_type="tool_call",
            data={"message": "Calling search tool", "tool": "web_search", "query": "AI agents"},
        ),
        ProgressEvent(
            event_type="agent_complete",
            data={"message": "Agent finished", "results": 10, "duration_ms": 1234},
        ),
    ]
    
    print("Custom progress events:")
    for event in events:
        print(f"\n  Event Type: {event.event_type}")
        print(f"  Event ID: {event.event_id}")
        print(f"  Timestamp: {event.timestamp.isoformat()}")
        print(f"  Data: {event.data}")
        print(f"  JSON: {event.to_json()}")


def example_global_config() -> None:
    """Demonstrate global configuration."""
    section("13. Global Configuration")
    
    # Configure and get a global tracker
    tracker = configure_output(
        verbosity=Verbosity.VERBOSE,
        show_timestamps=True,
        use_colors=True,
    )
    
    print("Global tracker configured:")
    print(f"  Verbosity: {tracker.config.verbosity.name}")
    print(f"  Show timestamps: {tracker.config.show_timestamps}")
    print(f"  Use colors: {tracker.config.use_colors}")
    
    # Use the tracker
    tracker.update("This uses the global configuration")


def example_execution_strategy_integration() -> None:
    """Show how progress integrates with execution strategies."""
    section("14. Execution Strategy Integration")
    
    print("Available Execution Strategies:")
    for strategy in ExecutionStrategy:
        print(f"  - {strategy.name}: {strategy.value}")
    
    print("\nProgress tracking works with all strategies:")
    print("  - REACT: Shows thought/action/observation cycles")
    print("  - PLAN_EXECUTE: Shows plan creation and execution steps")
    print("  - DAG: Shows parallel task graph and dependencies")
    print("  - ADAPTIVE: Shows strategy switching decisions")
    
    config = OutputConfig(
        verbosity=Verbosity.DEBUG,
        show_timestamps=True,
        show_duration=True,
    )
    tracker = ProgressTracker(config=config)
    
    # Simulate DAG execution
    with tracker.task("DAG Execution"):
        tracker.update("Building task dependency graph...")
        time.sleep(0.05)
        
        with tracker.task("Layer 0 (parallel)"):
            tracker.wave_start(wave=0, total_waves=3, parallel_calls=2)
            tracker.update("Executing: research_1, research_2")
            time.sleep(0.1)
            
        with tracker.task("Layer 1 (parallel)"):
            tracker.wave_start(wave=1, total_waves=3, parallel_calls=2)
            tracker.update("Executing: analyze_1, analyze_2")
            time.sleep(0.1)
            
        with tracker.task("Layer 2"):
            tracker.wave_start(wave=2, total_waves=3, parallel_calls=1)
            tracker.update("Executing: synthesize")
            time.sleep(0.05)


def example_output_formats() -> None:
    """Demonstrate different output formats."""
    section("15. Output Formats")
    
    formats = [
        (OutputFormat.TEXT, "Plain text"),
        (OutputFormat.RICH, "Rich formatting with colors"),
        (OutputFormat.JSON, "JSON lines format"),
        (OutputFormat.STRUCTURED, "Structured data"),
        (OutputFormat.MINIMAL, "Minimal output"),
    ]
    
    for fmt, description in formats:
        print(f"\n--- {fmt.name}: {description} ---")
        config = OutputConfig(
            format=fmt,
            verbosity=Verbosity.VERBOSE,
            show_timestamps=True,
        )
        tracker = ProgressTracker(config=config)
        
        with tracker.task(f"Task with {fmt.name}"):
            tracker.update(f"Message in {fmt.name} format")
            time.sleep(0.05)


def main() -> None:
    """Run all examples."""
    print("\n" + "="*60)
    print("  AgenticFlow Progress System Examples")
    print("="*60)
    
    # Sync examples
    example_basic_tracking()
    example_output_presets()
    example_verbosity_levels()
    example_progress_styles()
    example_themes()
    example_nested_tasks()
    example_callbacks()
    example_dag_visualization()
    example_json_output()
    example_custom_config()
    example_progress_events()
    example_global_config()
    example_execution_strategy_integration()
    example_output_formats()
    
    # Async example
    asyncio.run(example_async_tracking())
    
    print("\n" + "="*60)
    print("  All Examples Complete!")
    print("="*60)
    print("\nKey Features Demonstrated:")
    print("  ✓ 5 output presets (verbose, minimal, debug, json, silent)")
    print("  ✓ 6 verbosity levels for fine-grained control")
    print("  ✓ 6 progress styles (spinner, bar, dots, steps, percent, none)")
    print("  ✓ 5 output formats (text, rich, json, structured, minimal)")
    print("  ✓ 5 themes (default, dark, light, minimal, colorful)")
    print("  ✓ Rich styling with colors and Unicode symbols")
    print("  ✓ Nested task tracking with context managers")
    print("  ✓ Async task tracking with async context managers")
    print("  ✓ Callback integration for agents/executors")
    print("  ✓ DAG visualization in ASCII")
    print("  ✓ JSON output for programmatic parsing")
    print("  ✓ Global configuration support")
    print("  ✓ Full execution strategy integration")


if __name__ == "__main__":
    main()
