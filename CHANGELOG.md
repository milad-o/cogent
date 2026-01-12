# Changelog

All notable changes to AgenticFlow will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.4.0] - 2026-01-11

### Added

#### Tool Return Type Visibility

- **Return Type Extraction**: The `@tool` decorator now extracts return type information and includes it in tool descriptions
  - Return type annotations (e.g., `-> dict[str, int]`) are converted to readable strings
  - Docstring `Returns:` sections are parsed and combined with type info
  - LLM sees: `"Get weather data. Returns: dict[str, int] - A dictionary with temp and humidity."`
  - Access via `tool.return_info` property
  - Helps LLM understand expected output format from each tool

#### External Event Sources & Sinks

- **`FileWatcherSource`**: Monitor directories for file changes, emit events for created/modified/deleted files
- **`WebhookSource`**: Receive HTTP webhooks as events (requires `starlette`, `uvicorn`)
- **`RedisStreamSource`**: Consume from Redis Streams with consumer group support (requires `redis`)
- **`WebhookSink`**: POST events to HTTP endpoints with pattern matching (requires `httpx`)
- **`EventFlow.source()`**: Register external event sources to inject events into reactive flows
- **`EventFlow.sink()`**: Register sinks to emit events to external systems

### Changed

#### Observability Renamed for Clarity (Breaking)

- **File Renames**:
  - `observability/event.py` → `trace_record.py`
  - `tests/test_events.py` → `test_traces.py`

- **Class/Function Renames** (observability module only):
  - `Event` → `Trace`
  - `EventType` → `TraceType`
  - `EventBus` → `TraceBus`
  - `get_event_bus()` → `get_trace_bus()`
  - `set_event_bus()` → `set_trace_bus()`

- **Core orchestration unchanged**: `agenticflow.events.Event` and `agenticflow.events.EventBus` remain for agent-to-agent routing

### Migration

```python
# Before (1.3.0): Observability
from agenticflow.observability import Event, EventType, EventBus, get_event_bus

# After (1.4.0): Observability
from agenticflow.observability import Trace, TraceType, TraceBus, get_trace_bus

# Core orchestration (unchanged)
from agenticflow.events import Event, EventBus
```

---

## [1.3.0] - 2026-01-10

### Changed

- **LLM Channel Now Opt-in**: `Observer.debug()` and `Observer.trace()` no longer include `Channel.LLM` by default
  - LLM request/response content requires explicit opt-in for privacy
  - Users must add `Channel.LLM` to their channels list to see raw LLM content
  - This is a **breaking change** for users who relied on debug/trace showing LLM payloads
  - Updated documentation to reflect opt-in behavior

### Migration

To restore previous behavior where debug/trace included LLM content:

```python
# Before (1.2.0): LLM content shown automatically
observer = Observer.debug()

# After (1.3.0): Explicitly opt-in to LLM content
observer = Observer(
    level=ObservabilityLevel.DEBUG,
    channels=[Channel.AGENTS, Channel.TOOLS, Channel.LLM, ...],
)
```

---

## [1.2.0] - 2026-01-03

### Added

#### Enhanced Observability Features

- **Token Usage Tracking**: Automatic tracking and display of LLM token consumption
  - Track input/output/total tokens per agent and globally
  - Display token counts in LLM response events: `[llm-response] (2.1s) ~850 tokens (650 in, 200 out)`
  - Detailed token breakdown in `observer.summary()` with per-agent statistics
  - Configurable via `track_tokens` and `show_token_usage` flags
  - Helps with cost monitoring, usage analytics, and budget tracking

- **Structured Event Export**: Export captured events to multiple formats for analysis
  - JSONL format: One event per line, ideal for streaming logs and log aggregation systems
  - JSON format: Complete event array with full structure for detailed analysis
  - CSV format: Tabular data perfect for spreadsheet analysis and reporting
  - Usage: `observer.export("trace.jsonl", format="jsonl|json|csv")`
  - Enables integration with monitoring systems, audit trails, and ML analysis

- **Progress Step Indicators**: Visual progress for multi-step agent operations
  - Show "Step N/M: description" during long-running workflows
  - Automatic tracking of current/total steps per agent
  - Configurable via `show_progress_steps` flag
  - Improves UX and helps identify bottlenecks in agent execution

- **Enhanced Error Context**: Actionable error messages with contextual suggestions
  - Smart pattern matching for common errors (permission denied, connection refused, timeout, etc.)
  - Automatic inclusion of file/line/tool context when available
  - Actionable suggestions displayed at DEBUG level
  - Supported patterns: permission, connection, timeout, not found, invalid credentials
  - Reduces debugging time and enables self-service problem resolution

- **State Change Diff Visualization**: Visual diffs for entity state changes
  - Shows `old_value → new_value` with color coding for AGENT_STATUS_CHANGED events
  - Tracks state snapshots per entity for comparison
  - Enabled at DETAILED level or higher
  - Ideal for reactive agents, task tracking, and debugging state transitions

### Changed

- **Observer Configuration**: Extended `ObserverConfig` with new settings
  - Added `track_tokens: bool = True` - Enable/disable token tracking
  - Added `show_token_usage: bool = True` - Display tokens in LLM events
  - Added `show_cost_estimates: bool = False` - Show estimated costs (future enhancement)
  - Added `show_progress_steps: bool = False` - Enable step progress indicators

- **Observer Internal State**: Enhanced tracking capabilities
  - Token usage tracking: `_token_usage` dict per agent, `_total_tokens` global
  - Progress tracking: `_progress_steps` dict for multi-step operations
  - State management: `_state_snapshots` dict for diff visualization
  - Error context: `_error_suggestions` dict with pattern-based recommendations

### Examples

- Added `examples/observability/enhanced_features.py` - Comprehensive demo of all new features
- Added `examples/observability/custom_truncation.py` - Configuration examples for truncation

### Documentation

- Enhanced docstrings for all new observer features
- Added inline examples for export functionality
- Documented token tracking configuration options

## [1.1.0] - Previous Release

### Added

- Professional observability formatting with bracket notation `[event-type]`
- Configurable truncation per content type (tool args, results, messages)
- Improved color scheme (grey labels, green success, blue tools)
- Increased max_iterations from 10 to 25 globally

### Changed

- Standardized all event output to professional bracket notation
- Separated completion status from output content
- Removed emoji-heavy output in favor of clean, professional format
- Enhanced visual hierarchy with consistent colors

### Fixed

- Agent name alignment with 12-character padding
- Duration formatting consistency across all events
- Truncation now respects word boundaries

---

## Version History

- **1.3.0** (2026-01-10) - LLM channel opt-in by default for privacy
- **1.2.0** (2026-01-03) - Enhanced observability with token tracking, export, and contextual features
- **1.1.0** (Previous) - Professional formatting and configuration improvements
