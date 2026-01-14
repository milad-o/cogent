# Streaming Reactions

Real-time token-by-token streaming from event-driven ReactiveFlow executions.

## Overview

Streaming reactions enable ReactiveFlow to yield output progressively as agents generate tokens, rather than waiting for complete responses. This provides:

- **Real-time feedback** during long-running agent operations
- **Better UX** with progressive output display
- **Lower perceived latency** by showing immediate progress
- **Agent tracking** to see which agent is currently active
- **Cancellation support** for in-flight operations

## Quick Start

```python
from agenticflow import Agent
from agenticflow.reactive import ReactiveFlow, react_to
from agenticflow.models import ChatModel

# Create agents with streaming-capable models
researcher = Agent(
    name="researcher",
    model=ChatModel(model="gpt-4o"),
    system_prompt="You research topics thoroughly.",
)

writer = Agent(
    name="writer",
    model=ChatModel(model="gpt-4o"),
    system_prompt="You write engaging content.",
)

# Create reactive flow
flow = ReactiveFlow()
flow.register(researcher, [react_to("task.created").emits("researcher.completed")])
flow.register(writer, [react_to("researcher.completed")])

# Stream execution - tokens arrive in real-time
async for chunk in flow.run_streaming("Research quantum computing"):
    print(f"[{chunk.agent_name}] {chunk.content}", end="", flush=True)
    
    if chunk.is_final:
        print()  # Newline after agent completes
```

## Core Concepts

### ReactiveStreamChunk

Each streaming chunk contains full context about the reactive flow execution:

```python
@dataclass
class ReactiveStreamChunk:
    agent_name: str           # Which agent is streaming
    event_id: str             # Event that triggered this agent
    event_name: str           # Type of triggering event
    content: str              # Token content
    delta: str                # Incremental text (same as content)
    is_final: bool           # Last chunk from this agent?
    finish_reason: str | None # Why stopped (stop, length, error)
    metadata: dict | None     # Additional context
```

**Key Properties**:
- `agent_name` — Identifies which agent in the flow is currently streaming
- `event_id` / `event_name` — Event context that triggered this agent
- `is_final` — True when agent completes (useful for UI formatting)
- `finish_reason` — "stop" (complete), "length" (max tokens), "error" (failed)

### run_streaming() vs run()

| Feature | `run()` | `run_streaming()` |
|---------|---------|-------------------|
| **Returns** | `ReactiveFlowResult` | `AsyncIterator[ReactiveStreamChunk]` |
| **Output** | Complete final output | Progressive tokens |
| **Latency** | Wait for completion | Immediate feedback |
| **Use Case** | Batch processing | Interactive UX |
| **Agent Execution** | Parallel (up to max_concurrent) | Sequential (preserves order) |

**When to use streaming**:
- Interactive applications (CLIs, web UIs, chatbots)
- Long-running multi-agent workflows
- Progress tracking and status updates
- User experience is priority

**When to use regular run()**:
- Batch processing or automation
- Final result is all that matters
- Parallel agent execution preferred
- Simpler code (no async iteration)

## Usage Patterns

### Basic Streaming

Simple single-agent streaming:

```python
from agenticflow.reactive import ReactiveFlow, react_to

agent = Agent(name="assistant", model=model)
flow = ReactiveFlow()
flow.register(agent, [react_to("task.created")])

# Stream tokens as they arrive
async for chunk in flow.run_streaming("Explain streaming"):
    print(chunk.content, end="", flush=True)
```

### Multi-Agent Streaming

Track which agent is speaking:

```python
current_agent = None

async for chunk in flow.run_streaming("Multi-step task"):
    # Detect agent transitions
    if chunk.agent_name != current_agent:
        if current_agent is not None:
            print()  # Newline before new agent
        print(f"\n[{chunk.agent_name}]:", end=" ")
        current_agent = chunk.agent_name
    
    print(chunk.content, end="", flush=True)
```

### Progress Indicators

Show pipeline progress:

```python
agents_completed = 0
total_agents = 3
current_agent = None

async for chunk in flow.run_streaming("Build web scraper"):
    # Update progress when agent changes
    if chunk.agent_name != current_agent:
        if current_agent is not None:
            agents_completed += 1
        
        current_agent = chunk.agent_name
        progress = f"[{agents_completed + 1}/{total_agents}]"
        print(f"\n{progress} {chunk.agent_name}:")
        print("-" * 50)
    
    print(chunk.content, end="", flush=True)
```

### Conditional Routing

Different agents stream based on event data:

```python
flow = ReactiveFlow()

# Python expert handles Python questions
flow.register(
    python_expert,
    [react_to("question.asked").when(lambda e: "python" in str(e.data).lower())],
)

# JavaScript expert handles JS questions
flow.register(
    js_expert,
    [react_to("question.asked").when(lambda e: "javascript" in str(e.data).lower())],
)

# Route based on question content
async for chunk in flow.run_streaming(
    "How do I use async/await in Python?",
    initial_event="question.asked",
    initial_data={"language": "python"},
):
    print(chunk.content, end="", flush=True)
```

### Error Handling

Gracefully handle errors during streaming:

```python
try:
    async for chunk in flow.run_streaming("Task with potential errors"):
        print(chunk.content, end="", flush=True)
        
        # Check for error metadata
        if chunk.metadata and chunk.metadata.get("error"):
            print(f"\n⚠️ Error: {chunk.metadata['error']}")
        
        if chunk.is_final:
            if chunk.finish_reason == "error":
                print("\n❌ Stream ended with error")
            else:
                print("\n✅ Stream completed")

except Exception as e:
    print(f"\n❌ Exception during streaming: {e}")
```

## Architecture

### How It Works

1. **Event Loop**: `run_streaming()` processes events like regular `run()`
2. **Agent Triggers**: Events trigger matching agents (same matching logic)
3. **Streaming Execution**: Agents execute with `stream=True` parameter
4. **Chunk Conversion**: Agent `StreamChunk` → `ReactiveStreamChunk` with event context
5. **Sequential Flow**: Agents execute sequentially to preserve output order
6. **Yield to Caller**: Each chunk is yielded immediately as it arrives

```
┌─────────────────────────────────────────────────────────────┐
│ ReactiveFlow.run_streaming("task")                          │
└─────────────────┬───────────────────────────────────────────┘
                  │
                  ▼
         ┌────────────────┐
         │ Event Loop     │ Processes events in rounds
         └────────┬───────┘
                  │
                  ▼
         ┌────────────────┐
         │ Match Agents   │ Find agents triggered by event
         └────────┬───────┘
                  │
                  ▼
         ┌─────────────────────────────────┐
         │ Execute Agent (stream=True)     │ Sequential execution
         └────────┬────────────────────────┘
                  │
                  ▼
         ┌──────────────────────────┐
         │ Agent Streams Tokens     │ token1, token2, token3...
         └────────┬─────────────────┘
                  │
                  ▼
         ┌──────────────────────────────────┐
         │ Convert to ReactiveStreamChunk   │ Add event context
         └────────┬─────────────────────────┘
                  │
                  ▼
         ┌──────────────────────────┐
         │ Yield to Caller          │ Immediate feedback
         └──────────────────────────┘
```

### Integration with Agent Streaming

ReactiveFlow streaming leverages the existing agent streaming infrastructure:

- **Agent.run(stream=True)** — Returns `AsyncIterator[StreamChunk]`
- **StreamChunk** — Token content from LLM provider
- **ReactiveStreamChunk** — Wraps StreamChunk with reactive flow context

This means:
- ✅ No duplicate streaming logic
- ✅ All LLM providers supported (OpenAI, Anthropic, etc.)
- ✅ Agent-level streaming configuration respected
- ✅ Consistent behavior with imperative Flow.stream()

## Performance Considerations

### Sequential vs Parallel Execution

**Regular `run()`** executes agents in parallel (up to `max_concurrent_agents`):
```python
# Agents run in parallel
result = await flow.run("task")  
# ⚡ Faster completion
# ❌ No real-time feedback
```

**`run_streaming()`** executes agents sequentially:
```python
# Agents run one at a time
async for chunk in flow.run_streaming("task"):
    print(chunk.content)
# ✅ Real-time feedback
# ⏱️ Slower completion (sequential)
```

**Trade-off**: Streaming sacrifices parallel speedup for user experience.

### Memory Usage

Streaming is more memory-efficient than batching:
- Regular `run()` accumulates full output in memory
- `run_streaming()` yields chunks immediately, allowing garbage collection

For very long outputs, streaming prevents memory buildup.

## Configuration

### Flow Configuration

Streaming respects `ReactiveFlowConfig` settings:

```python
from agenticflow.reactive import ReactiveFlowConfig

config = ReactiveFlowConfig(
    max_rounds=100,          # Maximum event processing rounds
    event_timeout=30.0,      # Timeout for waiting on events
    stop_on_idle=True,       # Stop when no more events
    stop_events=frozenset({"flow.completed"}),  # Events that end flow
)

flow = ReactiveFlow(config=config)

# Streaming obeys these limits
async for chunk in flow.run_streaming("task"):
    print(chunk.content)
```

### Agent Configuration

Enable streaming by default for specific agents:

```python
agent = Agent(
    name="assistant",
    model=model,
    stream=True,  # Always stream (even in non-streaming flows)
)

# This agent will stream in both run() and run_streaming()
```

## Examples

See [examples/reactive/streaming.py](../examples/reactive/streaming.py) for comprehensive demonstrations:

1. **Basic Streaming** — Simple token-by-token display
2. **Multi-Agent Streaming** — Track agent transitions
3. **Progress Indicators** — Pipeline progress visualization
4. **Conditional Streaming** — Event-driven routing
5. **Error Handling** — Graceful failure recovery

Run:
```bash
uv run python examples/reactive/streaming.py
```

## Testing

```python
import pytest
from agenticflow.reactive import ReactiveFlow, ReactiveStreamChunk

@pytest.mark.asyncio
async def test_streaming():
    flow = ReactiveFlow()
    flow.register(agent, [react_to("task.created")])
    
    chunks = []
    async for chunk in flow.run_streaming("Test"):
        assert isinstance(chunk, ReactiveStreamChunk)
        assert chunk.agent_name == "agent"
        chunks.append(chunk)
    
    assert len(chunks) > 1  # Multiple chunks received
```

See [tests/test_reactive_streaming.py](../tests/test_reactive_streaming.py) for full test suite (11 passing tests).

## API Reference

### ReactiveFlow.run_streaming()

```python
async def run_streaming(
    self,
    task: str,
    *,
    initial_event: str = "task.created",
    initial_data: dict[str, Any] | None = None,
    context: dict[str, Any] | None = None,
) -> AsyncIterator[ReactiveStreamChunk]:
    """
    Execute event-driven flow with streaming output.
    
    Args:
        task: The task/prompt to execute
        initial_event: Event type to emit at start
        initial_data: Additional data for initial event
        context: Shared context available to all agents
        
    Yields:
        ReactiveStreamChunk: Streaming chunks from agent executions
        
    Example:
        async for chunk in flow.run_streaming("Research topic"):
            print(f"[{chunk.agent_name}] {chunk.content}", end="")
    """
```

### ReactiveStreamChunk

```python
@dataclass
class ReactiveStreamChunk:
    """Streaming chunk from reactive agent execution."""
    
    agent_name: str
    """Name of the agent generating this chunk."""
    
    event_id: str
    """ID of the event that triggered this agent."""
    
    event_name: str
    """Name/type of the event that triggered this agent."""
    
    content: str
    """The text content of this streaming chunk."""
    
    delta: str
    """Incremental text added (same as content)."""
    
    is_final: bool = False
    """True if this is the last chunk from this reaction."""
    
    metadata: dict[str, Any] | None = None
    """Additional context about the streaming execution."""
    
    finish_reason: str | None = None
    """Reason streaming stopped (stop, length, tool_calls, error, etc.)."""
```

## Comparison with Imperative Flow Streaming

| Feature | ReactiveFlow.run_streaming() | Flow.stream() |
|---------|------------------------------|---------------|
| **Paradigm** | Event-driven | Imperative/Topology-based |
| **Chunk Type** | `ReactiveStreamChunk` | Status updates |
| **Agent Coordination** | Event triggers | Topology structure |
| **Event Context** | Full event metadata | Limited |
| **Use Case** | Reactive orchestration | Pipeline/Supervisor flows |

Both support real-time streaming, but ReactiveFlow provides richer event context.

## Best Practices

1. **Show agent names** — Help users understand which agent is active
2. **Handle transitions** — Add newlines/separators when agents change
3. **Progress indicators** — Show completion percentage for pipelines
4. **Error recovery** — Check `finish_reason` and `metadata` for errors
5. **Cancellation** — Use `asyncio.timeout()` to limit streaming duration
6. **Memory cleanup** — Process chunks immediately, don't accumulate all

```python
# ✅ Good - process immediately
async for chunk in flow.run_streaming("task"):
    await send_to_ui(chunk.content)

# ❌ Bad - accumulates memory
chunks = []
async for chunk in flow.run_streaming("task"):
    chunks.append(chunk)  # Memory grows
```

## Limitations

1. **Sequential execution** — Agents run one at a time in streaming mode
2. **No checkpointing** — Streaming flows don't support checkpoint saving (yet)
3. **Order dependency** — Agent order determined by event trigger order
4. **Token overhead** — More network roundtrips than batch mode

## Future Enhancements

- Parallel streaming with chunk interleaving
- Checkpoint support during streaming
- Stream pausing and resumption
- Custom chunk transformers/filters
- Streaming metrics and observability

## See Also

- [Reactive Flow Guide](reactive.md) — Event-driven orchestration
- [Agent Streaming](agent.md#streaming) — Agent-level streaming
- [Transport](transport.md) — Distributed event transport
- [Examples](../examples/reactive/) — Complete examples
