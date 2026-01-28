# Streaming

Real-time token-by-token streaming from agent executions.

## Overview

Streaming enables agents to yield output progressively as tokens are generated, rather than waiting for complete responses. This provides:

- **Real-time feedback** during long-running agent operations
- **Better UX** with progressive output display
- **Lower perceived latency** by showing immediate progress
- **Cancellation support** for in-flight operations

## Quick Start

```python
from cogent import Agent

agent = Agent(
    name="writer",
    model="gpt-4o",
    stream=True,  # Enable streaming
)

# Stream tokens as they arrive
async for chunk in agent.run_stream("Write a poem about AI"):
    print(chunk.content, end="", flush=True)
```

## Agent Streaming

### Enabling Streaming

Two ways to enable streaming:

```python
# Option 1: Set on agent creation
agent = Agent(
    name="writer",
    model="gpt-4o",
    stream=True,
)

result = await agent.run("Write about AI")  # Still works normally
async for chunk in agent.run_stream("Write a poem"):  # Stream tokens
    print(chunk.content, end="", flush=True)

# Option 2: Stream on demand (any agent)
agent = Agent(name="writer", model="gpt-4o")

async for chunk in agent.run_stream("Write a poem"):
    print(chunk.content, end="", flush=True)
```

### StreamChunk

Each streaming chunk contains:

```python
@dataclass
class StreamChunk:
    content: str              # Token content
    delta: str                # Incremental text (same as content)
    is_final: bool           # Last chunk?
    finish_reason: str | None # Why stopped (stop, length, tool_calls)
    metadata: dict | None     # Token usage, model info
```

**Key Properties**:
- `content` — The token text
- `delta` — Incremental text (alias for content)
- `is_final` — True when streaming completes
- `finish_reason` — "stop" (complete), "length" (max tokens), "tool_calls" (tool invocation)
- `metadata` — Includes token usage when available

### Streaming with Metadata

Access model metadata during streaming:

```python
async for chunk in agent.run_stream("Analyze data"):
    print(chunk.content, end="")
    
    if chunk.is_final and chunk.metadata:
        print(f"\nModel: {chunk.metadata.get('model')}")
        print(f"Tokens: {chunk.metadata.get('tokens')}")
```

---

## run_stream() vs run()

| Feature | `run()` | `run_stream()` |
|---------|---------|----------------|
| **Returns** | `Response` | `AsyncIterator[StreamChunk]` |
| **Output** | Complete final output | Progressive tokens |
| **Latency** | Wait for completion | Immediate feedback |
| **Use Case** | Batch processing | Interactive UX |

**When to use streaming**:
- Interactive applications (CLIs, web UIs, chatbots)
- Long-running agent operations
- Progress tracking and status updates
- User experience is priority

**When to use regular run()**:
- Batch processing or automation
- Final result is all that matters
- Simpler code (no async iteration)

---

## Usage Patterns

### Basic Streaming

```python
from cogent import Agent

agent = Agent(name="assistant", model="gpt-4o")

# Stream tokens as they arrive
async for chunk in agent.run_stream("Explain quantum computing"):
    print(chunk.content, end="", flush=True)

print()  # Newline at end
```

### Collecting Streamed Output

```python
full_response = []

async for chunk in agent.run_stream("Write a story"):
    full_response.append(chunk.content)
    print(chunk.content, end="", flush=True)

final_text = "".join(full_response)
```

### Progress Tracking

```python
async for chunk in agent.run_stream("Long analysis task"):
    print(chunk.content, end="", flush=True)
    
    if chunk.is_final:
        print("\n✅ Complete!")
```

### Error Handling

```python
try:
    async for chunk in agent.run_stream("Query"):
        print(chunk.content, end="", flush=True)
        
        if chunk.finish_reason == "length":
            print("\n⚠️ Response truncated (max tokens reached)")
except Exception as e:
    print(f"\n❌ Streaming error: {e}")
```

---

## Streaming with Tools

When an agent calls tools during streaming, the stream may pause while tools execute:

```python
from cogent import Agent, tool

@tool
def search(query: str) -> str:
    """Search for information."""
    return f"Results for: {query}"

agent = Agent(name="researcher", model="gpt-4o", tools=[search])

async for chunk in agent.run_stream("Search for AI news and summarize"):
    if chunk.content:
        print(chunk.content, end="", flush=True)
    
    if chunk.finish_reason == "tool_calls":
        print("\n[Tool calling...]")
```

---

## Web UI Integration

### FastAPI Streaming Endpoint

```python
from fastapi import FastAPI
from fastapi.responses import StreamingResponse

app = FastAPI()

@app.post("/chat/stream")
async def chat_stream(message: str):
    async def generate():
        async for chunk in agent.run_stream(message):
            yield chunk.content
    
    return StreamingResponse(generate(), media_type="text/plain")
```

### Server-Sent Events (SSE)

```python
from fastapi import FastAPI
from sse_starlette.sse import EventSourceResponse

@app.post("/chat/sse")
async def chat_sse(message: str):
    async def event_stream():
        async for chunk in agent.run_stream(message):
            yield {"event": "token", "data": chunk.content}
        yield {"event": "done", "data": ""}
    
    return EventSourceResponse(event_stream())
```

---

## Model Streaming Support

All major model providers support streaming:

| Provider | Streaming Support |
|----------|-------------------|
| OpenAI | ✅ Full support |
| Anthropic | ✅ Full support |
| Gemini | ✅ Full support |
| Groq | ✅ Full support |
| Ollama | ✅ Full support |
| Azure OpenAI | ✅ Full support |

---

## API Reference

### Agent.run_stream()

```python
async def run_stream(
    self,
    message: str,
    *,
    context: dict | None = None,
) -> AsyncIterator[StreamChunk]:
    """
    Stream agent response token-by-token.
    
    Args:
        message: The user message to process.
        context: Optional context dictionary.
        
    Yields:
        StreamChunk objects with progressive tokens.
    """
```

### StreamChunk

```python
@dataclass
class StreamChunk:
    content: str              # Token content
    delta: str                # Alias for content
    is_final: bool           # True on last chunk
    finish_reason: str | None # "stop", "length", "tool_calls"
    metadata: dict | None     # Model metadata
```

---

## Best Practices

1. **Use `end=""` and `flush=True`** — Ensure tokens display immediately
2. **Handle `is_final`** — Add newline or summary after completion
3. **Check `finish_reason`** — Detect truncation or tool calls
4. **Collect output** — Append chunks for final text if needed
5. **Error handling** — Wrap iteration in try/except for robustness
