# Getting Started

Get up and running with AgenticFlow in minutes.

---

## Installation

```bash
# Install from GitHub
pip install git+https://github.com/milad-o/agenticflow.git

# Or with uv (recommended)
uv add git+https://github.com/milad-o/agenticflow.git

# With optional dependencies
pip install "agenticflow[all] @ git+https://github.com/milad-o/agenticflow.git"
```

**Optional dependencies:**

- `[web]` — Web search, browser automation
- `[anthropic]` — Anthropic Claude models
- `[azure]` — Azure OpenAI support
- `[gemini]` — Google Gemini models
- `[all]` — All optional dependencies

---

## Quick Start

### Your First Agent

```python
import asyncio
from agenticflow import Agent, tool
from agenticflow.models import ChatModel

@tool
def get_weather(city: str) -> str:
    """Get current weather for a city."""
    return f"Weather in {city}: 72°F, sunny"

async def main():
    agent = Agent(
        name="Assistant",
        model=ChatModel(model="gpt-4o-mini"),
        tools=[get_weather],
    )
    
    result = await agent.run("What's the weather in Tokyo?")
    print(result.output)

asyncio.run(main())
```

### With Capabilities

Instead of defining individual tools, use pre-built capabilities:

```python
from agenticflow import Agent
from agenticflow.models import ChatModel
from agenticflow.capabilities import FileSystem, WebSearch, CodeSandbox

agent = Agent(
    name="Assistant",
    model=ChatModel(model="gpt-4o"),
    capabilities=[
        FileSystem(allowed_paths=["./project"]),
        WebSearch(),
        CodeSandbox(timeout=30),
    ],
)

result = await agent.run("Search for Python tutorials and create a summary file")
```

---

## Core Concepts

### Agents

Autonomous entities that think, act, and communicate:

```python
agent = Agent(
    name="Researcher",
    model=ChatModel(model="gpt-4o"),
    instructions="You are a thorough researcher.",
    tools=[search, summarize],
    verbose=True,
)
```

### Tools

Define tools with the `@tool` decorator:

```python
from agenticflow import tool

@tool
def search(query: str, max_results: int = 10) -> str:
    """Search the web for information.
    
    Args:
        query: Search query string.
        max_results: Maximum results to return.
    """
    return f"Found {max_results} results for: {query}"

# Async tools supported
@tool
async def fetch_data(url: str) -> str:
    """Fetch data from URL."""
    async with httpx.AsyncClient() as client:
        response = await client.get(url)
        return response.text
```

**Tool features:**
- Automatic schema extraction from type hints
- Docstring → description
- Sync and async support
- Context injection via `ctx: RunContext`

### Multi-Agent Flow

Coordinate multiple agents with built-in patterns:

```python
from agenticflow.flow import pipeline, supervisor, mesh

researcher = Agent(name="Researcher", model=model, instructions="Research thoroughly.")
writer = Agent(name="Writer", model=model, instructions="Write clearly.")
editor = Agent(name="Editor", model=model, instructions="Review and polish.")

# Sequential processing
flow = pipeline([researcher, writer, editor])

result = await flow.run("Create a blog post about quantum computing")
```

**Patterns:**
- **Pipeline** — Sequential agent execution
- **Supervisor** — Leader agent delegates to workers
- **Mesh** — Agents communicate peer-to-peer

---

## Model Providers

AgenticFlow supports all major LLM providers with native SDK integrations:

```python
from agenticflow.models import (
    OpenAIChat,
    AnthropicChat,
    GeminiChat,
    GroqChat,
    OllamaChat,
)

# OpenAI
model = OpenAIChat(model="gpt-4o")

# Anthropic Claude
model = AnthropicChat(model="claude-sonnet-4-20250514")

# Google Gemini
model = GeminiChat(model="gemini-2.0-flash-exp")

# Groq (fast inference)
model = GroqChat(model="llama-3.3-70b-versatile")

# Ollama (local models)
model = OllamaChat(model="llama3.2")

# Or use factory function
from agenticflow.models import create_chat

model = create_chat("anthropic", model="claude-sonnet-4-20250514")
```

---

## Streaming

Stream responses token-by-token:

```python
agent = Agent(
    name="Writer",
    model=model,
    stream=True,
)

async for chunk in agent.run_stream("Write a poem about AI"):
    print(chunk.content, end="", flush=True)
```

---

## Memory & Context

Agents maintain conversation history automatically:

```python
agent = Agent(
    name="Assistant",
    model=model,
    memory_enabled=True,  # Default: True
)

# First interaction
await agent.run("My name is Alice")

# Later interaction - agent remembers
await agent.run("What's my name?")  # "Your name is Alice"

# Clear memory
agent.clear_memory()
```

---

## Observability

Track execution with built-in observability:

```python
from agenticflow.observability import Observer

# Pre-configured observers
observer = Observer.trace()      # Maximum detail
observer = Observer.verbose()    # Key events
observer = Observer.minimal()    # Errors only

flow = Flow(
    agents=[...],
    observer=observer,
)

# Access execution traces
for event in observer.events():
    print(f"{event.type}: {event.message}")
```

---

## Interceptors

Control execution with middleware:

```python
from agenticflow.interceptors import BudgetGuard, RateLimiter, PIIShield

agent = Agent(
    name="Assistant",
    model=model,
    interceptors=[
        BudgetGuard(max_tokens=10000, max_cost=0.50),  # Token/cost limits
        RateLimiter(max_requests_per_minute=60),        # Rate limiting
        PIIShield(redact=True),                         # PII protection
    ],
)
```

---

## Next Steps

- [Agent Documentation](agent.md) — Deep dive into agents
- [Multi-Agent Flow](flow.md) — Build coordinated systems
- [Capabilities](capabilities.md) — Explore built-in tools
- [Graph Visualization](graph.md) — Visualize your agents
- [RAG & Retrieval](retrievers.md) — Build RAG systems
- [Examples](https://github.com/milad-o/agenticflow/tree/main/examples) — See working examples

---

## Need Help?

- **Documentation**: https://milad-o.github.io/agenticflow
- **Examples**: https://github.com/milad-o/agenticflow/tree/main/examples
- **Issues**: https://github.com/milad-o/agenticflow/issues

---

## Capabilities & Extensions

| Module | Description |
|--------|-------------|
| [Capabilities](capabilities.md) | Plug-in capabilities (RAG, WebSearch, Browser, Filesystem, etc.) |
| [Interceptors](interceptors.md) | Middleware for security, budgets, PII, rate limiting |
| [Executors](executors.md) | Code execution environments (Python, Node.js, Shell) |

---

## Event-Driven Architecture

| Module | Description |
|--------|-------------|
| [Events](events.md) | Event types, EventBus, event-driven orchestration |
| [Flow](flow.md) | Event-driven reactive agent flows with patterns and reactors |
| [Reactors](reactors.md) | Reactor types and orchestration building blocks |
| [A2A Communication](a2a.md) | **Agent-to-agent delegation across all flow types** 🆕 |
| [Streaming](streaming.md) | Real-time token streaming from Flow executions |
| [Transport](transport.md) | Distributed event transport (LocalTransport, RedisTransport) |

---

## Observability & State

| Module | Description |
|--------|-------------|
| [Observability](observability.md) | Events, tracing, metrics, progress output, dashboards |
| [Memory](memory.md) | Persistent memory with scoping and semantic search |
| [Graph](graph.md) | Visualization for agents, patterns, and flows |

---

## Module Map

```
agenticflow/
├── agent/          # Core agent abstraction
├── capabilities/   # Plug-in capabilities (RAG, WebSearch, etc.)
├── core/           # Enums, message types, utilities
├── document/       # Document processing
├── executors/      # Code execution environments
├── flow/           # Event-driven orchestration + patterns
├── graph/          # Visualization
├── interceptors/   # Middleware (security, budgets, etc.)
├── memory/         # Persistent memory
├── models/         # LLM providers
├── observability/  # Events, tracing, metrics
├── retriever/      # Retrieval strategies
├── tasks/          # Task management
├── tools/          # Tool creation and registry
└── vectorstore/    # Vector storage and search
```

---

## Use Cases

### Simple Agent

```python
from agenticflow import Agent
from agenticflow.models import ChatModel

agent = Agent(
    name="assistant",
    model=ChatModel(model="gpt-4o"),
    instructions="You are a helpful assistant.",
)

result = await agent.run("What is Python?")
```

### Agent with Tools

```python
from agenticflow import Agent, tool
from agenticflow.models import ChatModel

@tool
def search(query: str) -> str:
    """Search the web."""
    return f"Results for: {query}"

agent = Agent(
    name="researcher",
    model=ChatModel(model="gpt-4o"),
    tools=[search],
)

result = await agent.run("Find info about AI trends")
```

### Multi-Agent Pipeline

```python
from agenticflow import Agent
from agenticflow.flow import pipeline

researcher = Agent(name="researcher", model=model)
writer = Agent(name="writer", model=model)
editor = Agent(name="editor", model=model)

flow = pipeline([researcher, writer, editor])

result = await flow.run("Create a blog post about quantum computing")
```

---

## License

MIT License. See [LICENSE](../LICENSE).
