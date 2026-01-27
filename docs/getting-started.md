# Getting Started

Get up and running with Cogent in minutes.

---

## Installation

```bash
# Minimal installation (core only)
pip install git+https://github.com/milad-o/cogent.git

# Or with uv (recommended)
uv add git+https://github.com/milad-o/cogent.git
```

**Optional dependency groups:**

Choose what you need:

```bash
# Vector stores (FAISS, Qdrant)
uv add "cogent[vector-stores] @ git+https://github.com/milad-o/cogent.git"

# Retrieval (BM25, rerankers)
uv add "cogent[retrieval] @ git+https://github.com/milad-o/cogent.git"

# Database backends (SQLAlchemy + drivers)
uv add "cogent[database] @ git+https://github.com/milad-o/cogent.git"

# Infrastructure (Redis)
uv add "cogent[infrastructure] @ git+https://github.com/milad-o/cogent.git"

# Web tools (search, scraping)
uv add "cogent[web] @ git+https://github.com/milad-o/cogent.git"

# LLM providers (Anthropic, Azure, Cohere, Groq)
uv add "cogent[all-providers] @ git+https://github.com/milad-o/cogent.git"

# All backends
uv add "cogent[all-backend] @ git+https://github.com/milad-o/cogent.git"

# Everything
uv add "cogent[all] @ git+https://github.com/milad-o/cogent.git"
```

**Development installation:**

```bash
# Development + testing
uv add --dev cogent[dev,test,test-backends,docs]
```

---

## Quick Start

### Your First Agent

```python
import asyncio
from cogent import Agent, tool

@tool
def get_weather(city: str) -> str:
    """Get current weather for a city."""
    return f"Weather in {city}: 72°F, sunny"

async def main():
    # Simple string model (recommended)
    agent = Agent(
        name="Assistant",
        model="gpt4",  # Auto-resolves to gpt-4o
        tools=[get_weather],
    )
    
    result = await agent.run("What's the weather in Tokyo?")
    print(result.output)

asyncio.run(main())
```

**Other model options:**
```python
# With provider prefix
agent = Agent(name="Assistant", model="anthropic:claude")

# Medium-level: Factory function
from cogent.models import create_chat
agent = Agent(name="Assistant", model=create_chat("gpt4"))

# Low-level: Full control
from cogent.models import OpenAIChat
agent = Agent(name="Assistant", model=OpenAIChat(model="gpt-4o", temperature=0.7))
```

### With Capabilities

Instead of defining individual tools, use pre-built capabilities:

```python
from cogent import Agent
from cogent.capabilities import FileSystem, WebSearch, CodeSandbox

agent = Agent(
    name="Assistant",
    model="gpt4",  # Simple string model
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
    model="gpt4",  # String model - auto-resolves to gpt-4o
    instructions="You are a thorough researcher.",
    tools=[search, summarize],
    verbose=True,
)
```

### Tools

Define tools with the `@tool` decorator:

```python
from cogent import tool

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
from cogent.flow import pipeline, supervisor, mesh

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

Cogent supports all major LLM providers with native SDK integrations:

```python
from cogent.models import (
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
from cogent.models import create_chat

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
from cogent.observability import Observer

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
from cogent.interceptors import BudgetGuard, RateLimiter, PIIShield

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

Now that you know the basics, explore:

- **[Agent Documentation](agent.md)** — Deep dive into agents, instructions, and configuration
- **[Multi-Agent Flow](flow.md)** — Build coordinated multi-agent systems
- **[Capabilities](capabilities.md)** — Explore built-in tools and capabilities
- **[Graph Visualization](graph.md)** — Visualize your agents and flows
- **[RAG & Retrieval](retrievers.md)** — Build retrieval-augmented generation systems
- **[Examples](https://github.com/milad-o/cogent/tree/main/examples)** — See working examples

---

## Need Help?

- **Documentation**: https://milad-o.github.io/cogent
- **Examples**: https://github.com/milad-o/cogent/tree/main/examples
- **Issues**: https://github.com/milad-o/cogent/issues

