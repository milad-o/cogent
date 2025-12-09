# AgenticFlow Documentation

Comprehensive documentation for the AgenticFlow framework.

## Quick Start

```python
from agenticflow import Agent
from agenticflow.models import ChatModel

model = ChatModel(model="gpt-4o")
agent = Agent(name="assistant", model=model)

result = await agent.run("Hello!")
print(result.output)
```

---

## Core Modules

| Module | Description |
|--------|-------------|
| [Agent](agent.md) | Core agent abstraction - autonomous entities that think, act, and communicate |
| [Core](core.md) | Foundational types, enums, and utilities (TaskStatus, EventType, etc.) |
| [Models](models.md) | LLM providers (OpenAI, Azure, Anthropic, Groq, Gemini, Ollama) |
| [Tools](tools.md) | Tool creation, registry, and deferred execution |
| [Context](context.md) | Invocation-scoped dependency injection for tools and interceptors |

---

## Multi-Agent Coordination

| Module | Description |
|--------|-------------|
| [Flow](flow.md) | Main entry point for orchestrating multiple agents |
| [Topologies](topologies.md) | Coordination patterns (Supervisor, Pipeline, Mesh, Hierarchical) |
| [Tasks](tasks.md) | Hierarchical task tracking with lifecycle management |

---

## RAG & Retrieval

| Module | Description |
|--------|-------------|
| [Retrievers](retrievers.md) | Comprehensive retrieval strategies (Dense, BM25, Hybrid, HyDE, etc.) |
| [VectorStore](vectorstore.md) | Semantic search and vector storage (InMemory, FAISS, Chroma, Qdrant) |
| [Document](document.md) | Document processing, chunking, and loaders |
| [RAG Architecture](rag_architecture.md) | End-to-end RAG system design |

---

## Capabilities & Extensions

| Module | Description |
|--------|-------------|
| [Capabilities](capabilities.md) | Plug-in capabilities (RAG, WebSearch, Browser, Filesystem, etc.) |
| [Interceptors](interceptors.md) | Middleware for security, budgets, PII, rate limiting |
| [Executors](executors.md) | Code execution environments (Python, Node.js, Shell) |

---

## Observability & State

| Module | Description |
|--------|-------------|
| [Observability](observability.md) | Events, tracing, metrics, progress output, dashboards |
| [Memory](memory.md) | Persistent memory with scoping and semantic search |
| [Graph](graph.md) | Visualization for agents, topologies, and flows |

---

## Module Map

```
agenticflow/
├── agent/          # Core agent abstraction
├── capabilities/   # Plug-in capabilities (RAG, WebSearch, etc.)
├── core/           # Enums, message types, utilities
├── document/       # Document processing
├── executors/      # Code execution environments
├── flow.py         # Multi-agent orchestration
├── context.py      # RunContext for DI
├── graph/          # Visualization
├── interceptors/   # Middleware (security, budgets, etc.)
├── memory/         # Persistent memory
├── models/         # LLM providers
├── observability/  # Events, tracing, metrics
├── retriever/      # Retrieval strategies
├── tasks/          # Task management
├── tools/          # Tool creation and registry
├── topologies/     # Coordination patterns
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
from agenticflow import Agent, Flow

researcher = Agent(name="researcher", model=model)
writer = Agent(name="writer", model=model)
editor = Agent(name="editor", model=model)

flow = Flow(
    name="content-pipeline",
    agents=[researcher, writer, editor],
    topology="pipeline",
)

result = await flow.run("Create an article about quantum computing")
```

### Agent with Memory

```python
from agenticflow import Agent
from agenticflow.memory import Memory, SQLAlchemyStore

memory = Memory(store=SQLAlchemyStore("sqlite+aiosqlite:///agent.db"))

agent = Agent(
    name="assistant",
    model=model,
    memory=memory,
)

# First conversation
await agent.run("My name is Alice")

# Later conversation
await agent.run("What's my name?")  # Remembers "Alice"
```

### With Observability

```python
from agenticflow import Agent
from agenticflow.observability import Observer

observer = Observer.debug()

agent = Agent(
    name="assistant",
    model=model,
    verbose=True,  # Simple progress output
)

result = await agent.run("Complex task", observer=observer)
```

### With Interceptors

```python
from agenticflow import Agent
from agenticflow.interceptors import BudgetGuard, PIIShield

agent = Agent(
    name="assistant",
    model=model,
    intercept=[
        BudgetGuard(max_model_calls=10),
        PIIShield(patterns=["email", "ssn"]),
    ],
)
```

---

## Installation

```bash
# Core
pip install agenticflow

# With optional dependencies
pip install agenticflow[qdrant]     # Qdrant vector store
pip install agenticflow[chroma]     # ChromaDB vector store
pip install agenticflow[faiss]      # FAISS vector store
pip install agenticflow[all]        # All optional deps
```

---

## Environment Variables

| Variable | Description |
|----------|-------------|
| `OPENAI_API_KEY` | OpenAI API key |
| `ANTHROPIC_API_KEY` | Anthropic API key |
| `GROQ_API_KEY` | Groq API key |
| `GOOGLE_API_KEY` | Google Gemini API key |
| `AZURE_OPENAI_ENDPOINT` | Azure OpenAI endpoint |
| `AZURE_OPENAI_API_KEY` | Azure OpenAI API key |

---

## Contributing

See [CONTRIBUTING.md](../CONTRIBUTING.md) for development setup and guidelines.
