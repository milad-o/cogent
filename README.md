# AgenticFlow

A production-grade event-driven multi-agent system framework for building sophisticated AI applications.

Built on top of LangChain and LangGraph, AgenticFlow adds value where it matters:
- **Multi-agent topologies** (supervisor, mesh, pipeline, hierarchical)
- **Intelligent resilience** (retry, circuit breakers, fallbacks)
- **Advanced execution strategies** (DAG, ReAct, Plan-Execute)
- **Full observability** (tracing, metrics, progress tracking)
- **Event-driven architecture** with pub/sub patterns

**Philosophy**: USE LangChain/LangGraph DIRECTLY. We don't wrap what they do well.

## Overview

AgenticFlow provides a robust foundation for building multi-agent systems with:

- **Event-Driven Architecture**: Central event bus for decoupled communication
- **Hierarchical Tasks**: Parent/child task relationships with dependency management
- **Agent Orchestration**: Coordinate multiple specialized agents
- **Full Observability**: Complete audit trail with timestamps and correlation IDs
- **WebSocket Support**: Real-time event streaming for live dashboards
- **Extensible Tools**: Easy-to-register tool system for agent capabilities

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                         ORCHESTRATOR                                │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │  Plans → Delegates → Monitors → Aggregates                    │   │
│  └──────────────────────────────────────────────────────────────┘   │
│         │                    │                    │                 │
│         ▼                    ▼                    ▼                 │
│    ┌─────────┐          ┌─────────┐          ┌─────────┐           │
│    │ Agent A │          │ Agent B │          │ Agent C │           │
│    │ (Writer)│          │(Analyst)│          │(Critic) │           │
│    └────┬────┘          └────┬────┘          └────┬────┘           │
│         │                    │                    │                 │
│         └────────────────────┼────────────────────┘                 │
│                              ▼                                      │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │                      EVENT BUS                                │   │
│  │  Events: TaskCreated, AgentInvoked, ToolCalled, ...          │   │
│  └──────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────┘
```

## Installation

```bash
# Basic installation
uv add agenticflow

# With WebSocket support
uv add agenticflow[websocket]

# With FastAPI server
uv add agenticflow[api]

# Full installation
uv add agenticflow[all]

# Development
uv add agenticflow[dev]
```

## Quick Start

```python
import asyncio
from langchain_openai import ChatOpenAI  # Use LangChain directly for models
from agenticflow import (
    Agent, AgentConfig, AgentRole,
    EventBus, TaskManager, Orchestrator,
    ToolRegistry, create_tool_from_function,
)

# Create the system
async def main():
    # Initialize components
    event_bus = EventBus()
    task_manager = TaskManager(event_bus)
    
    # Create tools
    def write_poem(subject: str) -> str:
        """Write a poem about a subject."""
        return f"A poem about {subject}..."
    
    tool_registry = ToolRegistry()
    tool_registry.register(create_tool_from_function(write_poem))
    
    # Create a model using LangChain directly
    model = ChatOpenAI(model="gpt-4o")
    
    # Create an agent
    writer = Agent(
        config=AgentConfig(
            name="Writer",
            role=AgentRole.WORKER,
            model=model,  # Pass LangChain model directly
            tools=["write_poem"],
        ),
        event_bus=event_bus,
        tool_registry=tool_registry,
    )
    
    # Create orchestrator
    orchestrator = Orchestrator(
        event_bus=event_bus,
        task_manager=task_manager,
        tool_registry=tool_registry,
    )
    orchestrator.register_agent(writer)
    
    # Run a request
    result = await orchestrator.run("Write a poem about the ocean")
    print(result)

asyncio.run(main())
```

## Core Concepts

### Agent

An autonomous entity that can think, act, and communicate with intelligent resilience:

```python
from langchain_openai import ChatOpenAI
from agenticflow import Agent, AgentConfig, AgentRole

# Create model using LangChain directly
model = ChatOpenAI(model="gpt-4o", temperature=0.3)

agent = Agent(
    config=AgentConfig(
        name="Analyst",
        role=AgentRole.SPECIALIST,
        description="Analyzes data and provides insights",
        model=model,  # Pass LangChain model directly
        system_prompt="You are a data analyst...",
        tools=["analyze_data", "create_chart"],
    ),
    event_bus=event_bus,
    tool_registry=tool_registry,
)
```

### Using LangChain Directly

For models, embeddings, vector stores, memory - use LangChain/LangGraph directly:

```python
# Models
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_google_genai import ChatGoogleGenerativeAI

model = ChatOpenAI(model="gpt-4o")
model = ChatAnthropic(model="claude-3-5-sonnet-latest")

# Embeddings
from langchain_openai import OpenAIEmbeddings
embeddings = OpenAIEmbeddings()

# Vector stores
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_community.vectorstores import FAISS, Chroma

vectorstore = InMemoryVectorStore(embeddings)

# Document loading
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

docs = WebBaseLoader("https://example.com").load()
chunks = RecursiveCharacterTextSplitter(chunk_size=1000).split_documents(docs)

# Graphs and memory - use LangGraph directly
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.store.memory import InMemoryStore
```

### Task

A unit of work with lifecycle tracking:

```python
from agenticflow import Task, TaskStatus, Priority

task = Task(
    name="Analyze sales data",
    description="Review Q4 sales and identify trends",
    tool="analyze_data",
    args={"dataset": "sales_q4"},
    priority=Priority.HIGH,
)
```

### Event

An immutable record of system activity:

```python
from agenticflow import Event, EventType

# Events are automatically emitted by the system
# Subscribe to events for custom handling
event_bus.subscribe(EventType.TASK_COMPLETED, my_handler)
```

### EventBus

Central pub/sub system:

```python
from agenticflow import EventBus

event_bus = EventBus()

# Subscribe to specific events
event_bus.subscribe(EventType.AGENT_ERROR, handle_error)

# Subscribe to all events (logging, metrics)
event_bus.subscribe_all(log_all_events)
```

## Task Lifecycle

```
PENDING ──► SCHEDULED ──► RUNNING ──► COMPLETED
                │             │            │
                │             ▼            │
                │         SPAWNING ────────┤
                │         (subtasks)       │
                ▼             │            ▼
             BLOCKED ◄────────┘         FAILED
             (waiting)                  (error)
```

## Agent Lifecycle

```
IDLE ──► THINKING ──► ACTING ──► IDLE
           │            │
           ▼            ▼
        WAITING      ERROR
```

## WebSocket Server

Stream events in real-time:

```python
from agenticflow.server import start_websocket_server

# Start WebSocket server
server = await start_websocket_server(event_bus, port=8765)
```

Connect from JavaScript:

```javascript
const ws = new WebSocket('ws://localhost:8765');
ws.onmessage = (event) => {
    const data = JSON.parse(event.data);
    console.log('Event:', data);
};
```

## Project Structure

```
agenticflow/
├── src/agenticflow/
│   ├── __init__.py          # Main exports
│   ├── core/                 # Core types and utilities
│   │   ├── enums.py         # Status enums, event types
│   │   ├── types.py         # Type definitions
│   │   └── utils.py         # Helper functions
│   ├── models/               # Data models
│   │   ├── event.py         # Event class
│   │   ├── message.py       # Message class
│   │   └── task.py          # Task class
│   ├── agents/               # Agent system
│   │   ├── base.py          # Agent class
│   │   ├── config.py        # AgentConfig
│   │   └── state.py         # AgentState
│   ├── events/               # Event system
│   │   ├── bus.py           # EventBus
│   │   └── handlers.py      # Built-in handlers
│   ├── tasks/                # Task management
│   │   └── manager.py       # TaskManager
│   ├── tools/                # Tool system
│   │   ├── registry.py      # ToolRegistry
│   │   └── builtin.py       # Built-in tools
│   ├── orchestrator/         # Orchestration
│   │   └── orchestrator.py  # Orchestrator class
│   ├── server/               # External interfaces
│   │   ├── websocket.py     # WebSocket server
│   │   └── api.py           # FastAPI server
│   └── cli.py               # CLI entry point
├── tests/                    # Test suite
├── examples/                 # Example applications
├── pyproject.toml
└── README.md
```

## Examples

See the `examples/` directory for complete examples:

- `basic_agent.py` - Simple single-agent setup
- `multi_agent.py` - Multiple agents working together
- `websocket_demo.py` - Real-time event streaming
- `custom_tools.py` - Creating custom tools

## CLI

```bash
# Run the demo
agenticflow demo

# Start WebSocket server
agenticflow serve --websocket --port 8765

# Start API server
agenticflow serve --api --port 8000
```

## Development

```bash
# Install dev dependencies
uv sync --extra dev

# Run tests
uv run pytest

# Run with coverage
uv run pytest --cov=agenticflow

# Type checking
uv run mypy src/agenticflow

# Linting
uv run ruff check src/agenticflow
```

## License

MIT License - see LICENSE file for details.
