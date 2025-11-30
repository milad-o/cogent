# AgenticFlow

A production-grade event-driven multi-agent system framework for building sophisticated AI applications.

**Native-first design**:
- **Native model wrappers** for OpenAI, Azure, Anthropic, Groq, Gemini, Ollama
- **Multi-agent topologies** (supervisor, mesh, pipeline, hierarchical)
- **Intelligent resilience** (retry, circuit breakers, fallbacks)
- **Advanced execution strategies** (DAG, ReAct, Plan-Execute)
- **Full observability** (tracing, metrics, progress tracking)
- **Event-driven architecture** with pub/sub patterns

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
from agenticflow import Agent, Flow, Observer
from agenticflow.models import ChatModel
from agenticflow.tools import tool

# Define tools using the native @tool decorator
@tool
def write_poem(subject: str) -> str:
    """Write a poem about a subject."""
    return f"A poem about {subject}..."

async def main():
    # Create a model
    model = ChatModel(model="gpt-4o")
    
    # Create an agent
    writer = Agent(
        name="Writer",
        model=model,
        tools=[write_poem],
        instructions="You are a creative writer.",
    )
    
    # Create a flow with observability
    flow = Flow(
        name="writing-team",
        agents=[writer],
        topology="pipeline",
        observer=Observer.verbose(),
    )
    
    # Run the flow
    result = await flow.run("Write a poem about the ocean")
    print(result)

asyncio.run(main())
```

## Native Models

AgenticFlow provides native model wrappers for all major providers:

```python
from agenticflow.models import ChatModel, create_chat, create_embedding
from agenticflow.models.azure import AzureChat, AzureEmbedding
from agenticflow.models.anthropic import AnthropicChat
from agenticflow.models.groq import GroqChat
from agenticflow.models.gemini import GeminiChat
from agenticflow.models.ollama import OllamaChat

# OpenAI (default)
model = ChatModel(model="gpt-4o")

# Azure OpenAI with Managed Identity
model = AzureChat(
    deployment="gpt-4o",
    azure_endpoint="https://my-resource.openai.azure.com",
    use_managed_identity=True,
)

# Anthropic
model = AnthropicChat(model="claude-sonnet-4-20250514")

# Groq (fast inference)
model = GroqChat(model="llama-3.1-70b-versatile")

# Gemini
model = GeminiChat(model="gemini-1.5-pro")

# Ollama (local)
model = OllamaChat(model="llama3.1")

# Factory function
model = create_chat("anthropic", model="claude-sonnet-4-20250514")
```

## Core Concepts

### Agent

An autonomous entity that can think, act, and communicate with intelligent resilience:

```python
from agenticflow import Agent
from agenticflow.models import ChatModel
from agenticflow.tools import tool

@tool
def analyze_data(dataset: str) -> str:
    """Analyze a dataset."""
    return f"Analysis of {dataset}..."

model = ChatModel(model="gpt-4o")

agent = Agent(
    name="Analyst",
    model=model,
    tools=[analyze_data],
    instructions="You are a data analyst. Provide clear insights.",
)

# Simple chat
response = await agent.chat("What's in the sales data?")

# Run with tools
result = await agent.run("Analyze the Q4 sales dataset")
```

### Multi-Agent Topologies

Coordinate multiple agents using different patterns:

```python
from agenticflow import Agent, Flow, Observer
from agenticflow.models import ChatModel

model = ChatModel(model="gpt-4o")

# Create specialized agents
researcher = Agent(name="Researcher", model=model, instructions="Research topics thoroughly.")
writer = Agent(name="Writer", model=model, instructions="Write clear, engaging content.")
editor = Agent(name="Editor", model=model, instructions="Review and improve content.")

# Pipeline: Sequential workflow
flow = Flow(
    name="content-team",
    agents=[researcher, writer, editor],
    topology="pipeline",
    observer=Observer.verbose(),
)

result = await flow.run("Create a blog post about AI trends")

# Supervisor: One agent delegates to others
flow = Flow(
    name="team",
    agents=[supervisor, researcher, writer],
    topology="supervisor",
    supervisor_name="supervisor",
)

# Mesh: All agents can communicate
flow = Flow(
    name="collaborative",
    agents=[agent1, agent2, agent3],
    topology="mesh",
)
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
from agenticflow.events import start_websocket_server

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
