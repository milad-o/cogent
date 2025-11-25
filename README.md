# AgenticFlow

A production-grade event-driven multi-agent system framework for building sophisticated AI applications.

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
from agenticflow import (
    Agent, AgentConfig, AgentRole,
    EventBus, TaskManager, Orchestrator,
    ToolRegistry, tool,
)

# Define a tool
@tool
def write_poem(subject: str) -> str:
    """Write a poem about a subject."""
    return f"A poem about {subject}..."

# Create the system
async def main():
    # Initialize components
    event_bus = EventBus()
    task_manager = TaskManager(event_bus)
    tool_registry = ToolRegistry().register(write_poem)
    
    # Create an agent
    writer = Agent(
        config=AgentConfig(
            name="Writer",
            role=AgentRole.WORKER,
            model="openai/gpt-4o",  # provider/model format
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

An autonomous entity that can think, act, and communicate:

```python
from agenticflow import Agent, AgentConfig, AgentRole

agent = Agent(
    config=AgentConfig(
        name="Analyst",
        role=AgentRole.SPECIALIST,
        description="Analyzes data and provides insights",
        model="openai/gpt-4o",  # provider/model format
        temperature=0.3,
        system_prompt="You are a data analyst...",
        tools=["analyze_data", "create_chart"],
    ),
    event_bus=event_bus,
    tool_registry=tool_registry,
)
```

### Model Providers

Flexible model specification supporting multiple providers:

```python
from agenticflow import AgentConfig
from agenticflow.providers import ModelSpec, AzureOpenAIProvider, AzureAuthMethod

# String format (simplest)
config = AgentConfig(name="Agent", model="openai/gpt-4o-mini")
config = AgentConfig(name="Agent", model="anthropic/claude-3-5-sonnet-latest")

# ModelSpec for full control
config = AgentConfig(
    name="PreciseAgent",
    model=ModelSpec(
        provider="openai",
        model="gpt-4o",
        temperature=0.3,
        max_tokens=2000,
    ),
)

# Azure with Managed Identity
provider = AzureOpenAIProvider(
    endpoint="https://my-resource.openai.azure.com",
    auth_method=AzureAuthMethod.MANAGED_IDENTITY,
)
config = AgentConfig(
    name="AzureAgent",
    model=provider.create_chat_model("gpt-4o-deployment"),
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
