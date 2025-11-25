# AgenticFlow Documentation

**AgenticFlow** is a production-grade event-driven multi-agent system framework for building sophisticated AI applications.

## Features

- 🤖 **Multi-Agent Orchestration** - Coordinate multiple specialized AI agents
- 🔄 **Event-Driven Architecture** - Decoupled pub/sub communication
- 🧠 **Intelligent Resilience** - Retry, circuit breakers, fallback tools
- 🔌 **Multi-Provider Support** - OpenAI, Azure (with Managed Identity), Anthropic, Google, Ollama
- 📊 **Multiple Topologies** - Supervisor, mesh, pipeline, hierarchical
- 💾 **Memory Systems** - Short-term, long-term, semantic search
- 🔍 **Full Observability** - Tracing, metrics, progress tracking
- ⚡ **Parallel Execution** - DAG-based parallel tool execution
- 📈 **Visualization** - Mermaid diagrams for agents and topologies

## Quick Navigation

| Section | Description |
|---------|-------------|
| [Quickstart](quickstart.md) | Get started in 5 minutes |
| [Providers Guide](api/providers.md) | Configure LLM providers (OpenAI, Azure, etc.) |
| [API Reference](api/index.md) | Complete API documentation |
| [Cookbook](cookbook.md) | Practical examples and patterns |
| [Examples](../examples/) | Runnable example scripts |

## Installation

```bash
# Basic installation
pip install agenticflow

# With all optional dependencies
pip install agenticflow[all]

# Using uv (recommended)
uv add agenticflow[all]
```

## Quick Example

```python
import asyncio
from agenticflow import (
    Agent, AgentConfig, AgentRole,
    EventBus, ToolRegistry,
    ResilienceConfig,
)

async def main():
    # Create infrastructure
    event_bus = EventBus()
    tool_registry = ToolRegistry()
    
    # Register a tool
    @tool_registry.register
    def search(query: str) -> str:
        """Search for information."""
        return f"Results for: {query}"
    
    # Create a resilient agent with provider/model format
    agent = Agent(
        config=AgentConfig(
            name="Researcher",
            role=AgentRole.WORKER,
            model="openai/gpt-4o",  # provider/model string format
            tools=["search"],
            resilience_config=ResilienceConfig.aggressive(),
        ),
        event_bus=event_bus,
        tool_registry=tool_registry,
    )
    
    # Execute with automatic retry and recovery
    result = await agent.act("search", {"query": "Python async"})
    print(result)

asyncio.run(main())
```

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                         ORCHESTRATOR                                │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │  Plans → Delegates → Monitors → Aggregates                    │   │
│  └──────────────────────────────────────────────────────────────┘   │
│         │                    │                    │                 │
│         ▼                    ▼                    ▼                 │
│    ┌─────────┐          ┌─────────┐          ┌─────────┐           │
│    │ Agent A │◄────────►│ Agent B │◄────────►│ Agent C │           │
│    │ (Writer)│          │(Analyst)│          │(Critic) │           │
│    └────┬────┘          └────┬────┘          └────┬────┘           │
│         │    RESILIENCE      │                    │                 │
│         │    ┌───────────────┴───────────────┐    │                 │
│         │    │ Retry │ Circuit │ Fallback    │    │                 │
│         │    └───────────────────────────────┘    │                 │
│         └────────────────────┼────────────────────┘                 │
│                              ▼                                      │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │                      EVENT BUS                                │   │
│  │  Events: TaskCreated, AgentInvoked, ToolCalled, ...          │   │
│  └──────────────────────────────────────────────────────────────┘   │
│                              │                                      │
│  ┌───────────────────────────┼───────────────────────────────────┐  │
│  │              OBSERVABILITY LAYER                               │  │
│  │  Progress │ Metrics │ Tracing │ Dashboard                     │  │
│  └───────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────┘
```

## Core Concepts

### Agents
Autonomous entities that think, act, and communicate. Each agent has:
- A unique identity and role
- LLM model for reasoning (supports OpenAI, Azure, Anthropic, Google, Ollama)
- Access to tools
- Resilience configuration for retry/recovery

### Events
Immutable records of system activity. The event bus provides:
- Pub/sub pattern for decoupled communication
- Event history with filtering
- Real-time streaming via WebSocket

### Tasks
Units of work with lifecycle tracking:
- Priority-based scheduling
- Parent/child relationships
- Dependency management

### Topologies
Pre-built coordination patterns:
- **Supervisor**: One agent delegates to workers
- **Mesh**: All agents communicate freely
- **Pipeline**: Sequential processing stages
- **Hierarchical**: Multi-level organization

### Resilience
Production-grade fault tolerance:
- **Retry**: Exponential backoff with jitter
- **Circuit Breaker**: Prevent cascading failures
- **Fallback**: Graceful degradation to alternatives
- **Learning**: Adapt based on failure patterns

## Version

Current version: **0.1.0**

## License

MIT License - see [LICENSE](../LICENSE) for details.
