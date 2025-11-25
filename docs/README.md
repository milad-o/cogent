# AgenticFlow Documentation

Welcome to the AgenticFlow documentation. AgenticFlow is a production-grade framework for building multi-agent AI systems with LangChain and LangGraph.

## Documentation Index

### Getting Started
- [**Quickstart Guide**](quickstart.md) - Get up and running in 5 minutes
- [**Installation**](installation.md) - Installation options and requirements

### Core Concepts
- [**API Reference**](api-reference.md) - Complete API documentation
- [**Agents**](agents.md) - Creating and configuring agents
- [**Topologies**](topologies.md) - Multi-agent coordination patterns

### Advanced Features
- [**Memory Systems**](memory.md) - Short-term, long-term, and semantic memory
- [**Events & Streaming**](events.md) - Event-driven architecture and real-time updates
- [**Observability**](observability.md) - Tracing, metrics, and logging
- [**Visualization**](visualization.md) - Mermaid diagrams for topologies

### Examples
- [**Examples Directory**](../examples/) - Runnable example scripts

## Quick Links

```python
# Most common imports
from agenticflow import (
    # Agents
    Agent, AgentConfig, AgentRole,
    
    # Topologies  
    SupervisorTopology, PipelineTopology, MeshTopology,
    TopologyConfig,
    
    # Events
    EventBus,
    
    # Memory
    memory_checkpointer, memory_store,
)
```

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                         TOPOLOGY                                     │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │  Supervisor / Pipeline / Mesh / Hierarchical / Custom        │    │
│  └─────────────────────────────────────────────────────────────┘    │
│         │              │              │              │               │
│         ▼              ▼              ▼              ▼               │
│    ┌─────────┐   ┌─────────┐   ┌─────────┐   ┌─────────┐           │
│    │ Agent A │   │ Agent B │   │ Agent C │   │ Agent D │           │
│    └────┬────┘   └────┬────┘   └────┬────┘   └────┬────┘           │
│         │              │              │              │               │
│         └──────────────┴──────────────┴──────────────┘               │
│                              │                                       │
│                              ▼                                       │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │                      EVENT BUS                               │    │
│  │  Real-time events • Streaming • Progress updates             │    │
│  └─────────────────────────────────────────────────────────────┘    │
│                              │                                       │
│         ┌────────────────────┼────────────────────┐                  │
│         ▼                    ▼                    ▼                  │
│    ┌─────────┐         ┌─────────┐         ┌─────────┐              │
│    │ Memory  │         │Observa- │         │  Tools  │              │
│    │ System  │         │ bility  │         │Registry │              │
│    └─────────┘         └─────────┘         └─────────┘              │
└─────────────────────────────────────────────────────────────────────┘
```

## Feature Highlights

| Feature | Description |
|---------|-------------|
| **Multi-Agent Topologies** | Supervisor, Pipeline, Mesh, Hierarchical, Custom |
| **Real-time Streaming** | Progress callbacks and async generators |
| **Memory Systems** | Short-term (checkpointer), Long-term (store), Semantic (vector) |
| **Policy-Based Routing** | Define who can talk to whom with rules |
| **Observability** | Tracing, metrics, structured logging |
| **Mermaid Visualization** | Auto-generated topology diagrams |
| **LangGraph Integration** | Built on LangGraph for production reliability |

## License

MIT License - see [LICENSE](../LICENSE) for details.
