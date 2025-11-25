# API Reference

Complete API documentation for AgenticFlow. This reference covers all public modules, classes, and functions.

## Module Overview

| Module | Description |
|--------|-------------|
| [Agents](agents.md) | Agent creation, configuration, state management, and resilience |
| [Providers](providers.md) | LLM and embedding model providers (OpenAI, Azure, Anthropic, etc.) |
| [Topologies](topologies.md) | Multi-agent orchestration patterns |
| [Events](events.md) | Event-driven communication between agents |
| [Tasks](tasks.md) | Task management and tracking |
| [Memory](memory.md) | Persistence, checkpointing, and vector stores |
| [Observability](observability.md) | Progress tracking, metrics, and tracing |
| [Graph](graph.md) | LangGraph integration and workflow building |
| [Tools](tools.md) | Tool registry and management |
| [Visualization](visualization.md) | Mermaid diagram generation |

---

## Quick Import Reference

```python
# Core imports
from agenticflow import (
    # Agents
    Agent,
    AgentConfig,
    AgentState,
    AgentExecutor,
    
    # Providers (new modular system)
    create_model,
    create_embeddings,
    ModelSpec,
    EmbeddingSpec,
    OpenAIProvider,
    AzureOpenAIProvider,
    AnthropicProvider,
    GoogleProvider,
    OllamaProvider,
    AzureAuthMethod,
    AzureConfig,
    Provider,
    
    # Resilience
    RetryPolicy,
    RetryStrategy,
    CircuitBreaker,
    CircuitState,
    FallbackRegistry,
    FailureMemory,
    ResilienceConfig,
    ToolResilience,
    
    # Events
    EventBus,
    Event,
    EventType,
    EventPriority,
    
    # Tasks
    Task,
    TaskManager,
    TaskStatus,
    TaskPriority,
    
    # Topologies
    SupervisorTopology,
    MeshTopology,
    PipelineTopology,
    HierarchicalTopology,
    CustomTopology,
    
    # Memory
    Checkpointer,
    MemoryCheckpointer,
    FileCheckpointer,
    SQLiteCheckpointer,
    Store,
    VectorStore,
    
    # Observability
    ProgressTracker,
    OutputConfig,
    Verbosity,
    OutputFormat,
    ProgressStyle,
    Theme,
    MetricsCollector,
    Tracer,
    
    # Graph
    GraphBuilder,
    GraphRunner,
    AgentNode,
    ToolNode,
    RouterNode,
    
    # Tools
    ToolRegistry,
    
    # Visualization
    MermaidGenerator,
)
```

---

## Core Enums

### ExecutionStrategy

```python
from agenticflow import ExecutionStrategy

class ExecutionStrategy(Enum):
    """Execution strategies for agent task processing."""
    REACT = "react"           # ReAct pattern: Reason + Act
    PLAN_EXECUTE = "plan"     # Plan then execute
    DAG = "dag"               # DAG-based execution
    ADAPTIVE = "adaptive"     # Adaptive strategy selection
```

### RetryStrategy

```python
from agenticflow import RetryStrategy

class RetryStrategy(Enum):
    """Retry backoff strategies."""
    NONE = "none"                         # No retry
    FIXED = "fixed"                       # Fixed delay
    LINEAR = "linear"                     # Linear backoff
    EXPONENTIAL = "exponential"           # Exponential backoff
    EXPONENTIAL_JITTER = "exponential_jitter"  # Exponential + jitter
```

### CircuitState

```python
from agenticflow import CircuitState

class CircuitState(Enum):
    """Circuit breaker states."""
    CLOSED = "closed"       # Normal operation
    OPEN = "open"           # Failing, reject calls
    HALF_OPEN = "half_open" # Testing recovery
```

### Verbosity

```python
from agenticflow import Verbosity

class Verbosity(Enum):
    """Output verbosity levels."""
    SILENT = 0    # No output
    MINIMAL = 1   # Errors only
    NORMAL = 2    # Standard output
    VERBOSE = 3   # Detailed output
    DEBUG = 4     # All information
```

### OutputFormat

```python
from agenticflow import OutputFormat

class OutputFormat(Enum):
    """Output format types."""
    RICH = "rich"     # Rich terminal output
    PLAIN = "plain"   # Plain text
    JSON = "json"     # JSON format
    MARKDOWN = "markdown"  # Markdown format
```

---

## Type Aliases

```python
from typing import Any, Callable, Coroutine

# Tool function types
ToolFunction = Callable[..., Any]
AsyncToolFunction = Callable[..., Coroutine[Any, Any, Any]]

# Event handler types
EventHandler = Callable[[Event], None]
AsyncEventHandler = Callable[[Event], Coroutine[Any, Any, None]]

# Message types
MessageDict = dict[str, Any]
Messages = list[MessageDict]
```

---

## Exception Classes

```python
from agenticflow.agents.resilience import (
    CircuitOpenError,      # Raised when circuit breaker is open
    MaxRetriesExceeded,    # Raised when all retries exhausted
    NoFallbackAvailable,   # Raised when no fallback tools available
)
```

---

## Version Information

```python
import agenticflow

print(agenticflow.__version__)  # "0.1.0"
```

---

## Next Steps

- **Getting Started**: See the [Quickstart Guide](../quickstart.md)
- **Agents**: Learn about [Agent Configuration](agents.md)
- **Topologies**: Explore [Multi-Agent Patterns](topologies.md)
- **Examples**: Check the [examples/](../../examples/) directory
