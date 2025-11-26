# API Reference

Complete API documentation for AgenticFlow. This reference covers all public modules, classes, and functions.

## Philosophy

AgenticFlow is built on top of LangChain and LangGraph. We ADD VALUE where it matters:
- **Agents**: Resilience, execution strategies, state management
- **Topologies**: Multi-agent coordination patterns
- **Events**: Pub/sub communication
- **Observability**: Progress tracking, metrics, tracing

For models, embeddings, vector stores, memory, graphs - **use LangChain/LangGraph directly**:

```python
# Models
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic

# Embeddings
from langchain_openai import OpenAIEmbeddings

# Vector stores
from langchain_core.vectorstores import InMemoryVectorStore

# Graphs
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
```

## Module Overview

| Module | Description |
|--------|-------------|
| [Agents](agents.md) | Agent creation, configuration, state management, and resilience |
| [Topologies](topologies.md) | Multi-agent orchestration patterns |
| [Observability](observability.md) | Progress tracking, metrics, and tracing |

---

## Quick Import Reference

```python
# Core imports
from agenticflow import (
    # Agents
    Agent,
    AgentConfig,
    AgentState,
    AgentRole,
    
    # Resilience
    RetryPolicy,
    RetryStrategy,
    CircuitBreaker,
    CircuitState,
    FallbackRegistry,
    ResilienceConfig,
    ToolResilience,
    
    # Execution strategies
    ExecutionStrategy,
    DAGExecutor,
    ReActExecutor,
    PlanExecutor,
    AdaptiveExecutor,
    
    # Events
    EventBus,
    Event,
    EventType,
    
    # Tasks
    Task,
    TaskManager,
    TaskStatus,
    Priority,
    
    # Topologies
    TopologyFactory,
    TopologyType,
    SupervisorTopology,
    MeshTopology,
    PipelineTopology,
    HierarchicalTopology,
    
    # Observability
    ProgressTracker,
    OutputConfig,
    Verbosity,
    OutputFormat,
    ProgressStyle,
    MetricsCollector,
    Tracer,
    
    # Tools
    ToolRegistry,
    create_tool_from_function,
    
    # Visualization
    MermaidConfig,
    AgentDiagram,
    TopologyDiagram,
)

# LangChain imports (use directly)
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_anthropic import ChatAnthropic
from langchain_core.vectorstores import InMemoryVectorStore

# LangGraph imports (use directly)
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.store.memory import InMemoryStore
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
from agenticflow.agent.resilience import (
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
