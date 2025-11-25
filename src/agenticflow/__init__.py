# Suppress Pydantic V1 compatibility warning on Python 3.14+
# This is a LangChain internal issue, not our code
# Must be done BEFORE any langchain imports
import warnings

warnings.filterwarnings(
    "ignore",
    message="Core Pydantic V1 functionality",
    category=UserWarning,
)

"""
AgenticFlow - Event-Driven Multi-Agent System Framework
========================================================

A production-grade framework for building multi-agent systems with:
- Event-driven architecture with pub/sub patterns
- Hierarchical task management with dependencies
- Multi-agent topologies (supervisor, mesh, pipeline, hierarchical)
- Memory systems (short-term, long-term, shared)
- LangGraph integration with Command and interrupt
- Full observability (tracing, metrics, logging)
- Real-time WebSocket streaming

Quick Start:
    ```python
    from agenticflow import (
        Agent, AgentConfig, AgentRole,
        EventBus, TaskManager, Orchestrator,
        ToolRegistry,
    )
    
    # Create infrastructure
    event_bus = EventBus()
    task_manager = TaskManager(event_bus)
    tool_registry = ToolRegistry()
    
    # Create an agent
    agent = Agent(
        config=AgentConfig(
            name="Assistant",
            role=AgentRole.WORKER,
            model_name="gpt-4o",
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
    orchestrator.register_agent(agent)
    ```

Advanced Usage (Multi-Agent Topology):
    ```python
    from agenticflow.topologies import TopologyFactory, TopologyType
    from agenticflow.memory import memory_checkpointer, memory_store
    from agenticflow.observability import Tracer, MetricsCollector
    
    # Create topology with automatic coordination
    topology = TopologyFactory.create(
        TopologyType.SUPERVISOR,
        "content-team",
        agents=[supervisor, researcher, writer],
        supervisor_name="supervisor",
        checkpointer=memory_checkpointer(),  # short-term memory
        store=memory_store(),  # long-term memory
    )
    
    # Run with memory and observability
    result = await topology.run("Create a technical blog post")
    ```
"""

__version__ = "0.1.0"

# Core enums and utilities
from agenticflow.core.enums import (
    AgentRole,
    AgentStatus,
    EventType,
    Priority,
    TaskStatus,
)
from agenticflow.core.utils import generate_id, now_utc

# Model providers (new modular system)
from agenticflow.providers import (
    # Factory functions
    create_model,
    create_embeddings,
    acreate_model,
    acreate_embeddings,
    get_provider,
    parse_model_spec,
    list_providers,
    # Spec classes
    ModelSpec,
    EmbeddingSpec,
    # Base class
    BaseProvider,
    ProviderRegistry,
    # Provider implementations
    OpenAIProvider,
    AzureOpenAIProvider,
    AnthropicProvider,
    GoogleProvider,
    OllamaProvider,
    # Enums
    Provider,
    AzureAuthMethod,
    AzureConfig,
)

# Legacy provider aliases (deprecated, use providers module)
from agenticflow.core.providers import (
    create_chat_model,  # Use create_model instead
    openai_chat,
    azure_chat,
    anthropic_chat,
    ollama_chat,
    openai_embeddings,
    azure_embeddings,
)

# Models
from agenticflow.models.event import Event
from agenticflow.models.message import Message, MessageType
from agenticflow.models.task import Task

# Agents
from agenticflow.agents.base import Agent
from agenticflow.agents.config import AgentConfig
from agenticflow.agents.state import AgentState
from agenticflow.agents.executor import (
    ExecutionStrategy,
    ExecutionPlan,
    ToolCall,
    DAGExecutor,
    ReActExecutor,
    PlanExecutor,
    AdaptiveExecutor,
    create_executor,
)

# Events
from agenticflow.events.bus import EventBus, get_event_bus, set_event_bus
from agenticflow.events.handlers import (
    ConsoleEventHandler,
    FileEventHandler,
    FilteringEventHandler,
    MetricsEventHandler,
)

# Tasks
from agenticflow.tasks.manager import TaskManager

# Tools
from agenticflow.tools.registry import ToolRegistry, create_tool_from_function

# Orchestrator
from agenticflow.orchestrator.orchestrator import Orchestrator

# Memory (minimal wrappers around LangChain/LangGraph)
from agenticflow.memory import (
    # Checkpointers (short-term)
    create_checkpointer,
    memory_checkpointer,
    MemorySaver,
    BaseCheckpointSaver,
    # Stores (long-term)
    create_store,
    memory_store,
    semantic_store,
    BaseStore,
    InMemoryStore,
    # Vector stores (semantic)
    create_vectorstore,
    memory_vectorstore,
    VectorStore,
    InMemoryVectorStore,
    Document,
)

# Topologies
from agenticflow.topologies import (
    BaseTopology,
    TopologyConfig,
    TopologyState,
    SupervisorTopology,
    MeshTopology,
    PipelineTopology,
    HierarchicalTopology,
    TopologyFactory,
    TopologyType,
)

# Observability
from agenticflow.observability import (
    Tracer,
    Span,
    SpanContext,
    SpanKind,
    MetricsCollector,
    Counter,
    Gauge,
    Histogram,
    Timer,
    ObservabilityLogger,
    LogLevel,
    LogEntry,
    Dashboard,
    DashboardConfig,
    SystemInspector,
    AgentInspector,
    TaskInspector,
    EventInspector,
    # Progress & Output system
    OutputConfig,
    Verbosity,
    OutputFormat,
    ProgressStyle,
    ProgressTracker,
    ProgressEvent,
    Styler,
    Colors,
    Symbols,
    create_on_step_callback,
    create_executor_callback,
    configure_output,
    render_dag_ascii,
)

# Graph (LangGraph integration)
from agenticflow.graph import (
    GraphBuilder,
    NodeConfig,
    EdgeConfig,
    AgentGraphState,
    create_state_schema,
    merge_states,
    AgentNode,
    ToolNode,
    RouterNode,
    HumanNode,
    Handoff,
    HandoffType,
    create_handoff,
    create_interrupt,
    GraphRunner,
    RunConfig,
    StreamMode,
)

# Visualization (Mermaid diagrams)
from agenticflow.visualization import (
    MermaidConfig,
    MermaidRenderer,
    MermaidTheme,
    MermaidDirection,
    AgentDiagram,
    TopologyDiagram,
)

# All public exports
__all__ = [
    # Version
    "__version__",
    # Core enums
    "TaskStatus",
    "AgentStatus",
    "EventType",
    "Priority",
    "AgentRole",
    # Core utilities
    "generate_id",
    "now_utc",
    # Model providers (new system)
    "create_model",
    "create_embeddings",
    "acreate_model",
    "acreate_embeddings",
    "get_provider",
    "parse_model_spec",
    "list_providers",
    "ModelSpec",
    "EmbeddingSpec",
    "BaseProvider",
    "ProviderRegistry",
    "OpenAIProvider",
    "AzureOpenAIProvider",
    "AnthropicProvider",
    "GoogleProvider",
    "OllamaProvider",
    "Provider",
    "AzureAuthMethod",
    "AzureConfig",
    # Legacy aliases (deprecated)
    "create_chat_model",
    "openai_chat",
    "azure_chat",
    "anthropic_chat",
    "ollama_chat",
    "openai_embeddings",
    "azure_embeddings",
    # Models
    "Event",
    "Message",
    "MessageType",
    "Task",
    # Agents
    "Agent",
    "AgentConfig",
    "AgentState",
    # Execution strategies
    "ExecutionStrategy",
    "ExecutionPlan",
    "ToolCall",
    "DAGExecutor",
    "ReActExecutor",
    "PlanExecutor",
    "AdaptiveExecutor",
    "create_executor",
    # Events
    "EventBus",
    "get_event_bus",
    "set_event_bus",
    "ConsoleEventHandler",
    "FileEventHandler",
    "FilteringEventHandler",
    "MetricsEventHandler",
    # Tasks
    "TaskManager",
    # Tools
    "ToolRegistry",
    "create_tool_from_function",
    # Orchestrator
    "Orchestrator",
    # Memory (LangChain/LangGraph wrappers)
    "create_checkpointer",
    "memory_checkpointer",
    "MemorySaver",
    "BaseCheckpointSaver",
    "create_store",
    "memory_store",
    "semantic_store",
    "BaseStore",
    "InMemoryStore",
    "create_vectorstore",
    "memory_vectorstore",
    "VectorStore",
    "InMemoryVectorStore",
    "Document",
    # Topologies
    "BaseTopology",
    "TopologyConfig",
    "TopologyState",
    "SupervisorTopology",
    "MeshTopology",
    "PipelineTopology",
    "HierarchicalTopology",
    "TopologyFactory",
    "TopologyType",
    # Observability
    "Tracer",
    "Span",
    "SpanContext",
    "SpanKind",
    "MetricsCollector",
    "Counter",
    "Gauge",
    "Histogram",
    "Timer",
    "ObservabilityLogger",
    "LogLevel",
    "LogEntry",
    "Dashboard",
    "DashboardConfig",
    "SystemInspector",
    "AgentInspector",
    "TaskInspector",
    "EventInspector",
    # Progress & Output
    "OutputConfig",
    "Verbosity",
    "OutputFormat",
    "ProgressStyle",
    "ProgressTracker",
    "ProgressEvent",
    "Styler",
    "Colors",
    "Symbols",
    "create_on_step_callback",
    "create_executor_callback",
    "configure_output",
    "render_dag_ascii",
    # Graph (LangGraph integration)
    "GraphBuilder",
    "NodeConfig",
    "EdgeConfig",
    "AgentGraphState",
    "create_state_schema",
    "merge_states",
    "AgentNode",
    "ToolNode",
    "RouterNode",
    "HumanNode",
    "Handoff",
    "HandoffType",
    "create_handoff",
    "create_interrupt",
    "GraphRunner",
    "RunConfig",
    "StreamMode",
    # Visualization
    "MermaidConfig",
    "MermaidRenderer",
    "MermaidTheme",
    "MermaidDirection",
    "AgentDiagram",
    "TopologyDiagram",
]
