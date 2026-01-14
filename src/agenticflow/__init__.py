# ruff: noqa: E402

# Suppress Pydantic V1 compatibility warning on Python 3.14+
import warnings

warnings.filterwarnings(
    "ignore",
    message="Core Pydantic V1 functionality",
    category=UserWarning,
)

"""
AgenticFlow - Event-Driven Multi-Agent System Framework
========================================================

A lightweight, native multi-agent framework with:
- Multi-agent topologies (supervisor, mesh, pipeline, hierarchical)
- Intelligent resilience (retry, circuit breakers, fallbacks)
- Native model support for OpenAI, Azure, Anthropic, Groq, Gemini, Ollama
- Full observability (tracing, metrics, progress tracking)
- Event-driven architecture with pub/sub patterns
- Mermaid visualization for agents and topologies

Quick Start (Standalone Execution):
    ```python
    from agenticflow import run, tool

    @tool
    def search(query: str) -> str:
        '''Search the web.'''
        return f"Results for {query}"

    # Execute a task with tools - no Agent class needed!
    result = await run(
        "Search for Python tutorials",
        tools=[search],
        model="gpt-4o-mini",
    )
    ```

Native Models (Recommended):
    ```python
    from agenticflow.models.openai import OpenAIChat
    from agenticflow.models.azure import AzureOpenAIChat
    from agenticflow.models.anthropic import AnthropicChat
    from agenticflow.models import create_chat, create_embedding

    # Direct usage
    llm = OpenAIChat(model="gpt-4o")
    response = await llm.ainvoke([{"role": "user", "content": "Hello!"}])

    # Factory function
    llm = create_chat("anthropic", model="claude-sonnet-4-20250514")

    # Azure with Managed Identity (Entra ID)
    from agenticflow.models.azure import AzureEntraAuth
    llm = AzureOpenAIChat(
        deployment="gpt-4o",
        azure_endpoint="https://my-resource.openai.azure.com",
        entra=AzureEntraAuth(method="managed_identity"),
    )
    ```

With Agent Class:
    ```python
    from agenticflow import Agent, AgentConfig, EventBus
    from agenticflow.models.openai import OpenAIChat

    agent = Agent(
        config=AgentConfig(
            name="Assistant",
            model=OpenAIChat(model="gpt-4o"),
        ),
        event_bus=TraceBus(),
    )

    # Think with automatic retry on failures
    response = await agent.think("What should I do?")

    # Execute complex tasks
    result = await agent.run("Search and analyze data")
    ```

Multi-Agent Topology:
    ```python
    from agenticflow import TopologyFactory, TopologyType

    # Create a supervisor topology
    topology = TopologyFactory.create(
        TopologyType.SUPERVISOR,
        "team",
        agents=[supervisor, researcher, writer],
        supervisor_name="supervisor",
    )

    # Run with progress tracking
    result = await topology.run("Create a blog post")
    ```
"""

__version__ = "1.4.0"

# Core enums and utilities
# Graph API (unified visualization)
from agenticflow import graph

# Agents (THIS IS WHERE WE ADD VALUE)
from agenticflow.agent.base import Agent
from agenticflow.agent.config import AgentConfig
from agenticflow.agent.roles import (
    AutonomousRole,
    CustomRole,
    ReviewerRole,
    RoleConfig,
    SupervisorRole,
    WorkerRole,
)
from agenticflow.agent.hitl import (
    AbortedException,
    DecisionRequiredException,
    DecisionType,
    GuidanceResult,
    HumanDecision,
    HumanResponse,
    InterruptedException,
    InterruptedState,
    InterruptReason,
    PendingAction,
    should_interrupt,
)
from agenticflow.agent.memory import (
    AgentMemory,
    InMemoryCheckpointer,  # Backward compat alias
    InMemorySaver,
    MemoryCheckpoint,  # Backward compat alias
    MemorySnapshot,
    ThreadConfig,
)
from agenticflow.agent.output import (
    OutputMethod,
    ResponseSchema,
    StructuredResult,
)
from agenticflow.agent.resilience import (
    CircuitBreaker,
    CircuitState,
    FallbackRegistry,
    ResilienceConfig,
    RetryPolicy,
    RetryStrategy,
    ToolResilience,
)
from agenticflow.agent.state import AgentState
from agenticflow.agent.streaming import (
    CollectorStreamCallback,
    PrintStreamCallback,
    StreamCallback,
    StreamChunk,
    StreamConfig,
    StreamEvent,
    StreamTraceType,
    ToolCallChunk,
    chunk_from_message,
    collect_stream,
    extract_tool_calls,
    print_stream,
)

# Context - invocation-scoped data
from agenticflow.context import EMPTY_CONTEXT, RunContext
from agenticflow.core.enums import (
    AgentRole,
    AgentStatus,
    Priority,
    TaskStatus,
    get_role_capabilities,
)
from agenticflow.core.utils import generate_id, now_utc

# Core orchestration events (separate from observability)
# Graphs - Execution strategies
from agenticflow.executors import (
    ExecutionPlan,
    ExecutionStrategy,
    NativeExecutor,
    SequentialExecutor,
    ToolCall,
    TreeSearchExecutor,
    create_executor,
    run,
)

# Flow - THE MAIN ENTRY POINT
from agenticflow.flow import (
    # Base classes
    BaseFlow,
    # Imperative Flow
    Flow,
    FlowConfig,
    FlowProtocol,
    FlowResult,
    create_flow,
    mesh_flow,
    pipeline_flow,
    supervisor_flow,
)
from agenticflow.graph import (
    GraphConfig,
    GraphDirection,
    GraphTheme,
    GraphView,
)

# Interceptors (execution flow control)
from agenticflow.interceptors import (
    AuditEvent,
    AuditTraceType,
    Auditor,
    BudgetGuard,
    ContentFilter,
    ContextCompressor,
    ContextPrompt,
    ConversationGate,
    ConversationPrompt,
    Failover,
    InterceptContext,
    Interceptor,
    InterceptResult,
    LambdaPrompt,
    PermissionGate,
    Phase,
    PIIAction,
    PIIShield,
    PromptAdapter,
    RateLimiter,
    StopExecution,
    ThrottleInterceptor,
    TokenLimiter,
    # Context Layer
    ToolGate,
    ToolGuard,
    run_interceptors,
)

# LLM & Embedding Models (native)
from agenticflow.models import (
    ChatModel,
    EmbeddingModel,
    create_chat,
    create_embedding,
)

# Provider-specific model imports for convenience
from agenticflow.models.azure import AzureOpenAIChat, AzureOpenAIEmbedding

# Observability (THIS IS WHERE WE ADD VALUE)
from agenticflow.observability import (
    AgentInspector,
    Channel,
    Colors,
    Counter,
    Dashboard,
    DashboardConfig,
    EventInspector,
    Gauge,
    Histogram,
    LogEntry,
    LogLevel,
    MetricsCollector,
    ObservabilityLevel,
    ObservabilityLogger,
    # Observer (unified observability for agents, flows, teams)
    Observer,
    # Progress & Output system
    OutputConfig,
    OutputFormat,
    ProgressEvent,
    ProgressStyle,
    ProgressTracker,
    Span,
    SpanContext,
    SpanKind,
    Styler,
    Symbols,
    SystemInspector,
    TaskInspector,
    Timer,
    Tracer,
    Verbosity,
    configure_output,
    create_executor_callback,
    create_on_step_callback,
    render_dag_ascii,
)

# Events (THIS IS WHERE WE ADD VALUE)
from agenticflow.observability.bus import TraceBus, get_trace_bus, set_trace_bus

# Models
from agenticflow.observability.trace_record import Trace, TraceType
from agenticflow.observability.handlers import (
    ConsoleEventHandler,
    FileEventHandler,
    FilteringEventHandler,
    MetricsEventHandler,
)

# Reactive Flow (event-driven multi-agent orchestration)
from agenticflow.reactive import (
    AgentTriggerConfig,
    # Mid-level API
    Chain,
    # Low-level API
    EventFlow,
    EventFlowConfig,
    EventFlowResult,
    FanIn,
    FanOut,
    # New names
    ReactiveFlow,
    ReactiveFlowConfig,
    ReactiveFlowResult,
    Router,
    Saga,
    Trigger,
    # High-level API (recommended)
    chain,
    fanout,
    on,  # Backward compat alias
    route,
    when,
)

# Tasks
from agenticflow.tasks.manager import TaskManager
from agenticflow.tasks.task import Task
from agenticflow.tools.base import BaseTool, tool
from agenticflow.tools.deferred import (
    DeferredManager,
    DeferredResult,
    DeferredRetry,
    DeferredStatus,
    is_deferred,
)

# Tools (THIS IS WHERE WE ADD VALUE)
from agenticflow.tools.registry import ToolRegistry, create_tool_from_function

# Topologies (simple coordination patterns)
from agenticflow.topologies import (
    # Core classes
    AgentConfig as TopologyAgentConfig,
)
from agenticflow.topologies import (
    BaseTopology,
    Hierarchical,
    Mesh,
    Pipeline,
    # Pattern classes
    Supervisor,
    TopologyResult,
    TopologyType,
    mesh,
    pipeline,
    # Convenience functions
    supervisor,
)

# Backwards-compatible aliases for visualization module
MermaidConfig = GraphConfig
MermaidTheme = GraphTheme
MermaidDirection = GraphDirection
MermaidRenderer = GraphView  # Closest equivalent
AgentDiagram = GraphView  # Use GraphView.from_agent() instead
TopologyDiagram = GraphView  # Use GraphView.from_topology() instead

# Capabilities (composable tools for agents)
# Document processing module
from agenticflow import document
from agenticflow.capabilities import (
    BaseCapability,
    KnowledgeGraph,
)

# Native message types (from core.messages)
from agenticflow.core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)

# All public exports
__all__ = [
    # Version
    "__version__",
    # Core enums
    "TaskStatus",
    "AgentStatus",
    "Trace",
    "TraceType",
    "Priority",
    "AgentRole",
    "get_role_capabilities",
    # Core utilities
    "generate_id",
    "now_utc",
    # Models
    "Task",
    # Agents
    "Agent",
    "AgentConfig",
    "AgentState",
    # Roles
    "RoleConfig",
    "SupervisorRole",
    "WorkerRole",
    "ReviewerRole",
    "AutonomousRole",
    "CustomRole",
    # Memory
    "AgentMemory",
    "MemorySnapshot",
    "MemoryCheckpoint",  # Backward compat alias
    "ThreadConfig",
    "InMemorySaver",
    "InMemoryCheckpointer",  # Backward compat alias
    # Execution strategies
    "ExecutionStrategy",
    "ExecutionPlan",
    "ToolCall",
    "NativeExecutor",
    "SequentialExecutor",
    "TreeSearchExecutor",
    "run",
    "create_executor",
    # Resilience
    "ResilienceConfig",
    "RetryPolicy",
    "RetryStrategy",
    "CircuitBreaker",
    "CircuitState",
    "ToolResilience",
    "FallbackRegistry",
    # Human-in-the-Loop
    "InterruptReason",
    "DecisionType",
    "PendingAction",
    "HumanDecision",
    "InterruptedState",
    "InterruptedException",
    "DecisionRequiredException",
    "AbortedException",
    "GuidanceResult",
    "HumanResponse",
    "should_interrupt",
    # Streaming
    "StreamChunk",
    "StreamEvent",
    "StreamTraceType",
    "StreamConfig",
    "StreamCallback",
    "PrintStreamCallback",
    "CollectorStreamCallback",
    "ToolCallChunk",
    "chunk_from_message",
    "extract_tool_calls",
    "collect_stream",
    "print_stream",
    # Structured Output
    "ResponseSchema",
    "OutputMethod",
    "StructuredResult",
    # Interceptors
    "Interceptor",
    "Phase",
    "InterceptContext",
    "InterceptResult",
    "BudgetGuard",
    "StopExecution",
    "run_interceptors",
    "ContextCompressor",
    "TokenLimiter",
    "PIIAction",
    "PIIShield",
    "ContentFilter",
    "RateLimiter",
    "ThrottleInterceptor",
    "Auditor",
    "AuditEvent",
    "AuditTraceType",
    # Context Layer
    "ToolGate",
    "PermissionGate",
    "ConversationGate",
    "Failover",
    "ToolGuard",
    "PromptAdapter",
    "ContextPrompt",
    "ConversationPrompt",
    "LambdaPrompt",
    # Context
    "RunContext",
    "EMPTY_CONTEXT",
    # Events
    "TraceBus",
    "get_trace_bus",
    "set_trace_bus",
    "ConsoleEventHandler",
    "FileEventHandler",
    "FilteringEventHandler",
    "MetricsEventHandler",
    # Event-Driven Flow (reactive orchestration)
    # High-level API
    "chain",
    "fanout",
    "route",
    # Mid-level API
    "Chain",
    "FanIn",
    "FanOut",
    "Router",
    "Saga",
    # Low-level API (new names)
    "ReactiveFlow",
    "ReactiveFlowConfig",
    "ReactiveFlowResult",
    # Low-level API (legacy aliases)
    "EventFlow",
    "EventFlowConfig",
    "EventFlowResult",
    "Trigger",
    "AgentTriggerConfig",
    "on",
    "when",
    # Tasks
    "TaskManager",
    # Tools
    "ToolRegistry",
    "create_tool_from_function",
    "BaseTool",
    "tool",
    # Deferred Tools (event-driven completion)
    "DeferredResult",
    "DeferredStatus",
    "DeferredManager",
    "DeferredRetry",
    "is_deferred",
    # Flow (MAIN ENTRY POINT)
    "BaseFlow",
    "FlowProtocol",
    "FlowResult",
    "Flow",
    "FlowConfig",
    "create_flow",
    "supervisor_flow",
    "pipeline_flow",
    "mesh_flow",
    # Topologies (coordination patterns)
    "TopologyAgentConfig",
    "BaseTopology",
    "TopologyResult",
    "TopologyType",
    "Supervisor",
    "Pipeline",
    "Mesh",
    "Hierarchical",
    "supervisor",
    "pipeline",
    "mesh",
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
    # Flow Observer (integrated observability)
    "Observer",
    "ObservabilityLevel",
    "Channel",
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
    # Graph API (visualization)
    "graph",
    "GraphView",
    "GraphConfig",
    "GraphTheme",
    "GraphDirection",
    # Backwards-compatible visualization aliases
    "MermaidConfig",
    "MermaidRenderer",
    "MermaidTheme",
    "MermaidDirection",
    "AgentDiagram",
    "TopologyDiagram",
    # Capabilities
    "BaseCapability",
    "KnowledgeGraph",
    # LLM & Embedding Models (native)
    "ChatModel",
    "EmbeddingModel",
    "AzureOpenAIChat",
    "AzureOpenAIEmbedding",
    "create_chat",
    "create_embedding",
    # Native message types
    "BaseMessage",
    "HumanMessage",
    "AIMessage",
    "SystemMessage",
    "ToolMessage",
    # Document processing
    "document",
]
