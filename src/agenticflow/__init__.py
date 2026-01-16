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
from agenticflow.agent.roles import (
    AutonomousRole,
    CustomRole,
    ReviewerRole,
    RoleConfig,
    SupervisorRole,
    WorkerRole,
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

# Core Events (event-driven primitives)
from agenticflow.events import (
    Event,
    EventBus,
    EventMatcher,
    EventStore,
    FileEventStore,
    InMemoryEventStore,
    create_event_store,
    matches,
)

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
    ExecutionContext,
    # Core Flow (unified event-driven orchestration)
    Flow,
    FlowConfig,
    # Context
    FlowContext,
    FlowProtocol,
    FlowResult,
)
from agenticflow.flow import (
    brainstorm as flow_brainstorm,
)
from agenticflow.flow import (
    chain as flow_chain,
)
from agenticflow.flow import (
    collaborative as flow_collaborative,
)
from agenticflow.flow import (
    coordinator as flow_coordinator,
)
from agenticflow.flow import (
    mesh as flow_mesh,
)
from agenticflow.flow import (
    # Patterns (imported with aliases to avoid shadowing by legacy topologies)
    pipeline as flow_pipeline,
)
from agenticflow.flow import (
    supervisor as flow_supervisor,
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
    Auditor,
    AuditTraceType,
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

# Middleware (cross-cutting concerns)
from agenticflow.middleware import (
    AggressiveTimeoutMiddleware,
    BaseMiddleware,
    LoggingMiddleware,
    Middleware,
    MiddlewareChain,
    RetryMiddleware,
    SimpleRetryMiddleware,
    SimpleTracingMiddleware,
    TimeoutMiddleware,
    TracingMiddleware,
    VerboseMiddleware,
)
from agenticflow.middleware import (
    Span as MiddlewareSpan,  # Renamed to avoid conflict with observability.Span
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
from agenticflow.observability.handlers import (
    ConsoleEventHandler,
    FileEventHandler,
    FilteringEventHandler,
    MetricsEventHandler,
)

# Models
from agenticflow.observability.trace_record import Trace, TraceType

# Reactive Flow (event-driven multi-agent orchestration)
from agenticflow.flow.reactive import (
    EventFlow,
    EventFlowConfig,
    EventFlowResult,
    ReactiveFlow,
    ReactiveFlowConfig,
    ReactiveFlowResult,
)
from agenticflow.flow.triggers import (
    AgentTriggerConfig,
    Trigger,
    on,  # Backward compat alias
    react_to,
    when,
)
from agenticflow.flow.patterns import (
    # Pipeline
    chain,
    pipeline,
    # Supervisor
    coordinator,
    supervisor,
    # Mesh
    brainstorm,
    collaborative,
    mesh,
)
from agenticflow.flow.skills import (
    Skill,
    SkillBuilder,
    skill,
)

# Reactors (event handlers for flows)
from agenticflow.reactors import (
    AgentReactor,
    Aggregator,
    BaseReactor,
    CallbackGateway,
    ConditionalRouter,
    ErrorPolicy,
    FanInMode,
    FirstWins,
    FunctionReactor,
    Gateway,
    HandoverStrategy,
    HttpGateway,
    LogGateway,
    MapTransform,
    Reactor,
    ReactorConfig,
    Transform,
    WaitAll,
    function_reactor,
    wrap_agent,
)
from agenticflow.reactors import (
    Router as EventRouter,  # Renamed to avoid conflict with reactive.Router
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
    # Event-Driven Flow (event orchestration)
    # Orchestrator
    "ReactiveFlow",
    "ReactiveFlowConfig",
    "ReactiveFlowResult",
    # Legacy aliases
    "EventFlow",
    "EventFlowConfig",
    "EventFlowResult",
    # Triggers
    "Trigger",
    "AgentTriggerConfig",
    "on",
    "react_to",
    "when",
    # Skills
    "Skill",
    "SkillBuilder",
    "skill",
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
    "FlowContext",
    "ExecutionContext",
    # Flow Patterns (recommended API)
    "pipeline",
    "supervisor",
    "mesh",
    "chain",
    "coordinator",
    "collaborative",
    "brainstorm",
    # Flow Patterns (aliased to avoid shadowing)
    "flow_brainstorm",
    "flow_chain",
    "flow_collaborative",
    "flow_coordinator",
    "flow_mesh",
    "flow_pipeline",
    "flow_supervisor",
    # Core Events
    "Event",
    "EventBus",
    "EventStore",
    "InMemoryEventStore",
    "FileEventStore",
    "create_event_store",
    "matches",
    "EventMatcher",
    # Reactors
    "Reactor",
    "ReactorConfig",
    "BaseReactor",
    "FunctionReactor",
    "function_reactor",
    "Aggregator",
    "FirstWins",
    "WaitAll",
    "EventRouter",
    "ConditionalRouter",
    "Transform",
    "MapTransform",
    "Gateway",
    "HttpGateway",
    "LogGateway",
    "CallbackGateway",
    "AgentReactor",
    "wrap_agent",
    "FanInMode",
    "HandoverStrategy",
    "ErrorPolicy",
    # Middleware
    "Middleware",
    "BaseMiddleware",
    "MiddlewareChain",
    "LoggingMiddleware",
    "VerboseMiddleware",
    "RetryMiddleware",
    "SimpleRetryMiddleware",
    "TimeoutMiddleware",
    "AggressiveTimeoutMiddleware",
    "TracingMiddleware",
    "SimpleTracingMiddleware",
    "MiddlewareSpan",
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
