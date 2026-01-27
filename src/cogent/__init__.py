
# Suppress Pydantic V1 compatibility warning on Python 3.14+
import warnings

warnings.filterwarnings(
    "ignore",
    message="Core Pydantic V1 functionality",
    category=UserWarning,
)

"""
Cogent - Production Agent Framework
=========================================

A lightweight, single-agent-first framework with:
- Tool-augmented agent execution (research shows tools > multi-agent)
- Intelligent resilience (retry, circuit breakers, fallbacks)
- Native model support for OpenAI, Azure, Anthropic, Groq, Gemini, Ollama
- Full observability (tracing, metrics, progress tracking)
- Tactical multi-agent via Agent.as_tool() for simple delegation

Quick Start:
    ```python
    from cogent import Agent, tool

    @tool
    def search(query: str) -> str:
        '''Search the web.'''
        return f"Results for {query}"

    agent = Agent(
        name="Assistant",
        model="gpt-4o-mini",
        tools=[search],
    )

    result = await agent.run("Search for Python tutorials")
    ```

Native Models:
    ```python
    from cogent.models.openai import OpenAIChat
    from cogent.models.azure import AzureOpenAIChat
    from cogent.models.anthropic import AnthropicChat
    from cogent.models import create_chat, create_embedding

    # Direct usage
    llm = OpenAIChat(model="gpt-4o")
    response = await llm.ainvoke([{"role": "user", "content": "Hello!"}])

    # Factory function
    llm = create_chat("anthropic", model="claude-sonnet-4-20250514")

    # Azure with Managed Identity (Entra ID)
    from cogent.models.azure import AzureEntraAuth
    llm = AzureOpenAIChat(
        deployment="gpt-4o",
        azure_endpoint="https://my-resource.openai.azure.com",
        entra=AzureEntraAuth(method="managed_identity"),
    )
    ```

Simple Delegation (coming soon):
    ```python
    # Convert agent to tool for delegation
    researcher = Agent(name="Researcher", model="gpt-4o")
    writer = Agent(name="Writer", model="gpt-4o", tools=[researcher.as_tool()])

    # Writer can delegate to researcher as needed
    result = await writer.run("Research and write article on AI")
    ```
"""

__version__ = "1.14.1"

# Core enums and utilities
# Graph API (unified visualization)
from cogent import graph

# Agents (THIS IS WHERE WE ADD VALUE)
from cogent.agent.base import Agent
from cogent.agent.config import AgentConfig
from cogent.agent.hitl import (
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
# Memory components moved to cogent.memory
from cogent.memory import (
    InMemorySaver,
    MemorySnapshot,
    ThreadConfig,
)
from cogent.agent.output import (
    OutputMethod,
    ResponseSchema,
    StructuredResult,
)
from cogent.agent.resilience import (
    CircuitBreaker,
    CircuitState,
    FallbackRegistry,
    ResilienceConfig,
    RetryPolicy,
    RetryStrategy,
    ToolResilience,
)
from cogent.agent.state import AgentState
from cogent.agent.streaming import (
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
from cogent.core.context import EMPTY_CONTEXT, RunContext
from cogent.core.enums import (
    AgentRole,
    AgentStatus,
    Priority,
    TaskStatus,
    get_role_capabilities,
)
from cogent.core.utils import generate_id, now_utc

# Execution strategies
from cogent.executors import (
    ExecutionPlan,
    ExecutionStrategy,
    NativeExecutor,
    SequentialExecutor,
    ToolCall,
    create_executor,
    run,
)

from cogent.graph import (
    GraphConfig,
    GraphDirection,
    GraphTheme,
    GraphView,
)

# Interceptors (execution flow control)
from cogent.interceptors import (
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

# LLM & Embedding Models (native)
from cogent.models import (
    ChatModel,
    EmbeddingModel,
    create_chat,
    create_embedding,
)

# Provider-specific model imports for convenience
from cogent.models.azure import AzureOpenAIChat, AzureOpenAIEmbedding

# Observability (THIS IS WHERE WE ADD VALUE)
from cogent.observability import (
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
from cogent.observability.bus import TraceBus, get_trace_bus, set_trace_bus
from cogent.observability.handlers import (
    ConsoleEventHandler,
    FileEventHandler,
    FilteringEventHandler,
    MetricsEventHandler,
)

# Models
from cogent.observability.trace_record import Trace, TraceType

# Tools
from cogent.tools.base import BaseTool, tool
from cogent.tools.registry import ToolRegistry, create_tool_from_function

# Backwards-compatible aliases for visualization module
MermaidConfig = GraphConfig
MermaidTheme = GraphTheme
MermaidDirection = GraphDirection
MermaidRenderer = GraphView  # Closest equivalent
AgentDiagram = GraphView  # Use GraphView.from_agent() instead
TopologyDiagram = GraphView  # Use GraphView.from_topology() instead

# Capabilities (composable tools for agents)
# Document processing module
from cogent import documents
from cogent.capabilities import (
    BaseCapability,
    KnowledgeGraph,
)
from cogent.core import Document, DocumentMetadata

# Native message types (from core.messages)
from cogent.core.messages import (
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
    "AgentStatus",
    "Trace",
    "TraceType",
    "Priority",
    "AgentRole",
    "get_role_capabilities",
    # Core utilities
    "generate_id",
    "now_utc",
    # Agents
    "Agent",
    "AgentConfig",
    "AgentState",
    # Memory (checkpointing - from cogent.memory)
    "MemorySnapshot",
    "ThreadConfig",
    "InMemorySaver",
    # Execution strategies
    "ExecutionStrategy",
    "ExecutionPlan",
    "ToolCall",
    "NativeExecutor",
    "SequentialExecutor",
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
    # Tools
    "ToolRegistry",
    "create_tool_from_function",
    "BaseTool",
    "tool",
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
    # Observer (integrated observability)
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
    # Document types (core)
    "Document",
    "DocumentMetadata",
    # Document processing module
    "documents",
]
