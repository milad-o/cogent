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
    from agenticflow.models.azure import AzureChat
    from agenticflow.models.anthropic import AnthropicChat
    from agenticflow.models import create_chat, create_embedding
    
    # Direct usage
    llm = OpenAIChat(model="gpt-4o")
    response = await llm.ainvoke([{"role": "user", "content": "Hello!"}])
    
    # Factory function
    llm = create_chat("anthropic", model="claude-sonnet-4-20250514")
    
    # Azure with Managed Identity
    llm = AzureChat(
        deployment="gpt-4o",
        azure_endpoint="https://my-resource.openai.azure.com",
        use_managed_identity=True,
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
        event_bus=EventBus(),
    )
    
    # Think with automatic retry on failures
    response = await agent.think("What should I do?")
    
    # Execute complex tasks
    result = await agent.run_turbo("Search and analyze data")
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

__version__ = "0.1.0"

# Core enums and utilities
from agenticflow.core.enums import (
    AgentRole,
    AgentStatus,
    EventType,
    Priority,
    TaskStatus,
    get_role_capabilities,
)
from agenticflow.core.utils import generate_id, now_utc

# Models
from agenticflow.schemas.event import Event
from agenticflow.schemas.message import Message, MessageType
from agenticflow.schemas.task import Task

# Agents (THIS IS WHERE WE ADD VALUE)
from agenticflow.agent.base import Agent
from agenticflow.agent.config import AgentConfig
from agenticflow.agent.state import AgentState
from agenticflow.agent.memory import (
    AgentMemory,
    MemorySnapshot,
    MemoryCheckpoint,  # Backward compat alias
    ThreadConfig,
    InMemorySaver,
    InMemoryCheckpointer,  # Backward compat alias
)
from agenticflow.agent.resilience import (
    ResilienceConfig,
    RetryPolicy,
    RetryStrategy,
    CircuitBreaker,
    CircuitState,
    ToolResilience,
    FallbackRegistry,
)
from agenticflow.agent.hitl import (
    InterruptReason,
    DecisionType,
    PendingAction,
    HumanDecision,
    InterruptedState,
    InterruptedException,
    DecisionRequiredException,
    AbortedException,
    GuidanceResult,
    HumanResponse,
    should_interrupt,
)
from agenticflow.agent.streaming import (
    StreamChunk,
    StreamEvent,
    StreamEventType,
    StreamConfig,
    StreamCallback,
    PrintStreamCallback,
    CollectorStreamCallback,
    ObserverStreamCallback,
    ToolCallChunk,
    chunk_from_message,
    extract_tool_calls,
    collect_stream,
    print_stream,
)

# Graphs - Execution strategies
from agenticflow.executors import (
    ExecutionStrategy,
    ExecutionPlan,
    ToolCall,
    NativeExecutor,
    SequentialExecutor,
    TreeSearchExecutor,
    run,
    create_executor,
)

# Events (THIS IS WHERE WE ADD VALUE)
from agenticflow.events.bus import EventBus, get_event_bus, set_event_bus
from agenticflow.events.handlers import (
    ConsoleEventHandler,
    FileEventHandler,
    FilteringEventHandler,
    MetricsEventHandler,
)

# Tasks
from agenticflow.tasks.manager import TaskManager

# Tools (THIS IS WHERE WE ADD VALUE)
from agenticflow.tools.registry import ToolRegistry, create_tool_from_function
from agenticflow.tools.base import BaseTool, tool

# Flow - THE MAIN ENTRY POINT
from agenticflow.flow import (
    Flow,
    FlowConfig,
    create_flow,
    supervisor_flow,
    pipeline_flow,
    mesh_flow,
)

# Topologies (simple coordination patterns)
from agenticflow.topologies import (
    # Core classes
    AgentConfig as TopologyAgentConfig,
    BaseTopology,
    TopologyResult,
    TopologyType,
    # Pattern classes
    Supervisor,
    Pipeline,
    Mesh,
    Hierarchical,
    # Convenience functions
    supervisor,
    pipeline,
    mesh,
)

# Observability (THIS IS WHERE WE ADD VALUE)
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
    # Flow Observer (integrated observability)
    FlowObserver,
    ObservabilityLevel,
    Channel,
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

# LLM & Embedding Models (native, no LangChain required)
from agenticflow.models import (
    ChatModel,
    EmbeddingModel,
    create_chat,
    create_embedding,
)
# Provider-specific model imports for convenience
from agenticflow.models.azure import AzureChat, AzureEmbedding

# Visualization (THIS IS WHERE WE ADD VALUE)
from agenticflow.visualization import (
    MermaidConfig,
    MermaidRenderer,
    MermaidTheme,
    MermaidDirection,
    AgentDiagram,
    TopologyDiagram,
)

# Prebuilt agents (ready-to-use components)
from agenticflow.prebuilt import (
    Chatbot,
    RAGAgent,
    create_chatbot,
    create_rag_agent,
)

# Capabilities (composable tools for agents)
from agenticflow.capabilities import (
    BaseCapability,
    KnowledgeGraph,
)

# Native message types (from core.messages)
from agenticflow.core.messages import (
    BaseMessage,
    HumanMessage,
    AIMessage,
    SystemMessage,
    ToolMessage,
)

# Document processing module
from agenticflow import document

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
    "get_role_capabilities",
    # Core utilities
    "generate_id",
    "now_utc",
    # Models
    "Event",
    "Message",
    "MessageType",
    "Task",
    # Agents
    "Agent",
    "AgentConfig",
    "AgentState",
    # Memory (LangGraph compatible)
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
    "StreamEventType",
    "StreamConfig",
    "StreamCallback",
    "PrintStreamCallback",
    "CollectorStreamCallback",
    "ToolCallChunk",
    "chunk_from_message",
    "extract_tool_calls",
    "collect_stream",
    "print_stream",
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
    "BaseTool",
    "tool",
    # Flow (MAIN ENTRY POINT)
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
    "FlowObserver",
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
    # Visualization
    "MermaidConfig",
    "MermaidRenderer",
    "MermaidTheme",
    "MermaidDirection",
    "AgentDiagram",
    "TopologyDiagram",
    # Prebuilt agents
    "Chatbot",
    "RAGAgent",
    "create_chatbot",
    "create_rag_agent",
    # Capabilities
    "BaseCapability",
    "KnowledgeGraph",
    # LLM & Embedding Models (native)
    "ChatModel",
    "EmbeddingModel",
    "AzureChat",
    "AzureEmbedding",
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
