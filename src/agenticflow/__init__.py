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

Built on top of LangChain and LangGraph, AgenticFlow adds value where it matters:
- Multi-agent topologies (supervisor, mesh, pipeline, hierarchical)
- Intelligent resilience (retry, circuit breakers, fallbacks)
- Advanced execution strategies (DAG, ReAct, Plan-Execute)
- Full observability (tracing, metrics, progress tracking)
- Event-driven architecture with pub/sub patterns
- Mermaid visualization for agents and topologies

Philosophy: USE LangChain/LangGraph DIRECTLY. We don't wrap what they do well.

For models, embeddings, vector stores, memory, graphs - use LangChain/LangGraph:
    ```python
    # Models - use LangChain directly
    from langchain_openai import ChatOpenAI
    from langchain_anthropic import ChatAnthropic
    
    # Embeddings - use LangChain directly
    from langchain_openai import OpenAIEmbeddings
    
    # Vector stores - use LangChain directly
    from langchain_core.vectorstores import InMemoryVectorStore
    from langchain_community.vectorstores import FAISS
    
    # Document loading - use LangChain directly
    from langchain_community.document_loaders import WebBaseLoader
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    
    # Graphs - use LangGraph directly
    from langgraph.graph import StateGraph, START, END
    from langgraph.checkpoint.memory import MemorySaver
    from langgraph.store.memory import InMemoryStore
    ```

AgenticFlow adds value with:
    - Agent: Autonomous entity with resilience, execution strategies
    - EventBus: Pub/sub event system for agent communication
    - ToolRegistry: Tool management with permissions
    - Topologies: Multi-agent coordination patterns
    - Observability: Progress tracking, metrics, tracing
    - Visualization: Mermaid diagrams for agents/topologies

Quick Start:
    ```python
    from langchain_openai import ChatOpenAI
    from agenticflow import Agent, AgentConfig, EventBus
    
    # Create model using LangChain directly
    model = ChatOpenAI(model="gpt-4o")
    
    # Create an agent with AgenticFlow's resilience
    agent = Agent(
        config=AgentConfig(
            name="Assistant",
            model=model,  # Pass LangChain model directly
        ),
        event_bus=EventBus(),
    )
    
    # Think with automatic retry on failures
    response = await agent.think("What should I do?")
    
    # Act with circuit breakers and fallbacks
    result = await agent.act("search", {"query": "Python"})
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
from agenticflow.models.event import Event
from agenticflow.models.message import Message, MessageType
from agenticflow.models.task import Task

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

# Graphs - Execution strategies (DAG is our superior approach)
from agenticflow.graphs import (
    ExecutionStrategy,
    ExecutionPlan,
    ToolCall,
    DAGExecutor,
    ReActExecutor,
    PlanExecutor,
    AdaptiveExecutor,
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

# Flow - THE MAIN ENTRY POINT
from agenticflow.flow import (
    Flow,
    FlowConfig,
    TopologyPattern,
    create_flow,
    supervisor_flow,
    pipeline_flow,
    mesh_flow,
)

# Topologies (THIS IS WHERE WE ADD VALUE)
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
    # New API - Enums
    TopologyPattern,
    DelegationStrategy,
    CompletionCondition,
    # New API - Policies
    DelegationPolicy,
    EventHooks,
    # New API - Main spec
    TopologySpec,
    # New API - Factory functions (preferred)
    supervisor_topology,
    coordinator_topology,
    pipeline_topology,
    mesh_topology,
    hierarchical_topology,
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

# LangChain message types (re-export for convenience)
from langchain_core.messages import (
    BaseMessage,
    HumanMessage,
    AIMessage,
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
    "DAGExecutor",
    "ReActExecutor",
    "PlanExecutor",
    "AdaptiveExecutor",
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
    # Flow (MAIN ENTRY POINT)
    "Flow",
    "FlowConfig",
    "TopologyPattern",
    "create_flow",
    "supervisor_flow",
    "pipeline_flow",
    "mesh_flow",
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
    # Topology API - Enums
    "TopologyPattern",
    "DelegationStrategy",
    "CompletionCondition",
    # Topology API - Policies
    "DelegationPolicy",
    "EventHooks",
    # Topology API - Main spec
    "TopologySpec",
    # Topology API - Factory functions (preferred)
    "supervisor_topology",
    "coordinator_topology",
    "pipeline_topology",
    "mesh_topology",
    "hierarchical_topology",
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
    # LangChain messages
    "BaseMessage",
    "HumanMessage",
    "AIMessage",
    "SystemMessage",
    "ToolMessage",
]
