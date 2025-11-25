# Agents API Reference

Agents are the primary autonomous entities in AgenticFlow. They can think (use LLMs), act (use tools), communicate (via events), and learn from failures.

## Overview

```python
from agenticflow import (
    # Core
    Agent,
    AgentConfig,
    AgentState,
    AgentExecutor,
    
    # Resilience
    RetryPolicy,
    RetryStrategy,
    CircuitBreaker,
    CircuitState,
    FallbackRegistry,
    FailureMemory,
    ResilienceConfig,
    ToolResilience,
    
    # Enums
    AgentRole,
    AgentStatus,
    ExecutionStrategy,
)
```

---

## Agent

The main autonomous entity that can think, act, and communicate.

### Constructor

```python
Agent(
    config: AgentConfig,
    event_bus: EventBus,
    tool_registry: ToolRegistry | None = None,
)
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `config` | `AgentConfig` | Agent configuration |
| `event_bus` | `EventBus` | Event bus for communication |
| `tool_registry` | `ToolRegistry \| None` | Registry of available tools |

### Properties

| Property | Type | Description |
|----------|------|-------------|
| `id` | `str` | Unique agent identifier |
| `name` | `str` | Agent's display name |
| `role` | `AgentRole` | Agent's role in the system |
| `status` | `AgentStatus` | Current agent status |
| `model` | `BaseChatModel` | Lazy-loaded LLM model (see [Providers](providers.md)) |
| `resilience` | `ToolResilience` | Resilience layer for the agent |

### Methods

#### `think(prompt, correlation_id=None) -> str`

Process a prompt through the agent's reasoning (LLM).

```python
# Simple thinking
response = await agent.think("What should I write about?")

# With correlation for tracing
response = await agent.think(
    "Analyze this data: [1, 2, 3, 4, 5]",
    correlation_id="task-123",
)
```

#### `act(tool_name, args, correlation_id=None, tracker=None, use_resilience=True) -> Any`

Execute an action using a tool with intelligent retry and recovery.

```python
# Simple tool call
result = await agent.act("search_web", {"query": "Python tutorials"})

# With progress tracking
from agenticflow import ProgressTracker, OutputConfig
tracker = ProgressTracker(OutputConfig.verbose())
result = await agent.act(
    "analyze_data",
    {"data": [1, 2, 3, 4, 5]},
    tracker=tracker,
)

# Without resilience (for testing)
result = await agent.act(
    "flaky_tool",
    {"input": "test"},
    use_resilience=False,
)
```

#### `act_many(tool_calls, correlation_id=None, fail_fast=False, tracker=None, use_resilience=True) -> list[Any]`

Execute multiple tool calls in parallel with intelligent retry.

```python
# Parallel execution
results = await agent.act_many([
    ("search_web", {"query": "Python async"}),
    ("read_file", {"path": "data.json"}),
    ("calculate", {"expr": "2 + 2"}),
])

# With fail-fast (stop on first error)
results = await agent.act_many(
    [("tool1", {}), ("tool2", {}), ("tool3", {})],
    fail_fast=True,
)
```

#### `execute_task(task, correlation_id=None) -> Any`

Execute a task assigned to this agent.

```python
from agenticflow import Task

task = Task(
    name="Write a poem",
    description="Write a haiku about nature",
)
result = await agent.execute_task(task)
```

#### `run(task, context=None, strategy="dag", on_step=None, tracker=None) -> Any`

Execute a complex task using an advanced execution strategy.

```python
# With DAG strategy (fastest, parallelizes independent steps)
result = await agent.run(
    "Search for Python and Rust tutorials, then compare them",
    strategy="dag",
)

# With progress tracking
tracker = ProgressTracker(OutputConfig.verbose())
result = await agent.run(
    "Research AI trends and summarize",
    strategy="adaptive",
    tracker=tracker,
)

# With step callback
def on_step(step_type, data):
    print(f"{step_type}: {data}")

result = await agent.run(
    "Analyze the codebase",
    strategy="plan",
    on_step=on_step,
)
```

**Execution Strategies:**

| Strategy | Description | Use Case |
|----------|-------------|----------|
| `"react"` | Think-act-observe loop | Simple sequential tasks |
| `"plan"` | Plan all steps, then execute | Predictable multi-step tasks |
| `"dag"` | Build dependency graph, execute in parallel | Complex tasks with independent subtasks |
| `"adaptive"` | Auto-select based on task complexity | General purpose |

#### `send_message(content, receiver_id=None, correlation_id=None) -> Message`

Send a message to another agent or broadcast.

```python
# Direct message
message = await agent.send_message(
    "Here's the analysis result...",
    receiver_id=other_agent.id,
)

# Broadcast to all agents
message = await agent.send_message(
    "Task completed!",
    receiver_id=None,  # Broadcast
)
```

#### `receive_message(message, correlation_id=None) -> str | None`

Receive and process a message.

```python
response = await agent.receive_message(message)
```

#### `is_available() -> bool`

Check if agent can accept new work.

```python
if agent.is_available():
    await agent.execute_task(task)
```

### Resilience Methods

#### `get_circuit_status(tool_name=None) -> dict`

Get circuit breaker status for tools.

```python
# Check specific tool
status = agent.get_circuit_status("web_search")
print(f"State: {status['state']}")
print(f"Failures: {status['failure_count']}")

# Check all tools
all_status = agent.get_circuit_status()
for tool, status in all_status.items():
    if status["state"] == "open":
        print(f"⚠️ {tool} circuit is open!")
```

#### `reset_circuit(tool_name=None) -> None`

Reset circuit breaker(s) to closed state.

```python
# Reset specific tool
agent.reset_circuit("web_search")

# Reset all circuits
agent.reset_circuit()
```

#### `get_failure_suggestions(tool_name) -> dict | None`

Get suggestions for a failing tool based on learned patterns.

```python
suggestions = agent.get_failure_suggestions("web_search")
if suggestions:
    print(f"Failure rate: {suggestions['failure_rate']:.1%}")
    print(f"Common errors: {suggestions['common_errors']}")
    for rec in suggestions['recommendations']:
        print(f"  • {rec}")
```

#### `clear_failure_memory() -> None`

Clear all learned failure patterns.

```python
agent.clear_failure_memory()
```

### Visualization

#### `draw_mermaid(**kwargs) -> str`

Generate a Mermaid diagram showing this agent and its tools.

```python
diagram = agent.draw_mermaid(
    theme="forest",
    direction="LR",
    title="My Agent",
    show_tools=True,
    show_roles=True,
)
print(diagram)
```

#### `draw_mermaid_png(**kwargs) -> bytes`

Generate a PNG image of the agent's diagram.

```python
png_bytes = agent.draw_mermaid_png(theme="dark")
with open("agent.png", "wb") as f:
    f.write(png_bytes)
```

---

## AgentConfig

Configuration for an Agent.

### Constructor

```python
@dataclass
class AgentConfig:
    name: str
    role: AgentRole = AgentRole.WORKER
    description: str = ""
    
    # LLM Configuration (see Providers Guide for all options)
    model: str | ModelSpec | BaseChatModel | dict | None = None
    temperature: float = 0.7
    max_tokens: int | None = None
    system_prompt: str | None = None
    model_kwargs: dict = field(default_factory=dict)
    
    # Capabilities
    tools: list[str] = field(default_factory=list)
    
    # Execution
    max_concurrent_tasks: int = 5
    timeout_seconds: float = 300.0
    
    # Resilience
    resilience_config: ResilienceConfig | None = None
    fallback_tools: dict[str, list[str]] = field(default_factory=dict)
```

### Model Specification

The `model` parameter supports multiple formats. See the [Providers Guide](providers.md) for complete documentation.

```python
from agenticflow import ModelSpec

# String: provider/model format (recommended)
config = AgentConfig(name="Agent", model="openai/gpt-4o")
config = AgentConfig(name="Agent", model="anthropic/claude-sonnet-4-20250514")
config = AgentConfig(name="Agent", model="google/gemini-2.0-flash")
config = AgentConfig(name="Agent", model="ollama/llama3.2")

# ModelSpec for full control
config = AgentConfig(
    name="Agent",
    model=ModelSpec(
        provider="openai",
        model="gpt-4o",
        temperature=0.5,
        max_tokens=4096,
    ),
)

# Azure OpenAI with Managed Identity
from agenticflow.providers import AzureAuthMethod

config = AgentConfig(
    name="Agent",
    model=ModelSpec(
        provider="azure_openai",
        model="gpt-4o",
        azure_endpoint="https://my-resource.openai.azure.com",
        azure_deployment="my-gpt4-deployment",
        azure_auth_method=AzureAuthMethod.MANAGED_IDENTITY,
    ),
)

# Direct LangChain model object
from langchain_openai import ChatOpenAI
config = AgentConfig(
    name="Agent",
    model=ChatOpenAI(model="gpt-4o", temperature=0.5),
)

# Configuration dict
config = AgentConfig(
    name="Agent",
    model={
        "provider": "openai",
        "model": "gpt-4o",
        "temperature": 0.7,
    },
)
```

### Methods

#### `with_tools(tools) -> AgentConfig`

Create a new config with additional tools.

```python
config = AgentConfig(name="Agent", model="openai/gpt-4o")
config = config.with_tools(["search", "calculate", "write"])
```

#### `with_system_prompt(prompt) -> AgentConfig`

Create a new config with a different system prompt.

```python
config = config.with_system_prompt(
    "You are a helpful data analyst. Be concise and accurate."
)
```

#### `with_resilience(config) -> AgentConfig`

Create a new config with resilience configuration.

```python
config = config.with_resilience(ResilienceConfig.aggressive())
```

#### `with_fallbacks(fallback_tools) -> AgentConfig`

Create a new config with fallback tool mappings.

```python
config = config.with_fallbacks({
    "web_search": ["cached_search", "local_search"],
    "gpt4_analyze": ["gpt35_analyze"],
})
```

#### `can_use_tool(tool_name) -> bool`

Check if agent is configured to use a tool.

```python
if config.can_use_tool("search"):
    # Agent can use search tool
    pass
```

### Complete Example

```python
from agenticflow import (
    Agent,
    AgentConfig,
    AgentRole,
    EventBus,
    ToolRegistry,
    ResilienceConfig,
)
from langchain_core.tools import tool

# Define tools
@tool
def search(query: str) -> str:
    """Search the web."""
    return f"Results for: {query}"

@tool
def cached_search(query: str) -> str:
    """Search cached results."""
    return f"Cached results for: {query}"

# Setup
event_bus = EventBus()
tool_registry = ToolRegistry()
tool_registry.register(search)
tool_registry.register(cached_search)

# Create resilient agent
config = AgentConfig(
    name="ResearchAgent",
    role=AgentRole.SPECIALIST,
    description="A resilient research agent",
    model="openai/gpt-4o",
    temperature=0.3,
    system_prompt="You are a meticulous researcher. Always verify facts.",
    tools=["search", "cached_search"],
    max_concurrent_tasks=3,
).with_resilience(
    ResilienceConfig.aggressive()
).with_fallbacks({
    "search": ["cached_search"],
})

agent = Agent(
    config=config,
    event_bus=event_bus,
    tool_registry=tool_registry,
)

# Use the agent
result = await agent.run(
    "Research the latest AI developments",
    strategy="dag",
)
```

---

## AgentState

Runtime state tracking for an agent.

### Properties

| Property | Type | Description |
|----------|------|-------------|
| `status` | `AgentStatus` | Current status |
| `active_tasks` | `set[str]` | Currently executing task IDs |
| `thinking_time_ms` | `float` | Total thinking time |
| `acting_time_ms` | `float` | Total acting time |
| `task_count` | `int` | Total tasks executed |
| `error_count` | `int` | Total errors encountered |

### Methods

```python
# Check availability
if state.is_available():
    # Agent is idle or thinking

if state.has_capacity(max_tasks=5):
    # Agent has room for more tasks

# Track tasks
state.start_task("task-123")
state.finish_task("task-123", success=True)

# Get history
messages = state.get_recent_history(limit=10)

# Record metrics
state.add_thinking_time(150.0)  # ms
state.add_acting_time(50.0)     # ms
state.record_error("Something went wrong")

# Export
data = state.to_dict()
```

---

## AgentRole

Enumeration of agent roles.

```python
from agenticflow import AgentRole

class AgentRole(Enum):
    SUPERVISOR = "supervisor"    # Manages other agents
    COORDINATOR = "coordinator"  # Coordinates between teams
    WORKER = "worker"           # Executes tasks
    SPECIALIST = "specialist"   # Domain expert
    ROUTER = "router"           # Routes tasks
    CRITIC = "critic"           # Reviews work
    PLANNER = "planner"         # Creates plans
```

---

## AgentStatus

Enumeration of agent statuses.

```python
from agenticflow import AgentStatus

class AgentStatus(Enum):
    IDLE = "idle"           # Ready for work
    THINKING = "thinking"   # Processing with LLM
    ACTING = "acting"       # Executing a tool
    WAITING = "waiting"     # Waiting for response
    ERROR = "error"         # Encountered an error
    OFFLINE = "offline"     # Not available
```

---

## Resilience System

AgenticFlow includes a comprehensive resilience system that helps agents recover from failures gracefully.

### RetryPolicy

Configurable retry with backoff strategies.

```python
from agenticflow import RetryPolicy, RetryStrategy

# Default balanced policy
policy = RetryPolicy()

# Aggressive for flaky APIs
policy = RetryPolicy(
    max_retries=5,
    strategy=RetryStrategy.EXPONENTIAL_JITTER,
    base_delay=0.5,
    max_delay=30.0,
    jitter_factor=0.3,
)

# Conservative for expensive operations
policy = RetryPolicy(
    max_retries=2,
    strategy=RetryStrategy.FIXED,
    base_delay=5.0,
)

# No retry (fail fast)
policy = RetryPolicy.no_retry()

# Preset constructors
policy = RetryPolicy.aggressive()
policy = RetryPolicy.conservative()
policy = RetryPolicy.default()
```

**RetryStrategy Options:**

| Strategy | Description |
|----------|-------------|
| `NONE` | No retry |
| `FIXED` | Fixed delay between retries |
| `LINEAR` | Linearly increasing delay |
| `EXPONENTIAL` | Exponential backoff (2^n) |
| `EXPONENTIAL_JITTER` | Exponential with random jitter (best) |

### CircuitBreaker

Prevents cascading failures by blocking calls to failing tools.

```python
from agenticflow import CircuitBreaker, CircuitState

breaker = CircuitBreaker(
    failure_threshold=3,    # Open after 3 failures
    success_threshold=2,    # Close after 2 successes in half-open
    reset_timeout=30.0,     # Wait 30s before testing recovery
    half_open_max_calls=3,  # Allow 3 test calls in half-open
)

# Check state
if breaker.can_execute():
    try:
        result = await tool.invoke(args)
        breaker.record_success()
    except Exception:
        breaker.record_failure()

# Get status
status = breaker.get_status()
print(f"State: {status['state']}")
print(f"Failures: {status['failure_count']}")

# Manual control
breaker.reset()       # Force close
breaker.force_open()  # Force open (maintenance)
```

**CircuitState:**

| State | Description |
|-------|-------------|
| `CLOSED` | Normal operation, calls pass through |
| `OPEN` | Failing, calls blocked immediately |
| `HALF_OPEN` | Testing recovery, limited calls allowed |

### FallbackRegistry

Register alternative tools for graceful degradation.

```python
from agenticflow import FallbackRegistry

registry = FallbackRegistry()

# Register fallback chain
registry.register(
    "web_search",                                    # Primary tool
    ["cached_search", "local_search", "llm_search"], # Fallbacks in order
)

# Register with argument transformation
def transform_for_fallback(fallback_tool: str, args: dict) -> dict:
    if fallback_tool == "cached_search":
        return {"query": args["query"], "max_age_hours": 24}
    return args

registry.register(
    "expensive_api",
    ["cheap_api", "local_cache"],
    transform_args=transform_for_fallback,
)

# Query fallbacks
fallback = registry.get_fallback("web_search", attempt=0)  # "cached_search"
fallback = registry.get_fallback("web_search", attempt=1)  # "local_search"

all_fallbacks = registry.get_all_fallbacks("web_search")
has_fallback = registry.has_fallback("web_search")  # True
```

### FailureMemory

Learn from failures to adapt future behavior.

```python
from agenticflow import FailureMemory

memory = FailureMemory(
    max_records=1000,       # Maximum failure records to keep
    learning_threshold=3,   # Failures before marking pattern as problematic
)

# Record failures and successes
memory.record_failure(
    tool_name="web_search",
    exception=TimeoutError("Request timed out"),
    args={"query": "complex search"},
    attempt=1,
)
memory.record_success("web_search")

# Check if pattern should be avoided
if memory.should_avoid("web_search", {"query": "complex search"}):
    # This pattern has failed repeatedly, try alternative
    pass

# Get analytics
failure_rate = memory.get_failure_rate("web_search")
common_errors = memory.get_common_errors("web_search")

# Get suggestions
suggestions = memory.get_suggestions("web_search")
print(f"Failure rate: {suggestions['failure_rate']:.1%}")
for rec in suggestions['recommendations']:
    print(f"  • {rec}")

# Clear memory
memory.clear()
```

### ResilienceConfig

Complete resilience configuration combining all components.

```python
from agenticflow import ResilienceConfig, RetryPolicy, CircuitBreaker, RecoveryAction

# Full custom configuration
config = ResilienceConfig(
    retry_policy=RetryPolicy(
        max_retries=5,
        strategy=RetryStrategy.EXPONENTIAL_JITTER,
        base_delay=1.0,
    ),
    circuit_breaker_enabled=True,
    circuit_breaker_config=CircuitBreaker(
        failure_threshold=3,
        reset_timeout=30.0,
    ),
    fallback_enabled=True,
    learning_enabled=True,
    timeout_seconds=60.0,
    on_failure=RecoveryAction.ADAPT,
)

# Per-tool overrides
config = ResilienceConfig(
    retry_policy=RetryPolicy.default(),
    tool_configs={
        "expensive_api": {
            "retry_policy": RetryPolicy.conservative(),
            "timeout_seconds": 120.0,
        },
        "fast_cache": {
            "retry_policy": RetryPolicy.no_retry(),
            "timeout_seconds": 5.0,
        },
    },
)

# Preset configurations
config = ResilienceConfig.aggressive()  # Never give up
config = ResilienceConfig.balanced()    # Default balanced approach
config = ResilienceConfig.fast_fail()   # Fail quickly (for testing)
```

**RecoveryAction Options:**

| Action | Description |
|--------|-------------|
| `RETRY` | Retry the same tool |
| `FALLBACK` | Try a fallback tool |
| `SKIP` | Skip and continue (return error) |
| `ABORT` | Abort entire execution |
| `ADAPT` | Let agent decide (re-think) |

### ToolResilience

Unified resilience layer wrapping tool execution.

```python
from agenticflow import ToolResilience, FallbackRegistry, ResilienceConfig

# Setup
fallback_registry = FallbackRegistry()
fallback_registry.register("search", ["cached_search", "llm_search"])

resilience = ToolResilience(
    config=ResilienceConfig.aggressive(),
    fallback_registry=fallback_registry,
)

# Execute with full resilience
result = await resilience.execute(
    tool_fn=search_tool.invoke,
    tool_name="search",
    args={"query": "Python tutorials"},
    fallback_fn=lambda name: tool_registry.get(name).invoke,
)

if result.success:
    print(f"Result: {result.result}")
    print(f"Tool used: {result.tool_used}")
    print(f"Attempts: {result.attempts}")
else:
    print(f"Failed: {result.error}")
    print(f"Fallback chain tried: {result.fallback_chain}")

# Circuit breaker management
status = resilience.get_circuit_status("search")
resilience.reset_circuit("search")
resilience.reset_all_circuits()

# Failure insights
suggestions = resilience.get_failure_suggestions("search")
```

### ExecutionResult

Result from resilient tool execution.

```python
@dataclass
class ExecutionResult:
    success: bool                           # Whether execution succeeded
    result: Any = None                      # The result (if success)
    error: Exception | None = None          # The error (if failed)
    tool_used: str = ""                     # Actual tool used (may be fallback)
    attempts: int = 1                       # Total attempts made
    total_time_ms: float = 0.0              # Total execution time
    used_fallback: bool = False             # Whether fallback was used
    fallback_chain: list[str] = []          # Fallbacks tried
    circuit_state: CircuitState | None      # Circuit breaker state
    recovery_action: RecoveryAction | None  # Action taken on failure
```

---

## Complete Resilience Example

```python
import asyncio
from agenticflow import (
    Agent,
    AgentConfig,
    AgentRole,
    EventBus,
    ToolRegistry,
    ResilienceConfig,
    RetryPolicy,
    RetryStrategy,
    CircuitBreaker,
    FallbackRegistry,
    ProgressTracker,
    OutputConfig,
)
from langchain_core.tools import tool

# Define tools with varying reliability
call_count = {"search": 0}

@tool
def unreliable_search(query: str) -> str:
    """Search that fails occasionally."""
    call_count["search"] += 1
    if call_count["search"] % 3 != 0:  # Fails 2 out of 3 times
        raise TimeoutError("Service temporarily unavailable")
    return f"Results for: {query}"

@tool
def cached_search(query: str) -> str:
    """Reliable cached search."""
    return f"Cached results for: {query}"

@tool
def local_search(query: str) -> str:
    """Local search fallback."""
    return f"Local results for: {query}"

async def main():
    # Setup
    event_bus = EventBus()
    tool_registry = ToolRegistry()
    tool_registry.register(unreliable_search)
    tool_registry.register(cached_search)
    tool_registry.register(local_search)
    
    # Create resilient config
    config = AgentConfig(
        name="ResearchBot",
        role=AgentRole.SPECIALIST,
        model="openai/gpt-4o",
        tools=["unreliable_search", "cached_search", "local_search"],
    ).with_resilience(
        ResilienceConfig(
            retry_policy=RetryPolicy(
                max_retries=3,
                strategy=RetryStrategy.EXPONENTIAL_JITTER,
                base_delay=0.5,
            ),
            circuit_breaker_enabled=True,
            circuit_breaker_config=CircuitBreaker(
                failure_threshold=2,
                reset_timeout=10.0,
            ),
            fallback_enabled=True,
            learning_enabled=True,
        )
    ).with_fallbacks({
        "unreliable_search": ["cached_search", "local_search"],
    })
    
    agent = Agent(
        config=config,
        event_bus=event_bus,
        tool_registry=tool_registry,
    )
    
    # Setup progress tracking
    tracker = ProgressTracker(OutputConfig.verbose())
    
    # Execute - will retry, use circuit breaker, and fall back as needed
    with tracker:
        tracker.start_workflow("resilience-demo", "Resilient Search")
        
        for i in range(5):
            tracker.step(f"search-{i}", f"Search attempt {i+1}")
            try:
                result = await agent.act(
                    "unreliable_search",
                    {"query": f"test query {i}"},
                    tracker=tracker,
                )
                tracker.success(f"Got result: {result[:50]}...")
            except Exception as e:
                tracker.error(str(e))
        
        # Check circuit status
        status = agent.get_circuit_status("unreliable_search")
        print(f"\nCircuit status: {status}")
        
        # Get failure insights
        suggestions = agent.get_failure_suggestions("unreliable_search")
        if suggestions:
            print(f"Failure rate: {suggestions['failure_rate']:.1%}")
            for rec in suggestions.get('recommendations', []):
                print(f"  • {rec}")

if __name__ == "__main__":
    asyncio.run(main())
```

---

## Next Steps

- [Providers](providers.md) - Configure LLM providers (OpenAI, Azure, Anthropic, etc.)
- [Topologies](topologies.md) - Multi-agent orchestration patterns
- [Events](events.md) - Event-driven communication
- [Observability](observability.md) - Progress tracking and metrics
