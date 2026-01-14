# Agent-to-Agent (A2A) Communication

The A2A module enables direct agent-to-agent delegation and communication across **all flow types** — both ReactiveFlow and Topologies. Agents can delegate tasks to each other, wait for responses, and coordinate complex multi-agent workflows.

## Overview

Agent-to-Agent (A2A) communication allows agents to:
- **Delegate tasks** to specialized agents
- **Wait for responses** or fire-and-forget
- **Track requests** with correlation IDs
- **Handle replies** with success/error status
- **Configure declaratively** via `can_delegate` and `can_reply` parameters

This enables powerful patterns like coordinator-specialist, chains of delegation, and parallel fan-out.

**Supported in:**
- ✅ **ReactiveFlow** — Event-driven orchestration
- ✅ **Supervisor** — Coordinator delegates to workers
- ✅ **Pipeline** — Stages delegate to specialists
- ✅ **Mesh** — Collaborative agents with external specialists
- ✅ **Hierarchical** — Multi-level delegation (coming in Phase 2.3)

---

## Quick Start

### ReactiveFlow Delegation

```python
from agenticflow import Agent
from agenticflow.reactive import ReactiveFlow

model = get_model()

# Coordinator delegates to specialist
coordinator = Agent(
    name="coordinator",
    model=model,
    system_prompt="You coordinate tasks. Delegate specialized work.",
)

data_analyst = Agent(
    name="data_analyst",
    model=model,
    system_prompt="You analyze data and provide insights.",
)

flow = ReactiveFlow()

# Declarative delegation configuration
flow.register(
    coordinator,
    on="task.created",
    can_delegate=["data_analyst"]  # Enable delegation to analyst
)
flow.register(
    data_analyst,
    handles=True,  # Responds to agent.request for "data_analyst"
    can_reply=True  # Can send responses back
)

# Framework auto-injects:
# - delegate_to tool for coordinator
# - reply_with_result tool for data_analyst

result = await flow.run(
    "Analyze our sales data from last quarter",
    initial_event="task.created",
)
```

### Topology Delegation

```python
from agenticflow import Agent
from agenticflow.topologies import Supervisor, AgentConfig

# Create agents
manager = Agent(name="manager", model=model)
researcher = Agent(name="researcher", model=model)
writer = Agent(name="writer", model=model)

# Configure delegation in topology
supervisor = Supervisor(
    coordinator=AgentConfig(
        agent=manager,
        role="manager",
        can_delegate=["researcher", "writer"]  # Manager can delegate
    ),
    workers=[
        AgentConfig(agent=researcher, role="researcher", can_reply=True),
        AgentConfig(agent=writer, role="writer", can_reply=True),
    ]
)

# Framework auto-injects tools and enhances prompts
result = await supervisor.run("Create a market analysis report")
```

---

## Declarative Delegation Configuration

The framework provides declarative delegation configuration via `can_delegate` and `can_reply` parameters. This works across **all flow types**.

### Configuration Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `can_delegate` | `list[str] \| bool \| None` | Who this agent can delegate to. `True` = all workers/agents |
| `can_reply` | `bool` | Whether this agent can respond to delegated requests |

### Auto-Injection and Enhancement

When you configure delegation, the framework **automatically**:

1. **Injects tools** — Adds `delegate_to` and `reply_with_result` tools to agent
2. **Enhances prompts** — Appends delegation instructions to system prompts
3. **Enforces policy** — Validates delegation targets against allowed list

### ReactiveFlow Example

```python
flow = ReactiveFlow()

# Coordinator can delegate to specialists
flow.register(
    coordinator,
    on="task.created",
    can_delegate=["data_analyst", "writer"]  # Allowed targets
)

# Specialist handles delegated requests
flow.register(
    data_analyst,
    handles=True,  # Subscribes to agent.request:data_analyst
    can_reply=True  # Can send responses back
)

# Tools auto-injected:
# - coordinator gets: delegate_to(target: "data_analyst" | "writer", task, wait)
# - data_analyst gets: reply_with_result(request_id, result, success, error)
```

### Topology Example

```python
from agenticflow.topologies import Supervisor, AgentConfig

supervisor = Supervisor(
    coordinator=AgentConfig(
        agent=manager,
        role="manager",
        can_delegate=["researcher", "writer"]  # or True for all workers
    ),
    workers=[
        AgentConfig(
            agent=researcher,
            role="researcher",
            can_reply=True,  # Can respond to delegated tasks
            can_delegate=["fact_checker"]  # Can sub-delegate
        ),
        AgentConfig(agent=writer, role="writer", can_reply=True),
        AgentConfig(agent=fact_checker, role="fact_checker", can_reply=True),
    ]
)

# Delegation hierarchy:
# manager → researcher → fact_checker
# manager → writer
```

### Delegation Patterns

**Specific targets** (recommended for security):
```python
can_delegate=["specialist1", "specialist2"]
```

**All workers** (flexible, less restrictive):
```python
can_delegate=True  # In topologies: delegates to all workers
```

**Sub-delegation** (hierarchical):
```python
# Worker can delegate to specialist
AgentConfig(
    agent=worker,
    can_reply=True,  # Handles delegated tasks
    can_delegate=["specialist"]  # Can request help
)
```

**Specialist only** (no delegation, only replies):
```python
AgentConfig(agent=specialist, can_reply=True)
```

---

## Registration API

### Simple Syntax (Comprehensive)

The simple syntax now supports all advanced features without fluent API:

```python
# Basic event subscription
flow.register(coordinator, on="task.created")

# Event patterns with wildcards
flow.register(monitor, on="task.*")

# Multiple events
flow.register(monitor, on=["task.created", "task.completed"])

# With condition filter
flow.register(
    agent,
    on="order.placed",
    when=lambda e: e.data.get("value") > 100,
)

# With priority (higher runs first)
flow.register(urgent_handler, on="alert.*", priority=10)

# Auto-emit event after completion
flow.register(processor, on="task.created", emits="task.processed")

# Combined features
flow.register(
    agent,
    on="order.placed",
    when=lambda e: e.data.get("urgent"),
    priority=10,
    emits="order.processed",
)

# A2A: Agent handles requests for itself
flow.register(specialist, handles=True)

# A2A: Agent handles requests for specific name
flow.register(specialist, handles="data_analyst")
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `on` | `str \| list[str]` | `None` | Event type(s) to react to. Supports wildcards like `"task.*"` |
| `handles` | `str \| bool` | `None` | For A2A - if `True`, uses `agent.name`; if string, uses that name |
| `when` | `Callable[[Event], bool]` | `None` | Condition function - only trigger if returns `True` |
| `priority` | `int` | `0` | Higher priority triggers execute first |
| `emits` | `str` | `None` | Event to emit after agent completes |

### Legacy Syntax (triggers parameter)

For backward compatibility, the `triggers` parameter still works:

```python
from agenticflow.reactive import react_to

# Legacy approach
flow.register(
    agent,
    triggers=[
        react_to("custom.*")
            .when(lambda e: e.data.get("urgent"))
            .with_priority(10)
            .emits("custom.processed")
    ]
)
```

**Recommendation**: Use the simple syntax with parameters instead of the legacy `triggers` parameter.

---

## ExecutionContext API

Agents receive an `ExecutionContext` during reactive execution, providing delegation methods.

### delegate_to()

Delegate a task to another agent.

```python
async def delegate_to(
    agent_name: str,
    task: str,
    *,
    data: dict[str, Any] | None = None,
    wait: bool = True,
    timeout_ms: int = 30000,
) -> AgentResponse | None
```

**Parameters:**
- `agent_name` — Name of the target agent
- `task` — Task description for the agent
- `data` — Optional data payload
- `wait` — If True, wait for response; if False, fire-and-forget
- `timeout_ms` — Response timeout in milliseconds (default: 30s)

**Returns:**
- `AgentResponse` if `wait=True` and response received
- `None` if `wait=False` or timeout

**Example:**
```python
# In agent's react() method or via ExecutionContext
response = await ctx.delegate_to(
    "data_analyst",
    "Analyze user engagement metrics",
    data={"period": "last_quarter"},
    wait=True,
    timeout_ms=60000,
)

if response and response.success:
    print(f"Analysis: {response.result}")
```

### reply()

Send a response back to the requesting agent.

```python
def reply(
    result: str | None = None,
    *,
    success: bool = True,
    error: str | None = None,
) -> None
```

**Parameters:**
- `result` — Response data/result
- `success` — Whether the request succeeded
- `error` — Error message if failed

**Example:**
```python
# In specialist agent
async def react(self, event, context):
    # Do work
    analysis = await self.analyze_data(event.data)
    
    # Reply to requester
    context.reply(
        result=analysis,
        success=True,
    )
```

---

## Data Structures

### AgentRequest

Request from one agent to another.

```python
@dataclass
class AgentRequest:
    from_agent: str          # Requesting agent name
    to_agent: str           # Target agent name
    task: str               # Task description
    data: dict | None       # Optional data payload
    correlation_id: str     # Unique 8-char ID for tracking
```

### AgentResponse

Response to an agent request.

```python
@dataclass
class AgentResponse:
    from_agent: str         # Responding agent name
    to_agent: str          # Original requester
    result: str | None     # Response data
    data: dict | None      # Optional structured data
    correlation_id: str    # Matches request ID
    success: bool          # Success status
    error: str | None      # Error message if failed
```

### Factory Functions

```python
from agenticflow.reactive.a2a import create_request, create_response

# Create request
request = create_request(
    from_agent="coordinator",
    to_agent="data_analyst",
    task="Analyze sales data",
    data={"period": "Q4"},
)

# Create response
response = create_response(
    request=request,
    result="Sales up 15% in Q4",
    success=True,
)
```

---

## Common Patterns

### Pattern 1: Coordinator → Specialist

One coordinator delegates specialized work.

```python
coordinator = Agent(name="coordinator", model=model)
specialist = Agent(name="specialist", model=model)

flow = ReactiveFlow()
flow.register(coordinator, on="task.created")
flow.register(specialist, handles=True)

# Coordinator delegates to specialist via agent.request events
```

### Pattern 2: Multi-Specialist Team

Coordinator routes to different specialists based on task type.

```python
coordinator = Agent(name="coordinator", model=model)
data_analyst = Agent(name="data_analyst", model=model)
writer = Agent(name="writer", model=model)
researcher = Agent(name="researcher", model=model)

flow = ReactiveFlow()
flow.register(coordinator, on="task.created")

# All specialists handle their own requests
for agent in [data_analyst, writer, researcher]:
    flow.register(agent, handles=True)

# Coordinator determines which specialist to use
```

### Pattern 3: Chain Delegation (A → B → C)

Agents delegate to each other in sequence.

```python
pm = Agent(name="project_manager", model=model)
architect = Agent(name="architect", model=model)
developer = Agent(name="developer", model=model)

flow = ReactiveFlow()
flow.register(pm, on="task.created")
flow.register(architect, handles=True)
flow.register(developer, handles=True)

# PM → Architect → Developer
```

### Pattern 4: Fan-Out Delegation

One agent delegates to multiple specialists in parallel.

```python
coordinator = Agent(name="coordinator", model=model)
security = Agent(name="security_reviewer", model=model)
performance = Agent(name="performance_reviewer", model=model)
style = Agent(name="style_reviewer", model=model)

flow = ReactiveFlow()
flow.register(coordinator, on="task.created")

for reviewer in [security, performance, style]:
    flow.register(reviewer, handles=True)

# Coordinator delegates to all reviewers simultaneously
```

### Pattern 5: Request-Response with Correlation

Explicit request/response tracking.

```python
requester = Agent(name="requester", model=model)
processor = Agent(name="processor", model=model)
response_handler = Agent(name="response_handler", model=model)

flow = ReactiveFlow()
flow.register(requester, on="task.created")
flow.register(processor, handles=True)
flow.register(response_handler, on="agent.response")

# Tracks correlation IDs automatically
```

---

## Event Flow

### Request Flow

1. Agent calls `ctx.delegate_to("specialist", "task")`
2. Framework creates `AgentRequest` with correlation ID
3. Framework emits `agent.request` event
4. Target agent (registered with `handles=True`) triggers
5. Target agent processes and calls `ctx.reply(result)`
6. Framework creates `AgentResponse` with matching correlation ID
7. Framework emits `agent.response` event
8. Original agent receives response (if `wait=True`)

### Events Emitted

| Event | When | Data |
|-------|------|------|
| `agent.request` | Agent calls `delegate_to()` | AgentRequest fields |
| `agent.response` | Agent calls `reply()` | AgentResponse fields |

---

## Wait vs Fire-and-Forget

### Synchronous (wait=True)

```python
# Wait for response
response = await ctx.delegate_to(
    "data_analyst",
    "Analyze data",
    wait=True,
    timeout_ms=60000,
)

if response:
    print(f"Result: {response.result}")
else:
    print("Timeout or no response")
```

### Asynchronous (wait=False)

```python
# Fire and forget - don't wait
await ctx.delegate_to(
    "logger",
    "Log this event",
    wait=False,
)

# Continues immediately without waiting
```

---

## Error Handling

### Request Timeout

```python
response = await ctx.delegate_to(
    "slow_agent",
    "Long task",
    wait=True,
    timeout_ms=5000,  # 5 seconds
)

if response is None:
    print("Request timed out")
elif response.success:
    print(f"Success: {response.result}")
else:
    print(f"Error: {response.error}")
```

### Failed Response

```python
# In specialist agent
try:
    result = await process_task(event.data)
    context.reply(result=result, success=True)
except Exception as e:
    context.reply(
        success=False,
        error=str(e),
    )
```

---

## Observability

A2A operations emit standard observability events:

```
[coordinator] [triggered] by task.created
[coordinator] [starting]
[coordinator] [thinking]
[coordinator] [completed] (1.2s)
[data_analyst] [triggered] by agent.request
[data_analyst] [starting]
[data_analyst] [thinking]
[data_analyst] [completed] (2.5s)
[Flow] [completed] in 3.8s (2 reactions, 3 rounds, 4 events)
```

Use `Observer` for detailed tracking:

```python
from agenticflow.observability import Observer, ObservabilityLevel, Channel

flow = ReactiveFlow(
    observer=Observer(
        level=ObservabilityLevel.PROGRESS,
        channels=[Channel.REACTIVE, Channel.AGENTS],
        show_duration=True,
    )
)
```

---

## Migration Guide

### Before (Legacy Fluent API)

```python
from agenticflow.reactive import react_to, for_agent

# Old way - verbose fluent API
flow.register(
    data_analyst,
    triggers=[react_to("agent.request", for_agent(data_analyst.name))]
)
flow.register(
    coordinator,
    triggers=[react_to("task.created")]
)

# With advanced features
flow.register(
    agent,
    triggers=[
        react_to("order.placed")
            .when(lambda e: e.data.get("value") > 100)
            .with_priority(10)
            .emits("order.processed")
    ]
)
```

### After (Simple Parameters)

```python
# New way - clean parameter-based syntax
flow.register(data_analyst, handles=True)
flow.register(coordinator, on="task.created")

# Advanced features with parameters (no fluent API needed!)
flow.register(
    agent,
    on="order.placed",
    when=lambda e: e.data.get("value") > 100,
    priority=10,
    emits="order.processed",
)
```

**Note**: The legacy `triggers` parameter still works for backward compatibility, but the new parameter-based syntax is recommended.

---

## Best Practices

1. **Use `handles=True` for specialists** — Let framework manage agent.request routing
2. **Prefer synchronous delegation** — `wait=True` for better error handling
3. **Set appropriate timeouts** — Consider agent complexity
4. **Use correlation IDs** — Automatically tracked for request/response matching
5. **Handle timeouts gracefully** — Check for `None` response
6. **Reply with success status** — Always indicate success/failure
7. **Use observability** — Track delegation chains with Observer

---

## Complete Example

```python
from agenticflow import Agent
from agenticflow.reactive import ReactiveFlow
from agenticflow.observability import Observer, ObservabilityLevel, Channel

model = get_model()

# Define agents
coordinator = Agent(
    name="coordinator",
    model=model,
    system_prompt="You coordinate a team. Delegate specialized tasks.",
)

data_analyst = Agent(
    name="data_analyst",
    model=model,
    system_prompt="You analyze data and provide insights.",
)

report_generator = Agent(
    name="report_generator",
    model=model,
    system_prompt="You generate professional reports.",
)

# Create flow with observability
flow = ReactiveFlow(
    observer=Observer(
        level=ObservabilityLevel.PROGRESS,
        channels=[Channel.REACTIVE, Channel.AGENTS],
        show_duration=True,
    )
)

# Register agents with simple syntax
flow.register(coordinator, on="task.created")
flow.register(data_analyst, handles=True)
flow.register(report_generator, handles=True)

# Run flow
result = await flow.run(
    "Analyze sales data and generate a report",
    initial_event="task.created",
)

print(result.output)
```

---

## Topology Delegation

A2A delegation works across all topology patterns with the same declarative configuration.

### Supervisor Pattern

```python
from agenticflow.topologies import Supervisor, AgentConfig

supervisor = Supervisor(
    coordinator=AgentConfig(
        agent=manager,
        role="manager",
        can_delegate=["researcher", "writer"]  # Manager can delegate
    ),
    workers=[
        AgentConfig(agent=researcher, role="researcher", can_reply=True),
        AgentConfig(agent=writer, role="writer", can_reply=True),
    ]
)

result = await supervisor.run("Create a market analysis report")
```

**Hierarchical Sub-Delegation:**

```python
supervisor = Supervisor(
    coordinator=AgentConfig(
        agent=manager,
        role="manager",
        can_delegate=True  # Can delegate to all workers
    ),
    workers=[
        AgentConfig(
            agent=team_lead,
            role="team_lead",
            can_reply=True,
            can_delegate=["specialist"]  # Can sub-delegate
        ),
        AgentConfig(agent=specialist, role="specialist", can_reply=True),
    ]
)

# Delegation hierarchy: manager → team_lead → specialist
```

### Pipeline Pattern

```python
from agenticflow.topologies import Pipeline, AgentConfig

pipeline = Pipeline(
    stages=[
        AgentConfig(agent=researcher, role="research"),
        AgentConfig(
            agent=analyzer,
            role="analyze",
            can_delegate=["statistician"]  # Can delegate to specialist
        ),
        AgentConfig(agent=writer, role="write"),
        AgentConfig(agent=statistician, role="statistician", can_reply=True),
    ]
)

# Stage 2 (analyzer) can delegate complex stats to specialist
result = await pipeline.run("Analyze survey data and create report")
```

### Mesh Pattern

```python
from agenticflow.topologies import Mesh, AgentConfig

mesh = Mesh(
    agents=[
        AgentConfig(
            agent=business_analyst,
            role="business",
            can_delegate=["finance_specialist"]
        ),
        AgentConfig(agent=tech_analyst, role="tech"),
        AgentConfig(agent=ux_analyst, role="ux"),
        AgentConfig(agent=finance_specialist, role="finance", can_reply=True),
    ],
    max_rounds=2
)

# Business analyst can request financial analysis from specialist
result = await mesh.run("Evaluate new product viability")
```

### Cross-Topology Patterns

Same agents can participate in different topologies with different policies:

```python
# In Pipeline: No delegation (linear flow)
pipeline = Pipeline(stages=[
    AgentConfig(agent=researcher, role="research"),
    AgentConfig(agent=writer, role="write"),
])

# In Supervisor: Analyst coordinates with delegation
supervisor = Supervisor(
    coordinator=AgentConfig(
        agent=analyst,
        role="coordinator",
        can_delegate=["researcher", "writer"]
    ),
    workers=[
        AgentConfig(agent=researcher, role="researcher", can_reply=True),
        AgentConfig(agent=writer, role="writer", can_reply=True),
    ]
)

# Same agents, different delegation policies per topology
```

### Topology Examples

See [examples/topologies/delegation.py](../examples/topologies/delegation.py) for complete examples:
- Supervisor with hierarchical delegation
- Pipeline with specialist delegation
- Mesh with external specialists
- Dynamic delegation policies
- Cross-topology delegation patterns

---

## Best Practices

1. **Use declarative configuration** — `can_delegate` and `can_reply` parameters over manual tool injection
2. **Prefer specific targets** — `can_delegate=["specialist1", "specialist2"]` over `can_delegate=True` for security
3. **Use synchronous delegation** — `wait=True` for better error handling
4. **Set appropriate timeouts** — Consider agent complexity
5. **Handle timeouts gracefully** — Check for `None` response
6. **Reply with success status** — Always indicate success/failure
7. **Use observability** — Track delegation chains with Observer
8. **Design clear hierarchies** — Avoid circular delegation
9. **Limit delegation depth** — Prevent runaway delegation chains
10. **Use topology delegation** — Let framework manage multi-agent patterns

---

## API Reference

See also:
- [Reactive Module](reactive.md) — Event-driven orchestration
- [Topologies Module](topologies.md) — Multi-agent coordination patterns
- [Flow Module](flow.md) — Flow orchestration patterns
- [Observability](observability.md) — Monitoring and tracing
