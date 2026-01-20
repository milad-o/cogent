# Changelog

All notable changes to AgenticFlow will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [1.8.8] - 2026-01-20

### Added

#### Memory: Conversation History Search

- **New Tool**: `search_conversation(query, max_results=5)` - Search through conversation history for relevant context
  - Critical for long conversations that exceed context window limits
  - Enables agents to find information discussed earlier without explicit `remember()` calls
  - Searches through past messages in the current namespace/thread
  - Complementary to `search_memories()` which searches long-term facts

**Enhanced `search_memories()`:**
  - Now uses semantic search when VectorStore is configured
  - Falls back to keyword search when no VectorStore available
  - Enables natural language queries over stored facts

**Updated System Prompt:**
  - Agents now instructed to search conversation history for long conversations
  - Clear guidance on when to use `search_conversation()` vs `search_memories()`
  - Agents search before claiming "I don't know"

**Use Cases:**
```python
# Long conversation - agent can still find earlier context
memory = Memory()
agent = Agent(model=model, memory=memory)

# Many messages later...
await agent.run("What were the three projects I mentioned earlier?")
# → Agent calls: search_conversation("three projects")

# Semantic search over facts (requires VectorStore)
memory = Memory(vectorstore=VectorStore())
await agent.run("What do you know about my hobbies?")
# → Agent calls: search_memories("hobbies") with semantic search
```

## [1.8.7] - 2026-01-20

### Changed

#### Memory is Always Agentic

**BREAKING:** Removed `agentic` parameter from Memory class
- Memory is now always agentic - tools are always exposed to agents
- Removed confusing distinction between agentic and non-agentic modes
- In a truly agentic framework, memory should always be agentic
- Memory tools (`remember`, `recall`, `forget`, `search_memories`) are always available
- Simplifies API and reduces cognitive overhead

**Migration:**
```python
# Before (v1.8.6)
memory = Memory(agentic=True)  # Explicit
memory = Memory(agentic=False)  # No tools

# After (v1.8.7)
memory = Memory()  # Always has tools - simpler!
```

**Rationale:** In fully agentic systems, manually controlling memory defeats the purpose. Agents should autonomously manage their own memory. For programmatic control, use the context parameter instead.

## [1.8.6] - 2026-01-20

### Added

#### KnowledgeGraph Memory Backend Auto-Save

**Consistent Auto-Save Across All Backends:**
- `InMemoryGraph` now supports optional `path` and `auto_save` parameters
- Memory backend auto-saves to file after each modification when `auto_save=True`
- Automatically loads from file on initialization if path exists
- All backends now have consistent real-time persistence behavior

**Usage:**
```python
# Memory with auto-save - best of both worlds
kg = KnowledgeGraph(backend="memory", path="data.json", auto_save=True)
kg.remember("Alice", "Person")  # Automatically saved to data.json

# Load pre-saved knowledge graphs
kg = KnowledgeGraph.from_file("company.db")  # SQLite
kg = KnowledgeGraph.from_file("knowledge.json")  # JSON

# Pass to agent
agent = Agent(name="Assistant", model=model, capabilities=[kg])
```

**Real-time Persistence Summary:**
- `memory` with auto_save=True: ✅ Saves to file after each change
- `sqlite`: ✅ Always commits after each change
- `json` with auto_save=True: ✅ Saves after each change
- `neo4j`: ✅ Auto-commits on session close

## [1.8.5] - 2026-01-20

### Added

#### KnowledgeGraph Backend Switching

**Dynamic Backend Management:**
- `kg.set_backend(backend, path, migrate=True)` — Switch backends on existing instances
- Optional data migration when changing backends
- Support for custom backend instances via `GraphBackend` type hint
- Updated documentation with backend switching examples

**Use Cases:**
- Start with in-memory during development, switch to SQLite for persistence
- Migrate from JSON to SQLite as data grows
- Test with memory, deploy with production backends

```python
kg = KnowledgeGraph()  # Start in-memory
kg.set_backend("sqlite", path="db.db", migrate=True)  # Switch with migration
```

## [1.8.4] - 2026-01-20

### Added

#### KnowledgeGraph Three-Level Visualization API

**Convenience Methods:**
- `kg.mermaid(**kwargs)` — Low-level: returns raw Mermaid code string
- `kg.render(format, **kwargs)` — Medium-level: renders to mermaid/ascii/html/png/svg formats
- `kg.display(**kwargs)` — High-level: inline Jupyter notebook rendering
- `kg.visualize(**kwargs)` — Returns `GraphView` for full control

**Example:**
```python
from agenticflow.capabilities import KnowledgeGraph

kg = KnowledgeGraph()
kg.remember("Alice", "Person", {"role": "Engineer"})
kg.remember("TechCorp", "Company")
kg.connect("Alice", "works_at", "TechCorp")

# Low-level: raw code
code = kg.mermaid(direction="LR")

# Medium-level: multiple formats
ascii = kg.render("ascii")
html = kg.render("html")

# High-level: Jupyter inline
kg.display()  # Renders directly in notebook

# Full control
view = kg.visualize(direction="LR", group_by_type=True)
view.save("graph.png")
```

### Fixed

- **Redundant Title in HTML Output** — Removed duplicate title from HTML wrapper in `MermaidBackend.to_html()`. Title is now only shown via `_repr_markdown_()` in GraphView, eliminating the "Knowledge Graph (N entities, M relationships)" redundancy in Jupyter notebooks.

### Changed

- **Updated Example** — `examples/capabilities/kg_agent_viz.py` now demonstrates all three visualization API levels
- **Documentation** — Updated `docs/capabilities.md` with comprehensive three-level API documentation

## [1.8.3] - 2026-01-19

### Added

#### Knowledge Graph Visualization

**GraphView Integration:**
- Added `KnowledgeGraph.visualize()` method for graph visualization
- Returns `GraphView` instance with full rendering capabilities
- Supports all GraphView methods: `.mermaid()`, `.ascii()`, `.dot()`, `.url()`, `.html()`, `.png()`, `.svg()`, `.save()`
- Layout options: direction (LR/TB/BT/RL), grouping by entity type, attribute display
- Color-coded entity types: Person (blue), Company (green), Location (orange), Event (purple), Generic (gray)
- Automatic subgraph grouping for organized visualization

**Example:**
```python
from agenticflow.capabilities import KnowledgeGraph

kg = KnowledgeGraph()
# ... add entities and relationships ...

# Visualize with left-right layout and type grouping
view = kg.visualize(direction="LR", group_by_type=True)

# Generate different formats
print(view.mermaid())  # Mermaid diagram
print(view.ascii())    # Terminal-friendly ASCII art
print(view.url())      # Shareable mermaid.ink URL

# Save to files
view.save("graph.mmd")   # Mermaid source
view.save("graph.html")  # Interactive HTML
view.save("graph.png")   # PNG image
```

**Documentation:**
- Added visualization examples to `examples/capabilities/kg_agent_viz.py`
- Added layout comparison demo in `examples/capabilities/kg_layout_demo.py`

### Changed

#### Tool API Improvements

**KnowledgeGraph Tool Redesign:**
- **`query_knowledge`**: Changed from string pattern syntax to structured parameters
  - Before: `query_knowledge(pattern="? -works_at-> TechCorp")`  
  - After: `query_knowledge(source=None, relation="works_at", target="TechCorp")`
  - LLMs now use natural function parameters instead of custom DSL
  - Improved reliability and reduced errors

- **`remember`**: Changed from JSON string to dict/string hybrid
  - Before: `remember(entity="Alice", entity_type="Person", facts='{"role": "CEO"}')`
  - After: `remember(entity="Alice", entity_type="Person", attributes={"role": "CEO"})`
  - Accepts both dict (preferred) and JSON string (backward compatible)
  - Eliminates JSON parsing errors

**Rationale:** LLMs are trained on function calls with typed parameters, not custom string formats. Structured APIs dramatically improve success rates.

#### Executor Improvements

**Per-Turn Tool Call Limits:**
- Changed from cumulative total limit to per-turn limit
  - Before: Max 20 tool calls across entire conversation (cumulative)
  - After: Max 50 tool calls per LLM response turn
  - Prevents agents from hitting artificial limits during productive work
  - `max_iterations` already prevents infinite loops

**Semaphore-Based Concurrency:**
- Added `max_concurrent_tools` parameter (default: 20)
- Parallel tool execution now uses semaphore for concurrency limiting
- Prevents overwhelming rate limits when LLM requests many tools
- Better resource utilization - fast tools don't wait for slow ones
- Smooth execution with constant ~20 concurrent operations

**NativeExecutor Changes:**
```python
executor = NativeExecutor(
    agent,
    max_tool_calls_per_turn=50,    # Was: max_tool_calls=20 (cumulative)
    max_concurrent_tools=20,        # NEW: Concurrency limiting
)
```

### Fixed

- Fixed tool call limit stopping execution before running batched tool calls
- Fixed `GraphView` rendering methods (were incomplete stubs)
- Fixed `KnowledgeGraph` entity type inference in visualization
- Fixed Observer API usage in examples (`observer.events()` not `observer.get_trace()`)

## [1.8.2] - 2026-01-18

### Changed

#### Consolidated ReactiveFlow/EventFlow into Flow

**Unified Flow API:**
- **Consolidated Classes**: Merged `ReactiveFlow` and `EventFlow` functionality into `agenticflow.flow.Flow`.
- **Removed Modules**:
  - `agenticflow.flow.reactive.py` (Deleted)
  - `agenticflow.reactive.py` (Deleted compatibility shim)
  - `agenticflow.agent.flow_helpers.py` (Deleted)
- **Updated Terminology**: Replaced all "reactive" terminology with "flow" or "event-driven".

**Bug Fixes:**
- **Streaming**: Patched `Flow.run_streaming` to correctly support event chaining (`binding.emits`) in fallback mode.
- **Roles**: Fixed indentation bug in `Agent` prompt generation for roles.
- **Observability**: Fixed `Event` vs `Trace` class usage error in tests.

## [1.8.1] - 2026-01-16

### Changed

#### Agent API Simplification

**Removed Dual API Pattern:**
- **Eliminated `config=` parameter** from `Agent.__init__`
  - Removed `@overload` signatures for `config: AgentConfig` parameter
  - `AgentConfig` now internal implementation detail only
  - Simplified public API: `Agent(name, model, tools, ...)`
  - No breaking changes to direct parameter usage

**Flow Pattern Helpers Enhanced:**
- **Added `observer` parameter** to pattern helper functions:
  - `pipeline(stages, observer=None)` — Sequential agent processing
  - `supervisor(coordinator, workers, observer=None)` — Coordinator delegation pattern
  - Enables observability without manual Flow construction

**Examples Updated:**
- Replaced deprecated `Flow(topology=..., agents=...)` API
- Now use pattern helpers: `pipeline([agents])`, `supervisor(coordinator, workers)`
- Updated `examples/basics/hello_world.py` and `examples/basics/roles.py`

**Bug Fixes:**
- Fixed `reactive.py` compatibility module — corrected `Observer` import path
- Fixed `test_graph.py` — added missing `TraceBus` import

**Files Modified:**
- `src/agenticflow/agent/base.py` — Simplified constructor, removed dual API
- `src/agenticflow/flow/patterns/pipeline.py` — Added observer support
- `src/agenticflow/flow/patterns/supervisor.py` — Added observer support
- `examples/basics/hello_world.py` — Uses `pipeline()` helper
- `examples/basics/roles.py` — Uses pattern helpers
- `src/agenticflow/reactive.py` — Fixed import
- `tests/test_graph.py` — Fixed import

**Migration Guide:**
```python
# Before (no longer supported)
from agenticflow import Agent, AgentConfig, Flow

config = AgentConfig(name="Worker", model=model)
agent = Agent(config=config)

flow = Flow(name="basic", agents=[agent], topology="pipeline")

# After (recommended)
from agenticflow import Agent, pipeline

agent = Agent(name="Worker", model=model)
flow = pipeline([agent])
```

## [1.8.0] - 2026-01-16

### Changed

#### Module Reorganization for Better Separation of Concerns

**Core Module Cleanup:**
- **Deleted `core/models.py`** (222 lines) — Removed unused deprecated `ChatModel` wrapper
  - Zero imports found across entire codebase
  - Users now use native model implementations from `agenticflow.models`
  - Eliminated legacy OpenAI SDK wrapper

**Consolidated Utilities:**
- **Inlined `flow/threading.py`** into `flow/reactive.py` (39 lines)
  - Single-use `thread_id_from_data()` function moved to its only call site
  - Eliminated unnecessary micro-module
  - Cleaner reactive flow implementation

**Foundational Types Relocated to Core:**
- **Moved `flow/utils.py` → `core/utils.py`** (128 lines)
  - Generic primitives now in foundational layer:
    - `IdempotencyGuard` — Event deduplication
    - `RetryBudget` — Bounded retry tracking
    - `emit_later` — Delayed event emission
    - `jittered_delay` — Exponential backoff calculator
    - `Stopwatch` — Performance timing
  - These are framework primitives, not flow-specific logic

- **Moved `context.py` → `core/context.py`** (113 lines)
  - `RunContext` is foundational dependency injection mechanism
  - Used across executors, interceptors, tools (6 locations)
  - Now properly located in core module

**Import Updates:**
- Updated 6 import statements: `agenticflow.context` → `agenticflow.core.context`
  - `src/agenticflow/__init__.py`
  - `src/agenticflow/executors/native.py`
  - `src/agenticflow/tools/base.py`
  - `src/agenticflow/interceptors/base.py`
  - `tests/test_interceptors.py`
  - `README.md`

**Backward Compatibility:**
- Added `reactive.py` compatibility module at package root
  - Re-exports from `flow.reactive`, `flow.triggers`, `flow.skills`
  - Maintains compatibility for code importing from `agenticflow.reactive`
  - Zero breaking changes for existing users

**Core Module Exports:**
- `core/__init__.py` now exports reactive utilities:
  - `RunContext`, `EMPTY_CONTEXT`
  - `IdempotencyGuard`, `RetryBudget`
  - `emit_later`, `jittered_delay`, `Stopwatch`

**Architecture Improvements:**
- ✅ **Clear separation**: `core/` = foundational primitives, `flow/` = orchestration logic
- ✅ **502 lines** moved to correct architectural locations
- ✅ **261 lines** deleted (unused code and consolidation)
- ✅ **1,333 passing tests** — Zero regressions
- ✅ **No breaking changes** — Backward compatibility maintained

### Migration Guide

**Option 1: No changes required** (recommended)
```python
# All imports from main package continue to work
from agenticflow import RunContext, EMPTY_CONTEXT
from agenticflow.reactive import ReactiveFlow  # Compatibility module
```

**Option 2: Update to new locations** (optional)
```python
# Old → New
from agenticflow.context import RunContext  
# → from agenticflow.core.context import RunContext

from agenticflow.flow.utils import IdempotencyGuard  
# → from agenticflow.core.utils import IdempotencyGuard
```

The `reactive.py` compatibility module maintains backward compatibility for all `agenticflow.reactive` imports.

### Summary

This release focuses on architectural cleanup and proper separation of concerns. Generic utilities and foundational types have been moved from `flow/` to `core/`, while unused deprecated code has been removed. The module structure now clearly reflects the framework's layered architecture. **No action required for existing users.**

---

## [1.7.0] - 2026-01-14

### Added

#### Agent Request/Response (A2A) Communication (Phase 2.2)

**Core Infrastructure:**

- **`AgentRequest` and `AgentResponse`**: Dataclasses for structured agent-to-agent communication
  - `AgentRequest(from_agent, to_agent, task, data, correlation_id)` — Request with correlation tracking
  - `AgentResponse(from_agent, to_agent, result, data, correlation_id, success, error)` — Response with success/error handling
  - Automatic correlation ID generation (`uuid.uuid4().hex[:8]`)
  - Factory functions: `create_request()` and `create_response()`
  - `to_event()` methods for emitting as events

- **`ExecutionContext.delegate_to()`**: Direct agent-to-agent delegation
  - `async delegate_to(agent_name, task, data, wait, timeout_ms)` — Delegate task to another agent
  - Wait for response or fire-and-forget
  - Timeout support for synchronous delegation
  - Event queue management for pending responses
  - Emits `agent.request` events automatically

- **`ExecutionContext.reply()`**: Send responses back to requesting agents
  - `reply(result, success, error)` — Reply to agent request
  - Automatic correlation ID tracking
  - Success/error status handling
  - Emits `agent.response` events

**Declarative Delegation (Unified Architecture):**

- **`DelegationMixin`**: Single source of truth for delegation configuration
  - `configure_delegation(agent, can_delegate, can_reply, trigger_config)` — Unified method
  - Auto-injects `delegate_to` and `reply_with_result` tools based on policy
  - Auto-enhances agent system prompts with delegation instructions
  - Policy enforcement — validates delegation targets against allowed list
  - Works across ALL flow types (reactive flows and topologies)

- **`BaseFlow` and `BaseTopology`**: Now inherit from `DelegationMixin`
  - All flows automatically support A2A delegation
  - Single implementation, zero code duplication
  - DRY architecture — fix bugs once, benefits everywhere

- **Delegation Tools**: Auto-generated based on configuration
  - `create_delegate_tool(flow, agent_name, specialists)` — Creates `delegate_to` tool
  - `create_reply_tool(flow, agent_name)` — Creates `reply_with_result` tool
  - Policy enforcement in tool execution
  - Event-based communication via AgenticFlow event system

**Reactive Flow Updates:**

- **Simplified Registration API**: Intuitive syntax for common A2A patterns
  - `flow.register(agent, handles=True)` — Agent handles requests for itself (uses `agent.name`)
  - `flow.register(agent, on="task.created", can_delegate=["specialist"])` — Declarative delegation
  - `flow.register(agent, on="request.*", can_reply=True)` — Enable reply capability
  - Backward compatible with advanced trigger syntax
  - No need to import `react_to()` or `for_agent()` for simple cases

- **ReactiveFlow refactored**: Removed ~70 lines of duplicate code
  - Now uses inherited `configure_delegation()` from `DelegationMixin`
  - Specialists auto-discovered from registered handlers

**Topology Updates:**

- **`AgentConfig` enhanced**: Delegation parameters
  - `can_delegate: list[str] | bool | None` — Who agent can delegate to
  - `can_reply: bool` — Whether agent handles delegated requests
  - Backward compatibility with `can_delegate_to` (deprecated)
  - Legacy support in `__post_init__` for smooth migration

- **All topology patterns support delegation**:
  - **Supervisor** — Coordinator delegates to workers, hierarchical sub-delegation
  - **Pipeline** — Stages can delegate to external specialists
  - **Mesh** — Collaborative agents with specialist delegation
  - All patterns call `super().__post_init__()` to apply delegation config

- **`BaseTopology`**: Inherits from `DelegationMixin`
  - `__post_init__()` configures delegation for all agents in topology
  - Resolves specialists based on topology type (workers vs all agents)
  - Consistent API across all coordination patterns

**Examples and Tests:**

- **Reactive examples**: 5 patterns in [examples/reactive/a2a_delegation.py](examples/reactive/a2a_delegation.py)
  - Simple delegation: coordinator → specialist
  - Multi-specialist team: parallel routing by task type
  - Chain delegation: PM → Architect → Developer
  - Parallel delegation: coordinator → multiple specialists
  - Request-response: bidirectional communication

- **Topology examples**: 5 patterns in [examples/topologies/delegation.py](examples/topologies/delegation.py)
  - Supervisor with hierarchical delegation
  - Pipeline with specialist delegation
  - Mesh with external specialists
  - Dynamic delegation policies
  - Cross-topology delegation patterns

- **Tests**: 13/15 passing tests in `tests/test_a2a.py`
  - AgentRequest/AgentResponse creation and serialization
  - ExecutionContext delegation and reply
  - Correlation ID tracking
  - Wait/timeout behavior (2 tests skipped pending mocks)
  - All topologies verified with delegation working

**Documentation:**

- **[docs/a2a.md](docs/a2a.md)**: Complete A2A delegation guide
  - Registration API (simple and advanced syntax)
  - Declarative configuration with `can_delegate` and `can_reply`
  - Delegation patterns (coordinator-specialist, chains, parallel)
  - ExecutionContext API reference
  - Request/response tracking
  - Examples across reactive flows and topologies

- **[docs/topologies.md](docs/topologies.md)**: Updated with A2A delegation
  - New section on Agent-to-Agent Delegation
  - Configuration parameters (`can_delegate`, `can_reply`)
  - Auto-injection and auto-enhancement explanation
  - Delegation patterns (coordinator, hierarchical, specialist)
  - Examples for all topology types

- **[README.md](README.md)**: Updated topologies section
  - Added A2A delegation example in topologies overview
  - Highlights declarative configuration
  - References comprehensive documentation

### Changed

- **`ReactiveFlow.register()`**: Enhanced with delegation parameters
  - Added `can_delegate` parameter for declarative delegation policy
  - Added `can_reply` parameter to enable response handling
  - `on`, `handles`, `when`, `priority`, `emits` parameters for simple syntax
  - Backward compatible with `triggers` parameter for advanced usage
  - Removed redundant `initial_event` default (no longer adds "task.created" automatically)

- **Context Consolidation**: Unified ExecutionContext in `flow/context.py`
  - `ExecutionContext` — Unified context for reactive agents (delegate_to, reply)
  - `RunContext` — Dependency injection (unchanged, separate purpose)
  - `ContextStrategy` — Multi-round history for topologies (unchanged, separate purpose)
  - `ReactiveContext` — Backward compatibility alias for ExecutionContext

- **Code Architecture**: DRY principle with DelegationMixin
  - Created `DelegationMixin` in `flow/delegation.py`
  - `BaseFlow` and `BaseTopology` both inherit from `DelegationMixin`
  - Removed ~70 lines of duplicate delegation code from `ReactiveFlow`
  - Single implementation benefits all flow types

## [1.6.0] - 2026-01-14

### Added

#### Streaming Reactions for ReactiveFlow (Phase 2.1)

- **`ReactiveFlow.run_streaming()`**: Real-time token-by-token streaming from event-driven flows
  - Returns `AsyncIterator[ReactiveStreamChunk]` for progressive output display
  - Leverages existing agent streaming infrastructure (`agent.run(stream=True)`)
  - Full event context in each chunk: agent name, event ID, event type
  - Sequential agent execution in streaming mode to preserve order
  - Example: [examples/reactive/streaming.py](examples/reactive/streaming.py)

- **`ReactiveStreamChunk`**: Streaming chunk with reactive flow context
  - `agent_name: str` — Which agent is currently streaming
  - `event_id: str` and `event_name: str` — Event that triggered the agent
  - `content: str` and `delta: str` — Token content
  - `is_final: bool` — Whether this is the last chunk from the agent
  - `finish_reason: str | None` — Why streaming stopped (stop, length, error, etc.)
  - `metadata: dict[str, Any]` — Additional context (round number, etc.)

- **Multi-Agent Streaming**: Track which agent is speaking in real-time
  - Agent name changes signal transition to next agent in flow
  - Enables progress indicators and agent-specific UI styling
  - Supports conditional routing, fan-out, and chained patterns

- **Tests**: 11 passing tests with real LLM in `tests/test_reactive_streaming.py`
  - Basic streaming, chunk properties, multi-agent coordination
  - Event context preservation, conditional triggers
  - Configuration respect, backward compatibility

#### Distributed Transport (Phase 1.3)

- **Transport Protocol**: Pluggable event transport for cross-process communication
  - Abstract `Transport` interface: connect, disconnect, publish, subscribe, unsubscribe
  - Pattern matching with wildcards: `task.*` (single-level), `**` (multi-level)
  - Multiple subscribers per pattern with subscription management

- **`LocalTransport`**: In-memory asyncio.Queue-based transport (zero dependencies)
  - Single-process event routing with pattern matching
  - Ideal for development and testing

- **`RedisTransport`**: Distributed Redis Pub/Sub transport (optional `redis>=5.0.0`)
  - Cross-process agent communication
  - Event serialization with `dataclasses.asdict()`
  - Automatic reconnection and error handling

- **`EventBus` Integration**: Optional transport parameter for distributed routing
  - `EventBus(transport=RedisTransport(...))` enables distributed events
  - Backward compatible — defaults to local behavior

- **Tests**: 8 passing LocalTransport tests, 3 skipped Redis integration tests
- **Example**: [examples/reactive/transport.py](examples/reactive/transport.py) with mock Redis fallback

## [1.5.0] - 2026-01-14

### Added

#### Flow-Level Checkpointing for Imperative Flows

- **Imperative Flow Checkpointing**: Added crash recovery support for `Flow` (Supervisor, Pipeline, Mesh, Hierarchical)
  - `FlowConfig.checkpoint_every: int` — Save checkpoint after every N steps
  - `FlowConfig.flow_id: str | None` — Unique flow identifier for checkpoint tracking
  - `Flow.resume(checkpoint_id)` — Resume execution from a saved checkpoint
  - Works with all checkpoint backends: `FileCheckpointer`, `MemoryCheckpointer`, `PostgresCheckpointer`
  - Example: [examples/flow/checkpointing_demo.py](examples/flow/checkpointing_demo.py)

- **Human-in-the-Loop (HITL) for Reactive Flows**: Integrated existing HITL system into `EventFlow`
  - `ReactionType.AWAIT_HUMAN` — Pause flow and request human approval
  - `Trigger.breakpoint: Any` — Attach metadata for approval context
  - `EventFlow.hitl_handler` — Configure approval handler (console, API, etc.)
  - Emits `flow.paused` and `flow.resumed` events
  - Example: [examples/reactive/hitl_approval.py](examples/reactive/hitl_approval.py)

#### Persistent Checkpointing for ReactiveFlow

- **`FlowState`**: Serializable snapshot of flow execution state for persistence
  - Captures task, events processed, pending events, context, and output
  - Serializable to/from dict with `to_dict()` and `from_dict()`
  - Includes `flow_id`, `checkpoint_id`, and `round` tracking

- **`Checkpointer` Protocol**: Abstract interface for checkpoint storage backends
  - `save(state)` / `load(checkpoint_id)` / `load_latest(flow_id)`
  - `list_checkpoints(flow_id)` / `delete(checkpoint_id)`

- **`MemoryCheckpointer`**: In-memory checkpointer for development and testing
  - Automatic pruning with configurable `max_checkpoints_per_flow`

- **`FileCheckpointer`**: File-based JSON checkpointer for simple persistence
  - Stores each checkpoint as `{checkpoint_id}.json`

- **`ReactiveFlow.resume(state)`**: Resume a flow from a saved checkpoint
  - Restores pending events, context, and continues processing
  - Supports crash recovery for long-running flows

- **`ReactiveFlowConfig` Extensions**:
  - `flow_id: str | None` — Fixed flow ID (auto-generated if None)
  - `checkpoint_every: int = 0` — Checkpoint every N rounds (0 = disabled)

- **`ReactiveFlowResult` Extensions**:
  - `flow_id: str | None` — Flow ID for this execution
  - `checkpoint_id: str | None` — Last checkpoint ID if checkpointing enabled

### Example

```python
from agenticflow.reactive import (
    ReactiveFlow, ReactiveFlowConfig,
    MemoryCheckpointer, react_to
)

# Enable checkpointing
checkpointer = MemoryCheckpointer()
config = ReactiveFlowConfig(checkpoint_every=1)

flow = ReactiveFlow(config=config, checkpointer=checkpointer)
flow.register(agent, [react_to("task.created")])

result = await flow.run("Process data")
print(f"Flow: {result.flow_id}, Checkpoint: {result.checkpoint_id}")

# Resume after crash
state = await checkpointer.load_latest(result.flow_id)
if state:
    result = await flow.resume(state)
```

### Changed

#### Checkpointer Module Reorganization (Non-Breaking)

- **Module Location**: Moved checkpointer from `agenticflow.reactive.checkpointer` to `agenticflow.flow.checkpointer`
  - Reason: Checkpointing is shared infrastructure used by both imperative (`Flow`) and reactive (`EventFlow`) orchestration
  - Clearer architecture: Flow-level persistence lives in `flow/`, agent-level memory stays in `agent/`
  - **Backward compatible**: Old imports from `reactive.checkpointer` still work via re-exports

### Migration

```python
# Recommended (new location)
from agenticflow.flow.checkpointer import FileCheckpointer, FlowState

# Still works (re-exported for backward compatibility)
from agenticflow.reactive.checkpointer import FileCheckpointer, FlowState
```

---

## [1.4.0] - 2026-01-11


### Added

#### Tool Return Type Visibility

- **Return Type Extraction**: The `@tool` decorator now extracts return type information and includes it in tool descriptions
  - Return type annotations (e.g., `-> dict[str, int]`) are converted to readable strings
  - Docstring `Returns:` sections are parsed and combined with type info
  - LLM sees: `"Get weather data. Returns: dict[str, int] - A dictionary with temp and humidity."`
  - Access via `tool.return_info` property
  - Helps LLM understand expected output format from each tool

#### External Event Sources & Sinks

- **`FileWatcherSource`**: Monitor directories for file changes, emit events for created/modified/deleted files
- **`WebhookSource`**: Receive HTTP webhooks as events (requires `starlette`, `uvicorn`)
- **`RedisStreamSource`**: Consume from Redis Streams with consumer group support (requires `redis`)
- **`WebhookSink`**: POST events to HTTP endpoints with pattern matching (requires `httpx`)
- **`EventFlow.source()`**: Register external event sources to inject events into reactive flows
- **`EventFlow.sink()`**: Register sinks to emit events to external systems

### Changed

#### Observability Renamed for Clarity (Breaking)

- **File Renames**:
  - `observability/event.py` → `trace_record.py`
  - `tests/test_events.py` → `test_traces.py`

- **Class/Function Renames** (observability module only):
  - `Event` → `Trace`
  - `EventType` → `TraceType`
  - `EventBus` → `TraceBus`
  - `get_event_bus()` → `get_trace_bus()`
  - `set_event_bus()` → `set_trace_bus()`

- **Core orchestration unchanged**: `agenticflow.events.Event` and `agenticflow.events.EventBus` remain for agent-to-agent routing

### Migration

```python
# Before (1.3.0): Observability
from agenticflow.observability import Event, EventType, EventBus, get_event_bus

# After (1.4.0): Observability
from agenticflow.observability import Trace, TraceType, TraceBus, get_trace_bus

# Core orchestration (unchanged)
from agenticflow.events import Event, EventBus
```

---

## [1.3.0] - 2026-01-10

### Changed

- **LLM Channel Now Opt-in**: `Observer.debug()` and `Observer.trace()` no longer include `Channel.LLM` by default
  - LLM request/response content requires explicit opt-in for privacy
  - Users must add `Channel.LLM` to their channels list to see raw LLM content
  - This is a **breaking change** for users who relied on debug/trace showing LLM payloads
  - Updated documentation to reflect opt-in behavior

### Migration

To restore previous behavior where debug/trace included LLM content:

```python
# Before (1.2.0): LLM content shown automatically
observer = Observer.debug()

# After (1.3.0): Explicitly opt-in to LLM content
observer = Observer(
    level=ObservabilityLevel.DEBUG,
    channels=[Channel.AGENTS, Channel.TOOLS, Channel.LLM, ...],
)
```

---

## [1.2.0] - 2026-01-03

### Added

#### Enhanced Observability Features

- **Token Usage Tracking**: Automatic tracking and display of LLM token consumption
  - Track input/output/total tokens per agent and globally
  - Display token counts in LLM response events: `[llm-response] (2.1s) ~850 tokens (650 in, 200 out)`
  - Detailed token breakdown in `observer.summary()` with per-agent statistics
  - Configurable via `track_tokens` and `show_token_usage` flags
  - Helps with cost monitoring, usage analytics, and budget tracking

- **Structured Event Export**: Export captured events to multiple formats for analysis
  - JSONL format: One event per line, ideal for streaming logs and log aggregation systems
  - JSON format: Complete event array with full structure for detailed analysis
  - CSV format: Tabular data perfect for spreadsheet analysis and reporting
  - Usage: `observer.export("trace.jsonl", format="jsonl|json|csv")`
  - Enables integration with monitoring systems, audit trails, and ML analysis

- **Progress Step Indicators**: Visual progress for multi-step agent operations
  - Show "Step N/M: description" during long-running workflows
  - Automatic tracking of current/total steps per agent
  - Configurable via `show_progress_steps` flag
  - Improves UX and helps identify bottlenecks in agent execution

- **Enhanced Error Context**: Actionable error messages with contextual suggestions
  - Smart pattern matching for common errors (permission denied, connection refused, timeout, etc.)
  - Automatic inclusion of file/line/tool context when available
  - Actionable suggestions displayed at DEBUG level
  - Supported patterns: permission, connection, timeout, not found, invalid credentials
  - Reduces debugging time and enables self-service problem resolution

- **State Change Diff Visualization**: Visual diffs for entity state changes
  - Shows `old_value → new_value` with color coding for AGENT_STATUS_CHANGED events
  - Tracks state snapshots per entity for comparison
  - Enabled at DETAILED level or higher
  - Ideal for reactive agents, task tracking, and debugging state transitions

### Changed

- **Observer Configuration**: Extended `ObserverConfig` with new settings
  - Added `track_tokens: bool = True` - Enable/disable token tracking
  - Added `show_token_usage: bool = True` - Display tokens in LLM events
  - Added `show_cost_estimates: bool = False` - Show estimated costs (future enhancement)
  - Added `show_progress_steps: bool = False` - Enable step progress indicators

- **Observer Internal State**: Enhanced tracking capabilities
  - Token usage tracking: `_token_usage` dict per agent, `_total_tokens` global
  - Progress tracking: `_progress_steps` dict for multi-step operations
  - State management: `_state_snapshots` dict for diff visualization
  - Error context: `_error_suggestions` dict with pattern-based recommendations

### Examples

- Added `examples/observability/enhanced_features.py` - Comprehensive demo of all new features
- Added `examples/observability/custom_truncation.py` - Configuration examples for truncation

### Documentation

- Enhanced docstrings for all new observer features
- Added inline examples for export functionality
- Documented token tracking configuration options

## [1.1.0] - Previous Release

### Added

- Professional observability formatting with bracket notation `[event-type]`
- Configurable truncation per content type (tool args, results, messages)
- Improved color scheme (grey labels, green success, blue tools)
- Increased max_iterations from 10 to 25 globally

### Changed

- Standardized all event output to professional bracket notation
- Separated completion status from output content
- Removed emoji-heavy output in favor of clean, professional format
- Enhanced visual hierarchy with consistent colors

### Fixed

- Agent name alignment with 12-character padding
- Duration formatting consistency across all events
- Truncation now respects word boundaries

---

## Version History

- **1.3.0** (2026-01-10) - LLM channel opt-in by default for privacy
- **1.2.0** (2026-01-03) - Enhanced observability with token tracking, export, and contextual features
- **1.1.0** (Previous) - Professional formatting and configuration improvements
