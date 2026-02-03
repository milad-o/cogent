# Changelog

All notable changes to Cogent will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [1.15.0] - 2026-02-03

### Added

- **Native Subagent Support** — Built-in multi-agent coordination via `subagents=` parameter
  - New `Agent(subagents={"name": agent})` parameter for delegation to specialized agents
  - LLM automatically calls subagents as tools - no custom syntax needed
  - Full metadata preservation: tokens, model calls, and delegation chain tracking
  - Automatic metadata aggregation across all agents in the hierarchy
  - New response fields: `response.metadata.subagent_calls`, `response.metadata.delegation_chain`
  - Comprehensive documentation: See [Multi-Agent Guide](https://docs.cogent.ai/subagents.html)

- **Enhanced Observability for Subagents** — Clear distinction between tools and subagents
  - Observability events now include `is_subagent` flag
  - Console formatter shows `[subagent-call]` and `[subagent-result]` labels
  - Tool decision events separate subagents from regular tools
  - Better visibility into delegation patterns and agent interactions

- **Gemini 3 thinking support** — Gemini 3 models now support thinking with `thinking_budget`
  - Added `gemini-3-pro-preview`, `gemini-3-flash-preview`, `gemini-3-pro-image-preview` to `THINKING_BUDGET_MODELS`
  - `thought_signature` field now parsed and stored in Gemini responses
  - Use `thinking_budget` parameter to control thinking token budget (e.g., 8192)

- **Comprehensive structured output types** — Full type system support
  - Collections: `list[T]`, `set[T]`, `tuple[T, ...]` (wrap in models for reliability)
  - Union types: `Union[A, B]` for polymorphic responses where agent chooses schema
  - Enum types: `class Priority(str, Enum)` for type-safe choices with behavior
  - None type: `type(None)` for confirmation responses
  - All types work with automatic validation and retry
  
- **dict type support for dynamic schemas** — `output=dict` allows agent to decide structure
  - Agent chooses fields and structure based on content
  - Useful when output structure varies or is unknown beforehand
  - Example: `output=dict` returns `{"sentiment": "positive", "score": 8, ...}`

### Deprecated

- **Agent.as_tool()** — Deprecated in favor of native `subagents=` parameter
  - Will be removed in v2.0.0
  - Migration: Replace `tools=[agent.as_tool()]` with `subagents={"name": agent}`
  - Benefits of migration:
    - Accurate token counting across all agents
    - Full Response metadata preserved (not just content string)
    - Automatic delegation chain tracking
    - Better observability with `[subagent-call]` labels
  - See migration guide: [Subagents Documentation](https://docs.cogent.ai/subagents.html#migration-guide)

### Fixed

- **Gemini tool schema conversion** — Pydantic BaseModel schemas now properly converted
  - Fixed `_tools_to_gemini()` to handle Pydantic models via `model_json_schema()`
  - LLMs now receive correct parameter schemas including required fields
  - Resolves issues with empty `{}` arguments on tool calls

## [1.0.4] - 2026-02-02

### Added

- **Context Propagation** — RunContext now flows automatically through agent delegations
  - Added `query` field to RunContext for tracking original user request
  - Auto-populated by native executor on first `agent.run()` call
  - Accessible in sub-agents via `ctx.query` throughout delegation chain
  - Enables sub-agents to understand broader context while working on subtasks

- **Agent.as_tool() API Improvements** — Better semantics for context handling
  - Changed parameter: `propagate_context` → `isolate_context` (inverted logic)
  - Default behavior: context flows automatically (`isolate_context=False`)
  - Makes agent-as-tool consistent with regular tools
  - Use `isolate_context=True` to create explicit context boundaries

- **model_kwargs Parameter** — Pass model-specific configuration to agents
  - New `Agent(model_kwargs={...})` parameter for model-specific settings
  - Only applies when using string model names
  - Example: `model_kwargs={"thinking_budget": 16384}` for Gemini thinking
  - Ignored when passing ChatModel instances (configure instance directly)

- **Common Context Patterns Documentation** — Practical extension patterns
  - Delegation depth tracking with `max_depth` protection
  - Retry tracking with adaptive strategies
  - Task lineage with parent-child relationships
  - Execution timing with deadlines
  - Composable patterns for comprehensive tracking

### Changed

- **Gemini Defaults** — Cost-efficient defaults for production use
  - Default model: `gemini-2.0-flash` → `gemini-2.5-flash` (latest stable)
  - Default `thinking_budget`: `None` → `0` (opt-in for cost efficiency)
  - Thinking only enables when `thinking_budget > 0` or `thinking_level` set
  - Prevents `thought_signature` errors with `budget=0`

- **Tool Calling Reliability** — Improved guidance for better LLM compliance
  - Added `[REQUIRED]` prefix to task parameter descriptions in agent-as-tool
  - Raises `ValueError` when task is empty to trigger retry mechanism
  - Auto-generates descriptive tool names with usage examples

### Breaking Changes

- `Agent.as_tool(propagate_context=True)` → `as_tool(isolate_context=False)`
  - Logic inverted: context flows by default now (like regular tools)
  - Migration: `propagate_context=False` → `isolate_context=True`
  - Migration: `propagate_context=True` → remove parameter (it's the default)

- `GeminiChat` thinking_budget default changed from `None` to `0`
  - Must explicitly enable: `thinking_budget > 0` (recommended: 8192-16384)
  - Cost-efficient default prevents accidental thinking token charges

## [1.0.3] - 2026-01-29

### Added

- **Bare Type Support for Structured Output** — Agents can now output primitive types directly
  - `output=str` — Return bare string values
  - `output=int` — Return bare integers
  - `output=bool` — Return bare booleans
  - `output=float` — Return bare floats
  - `output=Literal["A", "B"]` — Return bare Literal choices
  - Access with `result.content.data` (returns value directly, not wrapped in model)
  - LLM responses automatically unwrapped from single-key JSON dicts
  - Example: `examples/basics/literal_responses.py` demonstrates both bare types and single-field models

### Changed

- **Schema validation** — `schema_to_json()` now handles primitive types and Literal
- **Output parsing** — `validate_and_parse()` extracts values from single-key JSON objects

## [1.0.2] - 2026-01-28

### Added

- **Observer v2** — Redesigned observability system with simpler API
  - `Observer(level="progress")` — Clean console output for agent activity
  - `Observer(level="verbose")` — Show tool arguments and details
  - `Observer(level="debug")` — Include LLM request/response events
  - String-based event types: `agent.invoked`, `agent.thinking`, `tool.called`, `tool.result`, `agent.responded`
  - Proper tool call correlation with UUID-based `call_id` tracking
  - Consistent dict-style formatting for args `{key='value'}` and results
  - `observer.summary()` — Event count summary
  - `observer.on(pattern, handler)` — Subscribe to event patterns (e.g., `"tool.*"`)
  - `observer.on_all(handler)` — Subscribe to all events

### Changed

- **Executor event emission** — Events now flow through Observer v2 when attached
  - `emit_event()` helper prefers Observer over TraceBus
  - Tool calls include UUID-based `call_id` for result correlation
  - `llm.request` and `llm.response` events filtered to DEBUG level

### Fixed

- Duplicate `[completed]` events — Agent now emits single completion event with tokens
- Tool call/result tracking — UUIDs correlate parallel tool executions

## [1.0.1] - 2026-01-27

### Added

- **TaskBoard** — Agent task tracking and verification system
  - `taskboard=True` parameter to enable task tracking
  - Tools: `add_task`, `update_task`, `add_note`, `verify_task`, `get_taskboard_status`
  - Auto-injected instructions for complex task breakdown
  - Progress tracking and task verification
- **Token Usage Fix** — Response now shows actual token usage from model
  - `Response.metadata.tokens` properly aggregates from AIMessage metadata
  - Observer displays tokens in `[completed]` line
- **Observer(level="detailed")** — New preset showing tool calls with timestamps

### Fixed

- Pylance errors in `agent/base.py` (missing imports, legacy role references)
- Token usage was always `None` — now properly aggregated from executor messages

## [1.0.0] - 2026-01-27

### Added

**Core Agent System**
- `Agent` class with streaming, tools, and memory support
- Multi-provider LLM support: OpenAI, Anthropic, Google, Mistral, Cohere, Groq, Ollama, Azure, AWS Bedrock, Cloudflare Workers AI
- `@tool` decorator for creating agent tools from functions
- Automatic JSON schema generation from type hints and docstrings
- Context injection (`RunContext`) for tools
- NativeExecutor with parallel tool execution (asyncio.gather)

**Memory System**
- Fuzzy-first search with rapidfuzz (2,800× faster than semantic)
- Semantic search fallback for complex queries
- Configurable search strategy: `fuzzy_first`, `semantic_only`, `fuzzy_only`
- Thread-based conversation memory
- Multiple backends: in-memory, SQLite, PostgreSQL, Redis

**Capabilities (Pre-built Tool Classes)**
- `WebSearch` — Web search and URL fetching
- `FileSystem` — Sandboxed file operations
- `CodeSandbox` — Safe Python execution
- `KnowledgeGraph` — Entity/relationship memory with multi-hop reasoning
- `Browser` — Headless browser automation
- `PDF` — PDF reading and extraction
- `Shell` — Command execution
- `MCP` — Model Context Protocol integration
- `Spreadsheet` — CSV/Excel operations
- `Summarizer` — Text summarization

**Retrieval & RAG**
- Vector stores: FAISS, Qdrant, PgVector, Chroma, Redis
- Retrievers: BM25, semantic, ensemble, parent-child
- Cross-encoder reranking
- Document loaders and chunking strategies

**Observability**
- OpenTelemetry integration
- Structured logging with structlog
- Trace context propagation

**Resilience**
- Retry policies with exponential backoff
- Circuit breakers
- Rate limiting
- Graceful degradation

### Documentation
- Comprehensive guides for all modules
- Production-ready examples
- API reference

---

*This is the initial public release of Cogent, a complete rewrite focused on simplicity, performance, and production readiness.*
