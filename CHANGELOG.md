# Changelog

All notable changes to Cogent will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

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
