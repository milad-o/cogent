# Changelog

All notable changes to Cogent will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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
- **Observer.detailed()** — New preset showing tool calls with timestamps

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
