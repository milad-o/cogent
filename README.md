# Cogent

<p align="center">
  <strong>Build AI agents that actually work.</strong>
</p>

<p align="center">
  <a href="https://github.com/milad-o/cogent/releases"><img src="https://img.shields.io/badge/version-1.0.1-blue.svg" alt="Version"></a>
  <a href="https://github.com/milad-o/cogent/blob/main/LICENSE"><img src="https://img.shields.io/badge/license-MIT-green.svg" alt="License"></a>
  <a href="https://www.python.org/downloads/"><img src="https://img.shields.io/badge/python-3.13+-blue.svg" alt="Python"></a>
  <a href="https://milad-o.github.io/cogent"><img src="https://img.shields.io/badge/docs-latest-brightgreen.svg" alt="Documentation"></a>
  <a href="https://github.com/milad-o/cogent/actions"><img src="https://img.shields.io/badge/build-passing-brightgreen.svg" alt="Build"></a>
  <a href="https://github.com/milad-o/cogent/tree/main/tests"><img src="https://img.shields.io/badge/tests-1925-success.svg" alt="Tests"></a>
</p>

<p align="center">
  <a href="#installation">Installation</a> •
  <a href="#quick-start">Quick Start</a> •
  <a href="#core-architecture">Architecture</a> •
  <a href="#capabilities">Capabilities</a> •
  <a href="#examples">Examples</a>
</p>

---

Cogent is a **production AI agent framework** built on cutting-edge research in memory control and semantic caching. Unlike frameworks focused on multi-agent orchestration, Cogent emphasizes **bounded memory**, **reasoning artifacts caching**, and **tool augmentation** for superior performance and reliability.

**Why Cogent?**

- 🧠 **Memory Control** — Bio-inspired bounded memory prevents context drift and poisoning
- ⚡ **Semantic Caching** — Cache reasoning artifacts (intents, plans) at 80%+ hit rates
- 🚀 **Fast** — Parallel tool execution, cached model binding, direct SDK calls
- 🔧 **Simple** — Define tools with `@tool`, create agents in 3 lines, no boilerplate
- 🏭 **Production-ready** — Built-in resilience, observability, and security interceptors
- 📦 **Batteries included** — File system, web search, code sandbox, browser, PDF, and more

```python
from cogent import Agent, tool

@tool
async def search(query: str) -> str:
    """Search the web."""
    # Your search implementation
    return results

agent = Agent(name="Assistant", model="gpt-4o-mini", tools=[search])
result = await agent.run("Find the latest news on AI agents")
```

---

## 🎉 Latest Changes (v1.0.1 - January 2026)

**TaskBoard & Observability** 📋
- ✨ **TaskBoard** — Built-in task tracking system for complex multi-step workflows
- 🔧 **Token aggregation** — Fixed token usage reporting to aggregate from all executor messages
- 📊 **Observer(level="detailed")** — New preset for detailed observability output

**TaskBoard tools:**
- `add_task` — Add tasks with optional dependencies
- `update_task` — Update task status (pending → in_progress → completed/blocked)
- `add_note` — Add observations and notes during execution
- `verify_task` — Verify task completion with evidence
- `get_taskboard_status` — Get full taskboard state

```python
# Enable TaskBoard for complex tasks
agent = Agent(name="Planner", model="gpt-4o", taskboard=True)
result = await agent.run("Design a REST API with authentication")
```

See [CHANGELOG.md](CHANGELOG.md) for full version history and migration guide.

---

## Features

- **Native Executor** — High-performance parallel tool execution with zero framework overhead
- **Native Model Support** — OpenAI, Azure, Anthropic, Gemini, Groq, Ollama, Custom endpoints
- **Multi-Agent Patterns** — Supervisor, Pipeline, Mesh, Hierarchical
- **Capabilities** — Filesystem, Web Search, Code Sandbox, Browser, PDF, Shell, MCP, Spreadsheet, and more
- **RAG Pipeline** — Document loading, per-file-type splitting, embeddings, vector stores, retrievers
- **Memory & Persistence** — Conversation history, long-term memory with fuzzy matching ([docs/memory.md](docs/memory.md))
- **Memory Control (ACC)** — Bio-inspired bounded memory prevents drift ([docs/acc.md](docs/acc.md))
- **Semantic Caching** — Cache reasoning artifacts at 80%+ hit rates ([docs/memory.md#semantic-cache](docs/memory.md#semantic-cache))
- **Graph Visualization** — Mermaid, Graphviz, ASCII diagrams for agents and patterns
- **Observability** — Tracing, metrics, progress tracking, structured logging
- **Interceptors** — Budget guards, rate limiting, PII protection, tool gates
- **Resilience** — Retry policies, circuit breakers, fallbacks
- **Human-in-the-Loop** — Tool approval, guidance, interruption handling
- **Streaming** — Real-time token streaming with callbacks
- **Structured Output** — Type-safe responses with Pydantic schemas
- **Reasoning** — Extended thinking mode with chain-of-thought

---

## Modules

Cogent is organized into focused modules, each with multiple backends and implementations.

### `cogent.models` — LLM Providers

Native SDK wrappers for all major LLM providers with zero abstraction overhead.

| Provider | Chat | Embeddings | String Alias | Notes |
|----------|------|------------|--------------|-------|
| **OpenAI** | `OpenAIChat` | `OpenAIEmbedding` | `"gpt4"`, `"gpt-4o"`, `"gpt-4o-mini"` | GPT-4o series, o1, o3 |
| **Azure** | `AzureOpenAIChat` | `AzureOpenAIEmbedding` | — | Managed Identity, Azure AD support |
| **Anthropic** | `AnthropicChat` | — | `"claude"`, `"claude-opus"` | Claude 3.5 Sonnet, extended thinking |
| **Gemini** | `GeminiChat` | `GeminiEmbedding` | `"gemini"`, `"gemini-pro"` | Gemini 1.5 Pro/Flash, Gemini 2.0 |
| **Groq** | `GroqChat` | — | `"llama"`, `"mixtral"` | Fast inference, Llama 3.3, Mixtral |
| **Mistral** | `MistralChat` | `MistralEmbedding` | `"mistral"`, `"codestral"` | Mistral Large, Ministral |
| **Cohere** | `CohereChat` | `CohereEmbedding` | `"command"`, `"command-r"` | Command R+, Aya |
| **Ollama** | `OllamaChat` | `OllamaEmbedding` | `"ollama"` | Local models, any GGUF |
| **Custom** | `CustomChat` | `CustomEmbedding` | — | vLLM, Together AI, any OpenAI-compatible |

```python
# 3 ways to create models

# 1. Simple strings (recommended)
agent = Agent("Helper", model="gpt4")
agent = Agent("Helper", model="claude")
agent = Agent("Helper", model="gemini")

# 2. Factory functions
from cogent.models import create_chat
model = create_chat("gpt4")  # One argument
model = create_chat("anthropic", "claude-sonnet-4")  # Two arguments

# 3. Direct instantiation (full control)
from cogent.models import OpenAIChat
model = OpenAIChat(model="gpt-4o", temperature=0.7, api_key="sk-...")
```

### `cogent.capabilities` — Agent Capabilities

Composable tools that plug into any agent. Each capability adds related tools.

| Capability | Description | Tools Added |
|------------|-------------|-------------|
| **HTTPClient** | Full-featured HTTP client | `http_request`, `http_get`, `http_post` with retries, timeouts |
| **Database** | Async SQL database access | `execute_query`, `fetch_one`, `fetch_all` with connection pooling |
| **APITester** | HTTP endpoint testing | `test_endpoint`, `assert_status`, `assert_json` |
| **DataValidator** | Schema validation | `validate_data`, `validate_json`, `validate_dict` with Pydantic |
| **WebSearch** | Web search with caching | `web_search`, `news_search` with semantic cache |
| **Browser** | Playwright automation | `navigate`, `click`, `fill`, `screenshot` |
| **FileSystem** | Sandboxed file operations | `read_file`, `write_file`, `list_dir`, `search_files` |
| **CodeSandbox** | Safe Python execution | `execute_python`, `run_function` |
| **Shell** | Sandboxed shell commands | `run_command` |
| **PDF** | PDF processing | `read_pdf`, `create_pdf`, `merge_pdfs` |
| **Spreadsheet** | Excel/CSV operations | `read_spreadsheet`, `write_spreadsheet` |
| **MCP** | Model Context Protocol | Dynamic tools from MCP servers |

```python
from cogent.capabilities import FileSystem, CodeSandbox, WebSearch, HTTPClient, Database

agent = Agent(
    name="Assistant",
    model="gpt-4o-mini",
    capabilities=[
        FileSystem(allowed_paths=["./project"]),
        CodeSandbox(timeout=30),
        WebSearch(),
        HTTPClient(),
        Database("sqlite:///data.db"),
    ]
)
```

### `cogent.document` — Document Processing

Load, split, and process documents for RAG pipelines.

**Loaders** — Support for all common file formats:

| Loader | Formats | Notes |
|--------|---------|-------|
| `TextLoader` | `.txt`, `.rst` | Plain text extraction |
| `MarkdownLoader` | `.md` | Markdown with structure |
| `PDFLoader` | `.pdf` | Basic text extraction (pypdf/pdfplumber) |
| `PDFMarkdownLoader` | `.pdf` | Clean markdown output (pymupdf4llm) |

| `PDFVisionLoader` | `.pdf` | Vision model-based extraction |
| `WordLoader` | `.docx` | Microsoft Word documents |
| `HTMLLoader` | `.html`, `.htm` | HTML documents |
| `CSVLoader` | `.csv` | CSV files |
| `JSONLoader` | `.json`, `.jsonl` | JSON documents |
| `XLSXLoader` | `.xlsx` | Excel spreadsheets |
| `CodeLoader` | `.py`, `.js`, `.ts`, `.java`, `.go`, `.rs`, `.cpp`, etc. | Source code files |

**Splitters** — Multiple chunking strategies:

| Splitter | Strategy |
|----------|----------|
| `RecursiveCharacterSplitter` | Hierarchical separators (default) |
| `SentenceSplitter` | Sentence boundary detection |
| `MarkdownSplitter` | Markdown structure-aware |
| `HTMLSplitter` | HTML tag-based |
| `CodeSplitter` | Language-aware code splitting |
| `SemanticSplitter` | Embedding-based semantic chunking |
| `TokenSplitter` | Token count-based |

```python
from cogent.document import DocumentLoader, SemanticSplitter

loader = DocumentLoader()
docs = await loader.load_directory("./documents")

splitter = SemanticSplitter(model=model)
chunks = splitter.split_documents(docs)
```

### `cogent.vectorstore` — Vector Storage

Semantic search with pluggable backends and embedding providers.

**Backends:**

| Backend | Use Case | Persistence |
|---------|----------|-------------|
| `InMemoryBackend` | Development, small datasets | No |
| `FAISSBackend` | Large-scale local search | Optional |
| `ChromaBackend` | Persistent vector database | Yes |
| `QdrantBackend` | Production vector database | Yes |
| `PgVectorBackend` | PostgreSQL integration | Yes |

**Embedding Providers:**

| Provider | Model Examples |
|----------|----------------|
| `OpenAIEmbeddings` | `text-embedding-3-small`, `text-embedding-3-large` |
| `OllamaEmbeddings` | `nomic-embed-text`, `mxbai-embed-large` |
| `MockEmbeddings` | Testing only |

```python
from cogent.vectorstore import VectorStore, OpenAIEmbeddings
from cogent.vectorstore.backends import FAISSBackend

store = VectorStore(
    embeddings=OpenAIEmbeddings(model="text-embedding-3-large"),
    backend=FAISSBackend(dimension=3072),
)
```

### `cogent.memory` — Memory & Persistence

Long-term memory with fuzzy matching (semantic fallback optional), conversation history, and scoped views.

**Stores:**

| Store | Backend | Features |
|-------|---------|----------|
| `InMemoryStore` | Dict | Fast, no persistence |
| `SQLAlchemyStore` | SQLite, PostgreSQL, MySQL | Async, full SQL |
| `RedisStore` | Redis | Distributed, native TTL |

```python
from cogent.memory import Memory, SQLAlchemyStore

memory = Memory(store=SQLAlchemyStore("sqlite+aiosqlite:///./data.db"))

# Scoped views
user_mem = memory.scoped("user:alice")
team_mem = memory.scoped("team:research")
```

### `cogent.executors` — Execution Strategies

Pluggable execution strategies that define HOW agents process tasks.

| Executor | Strategy | Use Case |
|----------|----------|----------|
| `NativeExecutor` | Parallel tool execution | Default, high performance |
| `SequentialExecutor` | Sequential tool execution | Ordered dependencies |
| `TreeSearchExecutor` | LATS Monte Carlo tree search | Best accuracy, complex reasoning |

**Standalone execution** — bypass Agent class entirely:

```python
from cogent.executors import run

result = await run(
    "Search for Python tutorials and summarize",
    tools=[search, summarize],
    model="gpt-4o-mini",
)
```

**Tree Search (LATS)** — explores multiple reasoning paths with backtracking:

```python
from cogent.executors import TreeSearchExecutor

executor = TreeSearchExecutor(
    agent,
    max_iterations=10,
    exploration_weight=1.414,  # UCB1 exploration constant
)
result = await executor.execute("Complex multi-step task")
```

### `cogent.interceptors` — Middleware

Composable middleware for cross-cutting concerns.

| Category | Interceptors |
|----------|-------------|
| **Budget** | `BudgetGuard` (token/cost limits) |
| **Security** | `PIIShield`, `ContentFilter` |
| **Rate Limiting** | `RateLimiter`, `ThrottleInterceptor` |
| **Context** | `ContextCompressor`, `TokenLimiter` |
| **Gates** | `ToolGate`, `PermissionGate`, `ConversationGate` |
| **Resilience** | `Failover`, `CircuitBreaker`, `ToolGuard` |
| **Audit** | `Auditor` (event logging) |
| **Prompt** | `PromptAdapter`, `ContextPrompt`, `LambdaPrompt` |

```python
from cogent.interceptors import BudgetGuard, PIIShield, RateLimiter

agent = Agent(
    name="Safe",
    model="gpt-4o-mini",
    intercept=[
        BudgetGuard(max_model_calls=100),
        PIIShield(patterns=["email", "ssn"]),
        RateLimiter(requests_per_minute=60),
    ]
)
```

### `cogent.observability` — Monitoring & Tracing

Comprehensive monitoring for understanding system behavior.

| Component | Purpose |
|-----------|---------|
| `ExecutionTracer` | Deep execution tracing with spans |
| `MetricsCollector` | Counter, Gauge, Histogram, Timer |
| `ProgressTracker` | Real-time progress output |
| `Observer` | Unified observability with history capture |
| `Dashboard` | Visual inspection interface |
| `Inspectors` | Agent, Task, Event inspection |

**Renderers:** `TextRenderer`, `RichRenderer`, `JSONRenderer`, `MinimalRenderer`

```python
from cogent.observability import ExecutionTracer, ProgressTracker

tracer = ExecutionTracer()
async with tracer.trace("my-operation") as span:
    span.set_attribute("user_id", user_id)
    result = await do_work()
```

### `cogent.graph` — Visualization

Unified visualization API for agents and execution traces.

**Backends:**

| Method | Output |
|--------|--------|
| `.mermaid()` | Mermaid diagram code |
| `.ascii()` | Terminal-friendly text |
| `.dot()` | Graphviz DOT format |
| `.url()` | mermaid.ink shareable URL |
| `.html()` | Embeddable HTML |
| `.save()` | PNG, SVG, PDF, HTML |

```python
view = agent.graph()
print(view.mermaid())
view.save("diagram.png")
```

---

## Installation

```bash
# Minimal installation (core only)
uv add git+https://github.com/milad-o/cogent.git

# With vector stores & retrieval
uv add "cogent[vector-stores,retrieval] @ git+https://github.com/milad-o/cogent.git"

# With database backends
uv add "cogent[database] @ git+https://github.com/milad-o/cogent.git"

# With all backends (vector stores, retrieval, database, redis)
uv add "cogent[all-backend] @ git+https://github.com/milad-o/cogent.git"

# Full installation with all providers & capabilities
uv add "cogent[all] @ git+https://github.com/milad-o/cogent.git"
```

**Optional dependency groups:**

| Group | Purpose | Includes |
|-------|---------|----------|
| `vector-stores` | Vector databases | FAISS, Qdrant |
| `retrieval` | Retrieval libraries | BM25, sentence-transformers |
| `database` | SQL databases | SQLAlchemy, aiosqlite, psycopg2 |
| `infrastructure` | Infrastructure | Redis |
| `web` | Web tools | BeautifulSoup4, DuckDuckGo search |
| `browser` | Browser automation | Playwright |
| `document` | Document processing | PDF, Word, Markdown loaders |
| `api` | API framework | FastAPI, Uvicorn |
| `anthropic` | Claude models | Anthropic SDK |
| `azure` | Azure models | Azure OpenAI, Azure Identity |
| `cohere` | Cohere models | Cohere SDK |
| `groq` | Groq models | Groq SDK |
| `all-providers` | All LLM providers | All anthropic, azure, cohere, groq |
| `all-backend` | All backends | vector-stores, retrieval, database, infrastructure |
| `all` | Everything | All above |

**Development installation:**

```bash
# Core dev tools (linting, type checking)
uv add --dev cogent[dev]

# Add testing
uv add --dev cogent[dev,test]

# Add backend tests (vector stores, databases)
uv add --dev cogent[dev,test,test-backends]

# Add documentation
uv add --dev cogent[dev,test,test-backends,docs]
```

## Core Architecture

Cogent is built around a high-performance **Native Executor** that eliminates framework overhead while providing enterprise-grade features.

### Native Executor

The executor uses a direct asyncio loop with parallel tool execution—no graph frameworks, no unnecessary abstractions:

```python
from cogent import Agent, tool
from cogent.models import ChatModel

@tool
def search(query: str) -> str:
    """Search the web."""
    return f"Results for: {query}"

@tool
def calculate(expression: str) -> str:
    """Evaluate math expression."""
    return str(eval(expression))

agent = Agent(
    name="Assistant",
    model="gpt4",  # Simple string model
    tools=[search, calculate],
)

# Tools execute in parallel when independent
result = await agent.run("Search for Python and calculate 2^10")
```

**Key optimizations:**
- **Parallel tool execution** — Multiple tool calls run concurrently via `asyncio.gather`
- **Cached model binding** — Tools bound once at construction, zero overhead per call
- **Native SDK integration** — Direct OpenAI/Anthropic SDK calls, no translation layers
- **Automatic resilience** — Rate limit retries with exponential backoff built-in

### Tool System

Define tools with the `@tool` decorator—automatic schema extraction from type hints and docstrings:

```python
from cogent import tool
from cogent.core.context import RunContext

@tool
def search(query: str, max_results: int = 10) -> str:
    """Search the web for information.
    
    Args:
        query: Search query string.
        max_results: Maximum results to return.
    """
    return f"Found {max_results} results for: {query}"

# With context injection for user/session data
@tool
def get_user_preferences(ctx: RunContext) -> str:
    """Get preferences for the current user."""
    return f"Preferences for user {ctx.user_id}"

# Async tools supported
@tool
async def fetch_data(url: str) -> str:
    """Fetch data from URL."""
    async with httpx.AsyncClient() as client:
        response = await client.get(url)
        return response.text
```

**Tool features:**
- Type hints → JSON schema conversion
- Docstring → description extraction
- Sync and async function support
- Context injection via `ctx: RunContext` parameter
- Automatic error handling and retries

### Standalone Execution

For maximum performance, bypass the Agent class entirely:

```python
from cogent.executors import run

result = await run(
    "Search for Python tutorials and summarize the top 3",
    tools=[search, summarize],
    model="gpt-4o-mini",
)
```

## Quick Start

### Simple Agent

```python
import asyncio
from cogent import Agent, tool

@tool
def get_weather(city: str) -> str:
    """Get current weather for a city."""
    return f"Weather in {city}: 72°F, sunny"

async def main():
    agent = Agent(
        name="Assistant",
        model="gpt-4o-mini",
        tools=[get_weather],
    )
    
    result = await agent.run("What's the weather in Tokyo?")
    print(result)

asyncio.run(main())
```

## Streaming

```python
agent = Agent(
    name="Writer",
    model="gpt-4o-mini",
    stream=True,
)

async for chunk in agent.run_stream("Write a poem"):
    print(chunk.content, end="", flush=True)
```

## Human-in-the-Loop

```python
from cogent import Agent
from cogent.agent import InterruptedException

agent = Agent(
    name="Assistant",
    model="gpt-4o-mini",
    tools=[sensitive_tool],
    interrupt_on={"tools": ["sensitive_tool"]},  # Require approval
)

try:
    result = await agent.run("Do something sensitive")
except InterruptedException as e:
    # Handle approval flow
    decision = await get_human_decision(e.pending_action)
    result = await agent.resume(e.state, decision)
```

## Observability

```python
from cogent import Agent
from cogent.observability import Observer, ObservabilityLevel

# Verbosity levels for agents
agent = Agent(
    name="Assistant",
    model="gpt-4o-mini",
    verbosity="debug",  # off | result | progress | detailed | debug | trace
)

# Or use enum/int
agent = Agent(model=model, verbosity=ObservabilityLevel.DEBUG)  # Enum
agent = Agent(model=model, verbosity=4)  # Int (0-5)

# Boolean shorthand
agent = Agent(model=model, verbosity=True)  # → PROGRESS level

# With observer for history capture
observer = Observer(level="detailed", capture=["tool.result", "agent.*"])
result = await agent.run("Query", observer=observer)

# Access captured events
for event in observer.history():
    print(event)
```

## Interceptors

Control execution flow with middleware:

```python
from cogent.interceptors import (
    BudgetGuard,      # Token/cost limits
    RateLimiter,      # Request throttling
    PIIShield,        # Redact sensitive data
    ContentFilter,    # Block harmful content
    ToolGate,         # Conditional tool access
    PromptAdapter,    # Modify prompts dynamically
    Auditor,          # Audit logging
)

agent = Agent(
    name="Safe",
    model="gpt-4o-mini",
    intercept=[
        BudgetGuard(max_model_calls=100, max_tool_calls=500),
        PIIShield(patterns=["email", "ssn"]),
        RateLimiter(requests_per_minute=60),
    ],
)
```

## Structured Output

Type-safe responses with automatic validation:

```python
from pydantic import BaseModel
from typing import Literal
from cogent import Agent

# Structured models
class Analysis(BaseModel):
    sentiment: str
    confidence: float
    topics: list[str]

agent = Agent(
    name="Analyzer",
    model="gpt-4o-mini",
    output=Analysis,  # Enforce schema
)

result = await agent.run("Analyze: I love this product!")
print(result.content.data.sentiment)   # "positive"
print(result.content.data.confidence)  # 0.95

# Bare types - return primitive values directly
agent = Agent(name="Reviewer", model="gpt-4o-mini", output=Literal["APPROVE", "REJECT"])
result = await agent.run("Review this code")
print(result.content.data)  # "APPROVE" (bare string)

# Other bare types: str, int, bool, float
agent = Agent(name="Counter", model="gpt-4o-mini", output=int)
result = await agent.run("Count the items")
print(result.content.data)  # 5 (bare int)
```

## Reasoning

Extended thinking for complex problems with AI-controlled rounds:

```python
from cogent import Agent
from cogent.agent.reasoning import ReasoningConfig

# Simple: Enable with defaults
agent = Agent(
    name="Analyst",
    model="gpt-4o",
    reasoning=True,  # AI decides when ready (up to 10 rounds)
)

# Custom config
agent = Agent(
    name="DeepThinker",
    model="gpt-4o",
    reasoning=ReasoningConfig(
        max_thinking_rounds=15,  # Safety limit
        style=ReasoningStyle.CRITICAL,
    ),
)

# Per-call override
result = await agent.run(
    "Complex analysis task",
    reasoning=True,  # Enable for this call only
)
```

**Reasoning Styles:** `ANALYTICAL`, `EXPLORATORY`, `CRITICAL`, `CREATIVE`

## Resilience

```python
from cogent.agent import ResilienceConfig, RetryPolicy

agent = Agent(
    name="Resilient",
    model="gpt-4o-mini",
    resilience=ResilienceConfig(
        retry=RetryPolicy(max_attempts=3, backoff_multiplier=2.0),
        timeout=30.0,
        circuit_breaker=True,
    ),
)
```

## Configuration

Use environment variables or `.env`:

```bash
# LLM Provider
OPENAI_API_KEY=sk-...

# Azure
AZURE_OPENAI_ENDPOINT=https://...
AZURE_OPENAI_DEPLOYMENT=gpt-4o
AZURE_OPENAI_AUTH_TYPE=managed_identity
AZURE_OPENAI_CLIENT_ID=...  # optional (user-assigned managed identity)

# Azure (service principal / client secret)
# AZURE_OPENAI_AUTH_TYPE=client_secret
# AZURE_OPENAI_TENANT_ID=...
# AZURE_OPENAI_CLIENT_ID=...
# AZURE_OPENAI_CLIENT_SECRET=...

# Anthropic
ANTHROPIC_API_KEY=...

# Ollama (local)
OLLAMA_HOST=http://localhost:11434
```

## Examples

See `examples/` for complete examples:

| Example | Description |
|---------|-------------|
| `01_basic_usage.py` | Simple agent with tools |
| `flow/flow_basics.py` | Multi-agent patterns |
| `03_flow.py` | Flow orchestration |
| `04_events.py` | Event system |
| `05_observability.py` | Tracing and metrics |
| `06_memory.py` | Conversation persistence |
| `07_roles.py` | Agent roles |
| `08_mesh_writers.py` | Mesh collaboration |
| `09_hierarchical_roles.py` | Hierarchical teams |
| `10_agentic_rag.py` | RAG with agents |
| `11_supervisor_chat.py` | Supervisor pattern |
| `12_knowledge_graph.py` | Knowledge graphs |
| `13_codebase_analyzer.py` | Code analysis |
| `14_filesystem.py` | File operations |
| `15_web_search.py` | Web search capability |
| `16_code_sandbox.py` | Safe code execution |
| `17_ssis_analyzer.py` | SSIS analysis |
| `18_human_in_the_loop.py` | Approval workflows |
| `19_streaming.py` | Real-time streaming |
| `20_mcp.py` | MCP server integration |
| `21_deferred_tools.py` | Deferred tool execution |
| `22_rag_pipelines.py` | RAG pipelines |
| `23_pdf.py` | PDF processing |
| `24_structured_output.py` | Type-safe responses |
| `25_browser.py` | Web browsing |
| `26_spreadsheet.py` | Excel/CSV operations |
| `27_deep_observability.py` | Deep tracing |
| `28_reasoning.py` | Extended thinking |
| `29_interceptors.py` | Middleware patterns |
| `30_shell.py` | Shell commands |
| `31_context_layer.py` | Context management |
| `32_spawning_agents.py` | Dynamic agent creation |
| `33_graph_api.py` | Visualization API |
| `34_pdf_markdown.py` | PDF to Markdown |
| `35_pdf_html.py` | PDF to HTML (complex tables) |
| `35_pdf_rag.py` | PDF RAG pipeline |
| `36_summarizer.py` | Document summarization |
| `37_pdf_summarizer.py` | PDF summarization |

## Development

```bash
# Install with dev dependencies
uv sync --extra dev

# Run tests
uv run pytest

# Type checking
uv run mypy src/cogent

# Linting
uv run ruff check src/cogent
```

## License

MIT License
