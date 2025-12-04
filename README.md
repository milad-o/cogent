# AgenticFlow

<p align="center">
  <strong>Build AI agents that actually work.</strong>
</p>

<p align="center">
  <a href="#installation">Installation</a> •
  <a href="#quick-start">Quick Start</a> •
  <a href="#core-architecture">Architecture</a> •
  <a href="#multi-agent-topologies">Topologies</a> •
  <a href="#capabilities">Capabilities</a> •
  <a href="#examples">Examples</a>
</p>

---

AgenticFlow is a **production-grade multi-agent framework** designed for performance, simplicity, and real-world deployment. Unlike frameworks that wrap LangChain or add unnecessary abstractions, AgenticFlow uses **native SDK integrations** and a **zero-overhead executor** to deliver the fastest possible agent execution.

**Why AgenticFlow?**

- 🚀 **Fast** — Parallel tool execution, cached model binding, direct SDK calls
- 🔧 **Simple** — Define tools with `@tool`, create agents in 3 lines, no boilerplate
- 🏭 **Production-ready** — Built-in resilience, observability, and security interceptors
- 🤝 **Multi-agent** — Supervisor, Pipeline, Mesh, and Hierarchical coordination patterns
- 📦 **Batteries included** — File system, web search, code sandbox, browser, PDF, and more

```python
from agenticflow import Agent, tool
from agenticflow.models import ChatModel

@tool
def search(query: str) -> str:
    """Search the web."""
    return web_search(query)

agent = Agent(name="Assistant", model=ChatModel(), tools=[search])
result = await agent.run("Find the latest news on AI agents")
```

---

## Features

- **Native Executor** — High-performance parallel tool execution with zero framework overhead
- **Native Model Support** — OpenAI, Azure, Anthropic, Gemini, Groq, Ollama, Custom endpoints
- **Multi-Agent Topologies** — Supervisor, Pipeline, Mesh, Hierarchical
- **Capabilities** — Filesystem, Web Search, Code Sandbox, Browser, PDF, Shell, MCP, Spreadsheet, and more
- **RAG Pipeline** — Document loading, per-file-type splitting, embeddings, vector stores, retrievers
- **Memory & Persistence** — Conversation history, long-term memory with semantic search
- **Graph Visualization** — Mermaid, Graphviz, ASCII diagrams for agents and topologies
- **Observability** — Tracing, metrics, progress tracking, structured logging
- **Interceptors** — Budget guards, rate limiting, PII protection, tool gates
- **Resilience** — Retry policies, circuit breakers, fallbacks
- **Human-in-the-Loop** — Tool approval, guidance, interruption handling
- **Streaming** — Real-time token streaming with callbacks
- **Structured Output** — Type-safe responses with Pydantic schemas
- **Reasoning** — Extended thinking mode with chain-of-thought

---

## Modules

AgenticFlow is organized into focused modules, each with multiple backends and implementations.

### `agenticflow.models` — LLM Providers

Native SDK wrappers for all major LLM providers with zero abstraction overhead.

| Provider | Chat | Embeddings | Notes |
|----------|------|------------|-------|
| **OpenAI** | `OpenAIChat` | `OpenAIEmbedding` | Default provider, GPT-4o, o1, o3 |
| **Azure** | `AzureChat` | `AzureEmbedding` | Managed Identity, Azure AD support |
| **Anthropic** | `AnthropicChat` | — | Claude 4, extended thinking |
| **Gemini** | `GeminiChat` | `GeminiEmbedding` | Google AI, Vertex AI |
| **Groq** | `GroqChat` | — | Fast inference, Llama 3.3, Mixtral |
| **Ollama** | `OllamaChat` | `OllamaEmbedding` | Local models, any GGUF |
| **Custom** | `CustomChat` | `CustomEmbedding` | vLLM, Together AI, any OpenAI-compatible |

```python
from agenticflow.models import create_chat, create_embedding

# Factory functions for any provider
model = create_chat("anthropic", model="claude-sonnet-4-20250514")
embedder = create_embedding("openai", model="text-embedding-3-large")
```

### `agenticflow.capabilities` — Agent Capabilities

Composable tools that plug into any agent. Each capability adds related tools.

| Capability | Description | Tools Added |
|------------|-------------|-------------|
| **FileSystem** | Sandboxed file operations | `read_file`, `write_file`, `list_dir`, `search_files` |
| **WebSearch** | Web search (DuckDuckGo) | `web_search`, `news_search`, `fetch_page` |
| **CodeSandbox** | Safe Python execution | `execute_python`, `run_function` |
| **Browser** | Playwright automation | `navigate`, `click`, `fill`, `screenshot` |
| **PDF** | PDF processing | `read_pdf`, `create_pdf`, `merge_pdfs` |
| **Shell** | Sandboxed shell commands | `run_command` |
| **Spreadsheet** | Excel/CSV operations | `read_spreadsheet`, `write_spreadsheet` |
| **MCP** | Model Context Protocol | Dynamic tools from MCP servers |
| **KnowledgeGraph** | Entity/relationship memory | `remember`, `recall`, `query` |
| **Summarizer** | Document summarization | `summarize_text`, `summarize_file` |
| **CodebaseAnalyzer** | Python AST analysis | `analyze_code`, `find_usages` |
| **SSISAnalyzer** | SSIS package analysis | `analyze_package`, `trace_lineage` |

```python
from agenticflow.capabilities import FileSystem, CodeSandbox, WebSearch

agent = Agent(
    capabilities=[
        FileSystem(allowed_paths=["./project"]),
        CodeSandbox(timeout=30),
        WebSearch(),
    ]
)
```

### `agenticflow.document` — Document Processing

Load, split, and process documents for RAG pipelines.

**Loaders** — Support for all common file formats:

| Loader | Formats |
|--------|---------|
| `TextLoader` | `.txt`, `.rst` |
| `MarkdownLoader` | `.md` |
| `PDFLoader` | `.pdf` (with OCR fallback) |
| `WordLoader` | `.docx` |
| `HTMLLoader` | `.html`, `.htm` |
| `CSVLoader` | `.csv` |
| `JSONLoader` | `.json`, `.jsonl` |
| `XLSXLoader` | `.xlsx` |
| `CodeLoader` | `.py`, `.js`, `.ts`, `.java`, `.go`, `.rs`, `.cpp`, etc. |

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
from agenticflow.document import DocumentLoader, SemanticSplitter

loader = DocumentLoader()
docs = await loader.load_directory("./documents")

splitter = SemanticSplitter(model=model)
chunks = splitter.split_documents(docs)
```

### `agenticflow.vectorstore` — Vector Storage

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
from agenticflow.vectorstore import VectorStore, OpenAIEmbeddings
from agenticflow.vectorstore.backends import FAISSBackend

store = VectorStore(
    embeddings=OpenAIEmbeddings(model="text-embedding-3-large"),
    backend=FAISSBackend(dimension=3072),
)
```

### `agenticflow.memory` — Memory & Persistence

Long-term memory with semantic search, conversation history, and scoped views.

**Stores:**

| Store | Backend | Features |
|-------|---------|----------|
| `InMemoryStore` | Dict | Fast, no persistence |
| `SQLAlchemyStore` | SQLite, PostgreSQL, MySQL | Async, full SQL |
| `RedisStore` | Redis | Distributed, native TTL |

```python
from agenticflow.memory import Memory, SQLAlchemyStore

memory = Memory(store=SQLAlchemyStore("sqlite+aiosqlite:///./data.db"))

# Scoped views
user_mem = memory.scoped("user:alice")
team_mem = memory.scoped("team:research")
```

### `agenticflow.executors` — Execution Strategies

Pluggable execution strategies that define HOW agents process tasks.

| Executor | Strategy | Use Case |
|----------|----------|----------|
| `NativeExecutor` | Parallel tool execution | Default, high performance |
| `SequentialExecutor` | Sequential tool execution | Ordered dependencies |
| `TreeSearchExecutor` | LATS Monte Carlo tree search | Best accuracy, complex reasoning |

**Standalone execution** — bypass Agent class entirely:

```python
from agenticflow.executors import run

result = await run(
    "Search for Python tutorials and summarize",
    tools=[search, summarize],
    model="gpt-4o-mini",
)
```

**Tree Search (LATS)** — explores multiple reasoning paths with backtracking:

```python
from agenticflow.executors import TreeSearchExecutor

executor = TreeSearchExecutor(
    agent,
    max_iterations=10,
    exploration_weight=1.414,  # UCB1 exploration constant
)
result = await executor.execute("Complex multi-step task")
```

### `agenticflow.topologies` — Multi-Agent Patterns

Coordination patterns for multi-agent workflows.

| Pattern | Description | Use Case |
|---------|-------------|----------|
| `Supervisor` | Coordinator delegates to workers | Task routing, orchestration |
| `Pipeline` | Sequential A → B → C | Content creation, ETL |
| `Mesh` | All agents collaborate in rounds | Brainstorming, consensus |
| `Hierarchical` | Tree structure with team leads | Large organizations |

```python
from agenticflow.topologies import Pipeline, AgentConfig

pipeline = Pipeline(stages=[
    AgentConfig(agent=researcher, role="research"),
    AgentConfig(agent=writer, role="draft"),
    AgentConfig(agent=editor, role="polish"),
])
```

### `agenticflow.interceptors` — Middleware

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
from agenticflow.interceptors import BudgetGuard, PIIShield, RateLimiter

agent = Agent(
    intercept=[
        BudgetGuard(max_model_calls=100),
        PIIShield(patterns=["email", "ssn"]),
        RateLimiter(requests_per_minute=60),
    ]
)
```

### `agenticflow.observability` — Monitoring & Tracing

Comprehensive monitoring for understanding system behavior.

| Component | Purpose |
|-----------|---------|
| `ExecutionTracer` | Deep execution tracing with spans |
| `MetricsCollector` | Counter, Gauge, Histogram, Timer |
| `ProgressTracker` | Real-time progress output |
| `Observer` | Unified observability for flows |
| `Dashboard` | Visual inspection interface |
| `Inspectors` | Agent, Task, Event inspection |

**Renderers:** `TextRenderer`, `RichRenderer`, `JSONRenderer`, `MinimalRenderer`

```python
from agenticflow.observability import ExecutionTracer, ProgressTracker

tracer = ExecutionTracer()
async with tracer.trace("my-operation") as span:
    span.set_attribute("user_id", user_id)
    result = await do_work()
```

### `agenticflow.graph` — Visualization

Unified visualization API for agents, topologies, and flows.

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
# Install from GitHub
uv add git+https://github.com/milad-o/agenticflow.git

# Or with pip
pip install git+https://github.com/milad-o/agenticflow.git

# With optional dependencies
uv add "agenticflow[web] @ git+https://github.com/milad-o/agenticflow.git"
uv add "agenticflow[anthropic] @ git+https://github.com/milad-o/agenticflow.git"
uv add "agenticflow[azure] @ git+https://github.com/milad-o/agenticflow.git"
uv add "agenticflow[all] @ git+https://github.com/milad-o/agenticflow.git"
```

## Core Architecture

AgenticFlow is built around a high-performance **Native Executor** that eliminates framework overhead while providing enterprise-grade features.

### Native Executor

The executor uses a direct asyncio loop with parallel tool execution—no graph frameworks, no unnecessary abstractions:

```python
from agenticflow import Agent, tool
from agenticflow.models import ChatModel

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
    model=ChatModel(model="gpt-4o"),
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
from agenticflow import tool
from agenticflow.context import RunContext

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
from agenticflow.executors import run

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
from agenticflow import Agent, tool
from agenticflow.models import ChatModel

@tool
def get_weather(city: str) -> str:
    """Get current weather for a city."""
    return f"Weather in {city}: 72°F, sunny"

async def main():
    agent = Agent(
        name="Assistant",
        model=ChatModel(model="gpt-4o-mini"),
        tools=[get_weather],
    )
    
    result = await agent.run("What's the weather in Tokyo?")
    print(result)

asyncio.run(main())
```

### Multi-Agent Pipeline

```python
from agenticflow import Agent, Flow
from agenticflow.models import ChatModel

model = ChatModel(model="gpt-4o")

researcher = Agent(name="Researcher", model=model, instructions="Research topics thoroughly.")
writer = Agent(name="Writer", model=model, instructions="Write clear, engaging content.")
editor = Agent(name="Editor", model=model, instructions="Review and polish the content.")

flow = Flow(
    name="content-pipeline",
    agents=[researcher, writer, editor],
    topology="pipeline",
    verbose=True,  # Progress output
)

result = await flow.run("Create a blog post about quantum computing")
```

## Streaming

```python
agent = Agent(
    name="Writer",
    model=model,
    stream=True,
)

async for chunk in agent.run_stream("Write a poem"):
    print(chunk.content, end="", flush=True)
```

## Human-in-the-Loop

```python
from agenticflow import Agent
from agenticflow.agent import InterruptedException

agent = Agent(
    name="Assistant",
    model=model,
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
from agenticflow import Agent, Flow

# Verbosity levels for agents
agent = Agent(
    name="Assistant",
    model=model,
    verbose="debug",  # minimal | verbose | debug | trace
)

# Or for flows
flow = Flow(
    agents=[...],
    verbose=True,  # Progress output with timing
)

# Advanced: Custom observer
from agenticflow.observability import Observer

flow = Flow(
    agents=[...],
    observer=Observer.verbose(),
)
```

## Interceptors

Control execution flow with middleware:

```python
from agenticflow.interceptors import (
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
    model=model,
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
from agenticflow import Agent

class Analysis(BaseModel):
    sentiment: str
    confidence: float
    topics: list[str]

agent = Agent(
    name="Analyzer",
    model=model,
    response_schema=Analysis,
)

result = await agent.run("Analyze: I love this product!")
print(result.sentiment)   # "positive"
print(result.confidence)  # 0.95
```

## Resilience

```python
from agenticflow.agent import ResilienceConfig, RetryPolicy

agent = Agent(
    model=model,
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
AZURE_AUTH_TYPE=managed_identity

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
| `02_topologies.py` | Multi-agent patterns |
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
| `34_pdf_llm.py` | PDF with LLM |
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
uv run mypy src/agenticflow

# Linting
uv run ruff check src/agenticflow
```

## License

MIT License
