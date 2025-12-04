# AgenticFlow

A production-grade multi-agent framework for building AI applications with native model support, advanced coordination patterns, and full observability.

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

## Installation

```bash
# Basic
uv add agenticflow

# With optional dependencies
uv add "agenticflow[web]"       # Web search, browser
uv add "agenticflow[anthropic]" # Anthropic Claude
uv add "agenticflow[azure]"     # Azure OpenAI
uv add "agenticflow[all]"       # Everything

# Development
uv add "agenticflow[dev]"
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

## Models

Native wrappers for all major providers:

```python
from agenticflow.models import ChatModel, create_chat

# OpenAI (default)
model = ChatModel(model="gpt-4o")

# Anthropic
from agenticflow.models.anthropic import AnthropicChat
model = AnthropicChat(model="claude-sonnet-4-20250514")

# Azure with Managed Identity
from agenticflow.models.azure import AzureChat
model = AzureChat(
    deployment="gpt-4o",
    azure_endpoint="https://my-resource.openai.azure.com",
    use_azure_ad=True,
)

# Groq (fast inference)
from agenticflow.models.groq import GroqChat
model = GroqChat(model="llama-3.3-70b-versatile")

# Gemini
from agenticflow.models.gemini import GeminiChat
model = GeminiChat(model="gemini-2.0-flash-exp")

# Ollama (local)
from agenticflow.models.ollama import OllamaChat
model = OllamaChat(model="qwen2.5:7b")

# Factory function for any provider
model = create_chat("anthropic", model="claude-sonnet-4-20250514")
```

## Capabilities

Composable capabilities that add tools to agents:

```python
from agenticflow import Agent
from agenticflow.capabilities import (
    FileSystem,      # Read/write files, list directories
    WebSearch,       # Search web, fetch pages (DuckDuckGo)
    CodeSandbox,     # Execute Python safely
    Browser,         # Browse web pages with Playwright
    PDF,             # Read/create/merge PDFs
    Shell,           # Run shell commands (sandboxed)
    Spreadsheet,     # Read/write Excel files
    MCP,             # Connect to MCP servers
    Summarizer,      # Summarize documents (map-reduce, refine)
    KnowledgeGraph,  # Build and query knowledge graphs
)

agent = Agent(
    name="Developer",
    model=model,
    capabilities=[
        FileSystem(allowed_paths=["./project"]),
        CodeSandbox(timeout=30),
        Shell(allowed_commands=["git", "npm", "ls"]),
        WebSearch(),
        Summarizer(),
    ],
)
```

## RAG (Retrieval-Augmented Generation)

Full RAG pipeline with per-file-type processing:

```python
from agenticflow.prebuilt import create_rag_agent
from agenticflow.models import ChatModel
from agenticflow.vectorstore import OpenAIEmbeddings

rag = create_rag_agent(
    model=ChatModel(model="gpt-4o-mini"),
    embeddings=OpenAIEmbeddings(),
)

# Load documents - each file type uses optimal splitter
await rag.load_documents(["docs/", "report.pdf", "code.py"])

# Query with automatic retrieval
answer = await rag.query("What are the key findings?")
```

### Document Processing

```python
from agenticflow.document import (
    DocumentLoader,           # Load any file type
    RecursiveCharacterSplitter,
    MarkdownSplitter,
    CodeSplitter,
    SemanticSplitter,        # LLM-based chunking
    PDFMarkdownLoader,       # PDF to markdown
)

# Load and split
loader = DocumentLoader()
docs = await loader.load("technical_paper.pdf")

splitter = SemanticSplitter(model=model)
chunks = splitter.split_documents(docs)
```

### Vector Stores

```python
from agenticflow.vectorstore import VectorStore, OpenAIEmbeddings
from agenticflow.vectorstore.backends import (
    InMemoryBackend,
    FAISSBackend,
    ChromaBackend,
    QdrantBackend,
    PgVectorBackend,
)

store = VectorStore(
    embeddings=OpenAIEmbeddings(),
    backend=FAISSBackend(dimension=1536),
)

await store.add_documents(chunks)
results = await store.search("quantum entanglement", k=5)
```

## Multi-Agent Topologies

### Supervisor

One agent coordinates and delegates to specialists:

```python
flow = Flow(
    name="team",
    agents=[supervisor, researcher, coder, writer],
    topology="supervisor",
    supervisor="supervisor",
)
```

### Pipeline

Sequential processing through agents:

```python
flow = Flow(
    name="content",
    agents=[researcher, writer, editor],
    topology="pipeline",
)
```

### Mesh

All agents collaborate in rounds:

```python
flow = Flow(
    name="brainstorm",
    agents=[analyst1, analyst2, analyst3],
    topology="mesh",
    max_rounds=3,
)
```

### Hierarchical

Team leads coordinate sub-teams:

```python
from agenticflow.topologies import Hierarchical, AgentConfig

topology = Hierarchical(
    structure={
        "cto": ["frontend_lead", "backend_lead"],
        "frontend_lead": ["dev1", "dev2"],
        "backend_lead": ["dev3", "dev4"],
    },
    agents={...},
)
```

## Graph Visualization

Visualize agents and topologies as diagrams:

```python
# Get a graph from any entity
view = agent.graph()
view = topology.graph()

# Render in any format
print(view.mermaid())    # Mermaid code
print(view.ascii())      # Terminal-friendly
print(view.dot())        # Graphviz DOT
print(view.url())        # mermaid.ink URL

# Save to file (format auto-detected)
view.save("agent.png")
view.save("topology.svg")
view.save("flow.html")
```

## Memory & Persistence

```python
from agenticflow import Agent
from agenticflow.memory import InMemorySaver

agent = Agent(
    name="Assistant",
    model=model,
    memory=InMemorySaver(),  # Conversation history
)

# Continue conversation across calls
result = await agent.run("My name is Alice", thread_id="user-123")
result = await agent.run("What's my name?", thread_id="user-123")  # "Alice"

# Long-term memory with semantic search
from agenticflow.memory import Memory

agent = Agent(
    name="Assistant",
    model=model,
    store=Memory(embeddings=embeddings),  # Adds remember/recall tools
)
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
| `06_memory.py` | Conversation persistence |
| `10_agentic_rag.py` | RAG with agents |
| `14_filesystem.py` | File operations |
| `15_web_search.py` | Web search capability |
| `16_code_sandbox.py` | Safe code execution |
| `18_human_in_the_loop.py` | Approval workflows |
| `19_streaming.py` | Real-time streaming |
| `24_structured_output.py` | Type-safe responses |
| `25_browser.py` | Web browsing |
| `27_reasoning.py` | Extended thinking |
| `28_interceptors.py` | Middleware patterns |
| `32_graph_api.py` | Visualization API |
| `34_summarizer.py` | Document summarization |

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
