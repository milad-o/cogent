# AgenticFlow

A production-grade multi-agent framework for building AI applications with native model support, advanced execution strategies, and full observability.

## Features

- **Native Model Support** — OpenAI, Azure, Anthropic, Gemini, Groq, Ollama
- **Multi-Agent Topologies** — Supervisor, Pipeline, Mesh, Hierarchical
- **Execution Strategies** — DAG, ReAct, Plan-Execute, Tree Search
- **Capabilities** — Filesystem, Web Search, Code Sandbox, Browser, PDF, MCP, and more
- **RAG Pipeline** — Document loading, chunking, embeddings, vector stores, retrievers
- **Memory & Persistence** — Conversation history, checkpoints, long-term memory
- **Observability** — Tracing, metrics, progress tracking, structured logging
- **Resilience** — Retry policies, circuit breakers, fallbacks
- **Human-in-the-Loop** — Tool approval, guidance, interruption handling
- **Streaming** — Real-time token streaming with callbacks

## Installation

```bash
# Basic
uv add agenticflow

# With capabilities
uv add "agenticflow[web]"      # Web search, browser
uv add "agenticflow[pdf]"      # PDF processing
uv add "agenticflow[all]"      # Everything

# Development
uv add "agenticflow[dev]"
```

## Quick Start

### Simple Agent

```python
import asyncio
from agenticflow import Agent, tool
from agenticflow.models import OpenAIChat

@tool
def get_weather(city: str) -> str:
    """Get current weather for a city."""
    return f"Weather in {city}: 72°F, sunny"

async def main():
    agent = Agent(
        name="Assistant",
        model=OpenAIChat(model="gpt-4o-mini"),
        tools=[get_weather],
    )
    
    result = await agent.run("What's the weather in Tokyo?")
    print(result)

asyncio.run(main())
```

### Standalone Execution (No Agent)

```python
from agenticflow import run, tool

@tool
def calculate(expression: str) -> str:
    """Evaluate a math expression."""
    return str(eval(expression))

result = await run(
    "What is 25 * 4 + 10?",
    tools=[calculate],
    model="gpt-4o-mini",
)
```

### Multi-Agent Pipeline

```python
from agenticflow import Agent, Flow
from agenticflow.models import OpenAIChat

model = OpenAIChat(model="gpt-4o")

researcher = Agent(name="Researcher", model=model, instructions="Research topics thoroughly.")
writer = Agent(name="Writer", model=model, instructions="Write clear, engaging content.")
editor = Agent(name="Editor", model=model, instructions="Review and polish the content.")

flow = Flow(
    name="content-pipeline",
    agents=[researcher, writer, editor],
    topology="pipeline",
)

result = await flow.run("Create a blog post about quantum computing")
```

## Models

Native wrappers for all major providers:

```python
from agenticflow.models import (
    OpenAIChat,      # GPT-4o, GPT-4o-mini
    AnthropicChat,   # Claude 4
    GeminiChat,      # Gemini 2.0
    GroqChat,        # Llama, Mixtral (fast inference)
    OllamaChat,      # Local models
)
from agenticflow.models.azure import AzureChat  # Azure OpenAI

# OpenAI
model = OpenAIChat(model="gpt-4o")

# Anthropic
model = AnthropicChat(model="claude-sonnet-4-20250514")

# Azure with Managed Identity
model = AzureChat(
    deployment="gpt-4o",
    azure_endpoint="https://my-resource.openai.azure.com",
    use_managed_identity=True,
)

# Local with Ollama
model = OllamaChat(model="qwen2.5:7b")
```

## Capabilities

Composable capabilities that add tools to agents:

```python
from agenticflow import Agent
from agenticflow.capabilities import (
    Filesystem,      # Read/write files, list directories
    WebSearch,       # Search web, fetch pages
    CodeSandbox,     # Execute Python safely
    Browser,         # Browse web pages with Playwright
    PDFCapability,   # Extract text from PDFs
    Shell,           # Run shell commands
    Spreadsheet,     # Read/write Excel files
    MCPCapability,   # Connect to MCP servers
    Summarizer,      # Summarize documents
    KnowledgeGraph,  # Build and query knowledge graphs
)

agent = Agent(
    name="Developer",
    model=model,
    capabilities=[
        Filesystem(allowed_paths=["./project"]),
        CodeSandbox(timeout=30),
        Shell(allowed_commands=["git", "npm"]),
    ],
)
```

## RAG (Retrieval-Augmented Generation)

Full RAG pipeline with document processing and vector search:

```python
from agenticflow.prebuilt import RAGAgent
from agenticflow.models import OpenAIChat
from agenticflow.vectorstore import OpenAIEmbeddings

rag = RAGAgent(
    model=OpenAIChat(model="gpt-4o-mini"),
    embeddings=OpenAIEmbeddings(),
)

# Load documents (supports .pdf, .md, .txt, .html, .py, .json, .csv, ...)
await rag.load_documents(["docs/", "report.pdf"])

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
    PDFLoader,
    PDFMarkdownLoader,       # PDF to markdown with pymupdf4llm
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

One agent delegates to specialists:

```python
from agenticflow import Flow

flow = Flow(
    name="team",
    agents=[supervisor, researcher, coder, writer],
    topology="supervisor",
    supervisor_name="supervisor",
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

All agents can communicate freely:

```python
flow = Flow(
    name="brainstorm",
    agents=[agent1, agent2, agent3],
    topology="mesh",
)
```

### Hierarchical

Team leads coordinate sub-teams:

```python
from agenticflow.topologies import Hierarchical

topology = Hierarchical(
    name="org",
    teams={
        "research": [lead1, analyst1, analyst2],
        "engineering": [lead2, dev1, dev2],
    },
)
```

## Memory & Persistence

```python
from agenticflow import Agent
from agenticflow.memory import MemoryStore, InMemorySaver

agent = Agent(
    name="Assistant",
    model=model,
    memory=InMemorySaver(),  # Conversation history
    store=MemoryStore(),     # Long-term memory
)

# Continue conversation
result = await agent.run("Remember this: my name is Alice", thread_id="user-123")
result = await agent.run("What's my name?", thread_id="user-123")  # "Alice"
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
agent = Agent(
    name="Assistant",
    model=model,
    tools=[dangerous_tool],
    interrupt_on={"tools": ["dangerous_tool"]},  # Require approval
)

try:
    result = await agent.run("Do something risky")
except InterruptedException as e:
    # Handle approval flow
    decision = await get_human_decision(e.pending_action)
    result = await agent.resume(e.state, decision)
```

## Observability

```python
from agenticflow import Agent, Observer

# Verbose output with tool calls
agent = Agent(
    name="Assistant",
    model=model,
    verbose="debug",  # minimal | verbose | debug | trace
)

# Or use Observer for flows
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
)

agent = Agent(
    name="Safe",
    model=model,
    intercept=[
        BudgetGuard(max_tokens=10000),
        PIIShield(),
        RateLimiter(requests_per_minute=60),
    ],
)
```

## Execution Strategies

```python
from agenticflow import Agent

# DAG (default) - Parallel tool execution when possible
agent = Agent(model=model, strategy="dag")

# ReAct - Reason → Act → Observe loop
agent = Agent(model=model, strategy="react")

# Plan-Execute - Create plan, then execute steps
agent = Agent(model=model, strategy="plan")
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
LLM_PROVIDER=openai           # openai | anthropic | gemini | groq | azure | ollama
OPENAI_API_KEY=sk-...

# Embedding Provider  
EMBEDDING_PROVIDER=openai
OPENAI_EMBEDDING_MODEL=text-embedding-3-small

# Azure (if using)
AZURE_OPENAI_ENDPOINT=https://...
AZURE_OPENAI_DEPLOYMENT=gpt-4o
AZURE_AUTH_TYPE=managed_identity  # api_key | managed_identity | default

# Ollama (local)
OLLAMA_HOST=http://localhost:11434
OLLAMA_MODEL=qwen2.5:7b
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
| `25_browser.py` | Web browsing |
| `28_interceptors.py` | Middleware patterns |

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
