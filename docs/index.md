# AgenticFlow

<p align="center">
  <strong>Build AI agents that actually work.</strong>
</p>

---

AgenticFlow is a **production-grade multi-agent framework** designed for performance, simplicity, and real-world deployment. Unlike frameworks that wrap LangChain or add unnecessary abstractions, AgenticFlow uses **native SDK integrations** and a **zero-overhead executor** to deliver the fastest possible agent execution.

## Why AgenticFlow?

- 🚀 **Fast** — Parallel tool execution, cached model binding, direct SDK calls
- 🔧 **Simple** — Define tools with `@tool`, create agents in 3 lines, no boilerplate
- 🏭 **Production-ready** — Built-in resilience, observability, and security interceptors
- 🤝 **Multi-agent** — Supervisor, Pipeline, Mesh, and Hierarchical coordination patterns
- 📦 **Batteries included** — File system, web search, code sandbox, browser, PDF, knowledge graphs, and more

## Quick Example

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

## Installation

```bash
# Install from GitHub
pip install git+https://github.com/milad-o/agenticflow.git

# Or with uv (recommended)
uv add git+https://github.com/milad-o/agenticflow.git
```

**[Get Started →](getting-started.md)**

## Features

- **Native Executor** — High-performance parallel tool execution with zero framework overhead
- **Native Model Support** — OpenAI, Azure, Anthropic, Gemini, Groq, Ollama, Custom endpoints
- **Multi-Agent Patterns** — Supervisor, Pipeline, Mesh, Hierarchical
- **Capabilities** — Filesystem, Web Search, Code Sandbox, Browser, PDF, Shell, MCP, Spreadsheet, and more
- **RAG Pipeline** — Document loading, per-file-type splitting, embeddings, vector stores, retrievers
- **Memory & Persistence** — Conversation history, long-term memory with semantic search
- **Graph Visualization** — Mermaid, Graphviz, ASCII diagrams for agents, patterns, and flows
- **Observability** — Tracing, metrics, progress tracking, structured logging
- **Interceptors** — Budget guards, rate limiting, PII protection, tool gates
- **Resilience** — Retry policies, circuit breakers, fallbacks
- **Human-in-the-Loop** — Tool approval, guidance, interruption handling
- **Streaming** — Real-time token streaming with callbacks
- **Structured Output** — Type-safe responses with Pydantic schemas
- **Reasoning** — Extended thinking mode with chain-of-thought

## Next Steps

- [Getting Started](getting-started.md) — Get started in 5 minutes
- [Agent Documentation](agent.md) — Learn about the core Agent class
- [Multi-Agent Flow](flow.md) — Build coordinated multi-agent systems
- [Capabilities](capabilities.md) — Explore built-in capabilities
- [Examples](https://github.com/milad-o/agenticflow/tree/main/examples) — See working examples

## Latest Release (v1.8.5)

**Knowledge Graph Backend Switching & Improvements**

- 🔄 **Backend Switching** — `kg.set_backend()` to change backends on existing instances with optional migration
- 🎨 **Custom Backends** — Support for custom `GraphBackend` implementations
- ✨ **Three-Level Visualization API** — `kg.mermaid()`, `kg.render(format)`, `kg.display()` for easy Jupyter rendering
- 🧹 **Removed SSIS** — Cleaned up deprecated SSISAnalyzer capability

See [CHANGELOG](https://github.com/milad-o/agenticflow/blob/main/CHANGELOG.md) for full version history.
