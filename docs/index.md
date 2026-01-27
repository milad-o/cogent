# Cogent

<p align="center">
  <strong>Build AI agents that actually work.</strong>
</p>

---

Cogent is a **production-grade AI agent framework** designed for performance, simplicity, and real-world deployment. Unlike frameworks that wrap LangChain or add unnecessary abstractions, Cogent uses **native SDK integrations** and a **zero-overhead executor** to deliver the fastest possible agent execution.

## Why Cogent?

- 🚀 **Fast** — Parallel tool execution, cached model binding, direct SDK calls
- 🔧 **Simple** — Define tools with `@tool`, create agents in 3 lines, no boilerplate
- 🏭 **Production-ready** — Built-in resilience, observability, and security interceptors
- 🤝 **Multi-agent** — Supervisor, Pipeline, Mesh, and Hierarchical coordination patterns
- 📦 **Batteries included** — File system, web search, code sandbox, browser, PDF, knowledge graphs, and more

## Quick Example

```python
from cogent import Agent, tool

@tool
def search(query: str) -> str:
    """Search the web."""
    return web_search(query)

# v1.14.1: Simple string models!
agent = Agent(name="Assistant", model="gpt4", tools=[search])
result = await agent.run("Find the latest news on AI agents")
```

## Installation

```bash
# Install from GitHub
pip install git+https://github.com/milad-o/cogent.git

# Or with uv (recommended)
uv add git+https://github.com/milad-o/cogent.git
```

**[Get Started →](getting-started.md)**

## Features

- **Native Executor** — High-performance parallel tool execution with zero framework overhead
- **Native Model Support** — OpenAI, Azure, Anthropic, Gemini, Groq, Ollama, Custom endpoints
- **Multi-Agent Patterns** — Supervisor, Pipeline, Mesh, Hierarchical
- **Capabilities** — Filesystem, Web Search, Code Sandbox, Browser, PDF, Shell, MCP, Spreadsheet, and more
- **RAG Pipeline** — Document loading, per-file-type splitting, embeddings, vector stores, retrievers
- **Memory & Persistence** — Conversation history, long-term memory with fuzzy matching
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
- [Examples](https://github.com/milad-o/cogent/tree/main/examples) — See working examples

## Latest Release (v1.14.1)

**3-Tier Model API - String Models**

- 🎯 **Simple String Models** — `Agent(model="gpt4")` auto-resolves to gpt-4o
- 🏷️ **50+ Model Aliases** — `gpt5`, `gpt4`, `claude`, `gemini3`, `mistral`, `command-a`, etc.
- 🔗 **Provider Prefix** — `"anthropic:claude"`, `"groq:llama-70b"`
- ⚙️ **Auto-Configuration** — Loads API keys and model overrides from `.env`, TOML/YAML, or env vars
- 🔄 **Backward Compatible** — Existing code works unchanged
- 🧠 **3 API Tiers** — String (simple), Factory (4 patterns), Direct (full control)
- 🔍 **Auto Provider Detection** — Supports GPT-5, Gemini 3, Mistral Large 3, Command A, and all mainstream models
- ✅ **74 New Tests** — Comprehensive test coverage for all new features

See [CHANGELOG](https://github.com/milad-o/cogent/blob/main/CHANGELOG.md) for full version history.
