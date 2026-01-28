# Cogent Examples

Learn Cogent through hands-on examples organized by category.

---

## 🚀 Quick Start

### 1. Set Up API Keys

Cogent auto-discovers API keys from multiple sources. Choose one:

**Option A: `.env` file (recommended for development)**

```bash
# Create .env in project root
touch .env

# Add your API key(s)
echo "OPENAI_API_KEY=sk-..." >> .env
echo "ANTHROPIC_API_KEY=sk-ant-..." >> .env
echo "GEMINI_API_KEY=AIza..." >> .env
```

**Option B: Environment variables**

```bash
export OPENAI_API_KEY=sk-...
```

**Option C: Config file** (see [docs/models.md](../docs/models.md) for details)

```toml
# cogent.toml or ~/.cogent/config.toml
[models.openai]
api_key = "sk-..."
```

### 2. Run Your First Example

```bash
uv run python examples/basics/hello_world.py
```

That's it! No imports, no config files needed.

---

## 🔑 How API Key Auto-Discovery Works

Cogent automatically finds your API keys in this priority order:

| Priority | Source | Example |
|----------|--------|---------|
| 1 (highest) | Explicit parameter | `Agent(model="gpt4", api_key="sk-...")` |
| 2 | Environment variable | `OPENAI_API_KEY=sk-...` |
| 3 | `.env` file | Auto-loaded from project root |
| 4 | Project config | `cogent.toml` or `cogent.yaml` |
| 5 (lowest) | User config | `~/.cogent/config.toml` |

**Supported environment variables:**

| Provider | Variable |
|----------|----------|
| OpenAI | `OPENAI_API_KEY` |
| Anthropic | `ANTHROPIC_API_KEY` |
| Google Gemini | `GEMINI_API_KEY` or `GOOGLE_API_KEY` |
| Groq | `GROQ_API_KEY` |
| Mistral | `MISTRAL_API_KEY` |
| Cohere | `COHERE_API_KEY` |
| Azure OpenAI | `AZURE_OPENAI_API_KEY` + `AZURE_OPENAI_ENDPOINT` |
| Cloudflare | `CLOUDFLARE_API_TOKEN` + `CLOUDFLARE_ACCOUNT_ID` |
| Ollama | No key needed (local) |

📖 **Full documentation:** [docs/models.md](../docs/models.md)

---

## 🎯 Using Models in Examples

All examples use simple string-based models:

```python
from cogent import Agent

# Just use model names - keys are auto-discovered!
agent = Agent(name="Helper", model="gpt-4o-mini")
agent = Agent(name="Helper", model="claude-sonnet-4")
agent = Agent(name="Helper", model="gemini-2.5-flash")

# Or use aliases for convenience
agent = Agent(name="Helper", model="gpt4")      # → gpt-4o
agent = Agent(name="Helper", model="claude")    # → claude-sonnet-4
agent = Agent(name="Helper", model="gemini")    # → gemini-2.5-flash

# Provider prefix for explicit control
agent = Agent(name="Helper", model="groq:llama-3.1-70b")
agent = Agent(name="Helper", model="ollama:qwen2.5")
```

---

## 📁 Example Categories

### [basics/](basics/) — Start Here

Core concepts every user should know.

| Example | Description |
|---------|-------------|
| [hello_world.py](basics/hello_world.py) | Your first agent |
| [all_providers.py](basics/all_providers.py) | Test all model providers |
| [response.py](basics/response.py) | Response protocol with metadata |
| [memory.py](basics/memory.py) | Conversation memory |
| [streaming.py](basics/streaming.py) | Token-by-token streaming |
| [structured_output.py](basics/structured_output.py) | Pydantic/TypedDict responses |
| [roles.py](basics/roles.py) | Agent roles (worker, supervisor) |

---

### [flow/](flow/) — Orchestration

Event-driven multi-agent patterns.

| Example | Description |
|---------|-------------|
| [flow_basics.py](flow/flow_basics.py) | Introduction to Flow |
| [unified_flow.py](flow/unified_flow.py) | Unified orchestration |
| [event_sources.py](flow/event_sources.py) | External event sources |
| [checkpointing.py](flow/checkpointing.py) | State persistence |
| [source_filtering.py](flow/source_filtering.py) | Filter events by source |
| [pattern_syntax.py](flow/pattern_syntax.py) | Pattern matching (`event@source`) |

---

### [capabilities/](capabilities/) — Agent Tools

Pre-built tools that give agents superpowers.

| Example | Description |
|---------|-------------|
| [filesystem.py](capabilities/filesystem.py) | Read/write files safely |
| [web_search.py](capabilities/web_search.py) | Search the web |
| [code_sandbox.py](capabilities/code_sandbox.py) | Execute code safely |
| [browser.py](capabilities/browser.py) | Web browsing (Playwright) |
| [shell.py](capabilities/shell.py) | Shell command execution |
| [mcp.py](capabilities/mcp.py) | Model Context Protocol |
| [knowledge_graph.py](capabilities/knowledge_graph.py) | Graph-based knowledge |

---

### [retrieval/](retrieval/) — RAG

Retrieval-Augmented Generation patterns.

| Example | Description |
|---------|-------------|
| [retrievers.py](retrieval/retrievers.py) | Dense, sparse, hybrid retrievers |
| [hyde.py](retrieval/hyde.py) | Hypothetical Document Embeddings |
| [summarizer.py](retrieval/summarizer.py) | Document summarization |
| [pdf_summarizer.py](retrieval/pdf_summarizer.py) | PDF processing |

---

### [observability/](observability/) — Monitoring

See what your agents are doing.

| Example | Description |
|---------|-------------|
| [observer.py](observability/observer.py) | Observer v2 with levels |
| [custom_formatter.py](observability/custom_formatter.py) | Custom event formatters |
| [custom_sink.py](observability/custom_sink.py) | Custom output destinations |
| [response_metadata.py](observability/response_metadata.py) | Response metadata tracking |
| [deep_tracing.py](observability/deep_tracing.py) | Execution graphs |
| [graph.py](observability/graph.py) | Mermaid/ASCII diagrams |

---

### [advanced/](advanced/) — Production Patterns

Power-user features for production systems.

| Example | Description |
|---------|-------------|
| [reasoning.py](advanced/reasoning.py) | Extended thinking |
| [interceptors.py](advanced/interceptors.py) | Budget guards, PII masking |
| [human_in_the_loop.py](advanced/human_in_the_loop.py) | Tool approval workflows |
| [context_layer.py](advanced/context_layer.py) | Runtime configuration |
| [semantic_cache.py](advanced/semantic_cache.py) | Semantic caching |
| [acc.py](advanced/acc.py) | Context compression |

---

## 🛣️ Learning Path

### Beginner (30 min)

1. **[hello_world.py](basics/hello_world.py)** — Create your first agent
2. **[memory.py](basics/memory.py)** — Add conversation memory
3. **[structured_output.py](basics/structured_output.py)** — Get typed responses
4. **[observer.py](observability/observer.py)** — See what's happening

### Intermediate (1 hour)

5. **[flow_basics.py](flow/flow_basics.py)** — Multi-agent orchestration
6. **[filesystem.py](capabilities/filesystem.py)** — Give agents file access
7. **[retrievers.py](retrieval/retrievers.py)** — Add knowledge with RAG
8. **[streaming.py](basics/streaming.py)** — Real-time responses

### Advanced (2+ hours)

9. **[interceptors.py](advanced/interceptors.py)** — Production safeguards
10. **[reasoning.py](advanced/reasoning.py)** — Extended thinking
11. **[unified_flow.py](flow/unified_flow.py)** — Complex orchestration
12. **[custom_formatter.py](observability/custom_formatter.py)** — Custom observability

---

## 📂 Data Files

Sample files for examples are in `data/`:

| File | Used By |
|------|---------|
| `the_secret_garden.txt` | RAG examples |
| `company_knowledge.txt` | Knowledge base demos |
| `financial_report.pdf` | PDF processing |
| `mcp_server/` | MCP integration |

---

## 🔧 Troubleshooting

### "API key not found"

1. Check your `.env` file exists in project root
2. Verify the variable name matches (e.g., `OPENAI_API_KEY` not `OPENAI_KEY`)
3. Try setting it explicitly: `export OPENAI_API_KEY=sk-...`

### "Model not found"

1. Check the model name spelling
2. Use provider prefix for clarity: `openai:gpt-4o`
3. See [docs/models.md](../docs/models.md) for supported models

### Running examples

```bash
# Always run from project root
cd /path/to/cogent
uv run python examples/basics/hello_world.py
```

---

## 📚 More Resources

- **[docs/models.md](../docs/models.md)** — Complete model configuration guide
- **[docs/observability.md](../docs/observability.md)** — Observer v2 documentation
- **[docs/tools.md](../docs/tools.md)** — Creating custom tools
- **[docs/memory.md](../docs/memory.md)** — Memory systems
