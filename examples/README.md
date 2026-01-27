# Cogent Examples

Comprehensive examples organized by category to help you learn Cogent.

## 🆕 What's New in v1.14.1

**3-Tier Model API - Multiple Usage Patterns:**
- [`basics/config_file.py`](basics/config_file.py) - **NEW** API key management guide
- [`basics/all_providers.py`](basics/all_providers.py) - **NEW** Test all providers in one script
- **Pattern 1:** `create_chat("gpt-4o")` - model name only (auto-detects provider)
- **Pattern 2:** `create_chat("openai:gpt-4o")` - provider:model syntax
- **Pattern 3:** `create_chat("openai", "gpt-4o")` - separate arguments
- **Pattern 4:** `create_chat("gpt-4o", temperature=0.7)` - with configuration
- 30+ model aliases: `gpt4`, `claude`, `gemini`, `llama`, etc.
- Auto-loads from `.env`, config files, or environment variables

## 👍 What's New in v1.13.0

**Response Protocol with Rich Metadata:**
- [`basics/response.py`](basics/response.py) - Learn the Response[T] protocol
- [`observability/response_metadata.py`](observability/response_metadata.py) - See metadata in Observer output

## Quick Start

```bash
# 1. Create .env file in project root
cp examples/.env .env

# 2. Add your API key to .env
# OPENAI_API_KEY=sk-...
# ANTHROPIC_API_KEY=sk-ant-...
# GEMINI_API_KEY=...

# 3. Run any example - API keys load automatically!
uv run python examples/basics/hello_world.py

# Examples use direct string models - no config needed!
# Agent(name="...", model="gpt4")  ✅ Simple!
# Agent(name="...", model="claude")  ✅ Works!
```

---

## 📁 Categories

### [basics/](basics/) - Getting Started
Core concepts every user should know.

| Example | Description |
|---------|-------------|
| [hello_world.py](basics/hello_world.py) | Single agent basics |
| [config_file.py](basics/config_file.py) | **NEW v1.14.1** API key management (.env, TOML, YAML) |
| [all_providers.py](basics/all_providers.py) | **NEW v1.14.1** Test all model providers (OpenAI, Anthropic, Gemini, Groq) |
| [response.py](basics/response.py) | Response[T] protocol - typed results with metadata |
| [memory.py](basics/memory.py) | Thread-based conversation memory |
| [memory_layers.py](basics/memory_layers.py) | **NEW** 4-layer memory architecture (conversation, ACC, long-term, cache) |
| [roles.py](basics/roles.py) | Agent roles (worker, supervisor, etc.) |
| [role_configs.py](basics/role_configs.py) | RoleConfig objects (recommended API) |
| [streaming.py](basics/streaming.py) | Token-by-token LLM streaming |
| [structured_output.py](basics/structured_output.py) | Pydantic/TypedDict responses |
| [taskboard.py](basics/taskboard.py) | Built-in task management |

---

### [flow/](flow/) - Event-Driven Patterns
Modern Flow patterns and event-driven orchestration examples.

| Example | Description |
|---------|-------------|
| [flow_basics.py](flow/flow_basics.py) | Intro to Flow and patterns |
| [unified_flow.py](flow/unified_flow.py) | Unified Flow orchestration |
| [event_sources.py](flow/event_sources.py) | External event sources |
| [checkpointing.py](flow/checkpointing.py) | Checkpointing basics |
| [checkpointing_demo.py](flow/checkpointing_demo.py) | Checkpointing demo |
| [source_filtering.py](flow/source_filtering.py) | Source-based filtering (10 demos) |
| [pattern_syntax.py](flow/pattern_syntax.py) | Pattern syntax event@source (6 demos) |
| [source_groups.py](flow/source_groups.py) | Named source groups with :group syntax (5 demos) |

---

### [capabilities/](capabilities/) - Agent Tools & Skills
Pre-built capabilities that give agents superpowers.

| Example | Description |
|---------|-------------|
| [knowledge_graph.py](capabilities/knowledge_graph.py) | Graph-based knowledge storage |
| [codebase_analyzer.py](capabilities/codebase_analyzer.py) | Code understanding & analysis |
| [filesystem.py](capabilities/filesystem.py) | Read/write files safely |
| [web_search.py](capabilities/web_search.py) | Search the web |
| [code_sandbox.py](capabilities/code_sandbox.py) | Execute code safely |
| [mcp.py](capabilities/mcp.py) | Model Context Protocol integration |
| [browser.py](capabilities/browser.py) | Web browsing with Playwright |
| [spreadsheet.py](capabilities/spreadsheet.py) | Excel/CSV manipulation |
| [shell.py](capabilities/shell.py) | Shell command execution |

---

### [retrieval/](retrieval/) - RAG & Document Processing
Retrieval-Augmented Generation patterns.

| Example | Description |
|---------|-------------|
| [hyde.py](retrieval/hyde.py) | HyDE (Hypothetical Document Embeddings) |
| [summarizer.py](retrieval/summarizer.py) | Document summarization |
| [pdf_summarizer.py](retrieval/pdf_summarizer.py) | PDF-specific summarization |
| [retrievers.py](retrieval/retrievers.py) | All retriever types (dense, sparse, hybrid) |
| [pdf_vision_showcase.py](retrieval/pdf_vision_showcase.py) | Vision PDF extraction (metadata/TOC/timing + markdown/html/json) |

---

### [observability/](observability/) - Monitoring & Visualization
See what your agents are doing.

| Example | Description |
|---------|-------------|
| [events.py](observability/events.py) | Event callbacks and tracking |
| [observer.py](observability/observer.py) | Observer levels (verbose, debug, trace) |
| [response_metadata.py](observability/response_metadata.py) | **NEW** Response metadata in Observer (tokens, tools, errors) |
| [deep_tracing.py](observability/deep_tracing.py) | Execution graphs and timelines |
| [graph.py](observability/graph.py) | Visualize agents as Mermaid/ASCII diagrams |

---

### [advanced/](advanced/) - Power User Features
Advanced patterns for production systems.

| Example | Description |
|---------|-------------|
| [human_in_the_loop.py](advanced/human_in_the_loop.py) | Tool approval workflows |
| [deferred_tools.py](advanced/deferred_tools.py) | Webhook/callback-based tools |
| [reasoning.py](advanced/reasoning.py) | Extended thinking with AI-controlled rounds |
| [interceptors.py](advanced/interceptors.py) | BudgetGuard, PII masking, rate limiting |
| [context_layer.py](advanced/context_layer.py) | RunContext, ToolGate, Failover |

**Reasoning Features:**
- Basic: `reasoning=True` enables default config
- Custom: `reasoning=ReasoningConfig(max_thinking_rounds=15, style=...)`
- Per-call override: `agent.run(..., reasoning=True/False/Config)`
- AI-controlled: Agent decides when reasoning is complete

---

## 🔧 Configuration

### Recommended: Direct String Models (v1.14.1+)

All new examples use the **3-tier model API** - no `config.py` imports needed:

```python
from cogent import Agent

# Tier 1: Simple strings (recommended!) ⭐
agent = Agent(name="Helper", model="gpt4")
agent = Agent(name="Helper", model="claude")
agent = Agent(name="Helper", model="gemini")

# Tier 2: Provider prefix for explicit control
agent = Agent(name="Helper", model="anthropic:claude-sonnet-4")
agent = Agent(name="Helper", model="groq:llama-70b")

# Tier 3: Full control with model instances
from cogent.models import OpenAIChat
agent = Agent(name="Helper", model=OpenAIChat(model="gpt-4o", temperature=0.7))
```

**API keys auto-load from:**
1. `.env` file in project root (highest priority)
2. `cogent.toml` or `cogent.yaml` (project-level)
3. `~/.cogent/config.toml` or `config.yaml` (user-level)
4. Environment variables (lowest priority)

### Legacy: config.py Helper

**Note:** `examples/models.py` is now **legacy** and maintained only for backward compatibility with older examples. New code should use direct string models as shown above.

```python
# ❌ Old way (legacy)
from models import get_model
agent = Agent(name="Helper", model=get_model())

# ✅ New way (recommended)
agent = Agent(name="Helper", model="gpt4")
```

**Supported providers:** OpenAI, Anthropic, Azure OpenAI, Google Gemini, Groq, Cohere, Cloudflare, Ollama

---

## 📂 Data Files

The `data/` folder contains sample files used by examples:
- `the_secret_garden.txt` - Sample text for RAG examples
- `company_knowledge.txt` - Knowledge base sample
- `financial_report.pdf` - PDF processing sample
- `mcp_server/` - MCP server example

---

## 🚀 Running Examples

```bash
# Basic usage
uv run python examples/basics/hello_world.py

# With verbose output
uv run python examples/flow/flow_basics.py

# RAG example (needs embeddings)
uv run python examples/retrieval/retrievers.py
```

---

## 📚 Learning Path

**Beginners:**
1. `basics/hello_world.py` - Your first agent with string models
2. `basics/config_file.py` - **NEW** How API keys are loaded
3. `basics/memory.py` - Conversations that remember
4. `flow/flow_basics.py` - Flow patterns and orchestration

**Intermediate:**
4. `retrieval/retrievers.py` - Add knowledge to agents
5. `capabilities/filesystem.py` - File operations
6. `observability/observer.py` - See what's happening

**Advanced:**
7. `advanced/interceptors.py` - Production safeguards
8. `flow/unified_flow.py` - Unified Flow orchestration
9. `advanced/context_layer.py` - Runtime configuration
