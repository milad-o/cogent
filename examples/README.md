# AgenticFlow Examples

Comprehensive examples organized by category to help you learn AgenticFlow.

## Quick Start

```bash
# Set up your API key
export OPENAI_API_KEY="sk-..."  # or ANTHROPIC_API_KEY, etc.

# Run any example
uv run python examples/basics/hello_world.py
```

---

## 📁 Categories

### [basics/](basics/) - Getting Started
Core concepts every user should know.

| Example | Description |
|---------|-------------|
| [hello_world.py](basics/hello_world.py) | Single agent basics |
| [memory.py](basics/memory.py) | Thread-based conversation memory |
| [roles.py](basics/roles.py) | Agent roles (worker, supervisor, etc.) |
| [role_configs.py](basics/role_configs.py) | RoleConfig objects (recommended API) |
| [streaming.py](basics/streaming.py) | Token-by-token LLM streaming |
| [structured_output.py](basics/structured_output.py) | Pydantic/TypedDict responses |

---

### [topologies/](topologies/) - Multi-Agent Patterns
Coordinate multiple agents in different configurations.

| Example | Description |
|---------|-------------|
| [pipeline.py](topologies/pipeline.py) | Sequential: A → B → C |
| [flow.py](topologies/flow.py) | Flow streaming and events |
| [mesh.py](topologies/mesh.py) | All-to-all communication |
| [hierarchical.py](topologies/hierarchical.py) | Nested team structures |
| [supervisor.py](topologies/supervisor.py) | Manager delegates to workers |
| [spawning.py](topologies/spawning.py) | Dynamic agent creation |

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
| [ssis_analyzer.py](capabilities/ssis_analyzer.py) | SSIS package analysis |
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
| [deep_tracing.py](observability/deep_tracing.py) | Execution graphs and timelines |
| [graph.py](observability/graph.py) | Visualize agents as Mermaid/ASCII diagrams |

---

### [advanced/](advanced/) - Power User Features
Advanced patterns for production systems.

| Example | Description |
|---------|-------------|
| [human_in_the_loop.py](advanced/human_in_the_loop.py) | Tool approval workflows |
| [deferred_tools.py](advanced/deferred_tools.py) | Webhook/callback-based tools |
| [reasoning.py](advanced/reasoning.py) | Extended thinking (chain-of-thought) |
| [interceptors.py](advanced/interceptors.py) | BudgetGuard, PII masking, rate limiting |
| [context_layer.py](advanced/context_layer.py) | RunContext, ToolGate, Failover |

---

## 🔧 Configuration

All examples use `config.py` for model configuration:

```python
# examples/config.py loads from environment
export OPENAI_API_KEY="sk-..."
export ANTHROPIC_API_KEY="sk-ant-..."
export AZURE_OPENAI_ENDPOINT="https://..."
export AZURE_OPENAI_AUTH_TYPE="managed_identity"  # or api_key|default|client_secret

# Or create a .env file in the project root
```

Supported providers: OpenAI, Anthropic, Azure OpenAI, Google Gemini, Groq, Ollama

---

## 📂 Data Files

The `data/` folder contains sample files used by examples:
- `the_secret_garden.txt` - Sample text for RAG examples
- `company_knowledge.txt` - Knowledge base sample
- `financial_report.pdf` - PDF processing sample
- `ssis_project/` - SSIS analyzer samples
- `mcp_server/` - MCP server example

---

## 🚀 Running Examples

```bash
# Basic usage
uv run python examples/basics/hello_world.py

# With verbose output
uv run python examples/topologies/pipeline.py

# RAG example (needs embeddings)
uv run python examples/retrieval/retrievers.py
```

---

## 📚 Learning Path

**Beginners:**
1. `basics/hello_world.py` - Your first agent
2. `basics/memory.py` - Conversations that remember
3. `topologies/pipeline.py` - Chain agents together

**Intermediate:**
4. `retrieval/retrievers.py` - Add knowledge to agents
5. `capabilities/filesystem.py` - File operations
6. `observability/observer.py` - See what's happening

**Advanced:**
7. `advanced/interceptors.py` - Production safeguards
8. `topologies/spawning.py` - Dynamic agent creation
9. `advanced/context_layer.py` - Runtime configuration
