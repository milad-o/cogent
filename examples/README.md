# AgenticFlow Examples

Comprehensive examples organized by category to help you learn AgenticFlow.

## Quick Start

```bash
# Set up your API key
export OPENAI_API_KEY="sk-..."  # or ANTHROPIC_API_KEY, etc.

# Run any example
uv run python examples/basics/01_hello_world.py
```

---

## 📁 Categories

### [basics/](basics/) - Getting Started
Core concepts every user should know.

| Example | Description |
|---------|-------------|
| [01_hello_world.py](basics/01_hello_world.py) | Single agent basics |
| [02_memory.py](basics/02_memory.py) | Thread-based conversation memory |
| [03_roles.py](basics/03_roles.py) | Agent roles (worker, supervisor, etc.) |
| [04_streaming.py](basics/04_streaming.py) | Token-by-token LLM streaming |
| [05_structured_output.py](basics/05_structured_output.py) | Pydantic/TypedDict responses |

---

### [topologies/](topologies/) - Multi-Agent Patterns
Coordinate multiple agents in different configurations.

| Example | Description |
|---------|-------------|
| [01_pipeline.py](topologies/01_pipeline.py) | Sequential: A → B → C |
| [02_flow.py](topologies/02_flow.py) | Flow streaming and events |
| [03_mesh.py](topologies/03_mesh.py) | All-to-all communication |
| [04_hierarchical.py](topologies/04_hierarchical.py) | Nested team structures |
| [05_supervisor.py](topologies/05_supervisor.py) | Manager delegates to workers |
| [06_spawning.py](topologies/06_spawning.py) | Dynamic agent creation |

---

### [capabilities/](capabilities/) - Agent Tools & Skills
Pre-built capabilities that give agents superpowers.

| Example | Description |
|---------|-------------|
| [01_knowledge_graph.py](capabilities/01_knowledge_graph.py) | Graph-based knowledge storage |
| [02_codebase_analyzer.py](capabilities/02_codebase_analyzer.py) | Code understanding & analysis |
| [03_filesystem.py](capabilities/03_filesystem.py) | Read/write files safely |
| [04_web_search.py](capabilities/04_web_search.py) | Search the web |
| [05_code_sandbox.py](capabilities/05_code_sandbox.py) | Execute code safely |
| [06_ssis_analyzer.py](capabilities/06_ssis_analyzer.py) | SSIS package analysis |
| [07_mcp.py](capabilities/07_mcp.py) | Model Context Protocol integration |
| [08_browser.py](capabilities/08_browser.py) | Web browsing with Playwright |
| [09_spreadsheet.py](capabilities/09_spreadsheet.py) | Excel/CSV manipulation |
| [10_shell.py](capabilities/10_shell.py) | Shell command execution |

---

### [retrieval/](retrieval/) - RAG & Document Processing
Retrieval-Augmented Generation patterns.

| Example | Description |
|---------|-------------|
| [01_rag.py](retrieval/01_rag.py) | 5 RAG patterns (agentic, naive, programmatic) |
| [02_pdf.py](retrieval/02_pdf.py) | PDF loading and chunking |
| [03_pdf_llm.py](retrieval/03_pdf_llm.py) | LLM-based PDF extraction |
| [04_hyde.py](retrieval/04_hyde.py) | HyDE (Hypothetical Document Embeddings) |
| [05_summarizer.py](retrieval/05_summarizer.py) | Document summarization |
| [06_pdf_summarizer.py](retrieval/06_pdf_summarizer.py) | PDF-specific summarization |
| [07_retrievers.py](retrieval/07_retrievers.py) | All retriever types (dense, sparse, hybrid) |

---

### [observability/](observability/) - Monitoring & Visualization
See what your agents are doing.

| Example | Description |
|---------|-------------|
| [01_events.py](observability/01_events.py) | Event callbacks and tracking |
| [02_observer.py](observability/02_observer.py) | Observer levels (verbose, debug, trace) |
| [03_deep_tracing.py](observability/03_deep_tracing.py) | Execution graphs and timelines |
| [04_graph.py](observability/04_graph.py) | Visualize agents as Mermaid/ASCII diagrams |

---

### [advanced/](advanced/) - Power User Features
Advanced patterns for production systems.

| Example | Description |
|---------|-------------|
| [01_human_in_the_loop.py](advanced/01_human_in_the_loop.py) | Tool approval workflows |
| [02_deferred_tools.py](advanced/02_deferred_tools.py) | Webhook/callback-based tools |
| [03_reasoning.py](advanced/03_reasoning.py) | Extended thinking (chain-of-thought) |
| [04_interceptors.py](advanced/04_interceptors.py) | BudgetGuard, PII masking, rate limiting |
| [05_context_layer.py](advanced/05_context_layer.py) | RunContext, ToolGate, Failover |

---

## 🔧 Configuration

All examples use `config.py` for model configuration:

```python
# examples/config.py loads from environment
export OPENAI_API_KEY="sk-..."
export ANTHROPIC_API_KEY="sk-ant-..."
export AZURE_OPENAI_ENDPOINT="https://..."

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
uv run python examples/basics/01_hello_world.py

# With verbose output
uv run python examples/topologies/01_pipeline.py

# RAG example (needs embeddings)
uv run python examples/retrieval/01_rag.py
```

---

## 📚 Learning Path

**Beginners:**
1. `basics/01_hello_world.py` - Your first agent
2. `basics/02_memory.py` - Conversations that remember
3. `topologies/01_pipeline.py` - Chain agents together

**Intermediate:**
4. `retrieval/01_rag.py` - Add knowledge to agents
5. `capabilities/03_filesystem.py` - File operations
6. `observability/02_observer.py` - See what's happening

**Advanced:**
7. `advanced/04_interceptors.py` - Production safeguards
8. `topologies/06_spawning.py` - Dynamic agent creation
9. `advanced/05_context_layer.py` - Runtime configuration
