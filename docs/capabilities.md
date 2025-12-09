# Capabilities Module

The `agenticflow.capabilities` module provides composable tools that plug into any agent. Capabilities are reusable building blocks that add domain-specific functionality.

## Overview

Capabilities encapsulate related functionality and expose it through native tools. Each capability:
- Provides tools for the agent to use
- May maintain internal state (graphs, caches, connections)
- Can be initialized/shutdown with the agent lifecycle

```python
from agenticflow import Agent
from agenticflow.capabilities import (
    KnowledgeGraph, FileSystem, WebSearch, CodeSandbox,
    MCP, Spreadsheet, PDF, Browser, Shell, Summarizer,
)

agent = Agent(
    name="Assistant",
    model=model,
    capabilities=[
        KnowledgeGraph(),                     # Entity/relationship memory
        FileSystem(allowed_paths=["./data"]), # Sandboxed file operations
        WebSearch(),                          # Web search and fetching
        CodeSandbox(),                        # Safe Python execution
    ],
)
```

## Available Capabilities

### KnowledgeGraph

Entity/relationship memory with multi-hop reasoning and multiple storage backends.

```python
from agenticflow.capabilities import KnowledgeGraph

# In-memory (default)
kg = KnowledgeGraph()

# SQLite for persistence
kg = KnowledgeGraph(backend="sqlite", path="knowledge.db")

# JSON file with auto-save
kg = KnowledgeGraph(backend="json", path="knowledge.json")

agent = Agent(
    name="Researcher",
    model=model,
    capabilities=[kg],
)
```

**Tools provided:**
| Tool | Description |
|------|-------------|
| `remember` | Store entities and relationships |
| `recall` | Retrieve information about entities |
| `query` | Multi-hop relationship queries |
| `forget` | Remove entities or relationships |

**Backends:**
- `InMemoryGraph`: Fast, uses networkx if available
- `SQLiteGraph`: Persistent, handles large graphs
- `JSONFileGraph`: Simple persistence with auto-save

---

### FileSystem

Sandboxed file operations with security controls.

```python
from agenticflow.capabilities import FileSystem

# Read-only access to docs
fs = FileSystem(
    allowed_paths=["./docs"],
    allow_write=False,
)

# Full access with security
fs = FileSystem(
    allowed_paths=["./workspace"],
    deny_patterns=["*.env", "*.key", ".git/*"],
    allow_delete=True,
    max_file_size=10 * 1024 * 1024,  # 10MB
)

agent = Agent(name="Worker", model=model, capabilities=[fs])
```

**Tools provided:**
| Tool | Description |
|------|-------------|
| `read_file` | Read file contents |
| `write_file` | Write/update files |
| `list_directory` | List directory contents |
| `search_files` | Search by pattern/content |
| `copy_file` | Copy files |
| `move_file` | Move/rename files |
| `delete_file` | Delete files (if enabled) |

**Parameters:**
- `allowed_paths`: Directories the agent can access
- `deny_patterns`: Glob patterns to block (e.g., `["*.env", ".git/*"]`)
- `max_file_size`: Maximum file size in bytes (default: 10MB)
- `allow_write`: Enable write operations (default: True)
- `allow_delete`: Enable delete operations (default: False)

---

### WebSearch

Web search and page fetching using DuckDuckGo (free, no API key).

```python
from agenticflow.capabilities import WebSearch

# Default configuration
ws = WebSearch()

# Custom settings
ws = WebSearch(
    max_results=10,
    timeout=30,
    include_raw_html=False,
)

agent = Agent(name="Researcher", model=model, capabilities=[ws])
```

**Tools provided:**
| Tool | Description |
|------|-------------|
| `web_search` | Search the web |
| `news_search` | Search news articles |
| `fetch_page` | Fetch and extract page content |
| `fetch_multiple` | Fetch multiple URLs concurrently |

**Requires:** `uv add ddgs`

---

### CodeSandbox

Safe Python code execution with security controls.

```python
from agenticflow.capabilities import CodeSandbox

sandbox = CodeSandbox(
    timeout=30,           # Execution timeout in seconds
    max_output_size=10000, # Max output characters
    allow_imports=["math", "json", "re"],  # Allowed imports
)

agent = Agent(name="Coder", model=model, capabilities=[sandbox])
```

**Tools provided:**
| Tool | Description |
|------|-------------|
| `execute_python` | Run Python code |
| `run_function` | Execute a specific function |
| `eval_expression` | Evaluate an expression |

**Security features:**
- Blocked dangerous imports (os, subprocess, socket, etc.)
- Resource limits (time, memory)
- AST-based code analysis
- Sandboxed execution environment

---

### MCP (Model Context Protocol)

Connect to local and remote MCP servers.

```python
from agenticflow.capabilities import MCP

# Local server (stdio)
mcp_local = MCP.stdio(
    command="uv",
    args=["run", "my-mcp-server"],
)

# Remote server (HTTP/SSE)
mcp_http = MCP.http("https://api.example.com/mcp")

# WebSocket
mcp_ws = MCP.websocket("ws://localhost:8766")

# Multiple servers
agent = Agent(
    name="Assistant",
    model=model,
    capabilities=[
        MCP.stdio(command="npx", args=["-y", "@modelcontextprotocol/server-filesystem", "."]),
        MCP.http("https://weather-api.example.com/mcp"),
    ],
)
```

**Transport types:**
- `STDIO`: Local process communication
- `HTTP`: HTTP/Streamable HTTP
- `SSE`: Server-Sent Events
- `WEBSOCKET`: WebSocket connection

---

### Spreadsheet

Excel and CSV file operations.

```python
from agenticflow.capabilities import Spreadsheet

ss = Spreadsheet(
    default_format="xlsx",
    max_rows=100000,
)

agent = Agent(name="Analyst", model=model, capabilities=[ss])
```

**Tools provided:**
| Tool | Description |
|------|-------------|
| `read_spreadsheet` | Read Excel/CSV files |
| `write_spreadsheet` | Create/update spreadsheets |
| `query_spreadsheet` | Query with filters |
| `spreadsheet_stats` | Get statistics |

**Requires:** `uv add openpyxl pandas`

---

### Browser

Headless browser automation with Playwright.

```python
from agenticflow.capabilities import Browser

browser = Browser(
    headless=True,
    timeout=30000,
)

agent = Agent(name="WebAgent", model=model, capabilities=[browser])
```

**Tools provided:**
| Tool | Description |
|------|-------------|
| `browse_url` | Navigate to URL |
| `click_element` | Click on elements |
| `fill_form` | Fill form fields |
| `screenshot` | Take screenshots |
| `extract_content` | Extract page content |

**Requires:** `uv add playwright && playwright install`

---

### Shell

Sandboxed terminal command execution.

```python
from agenticflow.capabilities import Shell

shell = Shell(
    allowed_commands=["ls", "cat", "grep", "find"],
    working_dir="./workspace",
    timeout=30,
)

agent = Agent(name="SysAdmin", model=model, capabilities=[shell])
```

**Tools provided:**
| Tool | Description |
|------|-------------|
| `run_command` | Execute shell command |
| `run_script` | Execute shell script |

**Security features:**
- Command allowlist/blocklist
- Working directory sandboxing
- Timeout enforcement
- Output size limits

---

### Summarizer

Document summarization with multiple strategies.

```python
from agenticflow.capabilities import Summarizer, SummarizerConfig

summarizer = Summarizer(
    config=SummarizerConfig(
        strategy="map_reduce",  # or "refine", "hierarchical"
        chunk_size=4000,
        max_concurrency=5,
    ),
)

agent = Agent(name="Summarizer", model=model, capabilities=[summarizer])
```

**Tools provided:**
| Tool | Description |
|------|-------------|
| `summarize_text` | Summarize text content |
| `summarize_file` | Summarize file contents |
| `summarize_url` | Fetch and summarize URL |

**Strategies:**
- `map_reduce`: Parallel chunk summarization + combine
- `refine`: Iterative refinement through chunks
- `hierarchical`: Multi-level summarization tree

---

### RAG (Retrieval-Augmented Generation)

Document retrieval with citations and bibliography.

```python
from agenticflow.capabilities import RAG
from agenticflow.capabilities.rag import RAGConfig
from agenticflow.retriever import DenseRetriever
from agenticflow.vectorstore import VectorStore

# Simple usage
store = VectorStore(embeddings=embeddings)
await store.add_documents(chunks)
rag = RAG(DenseRetriever(store))

# With bibliography
rag = RAG(
    DenseRetriever(store),
    config=RAGConfig(
        top_k=5,
        bibliography=True,
        bibliography_fields=("author", "date"),
    ),
)

agent = Agent(name="Researcher", model=model, capabilities=[rag])
response = await agent.run("What are the key findings?")

# Add bibliography to response
formatted = rag.format_response_with_bibliography(response)
```

**Tools provided:**
| Tool | Description |
|------|-------------|
| `search_documents` | Search for relevant passages with citations |

**Configuration:**
| Option | Default | Description |
|--------|---------|-------------|
| `top_k` | `4` | Passages to retrieve |
| `bibliography` | `False` | Enable bibliography |
| `bibliography_fields` | `()` | Metadata to include |
| `citation_style` | `NUMERIC` | `NUMERIC`, `FOOTNOTE`, `INLINE`, `AUTHOR_YEAR` |

See [Retrievers Guide](retrievers.md) for advanced retriever options.

---

### CodebaseAnalyzer

Python codebase analysis with AST parsing.

```python
from agenticflow.capabilities import CodebaseAnalyzer

analyzer = CodebaseAnalyzer(
    root_path="./src",
    include_patterns=["*.py"],
)

agent = Agent(name="CodeReviewer", model=model, capabilities=[analyzer])
```

**Tools provided:**
| Tool | Description |
|------|-------------|
| `analyze_file` | Analyze Python file structure |
| `find_definitions` | Find classes/functions |
| `find_usages` | Find symbol usages |
| `get_dependencies` | Analyze imports |

---

### SSISAnalyzer

SQL Server Integration Services package analysis.

```python
from agenticflow.capabilities import SSISAnalyzer

ssis = SSISAnalyzer(project_path="./ssis_packages")

agent = Agent(name="ETLAnalyst", model=model, capabilities=[ssis])
```

**Tools provided:**
| Tool | Description |
|------|-------------|
| `analyze_package` | Analyze SSIS package |
| `trace_lineage` | Trace data lineage |
| `find_dependencies` | Find package dependencies |

---

## Creating Custom Capabilities

Extend `BaseCapability` to create your own:

```python
from agenticflow.capabilities.base import BaseCapability
from agenticflow.tools import tool, BaseTool

class MyCapability(BaseCapability):
    @property
    def name(self) -> str:
        return "my_capability"
    
    @property
    def description(self) -> str:
        return "Does something useful"
    
    @property
    def tools(self) -> list[BaseTool]:
        return [self._my_tool()]
    
    def _my_tool(self):
        @tool
        def do_something(x: str) -> str:
            '''Do something useful.'''
            return f"Did: {x}"
        return do_something
    
    async def initialize(self, agent) -> None:
        """Called when attached to agent."""
        pass
    
    async def shutdown(self) -> None:
        """Called when agent shuts down."""
        pass
```

## Exports

```python
from agenticflow.capabilities import (
    # Base class
    BaseCapability,
    # Capabilities
    Browser,
    CodebaseAnalyzer,
    CodeSandbox,
    FileSystem,
    KnowledgeGraph,
    MCP,
    MCPServerConfig,
    MCPTransport,
    PDF,
    RAG,
    Shell,
    Spreadsheet,
    SSISAnalyzer,
    Summarizer,
    SummarizerConfig,
    WebSearch,
)

from agenticflow.capabilities.rag import (
    RAGConfig,
    CitationStyle,
    CitedPassage,
)
```
