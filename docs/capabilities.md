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

# Custom backend instance
from agenticflow.capabilities.knowledge_graph.backends import GraphBackend
custom_backend = MyCustomBackend()  # Your implementation
kg = KnowledgeGraph(backend=custom_backend)

agent = Agent(
    name="Researcher",
    model=model,
    capabilities=[kg],
)
```

**Backend Switching:**

Switch backends dynamically with optional data migration:

```python
# Start with in-memory
kg = KnowledgeGraph()
kg.remember("Alice", "Person", {"role": "Engineer"})

# Switch to SQLite with migration
kg.set_backend("sqlite", path="knowledge.db", migrate=True)

# All data is now persisted, continue using same instance
kg.remember("Bob", "Person", {"role": "Manager"})

# Switch to JSON
kg.set_backend("json", path="knowledge.json", migrate=True)
```

**Tools provided:**
| Tool | Description |
|------|-------------|
| `remember` | Store entities with attributes (dict or JSON string) |
| `recall` | Retrieve information about entities |
| `connect` | Create relationships between entities |
| `query_knowledge` | Query relationships (source/relation/target params) |
| `forget` | Remove entities and their relationships |
| `list_knowledge` | List all entities, optionally filtered by type |

**Backends:**
- `memory`: Fast in-memory (uses networkx if available)
- `sqlite`: Persistent SQLite for large graphs
- `json`: Simple JSON file with auto-save
- `neo4j`: Production graph database (requires neo4j package)
- Custom: Extend `GraphBackend` for your own implementation

**Visualization:**

KnowledgeGraph provides a **three-level API** for visualization:

```python
kg = KnowledgeGraph()
# ... add entities and relationships ...

# 1. LOW-LEVEL: kg.mermaid() - raw Mermaid code
code = kg.mermaid(direction="LR")
print(code)  # Raw Mermaid diagram code

# 2. MEDIUM-LEVEL: kg.render(format) - multiple formats
ascii_art = kg.render("ascii")     # Terminal-friendly
html = kg.render("html")           # Interactive HTML
png_bytes = kg.render("png")       # PNG image bytes
svg_bytes = kg.render("svg")       # SVG vector bytes

# 3. HIGH-LEVEL: kg.display() - Jupyter inline rendering
kg.display()  # Renders inline in Jupyter notebook
kg.display(direction="TB", show_attributes=True)

# GRAPHVIEW: kg.visualize() - full control
view = kg.visualize(direction="LR", group_by_type=True)
view.mermaid()   # Mermaid code
view.ascii()     # ASCII art  
view.url()       # Shareable mermaid.ink URL
view.save("graph.mmd")   # Mermaid source
view.save("graph.html")  # Interactive HTML
view.save("graph.png")   # PNG image
view.save("graph.svg")   # SVG vector
view.save("graph.dot")   # Graphviz DOT
```

**Visualization options:**
- `direction`: Layout direction - `"LR"` (left-right), `"TB"` (top-bottom), `"BT"`, `"RL"`
- `group_by_type`: Group entities by type in subgraphs (default: True)
- `show_attributes`: Display entity attributes in labels (default: False)

**Entity colors:**
- Person: Blue (`#60a5fa`)
- Company/Organization: Green (`#7eb36a`)
- Location: Orange (`#f59e0b`)
- Event: Purple (`#9b59b6`)
- Generic: Gray (`#94a3b8`)

**Tool API (improved in v1.8.3):**
```python
# Query with structured parameters (NEW)
query_knowledge(source=None, relation="works_at", target="TechCorp")
# Returns: Who works at TechCorp?

# Remember with dict attributes (NEW - preferred)
remember(entity="Alice", entity_type="Person", attributes={"role": "CEO", "age": 35})

# Also accepts JSON string for backward compatibility
remember(entity="Alice", entity_type="Person", attributes='{"role": "CEO"}')
```

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
    Shell,
    Spreadsheet,
    Summarizer,
    SummarizerConfig,
    WebSearch,
)
```
