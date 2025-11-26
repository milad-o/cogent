# AgenticFlow Capabilities Workboard

## Goal
Build composable **capabilities** that plug into any agent, starting with KnowledgeGraph.

## Architecture
```
Agent(capabilities=[KnowledgeGraph(), WebSearch(), ...])
       ↓
  Auto-registers tools from each capability
```

## Progress

### Phase 1: Infrastructure ✅
- [x] `capabilities/base.py` - BaseCapability class
- [x] `capabilities/__init__.py` - Exports
- [x] Agent integration - `capabilities` param
- [x] Tests for capability infrastructure

### Phase 2: KnowledgeGraph ✅
- [x] `capabilities/knowledge_graph.py`
  - [x] Multiple backends: memory, sqlite, json
  - [x] In-memory graph (networkx fallback to dict)
  - [x] SQLite for persistence and large graphs
  - [x] JSON file with auto-save option
  - [x] Tools: remember, recall, query, connect, forget, list
  - [x] Multi-hop path finding
  - [x] Pattern-based queries
  - [x] Save/load for memory backend
  - [x] Context manager support
- [x] Tests: 50 tests passing (including persistence)
- [x] Example: `examples/12_knowledge_graph.py`

### Phase 3: CodebaseAnalyzer ✅
- [x] `capabilities/codebase.py`
  - [x] Python AST parsing
  - [x] Tools: find_classes, find_functions, find_callers, find_usages, find_subclasses, find_imports, get_definition
  - [x] Builds KnowledgeGraph of code structure
- [x] Tests: test_capabilities_extended.py
- [x] Example: `examples/13_codebase_analyzer.py`

### Phase 4: FileSystem ✅
- [x] `capabilities/filesystem.py`
  - [x] Sandboxed access (allowed_paths)
  - [x] Tools: read_file, write_file, list_directory, search_files, file_info, create_directory, copy_file, move_file, delete_file
  - [x] Path validation & security (deny patterns, traversal protection)
  - [x] Configurable write/delete permissions
- [x] Tests: 49 tests passing
- [x] Example: `examples/14_filesystem.py`

### Phase 5: WebSearch ✅
- [x] `capabilities/web_search.py`
  - [x] DuckDuckGo integration (free, no API key)
  - [x] Tools: web_search, news_search, fetch_webpage
  - [x] HTML content extraction (BeautifulSoup)
  - [x] Page caching
- [x] Tests: 27 tests (2 skipped integration)
- [x] Example: `examples/15_web_search.py`

### Phase 6: CodeSandbox ✅
- [x] `capabilities/code_sandbox.py`
  - [x] Safe Python execution with restricted builtins
  - [x] Configurable timeout and output limits
  - [x] Blocked dangerous operations (file/network/subprocess)
  - [x] Safe imports (math, json, datetime, statistics, etc.)
  - [x] Tools: execute_python, run_function
  - [x] Execution history tracking
- [x] Tests: 47 tests passing
- [x] Example: `examples/16_code_sandbox.py`

---
**Status**: All 6 capabilities complete, 465 tests passing
**Last Updated**: 2025-11-26
