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
  - [x] In-memory graph (networkx fallback to dict)
  - [x] Tools: remember, recall, query, connect, forget, list
  - [x] Multi-hop path finding
  - [x] Pattern-based queries
- [x] Tests: 27 tests passing
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

### Phase 5: WebSearch
- [ ] `capabilities/web_search.py`
  - [ ] Tavily integration
  - [ ] Search + fetch tools
  - [ ] Result deduplication

### Phase 6: CodeSandbox
- [ ] `capabilities/code_sandbox.py`
  - [ ] Safe Python execution
  - [ ] execute, test tools

---
**Status**: Phase 4 complete, 373 tests passing
**Last Updated**: 2025-11-26
