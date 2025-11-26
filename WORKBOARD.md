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

### Phase 3: WebSearch
- [ ] `capabilities/web_search.py`
  - [ ] Tavily integration
  - [ ] Search + fetch tools
  - [ ] Result deduplication

### Phase 4: CodeSandbox
- [ ] `capabilities/code_sandbox.py`
  - [ ] Safe Python execution
  - [ ] execute, test tools

---
**Status**: Phase 2 complete, 296 tests passing
**Last Updated**: 2025-11-25
