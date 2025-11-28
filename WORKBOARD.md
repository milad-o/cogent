# WORKBOARD: Remove LangChain/LangGraph Dependencies

## Goal
Completely remove langchain and langgraph dependencies. Use native SDK directly.
**NO backward compatibility with LangChain** - clean break.

## Status: Phase 7 - Native Topologies (In Progress)

### Completed ✅
- [x] Native message types (`core/messages.py`)
- [x] Native model wrappers for all providers
- [x] Native tools (`@tool` decorator, `BaseTool`)
- [x] Agent class uses native types only
- [x] All capabilities use native tools
- [x] Examples updated to native models
- [x] Prebuilt components updated

### In Progress 🔄 - Phase 7: Native Topologies
Replace LangGraph StateGraph with native implementation.

- [ ] **Create `topologies/engine.py`** - Native state machine engine
  - State management without LangGraph
  - Node execution with async support
  - Conditional routing
  - Checkpointing hooks
- [ ] **Update `topologies/base.py`** - Remove LangGraph imports
- [ ] **Update all topology classes** - Use native engine
- [ ] **Tests passing**

### Phase 8 - Native Memory System
Create comprehensive memory module with multiple backends.

- [ ] **Create `memory/` module**
  - `memory/base.py` - Abstract interfaces
  - `memory/short_term.py` - Conversation history, sliding window
  - `memory/long_term.py` - Persistent key-value storage
  - `memory/backends/` - Backend implementations
    - `inmemory.py` - In-memory (default)
    - `sqlite.py` - SQLite
    - `postgres.py` - PostgreSQL
    - `redis.py` - Redis
    - `filesystem.py` - File-based JSON/pickle
- [ ] **Update `agent/memory.py`** - Use new memory module
- [ ] **Remove LangGraph checkpoint imports**

### Phase 9 - Native Vector Store
Create vector store module with multiple backends.

- [ ] **Create `vectorstore/` module**
  - `vectorstore/base.py` - Abstract interfaces
  - `vectorstore/document.py` - Document class
  - `vectorstore/backends/`
    - `inmemory.py` - NumPy-based in-memory
    - `faiss.py` - FAISS (optional)
    - `chroma.py` - ChromaDB (optional)
    - `pinecone.py` - Pinecone (optional)
    - `qdrant.py` - Qdrant (optional)
    - `weaviate.py` - Weaviate (optional)
    - `pgvector.py` - PostgreSQL pgvector (optional)
- [ ] **Update `prebuilt/rag.py`** - Use native vector store
- [ ] **Remove LangChain document/vectorstore imports**

### Phase 10 - Final Cleanup
- [ ] Remove LangGraph from pyproject.toml dependencies
- [ ] Remove LangChain from pyproject.toml dependencies
- [ ] Remove `models/adapter.py`
- [ ] Final test pass
- [ ] Update README.md

## Architecture

### Native Topology Engine
```
TopologyEngine (replaces LangGraph StateGraph)
├── State: TypedDict with agent states
├── Nodes: Agent execution functions
├── Edges: Conditional routing
├── Checkpointer: Optional persistence
└── execute() → Async state machine loop
```

### Native Memory System
```
MemoryManager
├── ShortTermMemory (conversation history)
│   ├── add_message(role, content)
│   ├── get_history(limit) → list[Message]
│   └── clear()
├── LongTermMemory (key-value store)
│   ├── set(key, value)
│   ├── get(key) → value
│   └── search(query) → list[results]
└── Backends: InMemory, SQLite, PostgreSQL, Redis, Filesystem
```

### Native Vector Store
```
VectorStore
├── add_documents(docs: list[Document])
├── similarity_search(query, k) → list[Document]
├── delete(ids: list[str])
└── Backends: InMemory, FAISS, Chroma, Pinecone, Qdrant, Weaviate, PGVector
```

## Current LangGraph Usage in Topologies
```python
# topologies/base.py - TO BE REPLACED
from langgraph.graph import StateGraph, START, END
from langgraph.graph.state import CompiledStateGraph
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.store.base import BaseStore
from langgraph.types import Command, interrupt
```

### Completed ✅
- [x] Created `core/messages.py` - Native message types (HumanMessage, AIMessage, SystemMessage, ToolMessage)
- [x] Created `models/` module - Native model wrappers for all providers
  - [x] `openai.py` - OpenAIChat, OpenAIEmbedding
  - [x] `azure.py` - AzureChat, AzureEmbedding (with DefaultAzureCredential, ManagedIdentity)
  - [x] `anthropic.py` - AnthropicChat
  - [x] `groq.py` - GroqChat
  - [x] `gemini.py` - GeminiChat, GeminiEmbedding
  - [x] `ollama.py` - OllamaChat, OllamaEmbedding
  - [x] `custom.py` - CustomChat, CustomEmbedding (OpenAI-compatible endpoints)
- [x] Created `tools/base.py` - Native BaseTool and @tool decorator
- [x] Created standalone `run()` function - Quick execution without Agent class
- [x] Renamed `graphs/` → `executors/` module
- [x] Updated main `__init__.py` - Native-first documentation

### Phase 3 ✅ - Agent Migration (COMPLETE)
- [x] **Removed LangChain imports from agent/base.py** - Uses native messages/models only
- [x] **Removed LangChain imports from agent/config.py** - Native models only
- [x] **Removed LangChain imports from agent/state.py** - Native messages only
- [x] **Removed LangChain imports from agent/memory.py** - Native messages only
- [x] **Removed LangChain imports from agent/streaming.py** - Native AIMessage
- [x] **Consolidated AIMessage** - Single class in core/messages.py, removed duplicate from models/base.py

### Phase 4 ✅ - Capabilities & Tools (COMPLETE)
- [x] Removed LangChain from `capabilities/base.py`
- [x] Removed LangChain from all capability implementations (filesystem, web_search, code_sandbox, codebase, mcp, knowledge_graph, ssis)
- [x] Removed LangChain from `tools/registry.py`
- [x] Removed LangChain from `flow.py`
- [x] All 710 tests passing

### Phase 5 ✅ - Prebuilt & Examples (COMPLETE)
- [x] Rewrite `prebuilt/chatbot.py` with native models
- [x] Update `prebuilt/rag.py` docstrings (keeps LangChain for vector stores)
- [x] Update all examples to use native models
- [x] Update README.md

### Phase 6 ✅ - Cleanup (COMPLETE)
- [x] Update pyproject.toml dependencies
  - LangGraph required for topologies (state graph)
  - LangChain optional for RAGAgent/prebuilt components
- [x] Remove `models/adapter.py` backward compat layer → Kept for edge cases
- [x] Final test pass: 708 passed, 4 skipped

## Summary

**Core Library**: Native-first, no LangChain required for:
- Agent class
- Native models (OpenAI, Azure, Anthropic, Groq, Gemini, Ollama)
- Native tools (@tool decorator, BaseTool)
- Native messages (HumanMessage, AIMessage, SystemMessage, ToolMessage)
- Streaming
- Memory
- Resilience

**Still uses LangGraph**: Topologies (supervisor, mesh, pipeline, hierarchical)
- Future work: Create native topology implementation

**Optional LangChain**: RAGAgent, prebuilt components
- Install with: `uv add agenticflow[langchain]`

## Remaining LangChain Usage
```
topologies/base.py   - Uses LangGraph StateGraph for topology execution (REQUIRED)
models/adapter.py    - Backward compatibility adapter (OPTIONAL)
prebuilt/rag.py      - Uses LangChain for document loaders, vector stores (OPTIONAL)
examples/10_agentic_rag.py - Advanced RAG example (OPTIONAL)
```

## Quick API (Working ✅)
```python
from agenticflow import run, tool

@tool
def search(query: str) -> str:
    '''Search the web.'''
    return f"Results for {query}"

result = await run(
    "Search for Python tutorials",
    tools=[search],
    model="gpt-4o-mini",
)
```

