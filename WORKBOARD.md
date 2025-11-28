# WORKBOARD: Remove LangChain/LangGraph Dependencies

## Goal
Completely remove langchain and langgraph dependencies. Use native SDK directly.

## Status: Phase 2 - Purging LangChain from Executors

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

### In Progress 🔄
- [ ] **Purge LangChain from NativeExecutor** - Use native messages/models
- [ ] **Update main `__init__.py`** - Remove LangChain philosophy, update docs
- [ ] **Update `executors/__init__.py`** - Clean exports

### Phase 3 - Agent Migration (Next)
- [ ] Make Agent work with native models directly
- [ ] Support both LangChain and native tools in Agent
- [ ] Gradually replace langchain_core imports in agent/

### Phase 4 - Topologies (Later)
- [ ] Create native StateGraph replacement or simplify
- [ ] Remove LangGraph from topologies/ (big undertaking)

### Phase 5 - Full Migration
- [ ] Remove all langchain/langgraph imports
- [ ] Update pyproject.toml - remove dependencies
- [ ] Update all examples and tests

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
    model="gpt-4o-mini",  # or OpenAIChat instance
)
```

## Native Models API (Working ✅)
```python
from agenticflow.models.openai import OpenAIChat, OpenAIEmbedding
from agenticflow.models.azure import AzureChat
from agenticflow.models.anthropic import AnthropicChat
from agenticflow.models.groq import GroqChat
from agenticflow.models import create_chat, create_embedding

# Direct usage
llm = OpenAIChat(model="gpt-4o")
response = await llm.ainvoke([{"role": "user", "content": "Hello!"}])

# Factory function
llm = create_chat("groq", model="llama-3.3-70b-versatile")
```

