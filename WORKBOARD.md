# WORKBOARD: Remove LangChain/LangGraph Dependencies

## Goal
Completely remove langchain and langgraph dependencies. Use native OpenAI SDK directly.

## Status: Phase 1 - Native Types Created ✅

### Completed
- [x] Created `core/messages.py` - Native message types
- [x] Created `core/models.py` - Native ChatModel wrapper  
- [x] Created `tools/base.py` - Native BaseTool and @tool decorator
- [x] Renamed HyperExecutor → NativeExecutor
- [x] Removed old executors (dag, react, plan, turbo, adaptive)

### In Progress
- [ ] **Update NativeExecutor to work standalone** (without Agent's LangChain model)
- [ ] **Create standalone execution API** for quick testing
- [ ] **Update graphs/__init__.py and factory.py**

### Phase 2 - Agent Migration (Next)
- [ ] Add native model support to AgentConfig
- [ ] Support both LangChain and native tools in Agent
- [ ] Gradually replace langchain_core imports

### Phase 3 - Full Migration (Later)
- [ ] Remove all langchain imports from Agent
- [ ] Update pyproject.toml dependencies
- [ ] Update all examples and tests

## Quick Test API (Target)
```python
from agenticflow import quick_agent, tool

@tool
def search(query: str) -> str:
    '''Search the web.'''
    return f"Results for {query}"

result = await quick_agent(
    "Search for Python tutorials",
    tools=[search],
    model="gpt-4o-mini",
)
```

