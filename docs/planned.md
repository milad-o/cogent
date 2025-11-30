# Planned Features

Roadmap for future AgenticFlow capabilities.

---

## A2A Protocol Integration

**Priority:** Medium-Low  
**Status:** Planned  
**Target:** When cross-framework agent interop is needed

### What is A2A?

[Agent2Agent (A2A)](https://a2a-protocol.org) is a Linux Foundation protocol for agent-to-agent communication. It complements MCP:

| Protocol | Purpose | Use Case |
|----------|---------|----------|
| **MCP** | Agent → Tool | Access databases, APIs, file systems |
| **A2A** | Agent → Agent | Collaborate with external agents |

### Key Concepts

- **AgentCard**: JSON manifest with agent identity, skills, auth, endpoint
- **Task**: Stateful work unit (submitted → working → completed)
- **Message**: Communication turn with Parts (text, file, data)
- **Artifact**: Concrete deliverables from agents
- **Transports**: JSON-RPC, gRPC, HTTP+JSON

### Proposed API

```python
from agenticflow import Agent
from agenticflow.capabilities import A2A

# Call remote A2A agent
agent = Agent(
    name="coordinator",
    model=llm,
    capabilities=[
        A2A.client("https://external-agent.example.com")
    ]
)

# Expose agent as A2A server
from agenticflow.server import A2AServer

server = A2AServer(
    agent=my_agent,
    name="Research Agent",
    skills=[
        {"id": "search", "name": "Web Search", "description": "Search the web"}
    ]
)
await server.serve(port=8000)
```

### When to Implement

- External agent integration (LangGraph, CrewAI, Semantic Kernel)
- Enterprise cross-team agent interop
- Exposing AgenticFlow agents as public services

### Resources

- [A2A Protocol Docs](https://a2a-protocol.org/latest/)
- [Python SDK](https://github.com/a2aproject/a2a-python)
- [A2A + MCP Comparison](https://a2a-protocol.org/latest/topics/a2a-and-mcp/)

---

## Other Ideas

_Add future feature ideas below._
