# Models Module

The `agenticflow.models` module provides native LLM and embedding models using provider SDKs directly for high-performance inference.

## Overview

Supported providers:
- **OpenAI** - GPT-4o, GPT-4, GPT-3.5
- **Azure OpenAI** - With Azure AD support
- **Anthropic** - Claude 4, Claude 3.5
- **Groq** - Fast inference (Llama, Mixtral)
- **Google Gemini** - Gemini Pro, Flash
- **Ollama** - Local models
- **Custom** - Any OpenAI-compatible endpoint

```python
from agenticflow.models import ChatModel, create_chat

# Simple usage (OpenAI default)
model = ChatModel(model="gpt-4o")
response = await model.ainvoke([{"role": "user", "content": "Hello!"}])
print(response.content)

# Factory for any provider
model = create_chat("anthropic", model="claude-sonnet-4-20250514")
```

---

## OpenAI

The default provider for ChatModel and EmbeddingModel:

```python
from agenticflow.models import ChatModel, EmbeddingModel
from agenticflow.models.openai import OpenAIChat, OpenAIEmbedding

# Alias (recommended)
model = ChatModel(model="gpt-4o")

# Explicit
model = OpenAIChat(
    model="gpt-4o",
    temperature=0.7,
    max_tokens=2000,
    api_key="sk-...",  # Or OPENAI_API_KEY env var
)

# Embeddings
embeddings = EmbeddingModel(model="text-embedding-3-small")
vectors = await embeddings.embed_documents(["Hello world"])
```

**With tools:**

```python
from agenticflow.tools import tool

@tool
def search(query: str) -> str:
    """Search the web."""
    return f"Results for: {query}"

model = ChatModel(model="gpt-4o")
bound = model.bind_tools([search])

response = await bound.ainvoke([
    {"role": "user", "content": "Search for AI news"}
])

if response.tool_calls:
    for call in response.tool_calls:
        print(f"Tool: {call['name']}, Args: {call['args']}")
```

---

## Azure OpenAI

Enterprise Azure deployments with Azure AD support:

```python
from agenticflow.models.azure import AzureChat, AzureEmbedding, AzureEntraAuth

# With API key
model = AzureChat(
    azure_endpoint="https://your-resource.openai.azure.com",
    deployment="gpt-4o",
    api_key="your-api-key",
    api_version="2024-02-01",
)

# With Entra ID (DefaultAzureCredential)
model = AzureChat(
    azure_endpoint="https://your-resource.openai.azure.com",
    deployment="gpt-4o",
    entra=AzureEntraAuth(method="default"),  # Uses DefaultAzureCredential
)

# With Entra ID (Managed Identity)
# - System-assigned MI: omit client_id
# - User-assigned MI: set client_id (recommended when multiple identities exist)
model = AzureChat(
    azure_endpoint="https://your-resource.openai.azure.com",
    deployment="gpt-4o",
    entra=AzureEntraAuth(
        method="managed_identity",
        managed_identity_client_id="<USER_ASSIGNED_MANAGED_IDENTITY_CLIENT_ID>",
    ),
)

# Embeddings
embeddings = AzureEmbedding(
    azure_endpoint="https://your-resource.openai.azure.com",
    deployment="text-embedding-ada-002",
    entra=AzureEntraAuth(method="default"),
)
```

**Environment variables:**

```bash
AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com
AZURE_OPENAI_API_VERSION=2024-02-01
AZURE_OPENAI_DEPLOYMENT=gpt-4o
AZURE_OPENAI_EMBEDDING_DEPLOYMENT=text-embedding-ada-002

# Auth selection
AZURE_OPENAI_AUTH_TYPE=managed_identity  # api_key | default | managed_identity | client_secret

# API key auth
# AZURE_OPENAI_API_KEY=your-api-key

# Managed identity auth (user-assigned MI)
# AZURE_OPENAI_CLIENT_ID=...

# Service principal auth (client secret)
# AZURE_OPENAI_TENANT_ID=...
# AZURE_OPENAI_CLIENT_ID=...
# AZURE_OPENAI_CLIENT_SECRET=...
```

---

## Anthropic

Claude models with native SDK:

```python
from agenticflow.models.anthropic import AnthropicChat

model = AnthropicChat(
    model="claude-sonnet-4-20250514",
    max_tokens=4096,
    api_key="sk-ant-...",  # Or ANTHROPIC_API_KEY env var
)

response = await model.ainvoke([
    {"role": "user", "content": "Explain quantum computing"}
])
```

**Claude-specific features:**

```python
# System message
response = await model.ainvoke(
    messages=[{"role": "user", "content": "Hello"}],
    system="You are a helpful coding assistant.",
)

# With tools
model = AnthropicChat(model="claude-sonnet-4-20250514")
bound = model.bind_tools([search_tool])
```

---

## Groq

Ultra-fast inference for supported models:

```python
from agenticflow.models.groq import GroqChat

model = GroqChat(
    model="llama-3.3-70b-versatile",
    api_key="gsk_...",  # Or GROQ_API_KEY env var
)

response = await model.ainvoke([
    {"role": "user", "content": "Write a haiku about coding"}
])
```

**Available models:**

| Model | Description |
|-------|-------------|
| `llama-3.3-70b-versatile` | Llama 3.3 70B |
| `llama-3.1-8b-instant` | Fast Llama 3.1 8B |
| `mixtral-8x7b-32768` | Mixtral 8x7B |
| `gemma2-9b-it` | Gemma 2 9B |

---

## Google Gemini

Google's Gemini models:

```python
from agenticflow.models.gemini import GeminiChat, GeminiEmbedding

model = GeminiChat(
    model="gemini-2.0-flash",
    api_key="...",  # Or GOOGLE_API_KEY env var
)

response = await model.ainvoke([
    {"role": "user", "content": "What is the capital of France?"}
])

# Embeddings
embeddings = GeminiEmbedding(model="text-embedding-004")
```

---

## Ollama

Local models via Ollama:

```python
from agenticflow.models.ollama import OllamaChat, OllamaEmbedding

# Chat (requires `ollama run llama3.2`)
model = OllamaChat(
    model="llama3.2",
    base_url="http://localhost:11434",
)

response = await model.ainvoke([
    {"role": "user", "content": "Hello!"}
])

# Embeddings
embeddings = OllamaEmbedding(model="nomic-embed-text")
```

---

## Custom Endpoints

Any OpenAI-compatible endpoint (vLLM, Together AI, etc.):

```python
from agenticflow.models.custom import CustomChat, CustomEmbedding

# vLLM
model = CustomChat(
    base_url="http://localhost:8000/v1",
    model="meta-llama/Llama-3.2-3B-Instruct",
)

# Together AI
model = CustomChat(
    base_url="https://api.together.xyz/v1",
    model="meta-llama/Llama-3-70b-chat-hf",
    api_key="...",
)

# Custom embeddings
embeddings = CustomEmbedding(
    base_url="http://localhost:8000/v1",
    model="BAAI/bge-small-en-v1.5",
)
```

---

## Factory Function

Create models dynamically by provider:

```python
from agenticflow.models import create_chat, create_embedding

# OpenAI
model = create_chat("openai", model="gpt-4o")

# Azure
model = create_chat(
    "azure",
    deployment="gpt-4o",
    azure_endpoint="https://your-resource.openai.azure.com",
    entra=AzureEntraAuth(method="default"),
)

# Anthropic
model = create_chat("anthropic", model="claude-sonnet-4-20250514")

# Groq
model = create_chat("groq", model="llama-3.3-70b-versatile")

# Gemini
model = create_chat("gemini", model="gemini-2.0-flash")

# Ollama
model = create_chat("ollama", model="llama3.2")

# Custom
model = create_chat(
    "custom",
    base_url="http://localhost:8000/v1",
    model="my-model",
)
```

---

## Mock Models

For testing without API calls:

```python
from agenticflow.models import MockChatModel, MockEmbedding

# Predictable responses
model = MockChatModel(responses=["Hello!", "How can I help?"])

response = await model.ainvoke([{"role": "user", "content": "Hi"}])
print(response.content)  # "Hello!"

response = await model.ainvoke([{"role": "user", "content": "Help"}])
print(response.content)  # "How can I help?"

# Mock embeddings
embeddings = MockEmbedding(dimension=384)
vectors = await embeddings.embed_documents(["test"])
print(len(vectors[0]))  # 384
```

---

## Streaming

All models support streaming:

```python
from agenticflow.models import ChatModel

model = ChatModel(model="gpt-4o")

async for chunk in model.astream([
    {"role": "user", "content": "Write a story"}
]):
    print(chunk.content, end="", flush=True)
```

---

## Base Classes

### BaseChatModel

Protocol for all chat models:

```python
from agenticflow.models.base import BaseChatModel

class BaseChatModel(Protocol):
    async def ainvoke(
        self,
        messages: list[dict],
        **kwargs,
    ) -> AIMessage: ...
    
    async def astream(
        self,
        messages: list[dict],
        **kwargs,
    ) -> AsyncIterator[AIMessage]: ...
    
    def bind_tools(
        self,
        tools: list[BaseTool],
    ) -> BaseChatModel: ...
```

### AIMessage

Response type from chat models:

```python
from agenticflow.models.base import AIMessage

@dataclass
class AIMessage:
    content: str
    tool_calls: list[dict] | None = None
    usage: dict | None = None  # {"input_tokens": ..., "output_tokens": ...}
    raw: Any = None  # Original provider response
```

### BaseEmbedding

Protocol for embedding models:

```python
from agenticflow.models.base import BaseEmbedding

class BaseEmbedding(Protocol):
    async def embed_documents(
        self,
        texts: list[str],
    ) -> list[list[float]]: ...
    
    async def embed_query(
        self,
        text: str,
    ) -> list[float]: ...
```

---

## API Reference

### ChatModel Aliases

| Alias | Actual Class |
|-------|--------------|
| `ChatModel` | `OpenAIChat` |
| `EmbeddingModel` | `OpenAIEmbedding` |

### Provider Classes

| Provider | Chat Class | Embedding Class |
|----------|------------|-----------------|
| OpenAI | `OpenAIChat` | `OpenAIEmbedding` |
| Azure | `AzureChat` | `AzureEmbedding` |
| Anthropic | `AnthropicChat` | - |
| Groq | `GroqChat` | - |
| Gemini | `GeminiChat` | `GeminiEmbedding` |
| Ollama | `OllamaChat` | `OllamaEmbedding` |
| Custom | `CustomChat` | `CustomEmbedding` |

### Factory Functions

| Function | Description |
|----------|-------------|
| `create_chat(provider, **kwargs)` | Create chat model for any provider |
| `create_embedding(provider, **kwargs)` | Create embedding model for any provider |
