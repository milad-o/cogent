# Models Module

The `agenticflow.models` module provides a **3-tier API** for working with LLMs - from simple string-based models to full control with direct SDK access.

## 🎯 3-Tier Model API

AgenticFlow offers three levels of abstraction - choose based on your needs:

### Tier 1: High-Level (String Models) ⭐ **Recommended**

The simplest way to get started. Just use model name strings:

```python
from agenticflow import Agent

# Auto-resolves to gpt-4o
agent = Agent("Helper", model="gpt4")

# Auto-resolves to gemini-2.5-flash
agent = Agent("Helper", model="gemini")

# Auto-resolves to claude-sonnet-4
agent = Agent("Helper", model="claude")

# Provider prefix for explicit control
agent = Agent("Helper", model="anthropic:claude-opus-4")
agent = Agent("Helper", model="openai:gpt-4o")
```

**30+ Model Aliases:**
- `gpt4`, `gpt4-mini`, `gpt4-turbo`, `gpt35`
- `claude`, `claude-opus`, `claude-haiku`
- `gemini`, `gemini-flash`, `gemini-pro`
- `llama`, `llama-70b`, `llama-8b`, `mixtral`
- `ollama`

**API Key Loading** (Priority Order):
1. Explicit `api_key=` parameter (highest)
2. Environment variables (includes `.env` when loaded)
3. Config file `agenticflow.toml` / `agenticflow.yaml` or `~/.agenticflow/config.*` (lowest)

### Tier 2: Medium-Level (Factory Functions)

For when you need a model instance without an agent. Supports **4 flexible usage patterns**:

```python
from agenticflow.models import create_chat

# Pattern 1: Model name only (auto-detects provider)
llm = create_chat("gpt-4o")              # OpenAI
llm = create_chat("gemini-2.5-pro")      # Google Gemini
llm = create_chat("claude-sonnet-4")     # Anthropic
llm = create_chat("llama-3.1-8b-instant")  # Groq
llm = create_chat("mistral-small-latest")  # Mistral

# Pattern 2: Provider:model syntax (explicit provider prefix)
llm = create_chat("openai:gpt-4o")
llm = create_chat("gemini:gemini-2.5-flash")
llm = create_chat("anthropic:claude-sonnet-4-20250514")

# Pattern 3: Separate provider and model arguments
llm = create_chat("openai", "gpt-4o")
llm = create_chat("gemini", "gemini-2.5-pro")
llm = create_chat("anthropic", "claude-sonnet-4")

# Pattern 4: With additional configuration
llm = create_chat("gpt-4o", temperature=0.7, max_tokens=1000)
llm = create_chat("openai", "gpt-4o", api_key="sk-custom...")

# Use the model
response = await llm.ainvoke("What is 2+2?")
print(response.content)
```

**Auto-Detection:** Patterns 1 and 2 automatically detect the provider from model name prefixes:
- **OpenAI:** `gpt-`, `o1-`, `o3-`, `o4-`, `text-embedding-`, `gpt-audio`, `gpt-realtime`, `sora-`
- **Gemini:** `gemini-`, `text-embedding-`
- **Anthropic:** `claude-`
- **Mistral:** `mistral-`, `ministral-`, `magistral-`, `devstral-`, `codestral-`, `voxtral-`, `ocr-`
- **Cohere:** `command-`, `c4ai-aya-`, `embed-`, `rerank-`
- **Groq:** `llama-`, `mixtral-`, `qwen-`, `deepseek-`, `gemma-`
- **Cloudflare:** `@cf/`

### Tier 3: Low-Level (Direct Model Classes)

For maximum control over model configuration:

```python
from agenticflow.models import OpenAIChat, AnthropicChat, GeminiChat

# Full control over all parameters
model = OpenAIChat(
    model="gpt-4o",
    temperature=0.7,
    max_tokens=2000,
    api_key="sk-...",
    organization="org-...",
)

model = GeminiChat(
    model="gemini-2.5-flash",
    temperature=0.9,
    api_key="...",
)

model = AnthropicChat(
    model="claude-sonnet-4-20250514",
    max_tokens=4096,
    api_key="sk-ant-...",
)
```

**When to Use Each Tier:**

| Tier | Use Case | Example |
|------|----------|---------|
| **Tier 1** (Strings) | Quick prototyping, simple agents | `Agent(model="gpt4")` |
| **Tier 2** (Factory) | Reusable model instances | `create_chat("claude")` |
| **Tier 3** (Direct) | Custom config, advanced features | `OpenAIChat(temperature=0.9)` |

---

## Configuration

### .env File (Recommended for Development)

Create a `.env` file in your project root:

```bash
# .env
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
GEMINI_API_KEY=AIza...
GROQ_API_KEY=gsk_...
```

AgenticFlow automatically loads `.env` files using `python-dotenv`.

### Model Overrides (Environment + Config)

You can override default **chat** or **embedding** models via env vars or config files.

**Environment variables (highest):**
```bash
OPENAI_CHAT_MODEL=gpt-4.1
OPENAI_EMBEDDING_MODEL=text-embedding-3-large
GEMINI_CHAT_MODEL=gemini-2.5-flash
GEMINI_EMBEDDING_MODEL=gemini-embedding-001
MISTRAL_CHAT_MODEL=mistral-small-latest
MISTRAL_EMBEDDING_MODEL=mistral-embed
GROQ_CHAT_MODEL=llama-3.1-8b-instant
COHERE_CHAT_MODEL=command-a-03-2025
COHERE_EMBEDDING_MODEL=embed-english-v3.0
CLOUDFLARE_CHAT_MODEL=@cf/meta/llama-3.1-8b-instruct
CLOUDFLARE_EMBEDDING_MODEL=@cf/baai/bge-base-en-v1.5
GITHUB_CHAT_MODEL=gpt-4.1
GITHUB_EMBEDDING_MODEL=text-embedding-3-large
OLLAMA_CHAT_MODEL=qwen2.5:7b
OLLAMA_EMBEDDING_MODEL=nomic-embed-text
```

**Config file (fallback):**
```toml
[models.openai]
chat_model = "gpt-4.1"
embedding_model = "text-embedding-3-large"
```

### Config File (Recommended for Production)

Create a config file at one of these locations:

**TOML Format** (`agenticflow.toml` or `~/.agenticflow/config.toml`):

```toml
[models]
default = "gpt4"

[models.openai]
api_key = "sk-..."
organization = "org-..."

[models.anthropic]
api_key = "sk-ant-..."

[models.gemini]
api_key = "..."

[models.groq]
api_key = "gsk_..."
```

**YAML Format** (`agenticflow.yaml` or `~/.agenticflow/config.yaml`):

```yaml
models:
  default: gpt4
  
  openai:
    api_key: sk-...
    organization: org-...
  
  anthropic:
    api_key: sk-ant-...
  
  gemini:
    api_key: ...
```

### Environment Variables

```bash
export OPENAI_API_KEY=sk-...
export ANTHROPIC_API_KEY=sk-ant-...
export GEMINI_API_KEY=AIza...
export GROQ_API_KEY=gsk_...
```

---

## Provider Support

All chat models now accept multiple input formats for maximum convenience:

### 1. Simple String (Most Convenient)
```python
response = await model.ainvoke("What is the capital of France?")
```

### 2. List of Dicts (Standard Format)
```python
response = await model.ainvoke([
    {"role": "system", "content": "You are helpful"},
    {"role": "user", "content": "Hello"},
])
```

### 3. Message Objects (Type-Safe)
```python
from agenticflow.core.messages import SystemMessage, HumanMessage

response = await model.ainvoke([
    SystemMessage(content="You are helpful"),
    HumanMessage(content="Hello"),
])
```

---

## OpenAI

```python
from agenticflow.models import OpenAIChat, OpenAIEmbedding

# Tier 1: Simple string
agent = Agent("Helper", model="gpt4")

# Tier 2: Factory
model = create_chat("gpt4")
model = create_chat("openai", "gpt-4o")

# Tier 3: Direct
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
from agenticflow.models.azure import AzureEntraAuth, AzureOpenAIChat, AzureOpenAIEmbedding

# With API key
model = AzureOpenAIChat(
    azure_endpoint="https://your-resource.openai.azure.com",
    deployment="gpt-4o",
    api_key="your-api-key",
    api_version="2024-02-01",
)

# With Entra ID (DefaultAzureCredential)
model = AzureOpenAIChat(
    azure_endpoint="https://your-resource.openai.azure.com",
    deployment="gpt-4o",
    entra=AzureEntraAuth(method="default"),  # Uses DefaultAzureCredential
)

# With Entra ID (Managed Identity)
# - System-assigned MI: omit client_id
# - User-assigned MI: set client_id (recommended when multiple identities exist)
model = AzureOpenAIChat(
    azure_endpoint="https://your-resource.openai.azure.com",
    deployment="gpt-4o",
    entra=AzureEntraAuth(
        method="managed_identity",
        client_id="<USER_ASSIGNED_MANAGED_IDENTITY_CLIENT_ID>",
    ),
)

# Embeddings
embeddings = AzureOpenAIEmbedding(
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
| Azure | `AzureOpenAIChat` | `AzureOpenAIEmbedding` |
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
