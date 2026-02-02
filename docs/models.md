# Models Module

The `cogent.models` module provides a **3-tier API** for working with LLMs - from simple string-based models to full control with direct SDK access.

## 🎯 3-Tier Model API

Cogent offers three levels of abstraction - choose based on your needs:

### Tier 1: High-Level (String Models) ⭐ **Recommended**

The simplest way to get started. Just use model name strings:

```python
from cogent import Agent

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
- `gemini`, `gemini-flash`, `gemini-pro`, `gemini3` ⚠️
- `llama`, `llama-70b`, `llama-8b`, `mixtral`
- `ollama`

⚠️ = Preview model (not production-ready)

**API Key Loading** (Priority Order):
1. Explicit `api_key=` parameter (highest)
2. Environment variables (includes `.env` when loaded)
3. Config file `cogent.toml` / `cogent.yaml` or `~/.cogent/config.*` (lowest)

### Tier 2: Medium-Level (Factory Functions)

For when you need a model instance without an agent. Supports **4 flexible usage patterns**:

```python
from cogent.models import create_chat

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
- **xAI:** `grok-`
- **DeepSeek:** `deepseek-`
- **Cerebras:** `llama3.1-`, `llama-3.3-`, `qwen-3-`, `gpt-oss-`
- **Mistral:** `mistral-`, `ministral-`, `magistral-`, `devstral-`, `codestral-`, `voxtral-`, `ocr-`
- **Cohere:** `command-`, `c4ai-aya-`, `embed-`, `rerank-`
- **Groq:** `llama-`, `mixtral-`, `qwen-`, `gemma-`
- **Cloudflare:** `@cf/`

### Tier 3: Low-Level (Direct Model Classes)

For maximum control over model configuration:

```python
from cogent.models import OpenAIChat, AnthropicChat, GeminiChat

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

Cogent automatically loads `.env` files using `python-dotenv`.

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

**TOML Format** (`cogent.toml` or `~/.cogent/config.toml`):

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

**YAML Format** (`cogent.yaml` or `~/.cogent/config.yaml`):

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
from cogent.core.messages import SystemMessage, HumanMessage

response = await model.ainvoke([
    SystemMessage(content="You are helpful"),
    HumanMessage(content="Hello"),
])
```

---

## OpenAI

```python
from cogent.models import OpenAIChat, OpenAIEmbedding

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
embeddings = OpenAIEmbedding(model="text-embedding-3-small")

# Primary API with metadata
result = await embeddings.embed(["Hello world"])
print(result.embeddings)  # Vectors
print(result.metadata)    # Full metadata

# Convenience for single text
result = await embeddings.embed("Query")
vector = result.embeddings[0]
```

**With tools:**

```python
from cogent.tools import tool

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
    print(response.tool_calls)
```

---

## xAI (Grok)

```python
from cogent.models import XAIChat

# Latest flagship model
model = XAIChat(model="grok-4", api_key="xai-...")

# Fast agentic model (2M context)
model = XAIChat(model="grok-4-1-fast")

# Vision model
model = XAIChat(model="grok-2-vision-1212")

# With reasoning effort (grok-3-mini only)
model = XAIChat(model="grok-3-mini", reasoning_effort="high")
response = await model.ainvoke("What is 101 * 3?")
print(response.metadata.tokens.reasoning_tokens)
```

**Available Models:**
- `grok-4` (alias: `grok-4-0709`): Latest reasoning model, 256K context
- `grok-4-1-fast`: Frontier multimodal, 2M context, optimized for tool calling
- `grok-4-1-fast-reasoning`: With explicit reasoning
- `grok-4-1-fast-non-reasoning`: Faster, no reasoning
- `grok-3`, `grok-3-mini`: Previous generation
- `grok-2-vision-1212`: Vision model

**Environment Variable:** `XAI_API_KEY`

---

## DeepSeek

```python
from cogent.models import DeepSeekChat

# Standard chat model
model = DeepSeekChat(model="deepseek-chat", api_key="sk-...")

# Reasoning model with Chain of Thought
model = DeepSeekChat(model="deepseek-reasoner")
response = await model.ainvoke("9.11 and 9.8, which is greater?")

# Access reasoning content
if hasattr(response, 'reasoning'):
    print("Reasoning:", response.reasoning)
print("Answer:", response.content)
```

**Available Models:**
- `deepseek-chat`: General chat model with function calling
- `deepseek-reasoner`: Reasoning model with Chain of Thought (no function calling)

**Environment Variable:** `DEEPSEEK_API_KEY`

**Note:** DeepSeek Reasoner does NOT support function calling, temperature, or sampling parameters.

---

## Cerebras (Ultra-Fast Inference)

```python
from cogent.models import CerebrasChat

# Llama 3.1 8B (default)
model = CerebrasChat(model="llama3.1-8b", api_key="csk-...")

# Llama 3.3 70B
model = CerebrasChat(model="llama-3.3-70b")

# Streaming
async for chunk in model.astream(messages):
    print(chunk.content, end="")
```

**Available Models:**
- `llama3.1-8b`: Llama 3.1 8B (default)
- `llama-3.3-70b`: Llama 3.3 70B
- `qwen-3-32b`: Qwen 3 32B
- `gpt-oss-120b`: GPT OSS 120B (reasoning model)

**Environment Variable:** `CEREBRAS_API_KEY`

**Note:** Cerebras provides industry-leading inference speed using Wafer-Scale Engine (WSE-3).

---

## Cloudflare Workers AI

```python
from cogent.models import CloudflareChat, CloudflareEmbedding

# Chat models
model = CloudflareChat(
    model="@cf/meta/llama-3.3-70b-instruct",
    account_id="...",
    api_key="...",
)

# Embeddings
embeddings = CloudflareEmbedding(
    model="@cf/baai/bge-base-en-v1.5",
    account_id="...",
    api_key="...",
)
```

**Available Models:** All Cloudflare Workers AI models with `@cf/` prefix

**Environment Variables:** `CLOUDFLARE_ACCOUNT_ID`, `CLOUDFLARE_API_TOKEN`

---

## Azure AI Foundry (GitHub Models)

```python
from cogent.models.azure import AzureAIFoundryChat

# GitHub Models
model = AzureAIFoundryChat.from_github(
    model="meta/Meta-Llama-3.1-8B-Instruct",
    token=os.getenv("GITHUB_TOKEN"),
)

# Azure AI Foundry endpoint
model = AzureAIFoundryChat(
    model="gpt-4o-mini",
    endpoint="https://...",
    api_key="...",
)
```

**Available via GitHub Models:** Llama, Phi, Mistral, Cohere, and more

**Environment Variable:** `GITHUB_TOKEN`

---

## Previous Provider Sections Continue Below

if response.tool_calls:
    for call in response.tool_calls:
        print(f"Tool: {call['name']}, Args: {call['args']}")
```

**Responses API (Beta):**

OpenAI's Responses API is optimized for tool use and structured outputs. Use the `use_responses_api=True` parameter:

```python
from cogent.models.openai import OpenAIChat

# Standard Chat Completions API (default)
model = OpenAIChat(model="gpt-4o")

# Responses API (optimized for tool use)
model = OpenAIChat(model="gpt-4o", use_responses_api=True)

# Works seamlessly with tools
bound = model.bind_tools([search_tool, calc_tool])
response = await bound.ainvoke(messages)
```

The Responses API provides better performance for multi-turn tool conversations while maintaining the same interface.

---

## Azure OpenAI

Enterprise Azure deployments with Azure AD support:

```python
from cogent.models.azure import AzureEntraAuth, AzureOpenAIChat, AzureOpenAIEmbedding

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

result = await embeddings.embed(["Document text"])
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
from cogent.models.anthropic import AnthropicChat

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
from cogent.models.groq import GroqChat

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

**Responses API (Beta):**

Groq also supports OpenAI's Responses API for optimized tool use:

```python
from cogent.models.groq import GroqChat

# Standard Chat Completions API (default)
model = GroqChat(model="llama-3.3-70b-versatile")

# Responses API (optimized for tool use)
model = GroqChat(model="llama-3.3-70b-versatile", use_responses_api=True)

# Works seamlessly with tools
bound = model.bind_tools([search_tool])
response = await bound.ainvoke(messages)
```

---

## Google Gemini

Google's Gemini models:

```python
from cogent.models.gemini import GeminiChat, GeminiEmbedding

model = GeminiChat(
    model="gemini-2.0-flash",
    api_key="...",  # Or GOOGLE_API_KEY env var
)

response = await model.ainvoke([
    {"role": "user", "content": "What is the capital of France?"}
])

# Gemini 3 Preview (Not Production Ready)
model = GeminiChat(model="gemini-3-flash-preview")
# ⚠️ WARNING: Preview models may have breaking changes or be removed

# With thinking (Gemini 3 supports thinking_budget)
model = GeminiChat(
    model="gemini-3-flash-preview",
    thinking_budget=8192,  # Enable thinking
)

# Embeddings
embeddings = GeminiEmbedding(model="text-embedding-004")
```

**Available Models:**
- `gemini-2.5-pro`, `gemini-2.5-flash` (Stable, 1M context, thinking support)
- `gemini-2.0-flash` (Stable)
- `gemini-3-pro-preview`, `gemini-3-flash-preview` ⚠️ (Preview only, thinking support)

---

## Ollama

Local models via Ollama:

```python
from cogent.models.ollama import OllamaChat, OllamaEmbedding

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

## xAI (Grok)

Grok models with reasoning capabilities:

```python
from cogent.models.xai import XAIChat

# Flagship reasoning model (256K context)
model = XAIChat(
    model="grok-4",
    api_key="...",  # Or XAI_API_KEY env var
)

# Fast agentic model (2M context, optimized for tools)
model = XAIChat(model="grok-4-1-fast")

# Non-reasoning variant (faster, cheaper)
model = XAIChat(model="grok-4-1-fast-non-reasoning")

# With reasoning effort control (grok-3-mini only)
model = XAIChat(model="grok-3-mini", reasoning_effort="high")
# or use with_reasoning()
model = XAIChat(model="grok-3-mini").with_reasoning("high")

response = await model.ainvoke([
    {"role": "user", "content": "What is 101 * 3?"}
])

# Reasoning tokens tracked in metadata
if response.metadata.tokens:
    print(f"Reasoning tokens: {response.metadata.tokens.reasoning_tokens}")
```

**Available models:**

| Model | Context | Description |
|-------|---------|-------------|
| `grok-4` | 256K | Flagship reasoning model |
| `grok-4-1-fast` | 2M | Fast agentic, optimized for tools |
| `grok-4-1-fast-reasoning` | 2M | With explicit reasoning |
| `grok-4-1-fast-non-reasoning` | 2M | Without reasoning (faster) |
| `grok-3-mini` | - | Supports `reasoning_effort` (low/high) |
| `grok-2-vision-1212` | - | Image understanding |
| `grok-code-fast-1` | - | Code-optimized |

**Features:**
- Function/tool calling (all models)
- Structured outputs (JSON mode)
- Reasoning (grok-4, grok-4-1-fast-reasoning, grok-3-mini)
- Vision (grok-2-vision-1212)
- 2M context window (grok-4-1-fast models)

---

## DeepSeek

DeepSeek models with Chain of Thought reasoning:

```python
from cogent.models.deepseek import DeepSeekChat

# Standard chat model
model = DeepSeekChat(
    model="deepseek-chat",
    api_key="...",  # Or DEEPSEEK_API_KEY env var
)

# Reasoning model (exposes Chain of Thought)
model = DeepSeekChat(model="deepseek-reasoner")

response = await model.ainvoke("9.11 and 9.8, which is greater?")

# Access reasoning content (Chain of Thought)
if hasattr(response, 'reasoning'):
    print("Reasoning:", response.reasoning)
print("Answer:", response.content)
```

**Available models:**

| Model | Tools | Description |
|-------|-------|-------------|
| `deepseek-chat` | ✅ | General chat model with tool support |
| `deepseek-reasoner` | ❌ | Reasoning model with CoT (no tools) |

**Note:** `deepseek-reasoner` does NOT support:
- Function calling/tools
- `temperature`, `top_p`, `presence_penalty`, `frequency_penalty`

---

## Custom Endpoints

Any OpenAI-compatible endpoint (vLLM, Together AI, etc.):

```python
from cogent.models.custom import CustomChat, CustomEmbedding

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
from cogent.models import create_chat, create_embedding

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

# xAI (Grok)
model = create_chat("xai", model="grok-4-1-fast")

# DeepSeek
model = create_chat("deepseek", model="deepseek-chat")
model = create_chat("deepseek", model="deepseek-reasoner")  # Reasoning model

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
from cogent.models import MockChatModel, MockEmbedding

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

All models support streaming with **complete metadata**:

```python
from cogent.models import ChatModel

model = ChatModel(model="gpt-4o")

async for chunk in model.astream([
    {"role": "user", "content": "Write a story"}
]):
    print(chunk.content, end="", flush=True)
    
    # Access metadata in all chunks
    if chunk.metadata:
        print(f"\nModel: {chunk.metadata.model}")
        print(f"Response ID: {chunk.metadata.response_id}")
        
        # Token usage available in final chunk
        if chunk.metadata.tokens:
            print(f"Tokens: {chunk.metadata.tokens.total_tokens}")
            print(f"Finish: {chunk.metadata.finish_reason}")
```

### Streaming Metadata

**All 10 chat providers** return complete metadata during streaming:

| Provider | Model | Finish Reason | Token Usage | Notes |
|----------|-------|---------------|-------------|-------|
| OpenAI | ✅ | ✅ | ✅ | Uses `stream_options={"include_usage": True}` |
| Gemini | ✅ | ✅ | ✅ | Extracts from `usage_metadata` |
| Groq | ✅ | ✅ | ✅ | Compatible with OpenAI pattern |
| Mistral | ✅ | ✅ | ✅ | Metadata accumulation |
| Cohere | ✅ | ✅ | ✅ | Event-based streaming (`message-end`) |
| Anthropic | ✅ | ✅ | ✅ | Snapshot-based metadata |
| Cloudflare | ✅ | ✅ | ✅ | Stream options support |
| Ollama | ✅ | ✅ | ✅ | Local model metadata |
| Azure OpenAI | ✅ | ✅ | ✅ | Stream options support |
| Azure AI Foundry / GitHub | ✅ | ✅ | ✅ | Stream options via model_extras |

**Metadata Structure**:

```python
@dataclass
class MessageMetadata:
    id: str | None              # Response ID
    timestamp: str | None       # ISO 8601 timestamp
    model: str | None           # Model name/version
    tokens: TokenUsage | None   # Token counts
    finish_reason: str | None   # stop, length, error
    response_id: str | None     # Provider response ID
    duration: float | None      # Request duration (ms)
    correlation_id: str | None  # For tracing

@dataclass
class TokenUsage:
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
```

**Streaming Pattern**:

1. **Content chunks** — Include partial metadata (model, response_id, timestamp)
2. **Final chunk** — Empty content with complete metadata (finish_reason, tokens)

```python
# Example streaming flow
async for chunk in model.astream(messages):
    # Chunks 1-N: Content with partial metadata
    if chunk.content:
        print(chunk.content, end="")
    
    # Final chunk: Complete metadata
    if chunk.metadata and chunk.metadata.finish_reason:
        print(f"\n\nCompleted with {chunk.metadata.tokens.total_tokens} tokens")
```

---

## Embeddings

All 9 embedding providers support a **standardized API** with rich metadata and flexible usage patterns:

```python
from cogent.models import OpenAIEmbedding, GeminiEmbedding, OllamaEmbedding

embedder = OpenAIEmbedding(model="text-embedding-3-small")

# Primary API: embed() / aembed() - Returns EmbeddingResult with full metadata
result = await embedder.aembed(["Hello world", "Cogent"])
print(result.embeddings)            # list[list[float]] - the actual vectors
print(result.metadata.model)        # "text-embedding-3-small"
print(result.metadata.tokens)       # TokenUsage(prompt=4, completion=0, total=4)
print(result.metadata.dimensions)   # 1536
print(result.metadata.duration)     # 0.181 seconds
print(result.metadata.num_texts)    # 2

# Convenience: embed_one() / aembed_one() - Single text, returns vector only
vector = await embedder.aembed_one("Single text")
print(len(vector))  # 1536

# Sync versions
result = embedder.embed(["Text 1", "Text 2"])
vector = embedder.embed_one("Single text")

# VectorStore protocol: embed_texts() / embed_query() - Async, no metadata
vectors = await embedder.embed_texts(["Doc1", "Doc2"])  # list[list[float]]
query_vec = await embedder.embed_query("Search query")  # list[float]
```

**Standardized API Summary:**

| Method | Input | Returns | Async | Metadata |
|--------|-------|---------|-------|----------|
| `embed(texts)` | `list[str]` | `EmbeddingResult` | ❌ | ✅ |
| `aembed(texts)` | `list[str]` | `EmbeddingResult` | ✅ | ✅ |
| `embed_one(text)` | `str` | `list[float]` | ❌ | ❌ |
| `aembed_one(text)` | `str` | `list[float]` | ✅ | ❌ |
| `embed_texts(texts)` | `list[str]` | `list[list[float]]` | ✅ | ❌ |
| `embed_query(text)` | `str` | `list[float]` | ✅ | ❌ |
| `dimension` | property | `int` | - | - |

### Embedding Metadata

**All 9 embedding providers** return complete metadata:

| Provider | Token Usage | Notes |
|----------|-------------|-------|
| OpenAI | ✅ | Extracts from `response.usage.prompt_tokens` |
| Cohere | ✅ | Extracts from `response.meta.billed_units.input_tokens` |
| Mistral | ✅ | Uses OpenAI SDK, provides token counts |
| Azure OpenAI | ✅ | Extracts from `response.usage` like OpenAI |
| Gemini | ❌ | API doesn't provide token counts for embeddings |
| Ollama | ❌ | Local embeddings, no token tracking |
| Cloudflare | ❌ | API doesn't track tokens |
| Mock | ❌ | Test embedding, no real tokens |
| Custom | ⚡ | Conditional - depends on underlying API |

**Metadata Structure**:

```python
@dataclass
class EmbeddingMetadata:
    id: str                     # Unique request ID
    timestamp: str              # ISO 8601 timestamp
    model: str | None           # Model name/version
    tokens: TokenUsage | None   # Token usage (if available)
    duration: float             # Request duration (seconds)
    dimensions: int | None      # Vector dimensions
    num_texts: int              # Number of texts embedded

@dataclass
class EmbeddingResult:
    embeddings: list[list[float]]  # The actual embedding vectors
    metadata: EmbeddingMetadata    # Complete metadata
```

**Usage Examples**:

```python
# Use case 1: Need metadata for cost tracking
result = await embedder.aembed(["Text 1", "Text 2"])
vectors = result.embeddings
tokens = result.metadata.tokens  # Track token usage for billing
duration = result.metadata.duration  # Monitor performance

# Use case 2: Simple embedding without metadata
vector = await embedder.aembed_one("Query text")  # Just returns the vector

# Use case 3: VectorStore integration (protocol compliance)
# These methods are used internally by VectorStore
vectors = await embedder.embed_texts(["Document 1", "Document 2"])
query_vec = await embedder.embed_query("Search query")

# Use case 4: Sync batch embedding
result = embedder.embed(large_batch)  # Sync version for compatibility
```

**Observability Benefits**:

- **Cost tracking** — Monitor token usage across providers
- **Performance** — Track request duration and batch sizes
- **Debugging** — Trace requests with unique IDs and timestamps
- **Model versioning** — Know which embedding model version was used
- **Capacity planning** — Understand dimensions and text counts

---

## Streaming

All models support streaming with **complete metadata**:

```python
from cogent.models import ChatModel

model = ChatModel(model="gpt-4o")

async for chunk in model.astream([
    {"role": "user", "content": "Write a story"}
]):
    print(chunk.content, end="", flush=True)
    
    # Access metadata in all chunks
    if chunk.metadata:
        print(f"\nModel: {chunk.metadata.model}")
        print(f"Response ID: {chunk.metadata.response_id}")
        
        # Token usage available in final chunk
        if chunk.metadata.tokens:
            print(f"Tokens: {chunk.metadata.tokens.total_tokens}")
            print(f"Finish: {chunk.metadata.finish_reason}")
```

### Streaming Metadata

**All 10 chat providers** return complete metadata during streaming:

| Provider | Model | Finish Reason | Token Usage | Notes |
|----------|-------|---------------|-------------|-------|
| OpenAI | ✅ | ✅ | ✅ | Uses `stream_options={"include_usage": True}` |
| Gemini | ✅ | ✅ | ✅ | Extracts from `usage_metadata` |
| Groq | ✅ | ✅ | ✅ | Compatible with OpenAI pattern |

---

## Thinking & Reasoning

Several providers offer "reasoning" or "thinking" models that expose their chain-of-thought process. Cogent provides unified access to these capabilities.

### Feature Comparison

| Provider | Models | Control Parameter | Access Reasoning | Structured Output |
|----------|--------|-------------------|------------------|-------------------|
| **Anthropic** | `claude-sonnet-4`, `claude-opus-4` | `thinking_budget` | `msg.thinking` | ✅ via thinking |
| **OpenAI** | `o1`, `o3`, `o4-mini` | `reasoning_effort` | Hidden | ✅ |
| **Gemini** | `gemini-2.5-*` | `thinking_budget` | `msg.thinking` | ✅ |
| **xAI** | `grok-3-mini` | `reasoning_effort` | Hidden | ✅ |
| **DeepSeek** | `deepseek-reasoner` | Always on | `msg.reasoning` | ❌ |

### Anthropic Extended Thinking

Claude models support extended thinking with configurable token budgets:

```python
from cogent.models.anthropic import AnthropicChat

# Enable extended thinking with budget
model = AnthropicChat(
    model="claude-sonnet-4-20250514",
    thinking={"type": "enabled", "budget_tokens": 10000},
)

response = await model.ainvoke([
    {"role": "user", "content": "Solve this step by step: 15! / (12! * 3!)"}
])

# Access thinking content
if response.thinking:
    print("Thinking:", response.thinking)
print("Answer:", response.content)
```

**Using ReasoningConfig:**

```python
from cogent.models.anthropic import AnthropicChat
from cogent.reasoning import ReasoningConfig

# Create config
config = ReasoningConfig(budget_tokens=10000)

# Apply to model
model = AnthropicChat(model="claude-sonnet-4-20250514")
thinking_model = model.with_reasoning(config)

response = await thinking_model.ainvoke(messages)
```

**Features:**
- Thinking exposed in `msg.thinking` attribute
- Works with streaming (thinking streamed first)
- Compatible with `with_structured_output()` via thinking

### OpenAI Reasoning Models

OpenAI's o-series models (o1, o3, o4-mini) have built-in reasoning:

```python
from cogent.models.openai import OpenAIChat

# Reasoning effort: "low", "medium", "high"
model = OpenAIChat(
    model="o4-mini",
    reasoning_effort="high",  # More thorough reasoning
)

response = await model.ainvoke([
    {"role": "user", "content": "Prove that sqrt(2) is irrational"}
])
```

**Using ReasoningConfig:**

```python
from cogent.models.openai import OpenAIChat
from cogent.reasoning import ReasoningConfig

model = OpenAIChat(model="o4-mini")
reasoning_model = model.with_reasoning(ReasoningConfig(effort="high"))
```

**Notes:**
- Reasoning is internal (not exposed in response)
- No thinking budget - use `reasoning_effort` instead
- Supports structured output with `json_schema` response format

### Gemini Thinking

Gemini 2.5 and 3.0 models support thinking with budget control:

```python
from cogent.models.gemini import GeminiChat

model = GeminiChat(
    model="gemini-2.5-flash-preview-05-20",  # or gemini-3-flash-preview
    thinking_budget=8000,  # Token budget for thinking
)

response = await model.ainvoke([
    {"role": "user", "content": "What's the optimal strategy in this game?"}
])

# Access thinking
if response.thinking:
    print("Thought process:", response.thinking)
```

**Using ReasoningConfig:**

```python
from cogent.models.gemini import GeminiChat
from cogent.reasoning import ReasoningConfig

model = GeminiChat(model="gemini-2.5-flash-preview-05-20")
thinking_model = model.with_reasoning(ReasoningConfig(budget_tokens=8000))
```

### xAI Reasoning

Grok-3-mini supports reasoning effort control:

```python
from cogent.models.xai import XAIChat

# Enable reasoning with effort level
model = XAIChat(
    model="grok-3-mini",
    reasoning_effort="high",  # "low" or "high"
)

response = await model.ainvoke([
    {"role": "user", "content": "Explain the halting problem"}
])
```

**Using with_reasoning():**

```python
from cogent.models.xai import XAIChat

model = XAIChat(model="grok-3-mini")
reasoning_model = model.with_reasoning(effort="high")
```

**Notes:**
- Only `grok-3-mini` and `grok-3-mini-beta` support reasoning_effort
- Reasoning is internal (not exposed in response)

### DeepSeek Reasoner

DeepSeek's reasoner model exposes its chain-of-thought:

```python
from cogent.models.deepseek import DeepSeekChat

model = DeepSeekChat(model="deepseek-reasoner")

response = await model.ainvoke([
    {"role": "user", "content": "Prove the Pythagorean theorem"}
])

# Access reasoning content
if response.reasoning:
    print("Chain of thought:", response.reasoning)
print("Final answer:", response.content)
```

**Streaming reasoning:**

```python
async for chunk in model.astream(messages):
    if chunk.reasoning:
        print(f"[Reasoning] {chunk.reasoning}", end="", flush=True)
    if chunk.content:
        print(chunk.content, end="", flush=True)
```

**Notes:**
- Reasoning always enabled for `deepseek-reasoner`
- Does NOT support tools or structured output
- Use `deepseek-chat` for non-reasoning use cases

### ReasoningConfig

Unified configuration for reasoning across providers:

```python
from cogent.reasoning import ReasoningConfig

# Token budget (Anthropic, Gemini)
config = ReasoningConfig(budget_tokens=10000)

# Effort level (OpenAI, xAI)
config = ReasoningConfig(effort="high")

# Both (uses appropriate one per provider)
config = ReasoningConfig(budget_tokens=10000, effort="high")
```

**Provider mapping:**

| Provider | `budget_tokens` | `effort` |
|----------|-----------------|----------|
| Anthropic | ✅ `thinking.budget_tokens` | ❌ |
| OpenAI | ❌ | ✅ `reasoning_effort` |
| Gemini | ✅ `thinking_budget` | ❌ |
| xAI | ❌ | ✅ `reasoning_effort` |
| DeepSeek | ❌ (always on) | ❌ |

---

## Structured Output

Chat models support structured output via `with_structured_output()` for type-safe JSON responses.

### Provider Support

| Provider | Method | Strict Mode |
|----------|--------|-------------|
| **OpenAI** | `json_schema` | ✅ |
| **Anthropic** | Tool-based | ✅ |
| **Gemini** | `response_schema` | ✅ |
| **Groq** | `json_mode` | ❌ |
| **xAI** | `json_schema` | ✅ |
| **DeepSeek** | `deepseek-chat` only | ❌ |
| **Ollama** | `json_mode` | ❌ |

### Basic Usage

```python
from pydantic import BaseModel, Field
from cogent.models.openai import OpenAIChat

class Person(BaseModel):
    name: str = Field(description="Full name")
    age: int = Field(description="Age in years")

# Configure model for structured output
llm = OpenAIChat(model="gpt-4o").with_structured_output(Person)

response = await llm.ainvoke([
    {"role": "user", "content": "Extract: John Doe is 30 years old"}
])

# Response content is JSON matching schema
import json
data = json.loads(response.content)
print(data)  # {"name": "John Doe", "age": 30}
```

### Schema Types

```python
from dataclasses import dataclass
from typing import TypedDict

# Pydantic (recommended)
class PersonPydantic(BaseModel):
    name: str
    age: int

# Dataclass
@dataclass
class PersonDataclass:
    name: str
    age: int

# TypedDict
class PersonTypedDict(TypedDict):
    name: str
    age: int

# JSON Schema dict
person_schema = {
    "type": "object",
    "properties": {
        "name": {"type": "string"},
        "age": {"type": "integer"}
    },
    "required": ["name", "age"]
}

# All work with with_structured_output()
llm.with_structured_output(PersonPydantic)
llm.with_structured_output(PersonDataclass)
llm.with_structured_output(PersonTypedDict)
llm.with_structured_output(person_schema)
```

### Methods

```python
# json_schema (default, strict typing)
llm.with_structured_output(Person, method="json_schema")

# json_mode (less strict, more compatible)
llm.with_structured_output(Person, method="json_mode")
```

### With Tools

Structured output and tools can be combined (the model decides when to use each):

```python
@tool
def get_weather(location: str) -> str:
    """Get weather for a location."""
    return f"Sunny in {location}"

llm = OpenAIChat(model="gpt-4o")
llm = llm.bind_tools([get_weather])
llm = llm.with_structured_output(Person)
```

### Agent-Level Structured Output

For most use cases, use the Agent's `output` parameter instead:

```python
from cogent import Agent

agent = Agent(
    name="Extractor",
    model="gpt4",
    output=Person,  # Automatic validation and retry
)

result = await agent.run("Extract: John Doe, 30 years old")
print(result.data)  # Person(name="John Doe", age=30)
```

See [Agent Documentation](agent.md#structured-output) for more details.

---

## Base Classes

### BaseChatModel

Protocol for all chat models:

```python
from cogent.models.base import BaseChatModel

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
from cogent.models.base import AIMessage

@dataclass
class AIMessage:
    content: str
    tool_calls: list[dict] | None = None
    usage: dict | None = None  # {"input_tokens": ..., "output_tokens": ...}
    raw: Any = None  # Original provider response
```

### BaseEmbedding

Standardized protocol for all embedding models:

```python
from cogent.models.base import BaseEmbedding
from cogent.core.messages import EmbeddingResult

class BaseEmbedding(ABC):
    # Primary methods - return full metadata
    @abstractmethod
    def embed(self, texts: list[str]) -> EmbeddingResult:
        """Embed texts synchronously with metadata."""
        ...
    
    @abstractmethod
    async def aembed(self, texts: list[str]) -> EmbeddingResult:
        """Embed texts asynchronously with metadata."""
        ...
    
    # Convenience methods - single text, no metadata
    def embed_one(self, text: str) -> list[float]:
        """Embed single text synchronously, returns vector only."""
        ...
    
    async def aembed_one(self, text: str) -> list[float]:
        """Embed single text asynchronously, returns vector only."""
        ...
    
    # VectorStore protocol - async, no metadata
    async def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Embed texts for VectorStore (async, returns vectors only)."""
        ...
    
    async def embed_query(self, text: str) -> list[float]:
        """Embed query for VectorStore (async, returns vector only)."""
        ...
    
    @property
    def dimension(self) -> int:
        """Return embedding dimension."""
        ...
```

**All 9 providers implement this API:**
- OpenAIEmbedding
- AzureOpenAIEmbedding
- OllamaEmbedding
- CohereEmbedding
- GeminiEmbedding
- CloudflareEmbedding
- MistralEmbedding
- CustomEmbedding
- MockEmbedding

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
| xAI | `XAIChat` | - |
| DeepSeek | `DeepSeekChat` | - |
| Ollama | `OllamaChat` | `OllamaEmbedding` |
| Custom | `CustomChat` | `CustomEmbedding` |

### Factory Functions

| Function | Description |
|----------|-------------|
| `create_chat(provider, **kwargs)` | Create chat model for any provider |
| `create_embedding(provider, **kwargs)` | Create embedding model for any provider |

