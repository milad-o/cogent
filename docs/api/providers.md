# Providers API Reference

The providers module offers a modular, extensible system for creating LLM and embedding models. It supports multiple providers with a unified interface, including Azure Managed Identity authentication.

## Overview

```python
from agenticflow.providers import (
    # Factory functions (recommended)
    create_model,
    create_embeddings,
    acreate_model,          # async version
    acreate_embeddings,     # async version
    get_provider,
    parse_model_spec,
    list_providers,
    
    # Spec classes
    ModelSpec,
    EmbeddingSpec,
    
    # Provider classes
    OpenAIProvider,
    AzureOpenAIProvider,
    AnthropicProvider,
    GoogleProvider,
    OllamaProvider,
    
    # Azure authentication
    AzureAuthMethod,
    AzureConfig,
    
    # Base classes
    BaseProvider,
    ProviderRegistry,
    Provider,  # Enum
)
```

---

## Quick Start

### String Specs (Simplest)

```python
from agenticflow.providers import create_model, create_embeddings

# Format: "provider/model"
model = create_model("openai/gpt-4o-mini")
model = create_model("anthropic/claude-3-5-sonnet-latest")
model = create_model("google/gemini-1.5-flash")

# Auto-detect provider from model name
model = create_model("gpt-4o")      # Infers OpenAI
model = create_model("claude-3-opus")  # Infers Anthropic

# Embeddings
embeddings = create_embeddings("openai/text-embedding-3-small")
```

### With AgentConfig

```python
from agenticflow import AgentConfig, AgentRole

# String spec
config = AgentConfig(
    name="Assistant",
    role=AgentRole.WORKER,
    model="openai/gpt-4o-mini",
)

# ModelSpec for more control
from agenticflow.providers import ModelSpec

config = AgentConfig(
    name="PreciseAgent",
    model=ModelSpec(
        provider="openai",
        model="gpt-4o",
        temperature=0.3,
        max_tokens=2000,
    ),
)
```

---

## Model Specification Formats

The system accepts models in several formats:

| Format | Example | Description |
|--------|---------|-------------|
| `provider/model` | `"openai/gpt-4o"` | **Preferred** - explicit provider |
| `provider:model` | `"openai:gpt-4o"` | Legacy format (still supported) |
| `model` | `"gpt-4o"` | Auto-detects provider from model name |
| `ModelSpec` | `ModelSpec(...)` | Full programmatic control |
| `dict` | `{"provider": "openai", ...}` | Configuration dictionary |
| `BaseChatModel` | `ChatOpenAI(...)` | Direct LangChain object |

---

## Factory Functions

### create_model

Create a chat model from various input formats.

```python
def create_model(
    spec: str | ModelSpec | BaseChatModel | dict[str, Any],
    *,
    temperature: float = 0.7,
    max_tokens: int | None = None,
    **kwargs: Any,
) -> BaseChatModel:
```

**Examples:**

```python
from agenticflow.providers import create_model

# String specs
model = create_model("openai/gpt-4o-mini")
model = create_model("anthropic/claude-3-5-sonnet-latest", temperature=0.5)
model = create_model("ollama/llama3.2")

# With parameters
model = create_model(
    "openai/gpt-4o",
    temperature=0.3,
    max_tokens=2000,
)

# Dict configuration
model = create_model({
    "provider": "openai",
    "model": "gpt-4o",
    "temperature": 0.5,
})

# Pass through existing model
from langchain_openai import ChatOpenAI
existing = ChatOpenAI(model="gpt-4o")
model = create_model(existing)  # Returns as-is
```

### acreate_model

Async version for providers that need async initialization (e.g., Azure token fetching).

```python
async def acreate_model(
    spec: str | ModelSpec | BaseChatModel | dict[str, Any],
    **kwargs: Any,
) -> BaseChatModel:
```

### create_embeddings

Create embedding models from various formats.

```python
def create_embeddings(
    spec: str | EmbeddingSpec | Embeddings | dict[str, Any],
    *,
    dimensions: int | None = None,
    **kwargs: Any,
) -> Embeddings:
```

**Examples:**

```python
from agenticflow.providers import create_embeddings

# String specs
embeddings = create_embeddings("openai/text-embedding-3-small")
embeddings = create_embeddings("cohere/embed-english-v3.0")

# With reduced dimensions (text-embedding-3-* only)
embeddings = create_embeddings(
    "openai/text-embedding-3-large",
    dimensions=1024,  # Reduce from 3072
)
```

### get_provider

Get a provider instance by name.

```python
def get_provider(name: str, **config: Any) -> BaseProvider:
```

**Examples:**

```python
from agenticflow.providers import get_provider

# Get default singleton
provider = get_provider("openai")

# Get with custom config
provider = get_provider(
    "azure",
    endpoint="https://my-resource.openai.azure.com",
)
```

### list_providers

List all registered providers.

```python
from agenticflow.providers import list_providers

print(list_providers())
# ['anthropic', 'azure', 'azure_openai', 'gemini', 'google', 'ollama', 'openai']
```

### parse_model_spec

Parse a model string into provider and model name.

```python
from agenticflow.providers import parse_model_spec

parse_model_spec("openai/gpt-4o")  # ('openai', 'gpt-4o')
parse_model_spec("gpt-4o")         # ('openai', 'gpt-4o') - auto-detected
parse_model_spec("claude-3-opus")  # ('anthropic', 'claude-3-opus')
```

---

## ModelSpec

Type-safe specification for chat models.

```python
@dataclass
class ModelSpec:
    provider: str                    # Provider name
    model: str                       # Model name
    temperature: float = 0.7         # Sampling temperature
    max_tokens: int | None = None    # Max response tokens
    top_p: float | None = None       # Nucleus sampling
    stop: list[str] | None = None    # Stop sequences
    extra: dict[str, Any] = field(default_factory=dict)  # Provider-specific
```

**Methods:**

```python
# Create from string
spec = ModelSpec.from_string("openai/gpt-4o", temperature=0.5)

# Create model
model = spec.create()

# Async create (for Azure token fetching)
model = await spec.acreate()
```

**Examples:**

```python
from agenticflow.providers import ModelSpec

# Basic usage
spec = ModelSpec(
    provider="openai",
    model="gpt-4o-mini",
    temperature=0.7,
)
model = spec.create()

# With all options
spec = ModelSpec(
    provider="anthropic",
    model="claude-3-5-sonnet-latest",
    temperature=0.3,
    max_tokens=4096,
    top_p=0.9,
    stop=["END"],
)

# From string
spec = ModelSpec.from_string("openai/gpt-4o", temperature=0.5, max_tokens=2000)
```

---

## EmbeddingSpec

Type-safe specification for embedding models.

```python
@dataclass
class EmbeddingSpec:
    provider: str                    # Provider name
    model: str                       # Model name
    dimensions: int | None = None    # Output dimensions
    extra: dict[str, Any] = field(default_factory=dict)
```

**Examples:**

```python
from agenticflow.providers import EmbeddingSpec

# Basic
spec = EmbeddingSpec(
    provider="openai",
    model="text-embedding-3-small",
)

# With reduced dimensions
spec = EmbeddingSpec(
    provider="openai",
    model="text-embedding-3-large",
    dimensions=1024,
)

# From string
spec = EmbeddingSpec.from_string("openai/text-embedding-3-small")
```

---

## Provider Classes

### OpenAIProvider

```python
class OpenAIProvider(BaseProvider):
    name = "openai"
    default_chat_model = "gpt-4o-mini"
    default_embedding_model = "text-embedding-3-small"
```

**Constructor:**

```python
def __init__(
    self,
    *,
    api_key: str | None = None,      # Uses OPENAI_API_KEY env var if None
    organization: str | None = None,
    base_url: str | None = None,     # For proxies/compatible APIs
    timeout: float | None = None,
    max_retries: int = 2,
) -> None:
```

**Examples:**

```python
from agenticflow.providers import OpenAIProvider

# Default (uses env var)
provider = OpenAIProvider()

# Custom API key
provider = OpenAIProvider(api_key="sk-...")

# Create models
model = provider.create_chat_model("gpt-4o", temperature=0.5)
embeddings = provider.create_embeddings("text-embedding-3-small")
```

---

### AzureOpenAIProvider

Full Azure OpenAI support with multiple authentication methods.

```python
class AzureOpenAIProvider(BaseProvider):
    name = "azure"
    default_chat_model = "gpt-4o-mini"
    default_embedding_model = "text-embedding-3-small"
```

**Constructor:**

```python
def __init__(
    self,
    *,
    endpoint: str | None = None,                    # Azure endpoint URL
    api_key: str | None = None,                     # API key
    api_version: str = "2024-08-01-preview",
    auth_method: AzureAuthMethod = AzureAuthMethod.API_KEY,
    managed_identity_client_id: str | None = None,  # For user-assigned MI
    ad_token: str | None = None,                    # Direct AD token
    token_provider: Callable[[], str] | None = None,
    async_token_provider: Callable[[], Awaitable[str]] | None = None,
) -> None:
```

#### Authentication Methods

```python
from agenticflow.providers import AzureAuthMethod

class AzureAuthMethod(Enum):
    API_KEY = "api_key"                    # Traditional API key
    MANAGED_IDENTITY = "managed_identity"  # Azure Managed Identity
    DEFAULT_CREDENTIAL = "default_credential"  # DefaultAzureCredential
    TOKEN_PROVIDER = "token_provider"      # Custom token provider
    AD_TOKEN = "ad_token"                  # Direct AD token string
```

#### Examples

**API Key Authentication:**

```python
from agenticflow.providers import AzureOpenAIProvider

provider = AzureOpenAIProvider(
    endpoint="https://my-resource.openai.azure.com",
    api_key="your-api-key",
)

model = provider.create_chat_model("gpt-4o-deployment")
```

**Managed Identity (System-Assigned):**

```python
from agenticflow.providers import AzureOpenAIProvider, AzureAuthMethod

provider = AzureOpenAIProvider(
    endpoint="https://my-resource.openai.azure.com",
    auth_method=AzureAuthMethod.MANAGED_IDENTITY,
)

# The provider automatically fetches tokens using ManagedIdentityCredential
model = provider.create_chat_model("gpt-4o-deployment")
```

**Managed Identity (User-Assigned):**

```python
provider = AzureOpenAIProvider(
    endpoint="https://my-resource.openai.azure.com",
    auth_method=AzureAuthMethod.MANAGED_IDENTITY,
    managed_identity_client_id="your-client-id",
)
```

**DefaultAzureCredential (Automatic):**

```python
provider = AzureOpenAIProvider(
    endpoint="https://my-resource.openai.azure.com",
    auth_method=AzureAuthMethod.DEFAULT_CREDENTIAL,
)

# Tries multiple auth methods automatically:
# Environment vars -> Managed Identity -> Azure CLI -> etc.
```

**Custom Token Provider:**

```python
def my_token_provider() -> str:
    # Custom logic to get token
    return "your-token"

provider = AzureOpenAIProvider(
    endpoint="https://my-resource.openai.azure.com",
    auth_method=AzureAuthMethod.TOKEN_PROVIDER,
    token_provider=my_token_provider,
)
```

**Using AzureConfig:**

```python
from agenticflow.providers import AzureOpenAIProvider, AzureConfig, AzureAuthMethod

config = AzureConfig(
    endpoint="https://my-resource.openai.azure.com",
    auth_method=AzureAuthMethod.MANAGED_IDENTITY,
    api_version="2024-08-01-preview",
)

provider = AzureOpenAIProvider.from_config(config)
```

**With AgentConfig:**

```python
from agenticflow import AgentConfig
from agenticflow.providers import AzureOpenAIProvider, AzureAuthMethod

provider = AzureOpenAIProvider(
    endpoint="https://my-resource.openai.azure.com",
    auth_method=AzureAuthMethod.MANAGED_IDENTITY,
)

config = AgentConfig(
    name="AzureAgent",
    model=provider.create_chat_model("gpt-4o-deployment"),
)
```

---

### AnthropicProvider

```python
class AnthropicProvider(BaseProvider):
    name = "anthropic"
    supports_embeddings = False  # Anthropic doesn't provide embeddings
    default_chat_model = "claude-3-5-sonnet-latest"
```

**Examples:**

```python
from agenticflow.providers import AnthropicProvider

provider = AnthropicProvider()
model = provider.create_chat_model("claude-3-5-sonnet-latest")

# Note: Anthropic requires max_tokens
model = provider.create_chat_model(
    "claude-3-opus-latest",
    max_tokens=4096,
)
```

---

### GoogleProvider

```python
class GoogleProvider(BaseProvider):
    name = "google"  # Also registered as "gemini"
    default_chat_model = "gemini-1.5-flash"
    default_embedding_model = "models/embedding-001"
```

**Examples:**

```python
from agenticflow.providers import GoogleProvider

provider = GoogleProvider()
model = provider.create_chat_model("gemini-1.5-pro")
embeddings = provider.create_embeddings("models/embedding-001")
```

---

### OllamaProvider

For local models via Ollama.

```python
class OllamaProvider(BaseProvider):
    name = "ollama"
    default_chat_model = "llama3.2"
    default_embedding_model = "nomic-embed-text"
```

**Examples:**

```python
from agenticflow.providers import OllamaProvider

# Default localhost
provider = OllamaProvider()

# Custom server
provider = OllamaProvider(base_url="http://192.168.1.100:11434")

model = provider.create_chat_model("llama3.2")
embeddings = provider.create_embeddings("nomic-embed-text")
```

---

## BaseProvider

Abstract base class for creating custom providers.

```python
from abc import ABC, abstractmethod

class BaseProvider(ABC):
    name: ClassVar[str]                          # Provider identifier
    supports_chat: ClassVar[bool] = True
    supports_embeddings: ClassVar[bool] = True
    default_chat_model: ClassVar[str | None] = None
    default_embedding_model: ClassVar[str | None] = None

    @abstractmethod
    def create_chat_model(
        self,
        model: str | None = None,
        *,
        temperature: float = 0.7,
        max_tokens: int | None = None,
        **kwargs: Any,
    ) -> BaseChatModel:
        ...

    async def acreate_chat_model(self, ...) -> BaseChatModel:
        # Default calls sync version
        return self.create_chat_model(...)

    @abstractmethod
    def create_embeddings(
        self,
        model: str | None = None,
        *,
        dimensions: int | None = None,
        **kwargs: Any,
    ) -> Embeddings:
        ...

    async def acreate_embeddings(self, ...) -> Embeddings:
        # Default calls sync version
        return self.create_embeddings(...)
```

### Creating a Custom Provider

```python
from agenticflow.providers import BaseProvider

class MyProvider(BaseProvider):
    name = "myprovider"  # Auto-registers on class definition
    supports_chat = True
    supports_embeddings = False
    default_chat_model = "my-model-v1"

    def create_chat_model(self, model=None, **kwargs):
        model = model or self.default_chat_model
        # Your implementation
        return MyCustomLLM(model=model, **kwargs)

    def create_embeddings(self, model=None, **kwargs):
        raise NotImplementedError("MyProvider doesn't support embeddings")

# Now available via factory
from agenticflow.providers import create_model
model = create_model("myprovider/my-model-v1")
```

---

## ProviderRegistry

Manages provider registration and instances.

```python
class ProviderRegistry:
    @classmethod
    def register(cls, provider_class: type[BaseProvider]) -> None:
        """Register a provider class."""

    @classmethod
    def get(cls, name: str) -> BaseProvider:
        """Get singleton provider instance."""

    @classmethod
    def get_with_config(cls, name: str, **config) -> BaseProvider:
        """Get new instance with custom config."""

    @classmethod
    def list_providers(cls) -> list[str]:
        """List registered provider names."""

    @classmethod
    def is_registered(cls, name: str) -> bool:
        """Check if provider exists."""
```

---

## Provider Enum

```python
from agenticflow.providers import Provider

class Provider(str, Enum):
    OPENAI = "openai"
    AZURE = "azure"
    AZURE_OPENAI = "azure_openai"
    ANTHROPIC = "anthropic"
    GOOGLE = "google"
    GEMINI = "gemini"
    OLLAMA = "ollama"
    BEDROCK = "bedrock"
    COHERE = "cohere"
    MISTRAL = "mistral"
    GROQ = "groq"
    FIREWORKS = "fireworks"
    TOGETHER = "together"
    HUGGINGFACE = "huggingface"
    VOYAGE = "voyage"
```

**Usage:**

```python
from agenticflow.providers import Provider, create_model

# Use enum in f-strings
model = create_model(f"{Provider.OPENAI}/gpt-4o")
```

---

## Default Models

```python
from agenticflow.providers.enums import DEFAULT_CHAT_MODELS, DEFAULT_EMBEDDING_MODELS

DEFAULT_CHAT_MODELS = {
    "openai": "gpt-4o-mini",
    "azure": "gpt-4o-mini",
    "anthropic": "claude-3-5-sonnet-latest",
    "google": "gemini-1.5-flash",
    "ollama": "llama3.2",
    "mistral": "mistral-large-latest",
    "groq": "llama-3.3-70b-versatile",
    ...
}

DEFAULT_EMBEDDING_MODELS = {
    "openai": "text-embedding-3-small",
    "azure": "text-embedding-3-small",
    "cohere": "embed-english-v3.0",
    "ollama": "nomic-embed-text",
    ...
}
```

---

## Integration with Vector Stores

```python
from agenticflow.memory import create_vectorstore

# String spec (uses create_embeddings internally)
vectorstore = create_vectorstore(
    "memory",
    embeddings="openai/text-embedding-3-small",
)

# EmbeddingSpec
from agenticflow.providers import EmbeddingSpec

spec = EmbeddingSpec(
    provider="openai",
    model="text-embedding-3-large",
    dimensions=1024,
)
vectorstore = create_vectorstore("memory", embeddings=spec)
```

---

## Environment Variables

| Variable | Provider | Description |
|----------|----------|-------------|
| `OPENAI_API_KEY` | OpenAI | API key |
| `AZURE_OPENAI_ENDPOINT` | Azure | Endpoint URL |
| `AZURE_OPENAI_API_KEY` | Azure | API key (if using API key auth) |
| `ANTHROPIC_API_KEY` | Anthropic | API key |
| `GOOGLE_API_KEY` | Google | API key |

---

## Required Packages

| Provider | Package | Install Command |
|----------|---------|-----------------|
| OpenAI | `langchain-openai` | `uv add langchain-openai` |
| Azure | `langchain-openai`, `azure-identity` | `uv add langchain-openai azure-identity` |
| Anthropic | `langchain-anthropic` | `uv add langchain-anthropic` |
| Google | `langchain-google-genai` | `uv add langchain-google-genai` |
| Ollama | `langchain-ollama` | `uv add langchain-ollama` |

---

## See Also

- [Agent Configuration](agents.md) - Using providers with agents
- [Memory & Vector Stores](memory.md) - Embedding integration
- [Quickstart Guide](../quickstart.md) - Getting started
