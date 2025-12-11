"""
Test all configured models from .env

Tests each provider's chat model with a simple query to verify configuration.
"""

import asyncio
import os
import time
from pathlib import Path

# Load only examples/.env to avoid picking up other env files
env_path = Path(__file__).parent.parent / "examples" / ".env"
if env_path.exists():
    from dotenv import load_dotenv
    load_dotenv(env_path, override=False)


async def test_openai():
    """Test OpenAI chat model."""
    print("\n" + "=" * 70)
    print("Testing OpenAI")
    print("=" * 70)
    
    try:
        from agenticflow.models.openai import OpenAIChat
        
        model = os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini")
        print(f"Model: {model}")
        
        llm = OpenAIChat(model=model)
        response = await llm.ainvoke([{"role": "user", "content": "Say 'OpenAI works!' and nothing else."}])
        
        print(f"✓ Response: {response.content}")
        return True
    except Exception as e:
        print(f"✗ Error: {e}")
        return False


async def test_gemini():
    """Test Gemini chat model."""
    print("\n" + "=" * 70)
    print("Testing Gemini")
    print("=" * 70)
    
    try:
        from agenticflow.models.gemini import GeminiChat
        timeout_s = int(os.getenv("MODEL_TEST_TIMEOUT", "30"))
        
        model = os.getenv("GEMINI_CHAT_MODEL", "gemini-2.5-flash")
        print(f"Model: {model}")
        llm = GeminiChat(model=model, timeout=timeout_s)
        try:
            response = await asyncio.wait_for(
                llm.ainvoke([{"role": "user", "content": "Say 'Gemini works!' and nothing else."}]),
                timeout=timeout_s,
            )
            print(f"✓ Response: {response.content}")
            return True
        except asyncio.TimeoutError:
            print(f"⚠️  Gemini timed out after {timeout_s}s; skipping")
            return True
    except Exception as e:
        print(f"✗ Error: {e}")
        return False


async def test_mistral():
    """Test Mistral chat model."""
    print("\n" + "=" * 70)
    print("Testing Mistral")
    print("=" * 70)
    
    try:
        from agenticflow.models.mistral import MistralChat
        
        model = os.getenv("MISTRAL_CHAT_MODEL", "mistral-small-latest")
        print(f"Model: {model}")
        
        llm = MistralChat(model=model)
        response = await llm.ainvoke([{"role": "user", "content": "Say 'Mistral works!' and nothing else."}])
        
        print(f"✓ Response: {response.content}")
        return True
    except Exception as e:
        print(f"✗ Error: {e}")
        return False


async def test_groq():
    """Test Groq chat model."""
    print("\n" + "=" * 70)
    print("Testing Groq")
    print("=" * 70)
    
    try:
        from agenticflow.models.groq import GroqChat
        
        model = os.getenv("GROQ_CHAT_MODEL", "llama-3.1-8b-instant")
        print(f"Model: {model}")
        
        llm = GroqChat(model=model)
        response = await llm.ainvoke([{"role": "user", "content": "Say 'Groq works!' and nothing else."}])
        
        print(f"✓ Response: {response.content}")
        return True
    except Exception as e:
        print(f"✗ Error: {e}")
        return False


async def test_ollama():
    """Test Ollama chat model."""
    print("\n" + "=" * 70)
    print("Testing Ollama")
    print("=" * 70)
    
    try:
        from agenticflow.models.ollama import OllamaChat
        host = os.getenv("OLLAMA_HOST", "http://localhost:11434")
        model = os.getenv("OLLAMA_CHAT_MODEL", "llama3.2")
        print(f"Host: {host}")
        print(f"Model: {model}")
        llm = OllamaChat(model=model, host=host)
        response = await llm.ainvoke([{ "role": "user", "content": "Say 'Ollama works!' and nothing else." }])
        print(f"✓ Response: {response.content}")
        return True
    except Exception as e:
        print(f"✗ Error: {e}")
        return False


async def test_github():
    """Test GitHub Models (Azure AI Foundry)."""
    print("\n" + "=" * 70)
    print("Testing GitHub Models")
    print("=" * 70)
    
    try:
        from agenticflow.models.azure import AzureAIFoundryChat
        
        token = os.getenv("GITHUB_TOKEN")
        if not token:
            print("✗ GITHUB_TOKEN not set, skipping")
            return False
        
        model = os.getenv("GITHUB_CHAT_MODEL", "meta/Meta-Llama-3.1-8B-Instruct")
        print(f"Model: {model}")
        
        llm = AzureAIFoundryChat.from_github(
            model=model,
            token=token,
        )
        response = await llm.ainvoke([{"role": "user", "content": "Say 'GitHub Models works!' and nothing else."}])
        
        print(f"✓ Response: {response.content}")
        return True
    except Exception as e:
        print(f"✗ Error: {e}")
        return False


async def test_anthropic():
    """Test Anthropic chat model."""
    print("\n" + "=" * 70)
    print("Testing Anthropic")
    print("=" * 70)
    
    try:
        from agenticflow.models.anthropic import AnthropicChat
        
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            print("✗ ANTHROPIC_API_KEY not set, skipping")
            return False
        
        model = os.getenv("ANTHROPIC_CHAT_MODEL", "claude-sonnet-4-20250514")
        print(f"Model: {model}")
        
        llm = AnthropicChat(model=model)
        response = await llm.ainvoke([{"role": "user", "content": "Say 'Anthropic works!' and nothing else."}])
        
        print(f"✓ Response: {response.content}")
        return True
    except Exception as e:
        print(f"✗ Error: {e}")
        return False


async def test_cohere():
    """Test Cohere chat model."""
    print("\n" + "=" * 70)
    print("Testing Cohere")
    print("=" * 70)
    
    try:
        from agenticflow.models.cohere import CohereChat
        api_key = os.getenv("COHERE_API_KEY")
        if not api_key:
            print("✗ COHERE_API_KEY not set, skipping")
            return False
        model = os.getenv("COHERE_CHAT_MODEL", "command-r-plus")
        print(f"Model: {model}")
        llm = CohereChat(model=model, api_key=api_key)
        response = await llm.ainvoke([{ "role": "user", "content": "Say 'Cohere works!' and nothing else." }])
        print(f"✓ Response: {response.content}")
        return True
    except Exception as e:
        print(f"✗ Error: {e}")
        return False


async def test_cloudflare():
    """Test Cloudflare Workers AI chat model."""
    print("\n" + "=" * 70)
    print("Testing Cloudflare")
    print("=" * 70)
    
    try:
        from agenticflow.models.cloudflare import CloudflareChat
        token = os.getenv("CLOUDFLARE_API_TOKEN")
        account_id = os.getenv("CLOUDFLARE_ACCOUNT_ID")
        if not token or not account_id:
            print("✗ CLOUDFLARE_API_TOKEN or CLOUDFLARE_ACCOUNT_ID not set, skipping")
            return False
        model = os.getenv("CLOUDFLARE_CHAT_MODEL", "@cf/meta/llama-3.1-8b-instruct")
        print(f"Model: {model}")
        llm = CloudflareChat(model=model, api_key=token, account_id=account_id)
        response = await llm.ainvoke([{ "role": "user", "content": "Say 'Cloudflare works!' and nothing else." }])
        print(f"✓ Response: {response.content}")
        return True
    except Exception as e:
        print(f"✗ Error: {e}")
        return False


async def test_embeddings(timings: dict[str, float]):
    """Test embedding models."""
    print("\n" + "=" * 70)
    print("Testing Embeddings")
    print("=" * 70)
    
    results = {}
    
    # OpenAI embeddings
    start = time.perf_counter()
    try:
        from agenticflow.models.openai import OpenAIEmbedding
        
        model = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
        print(f"\nOpenAI Embedding: {model}")
        
        embedder = OpenAIEmbedding(model=model)
        vectors = await embedder.aembed(["test"])
        
        print(f"✓ Generated {len(vectors[0])} dimensional embedding")
        results["openai"] = True
    except Exception as e:
        print(f"✗ OpenAI Embedding Error: {e}")
        results["openai"] = False
    finally:
        timings["embed:openai"] = time.perf_counter() - start
    
    # Gemini embeddings
    start = time.perf_counter()
    try:
        from agenticflow.models.gemini import GeminiEmbedding
        
        model = os.getenv("GEMINI_EMBEDDING_MODEL", "text-embedding-004")
        print(f"\nGemini Embedding: {model}")
        
        embedder = GeminiEmbedding(model=model)
        vectors = await embedder.aembed(["test"])
        
        print(f"✓ Generated {len(vectors[0])} dimensional embedding")
        results["gemini"] = True
    except Exception as e:
        print(f"✗ Gemini Embedding Error: {e}")
        results["gemini"] = False
    finally:
        timings["embed:gemini"] = time.perf_counter() - start
    
    # Mistral embeddings
    start = time.perf_counter()
    try:
        from agenticflow.models.mistral import MistralEmbedding
        
        model = os.getenv("MISTRAL_EMBEDDING_MODEL", "mistral-embed")
        print(f"\nMistral Embedding: {model}")
        
        embedder = MistralEmbedding(model=model)
        vectors = await embedder.aembed(["test"])
        
        print(f"✓ Generated {len(vectors[0])} dimensional embedding")
        results["mistral"] = True
    except Exception as e:
        print(f"✗ Mistral Embedding Error: {e}")
        results["mistral"] = False
    finally:
        timings["embed:mistral"] = time.perf_counter() - start

    # Cloudflare embeddings
    start = time.perf_counter()
    try:
        from agenticflow.models.cloudflare import CloudflareEmbedding
        token = os.getenv("CLOUDFLARE_API_TOKEN")
        account_id = os.getenv("CLOUDFLARE_ACCOUNT_ID")
        if not token or not account_id:
            print("✗ Cloudflare Embedding Error: CLOUDFLARE_API_TOKEN or CLOUDFLARE_ACCOUNT_ID not set")
            results["cloudflare"] = False
        else:
            model = os.getenv("CLOUDFLARE_EMBEDDING_MODEL", "@cf/baai/bge-base-en-v1.5")
            print(f"\nCloudflare Embedding: {model}")
            embedder = CloudflareEmbedding(model=model, api_key=token, account_id=account_id)
            vectors = await embedder.aembed(["test"])
            print(f"✓ Generated {len(vectors[0])} dimensional embedding")
            results["cloudflare"] = True
    except Exception as e:
        print(f"✗ Cloudflare Embedding Error: {e}")
        results["cloudflare"] = False
    finally:
        timings["embed:cloudflare"] = time.perf_counter() - start

    # Cohere embeddings
    start = time.perf_counter()
    try:
        from agenticflow.models.cohere import CohereEmbedding
        api_key = os.getenv("COHERE_API_KEY")
        if not api_key:
            print("✗ Cohere Embedding Error: COHERE_API_KEY not set")
            results["cohere"] = False
        else:
            model = os.getenv("COHERE_EMBEDDING_MODEL", "embed-english-v3.0")
            print(f"\nCohere Embedding: {model}")
            embedder = CohereEmbedding(model=model, api_key=api_key)
            vectors = await embedder.aembed(["test"])
            print(f"✓ Generated {len(vectors[0])} dimensional embedding")
            results["cohere"] = True
    except Exception as e:
        print(f"✗ Cohere Embedding Error: {e}")
        results["cohere"] = False
    finally:
        timings["embed:cohere"] = time.perf_counter() - start

    # Ollama embeddings
    start = time.perf_counter()
    try:
        from agenticflow.models.ollama import OllamaEmbedding
        host = os.getenv("OLLAMA_HOST", "http://localhost:11434")
        model = os.getenv("OLLAMA_EMBEDDING_MODEL", "nomic-embed-text")
        print(f"\nOllama Embedding: {model} @ {host}")
        embedder = OllamaEmbedding(model=model, host=host)
        vectors = await embedder.aembed(["test"])
        print(f"✓ Generated {len(vectors[0])} dimensional embedding")
        results["ollama"] = True
    except Exception as e:
        msg = str(e)
        if "not found" in msg.lower() or "connection refused" in msg.lower():
            print(f"⚠️  Ollama embedding skipped: {e} (pull the model or set OLLAMA_EMBEDDING_MODEL)")
            results["ollama"] = True
        else:
            print(f"✗ Ollama Embedding Error: {e}")
            results["ollama"] = False
    finally:
        timings["embed:ollama"] = time.perf_counter() - start
    
    return results


async def main():
    """Run all model tests."""
    print("=" * 70)
    print("AgenticFlow Model Tests")
    print("=" * 70)
    print(f"\nLoaded env from: {env_path}")
    
    results = {}
    timings: dict[str, float] = {}

    async def timed(name: str, fn):
        start = time.perf_counter()
        result = await fn()
        timings[name] = time.perf_counter() - start
        return result
    
    # Test chat models
    results["openai"] = await timed("openai", test_openai)
    results["gemini"] = await timed("gemini", test_gemini)
    results["mistral"] = await timed("mistral", test_mistral)
    results["groq"] = await timed("groq", test_groq)
    results["cohere"] = await timed("cohere", test_cohere)
    results["cloudflare"] = await timed("cloudflare", test_cloudflare)
    results["ollama"] = await timed("ollama", test_ollama)
    results["github"] = await timed("github", test_github)
    
    # Skip anthropic if no key
    if os.getenv("ANTHROPIC_API_KEY"):
        results["anthropic"] = await timed("anthropic", test_anthropic)
    else:
        print("\n" + "=" * 70)
        print("Skipping Anthropic (ANTHROPIC_API_KEY not set)")
        print("=" * 70)
    
    # Test embeddings
    embedding_results = await test_embeddings(timings)
    results.update({f"embed:{k}": v for k, v in embedding_results.items()})
    results["embeddings"] = all(embedding_results.values()) if embedding_results else False
    
    # Summary
    print("\n" + "=" * 70)
    print("Test Summary")
    print("=" * 70)
    
    for provider, success in results.items():
        status = "✓ PASS" if success else "✗ FAIL"
        print(f"{provider:20} {status}")
    
    total = len(results)
    passed = sum(results.values())
    
    print(f"\nTotal: {passed}/{total} passed")
    
    if passed == total:
        print("\n🎉 All tests passed!")
    else:
        print(f"\n⚠️  {total - passed} test(s) failed")
    
    print("\nBenchmark (seconds)")
    for name, duration in sorted(timings.items(), key=lambda item: item[1]):
        status = "PASS" if results.get(name, False) else "FAIL"
        print(f"{name:20} {duration:6.2f}s ({status})")

    fastest = [(n, d) for n, d in timings.items() if results.get(n)]
    if fastest:
        best_name, best_time = min(fastest, key=lambda item: item[1])
        print(f"\nFastest (passed): {best_name} at {best_time:.2f}s")

    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(main())
