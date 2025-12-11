"""
Structured output smoke test for AgenticFlow providers.

- Loads env from examples/.env
- Invokes each provider's chat model with a realistic prompt
- Expects JSON matching TripPlan schema; validates with Pydantic
- Reports per-provider pass/fail and timings
"""

from __future__ import annotations

import asyncio
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from pydantic import BaseModel, Field, ValidationError

# Load env from examples/.env only
ENV_PATH = Path(__file__).parent.parent / "examples" / ".env"
if ENV_PATH.exists():
    from dotenv import load_dotenv

    load_dotenv(ENV_PATH, override=False)


class TripPlan(BaseModel):
    destination: str
    days: int = Field(gt=0, lt=15)
    budget_usd: float = Field(gt=50, lt=50000)
    must_do: list[str] = Field(min_length=1, max_length=8)
    notes: str | None = None


@dataclass
class TestResult:
    name: str
    ok: bool
    duration: float
    error: str | None = None


def _build_messages(retry_hint: str | None = None) -> list[dict[str, str]]:
    base = [
        {
            "role": "system",
            "content": (
                "You are a concise travel planner. Respond with ONLY raw JSON (no markdown) that"
                " exactly matches this schema and keys: {\"destination\": str, \"days\": int,"
                " \"budget_usd\": float, \"must_do\": [str], \"notes\": str}."
                " Do not include explanations or extra keys."
            ),
        },
        {
            "role": "user",
            "content": (
                "Plan a 3-day budget-conscious trip to Lisbon for two food-loving friends in May."
                " Give a short list of must-do items and a one-line note."
            ),
        },
    ]
    if retry_hint:
        base.append({
            "role": "system",
            "content": (
                "Your previous response was invalid. Output ONLY valid JSON with the required keys"
                " destination, days, budget_usd, must_do, notes. No prose, no markdown."
                f" Validation error: {retry_hint}"
            ),
        })
    return base


async def run_trip_plan(llm, *, max_attempts: int = 2) -> TripPlan:
    last_error: str | None = None
    for attempt in range(1, max_attempts + 1):
        messages = _build_messages(last_error)
        response = await llm.ainvoke(messages)
        text = response.content if hasattr(response, "content") else str(response)
        try:
            data = json.loads(text)
            return TripPlan.model_validate(data)
        except (json.JSONDecodeError, ValidationError) as exc:
            last_error = str(exc)
            if attempt == max_attempts:
                raise
            # Retry with stricter hint
    # Should not reach
    raise RuntimeError("Failed to produce TripPlan after retries")


def provider_configs() -> list[tuple[str, Callable[[], Any]]]:
    """List providers to test with constructors."""
    from agenticflow.models import create_chat

    configs: list[tuple[str, Callable[[], Any]]] = []

    configs.append(("openai", lambda: create_chat("openai", model=os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini"))))
    configs.append(("gemini", lambda: create_chat("gemini", model=os.getenv("GEMINI_CHAT_MODEL", "gemini-2.5-flash"))))
    configs.append(("mistral", lambda: create_chat("mistral", model=os.getenv("MISTRAL_CHAT_MODEL", "mistral-small-latest"))))
    configs.append(("groq", lambda: create_chat("groq", model=os.getenv("GROQ_CHAT_MODEL", "llama-3.1-8b-instant"))))
    configs.append(("cohere", lambda: create_chat("cohere", model=os.getenv("COHERE_CHAT_MODEL", "command-r-plus"))))
    configs.append(("cloudflare", lambda: create_chat("cloudflare", model=os.getenv("CLOUDFLARE_CHAT_MODEL", "@cf/meta/llama-3.1-8b-instruct"))))
    configs.append(("ollama", lambda: create_chat("ollama", model=os.getenv("OLLAMA_CHAT_MODEL", "qwen2.5:7b"), host=os.getenv("OLLAMA_HOST", "http://localhost:11434"))))
    configs.append(("github", lambda: create_chat("github", model=os.getenv("GITHUB_CHAT_MODEL", "gpt-4o-mini"), token=os.getenv("GITHUB_TOKEN"))))

    if os.getenv("ANTHROPIC_API_KEY"):
        configs.append(("anthropic", lambda: create_chat("anthropic", model=os.getenv("ANTHROPIC_CHAT_MODEL", "claude-sonnet-4-20250514"))))

    return configs


async def run_provider(name: str, ctor: Callable[[], Any], timeout_s: int = 45) -> TestResult:
    start = time.perf_counter()
    try:
        llm = ctor()
        plan = await asyncio.wait_for(run_trip_plan(llm), timeout=timeout_s)
        duration = time.perf_counter() - start
        print(f"✓ {name}: {plan.destination} for {plan.days} days | {duration:.2f}s")
        return TestResult(name=name, ok=True, duration=duration)
    except asyncio.TimeoutError:
        duration = time.perf_counter() - start
        print(f"⚠️  {name}: timed out after {timeout_s}s")
        return TestResult(name=name, ok=False, duration=duration, error="timeout")
    except ValidationError as ve:
        duration = time.perf_counter() - start
        print(f"✗ {name}: schema validation failed ({ve})")
        return TestResult(name=name, ok=False, duration=duration, error="validation")
    except Exception as exc:  # noqa: BLE001
        duration = time.perf_counter() - start
        print(f"✗ {name}: {exc}")
        return TestResult(name=name, ok=False, duration=duration, error=str(exc))


async def main():
    print("=" * 70)
    print("Structured Agent Test")
    print("=" * 70)
    print(f"Loaded env from: {ENV_PATH}")

    results: list[TestResult] = []
    for name, ctor in provider_configs():
        print("\n" + "-" * 70)
        print(f"Testing {name}")
        res = await run_provider(name, ctor)
        results.append(res)

    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)
    for r in results:
        status = "PASS" if r.ok else "FAIL"
        print(f"{r.name:12} {status:4} {r.duration:6.2f}s {('(' + r.error + ')') if r.error else ''}")

    passed = sum(1 for r in results if r.ok)
    total = len(results)
    print(f"\nTotal: {passed}/{total} passed")

    fastest = [r for r in results if r.ok]
    if fastest:
        best = min(fastest, key=lambda r: r.duration)
        print(f"Fastest: {best.name} at {best.duration:.2f}s")

    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(main())
