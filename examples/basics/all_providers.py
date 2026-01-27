"""Minimal provider smoke test (uses .env)."""

import asyncio

from cogent import Agent


async def main():
    question = "Name three layers of Earth's atmosphere from lowest to highest."
    providers = [
        ("OpenAI", "gpt-4.1"),
        ("Gemini", "gemini-2.5-flash-lite"),
        ("Groq", "llama-3.1-8b-instant"),
        ("Mistral", "mistral-small-latest"),
        ("Cohere", "command-a-03-2025"),
        ("Cloudflare", "cloudflare:@cf/meta/llama-3.1-8b-instruct"),
        ("GitHub", "github:gpt-4.1"),
    ]

    ok = 0
    for name, model in providers:
        try:
            result = await Agent(
                name=f"{name}Agent",
                model=model,
                instructions="Be concise. Answer the question directly. No extra text.",
            ).run(question)
            if not result.success:
                error_msg = result.error.message if result.error else "Unknown error"
                print(f"❌ {name}: {error_msg}")
                continue
            print(f"✅ {name}: {result.content}")
            ok += 1
        except Exception as exc:
            print(f"❌ {name}: {exc}")

    print(f"{ok}/{len(providers)} OK")


if __name__ == "__main__":
    asyncio.run(main())
