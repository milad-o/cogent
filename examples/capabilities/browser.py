"""
Example: Browser Capability

Agent scrapes and analyzes web content using headless browser.

Usage:
    uv run python examples/capabilities/browser.py

Requires: uv add playwright && playwright install chromium
"""

import asyncio

from agenticflow import Agent, Flow
from agenticflow.capabilities import Browser


async def main() -> None:
    browser = Browser(
        headless=True,
        blocked_domains=["facebook.com", "twitter.com"],
    )

    if not browser._has_playwright:
        print("Install playwright to run this example: uv add playwright && playwright install chromium")
        return

    model = "gpt4"
    researcher = Agent(
        name="WebResearcher",
        model="gpt4",
        instructions="You research topics by browsing the web. Extract relevant information from pages.",
        capabilities=[browser],
    )

    flow = Flow(
        name="web_research",
        agents=[researcher],
        verbosity=settings.verbose_level,
    )

    result = await flow.run(
        "Go to https://httpbin.org/html and extract the main content. "
        "Summarize what you find."
    )
    print(f"\n{result.output}")


if __name__ == "__main__":
    asyncio.run(main())
