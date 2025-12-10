"""
Example: Browser Capability

Agent scrapes and analyzes web content using headless browser.

Usage:
    uv run python examples/capabilities/browser.py

Requires: uv add playwright && playwright install chromium
"""

import asyncio
import sys
from pathlib import Path

# Add examples directory to path for config import
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import get_model, settings

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

    model = get_model()
    researcher = Agent(
        name="WebResearcher",
        model=model,
        instructions="You research topics by browsing the web. Extract relevant information from pages.",
        capabilities=[browser],
    )

    flow = Flow(
        name="web_research",
        agents=[researcher],
        verbose=settings.verbose_level,
    )

    result = await flow.run(
        "Go to https://httpbin.org/html and extract the main content. "
        "Summarize what you find."
    )
    print(f"\n{result.output}")


if __name__ == "__main__":
    asyncio.run(main())
