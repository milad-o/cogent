"""Test bare field responses"""
import asyncio
from typing import Literal
from cogent import Agent

async def test_bare_literal():
    """Try using a bare Literal type"""
    try:
        agent = Agent(
            name="Reviewer",
            model="gpt-4o-mini",
            output=Literal["PROCEED", "REVISE"],
        )
        result = await agent.run("Should we proceed with this implementation?")
        print(f"Bare Literal - Valid: {result.content.valid}")
        print(f"Bare Literal - Error: {result.content.error}")
        print(f"Bare Literal - Type: {type(result.content.data)}")
        print(f"Bare Literal - Value: {result.content.data}")
        print(f"Bare Literal - Raw: {result.content.raw}")
    except Exception as e:
        print(f"Bare Literal failed: {e}")
        import traceback
        traceback.print_exc()

async def test_bare_str():
    """Try using a bare str type"""
    try:
        agent = Agent(
            name="Summarizer",
            model="gpt-4o-mini",
            output=str,
        )
        result = await agent.run("Summarize in one word: Python is a programming language")
        print(f"\nBare str - Valid: {result.content.valid}")
        print(f"Bare str - Error: {result.content.error}")
        print(f"Bare str - Type: {type(result.content.data)}")
        print(f"Bare str - Value: {result.content.data}")
        print(f"Bare str - Raw: {result.content.raw}")
    except Exception as e:
        print(f"Bare str failed: {e}")
        import traceback
        traceback.print_exc()

async def test_bare_int():
    """Try using a bare int type"""
    try:
        agent = Agent(
            name="Counter",
            model="gpt-4o-mini",
            output=int,
        )
        result = await agent.run("How many words: Hello world test")
        print(f"\nBare int - Valid: {result.content.valid}")
        print(f"Bare int - Error: {result.content.error}")
        print(f"Bare int - Type: {type(result.content.data)}")
        print(f"Bare int - Value: {result.content.data}")
        print(f"Bare int - Raw: {result.content.raw}")
    except Exception as e:
        print(f"Bare int failed: {e}")
        import traceback
        traceback.print_exc()

async def main():
    await test_bare_literal()
    await test_bare_str()
    await test_bare_int()

if __name__ == "__main__":
    asyncio.run(main())
