"""Single Field Responses - Minimal Demo

Shows two patterns for minimal responses:
1. Bare types (str, int, Literal) - simplest possible
2. Single-field models - when you need field names
"""
import asyncio
from typing import Literal
from pydantic import BaseModel, Field
from cogent import Agent

# Pattern 1: Bare Types (Direct Values)
# ======================================

async def bare_types_demo():
    """Bare types return the value directly"""
    print("=" * 60)
    print("Pattern 1: Bare Types (Direct Values)")
    print("=" * 60)
    
    # Bare Literal
    agent = Agent(name="Reviewer", model="gpt-4o-mini", output=Literal["PROCEED", "REVISE"])
    result = await agent.run("Auth system 80% done, missing password reset")
    print(f"Bare Literal: {result.content.data}")  # Direct value: "REVISE"
    
    # Bare str
    agent = Agent(name="Summarizer", model="gpt-4o-mini", output=str)
    result = await agent.run("Summarize in one word: Python is a programming language")
    print(f"Bare str: {result.content.data}")  # Direct value: "Python"
    
    # Bare int
    agent = Agent(name="Counter", model="gpt-4o-mini", output=int)
    result = await agent.run("How many words: Hello world test")
    print(f"Bare int: {result.content.data}")  # Direct value: 3


# Pattern 2: Single-Field Models (Named Fields)
# ==============================================

class Decision(BaseModel):
    value: Literal["PROCEED", "REVISE"]

class YesNo(BaseModel):
    answer: Literal["YES", "NO"]

class Priority(BaseModel):
    level: Literal["CRITICAL", "HIGH", "MEDIUM", "LOW"]

class Score(BaseModel):
    rating: int = Field(ge=1, le=10)

async def single_field_models_demo():
    """Single-field models when you need named fields"""
    print("\n" + "=" * 60)
    print("Pattern 2: Single-Field Models (Named Fields)")
    print("=" * 60)
    
    agent = Agent(name="Reviewer", model="gpt-4o-mini", output=Decision, instructions="Review content.")
    result = await agent.run("Auth system 80% done, missing password reset")
    print(f"Decision: {result.content.data.value}")

    agent = Agent(name="Validator", model="gpt-4o-mini", output=YesNo)
    result = await agent.run("Is test@example.com valid?")
    print(f"Answer: {result.content.data.answer}")

    agent = Agent(name="Prioritizer", model="gpt-4o-mini", output=Priority)
    result = await agent.run("DB errors affecting 5% users")
    print(f"Priority: {result.content.data.level}")

    agent = Agent(name="Scorer", model="gpt-4o-mini", output=Score)
    result = await agent.run("Rate: Clear explanation with examples")
    print(f"Score: {result.content.data.rating}/10")


async def main():
    await bare_types_demo()
    await single_field_models_demo()

if __name__ == "__main__":
    asyncio.run(main())
