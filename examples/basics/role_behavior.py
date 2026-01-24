"""
Role Behavior - How Roles Actually Affect LLM Behavior

This demo shows the REAL impact of roles on agent behavior:
1. Capabilities (can_finish, can_delegate, can_use_tools)
2. System prompts that guide LLM behavior
3. Actual LLM responses showing the difference

Usage:
    uv run python examples/basics/role_behavior.py
"""

import asyncio

from agenticflow import Agent
from agenticflow.agent.roles import get_role_prompt
from agenticflow.core import AgentRole
from agenticflow.tools.base import BaseTool


# Simple tool for testing
async def search_web(query: str) -> str:
    """Search the web for information."""
    return f"Search results for '{query}': Python is a high-level programming language..."


search_tool = BaseTool(
    name="search_web",
    description="Search the web for information",
    func=search_web,
    args_schema={"query": {"type": "string"}},
)


async def show_role_prompts():
    """Show the actual system prompts for each role."""
    print("\n=== Role System Prompts (What the LLM Sees) ===\n")

    for role in AgentRole:
        prompt = get_role_prompt(role, has_tools=True)
        print(f"{role.value.upper()}:")
        print(f"{prompt[:200]}...")
        print("\nKey instructions:")
        if "FINAL ANSWER" in prompt:
            print("  ✅ Can finish tasks (FINAL ANSWER)")
        else:
            print("  ❌ Cannot finish tasks")
        if "DELEGATE" in prompt or "delegate" in prompt:
            print("  ✅ Can delegate to others")
        else:
            print("  ❌ Cannot delegate")
        if "tool" in prompt.lower():
            print("  ✅ Can use tools")
        else:
            print("  ❌ Cannot use tools")
        print("\n" + "-" * 70 + "\n")


async def demo_worker_vs_autonomous():
    """Show difference between WORKER and AUTONOMOUS with real LLM."""
    print("\n=== WORKER vs AUTONOMOUS (Real LLM Behavior) ===\n")

    task = "What is Python programming language?"

    # WORKER: Can use tools, but CANNOT finish
    print("1. WORKER Agent (can't finish):")
    worker = Agent(
        name="Worker",
        model="gpt4",
        role=AgentRole.WORKER,
        tools=[search_tool],
    )
    print(f"   Capabilities: finish={worker.can_finish}, tools={worker.can_use_tools}")
    print(f"   Task: {task}")

    result = await worker.run(task)
    print(f"   Response: {result.content[:200]}...")
    print(f"   Has 'FINAL ANSWER': {'FINAL ANSWER' in result.content}")

    # AUTONOMOUS: Can use tools AND finish
    print("\n2. AUTONOMOUS Agent (can finish):")
    autonomous = Agent(
        name="Autonomous",
        model="gpt4",
        role=AgentRole.AUTONOMOUS,
        tools=[search_tool],
    )
    print(f"   Capabilities: finish={autonomous.can_finish}, tools={autonomous.can_use_tools}")
    print(f"   Task: {task}")

    result = await autonomous.run(task)
    print(f"   Response: {result.content[:200]}...")
    print(f"   Has 'FINAL ANSWER': {'FINAL ANSWER' in result.content}")

    print("\n💡 Key Difference:")
    print("   - WORKER provides info but doesn't conclude")
    print("   - AUTONOMOUS provides FINAL ANSWER and concludes the task")


async def demo_supervisor_delegation():
    """Show supervisor trying to delegate (even without workers)."""
    print("\n\n=== SUPERVISOR Behavior (Delegation Tendency) ===\n")


    supervisor = Agent(
        name="Manager",
        model="gpt4",
        role=AgentRole.SUPERVISOR,
    )
    print(f"Capabilities: finish={supervisor.can_finish}, delegate={supervisor.can_delegate}, tools={supervisor.can_use_tools}")
    print("Task: Research Python and write a summary")

    result = await supervisor.run("Research Python programming and write a one-paragraph summary")
    print(f"\nResponse: {result.content[:300]}...")

    # Check if supervisor tried to delegate
    if "DELEGATE" in result.content.upper():
        print("\n✅ Supervisor tried to DELEGATE the work!")
    elif "FINAL ANSWER" in result.content:
        print("\n⚠️  Supervisor provided FINAL ANSWER (no workers to delegate to)")
    else:
        print("\n📝 Supervisor's response (no delegation, no conclusion)")


async def demo_reviewer_behavior():
    """Show reviewer evaluating work."""
    print("\n\n=== REVIEWER Behavior (Evaluation Mode) ===\n")


    reviewer = Agent(
        name="QA",
        model="gpt4",
        role=AgentRole.REVIEWER,
    )
    print(f"Capabilities: finish={reviewer.can_finish}, delegate={reviewer.can_delegate}, tools={reviewer.can_use_tools}")

    work_to_review = """
    Python is a programming language. It's used for many things.
    People like it because it's easy.
    """

    print(f"\nWork to review: {work_to_review.strip()}")

    result = await reviewer.run(f"Review this work:\n{work_to_review}\n\nIs it complete and accurate?")
    print(f"\nReviewer response: {result.content}")

    if "REVISION NEEDED" in result.content or "revision" in result.content.lower():
        print("\n✅ Reviewer asked for revisions (acting in reviewer mode)")
    elif "FINAL ANSWER" in result.content:
        print("\n✅ Reviewer approved (provided FINAL ANSWER)")


async def demo_tool_access_control():
    """Show that roles control tool access."""
    print("\n\n=== Tool Access Control by Role ===\n")

    task = "Search for information about Python"

    # WORKER: Can use tools
    print("1. WORKER (can_use_tools=True):")
    worker = Agent(name="Worker", model="gpt4", role=AgentRole.WORKER, tools=[search_tool])
    print(f"   Tools available: {[t.name for t in worker.all_tools]}")
    result = await worker.run(task)
    print(f"   Used tools: {[tc.tool_name for tc in result.tool_calls] if result.tool_calls else 'None'}")

    # SUPERVISOR: Cannot use tools
    print("\n2. SUPERVISOR (can_use_tools=False):")
    supervisor = Agent(name="Supervisor", model="gpt4", role=AgentRole.SUPERVISOR, tools=[search_tool])
    print(f"   Tools available: {[t.name for t in supervisor.all_tools]}")
    print("   But system prompt says: 'delegates tool work to workers'")
    result = await supervisor.run(task)
    print(f"   Used tools: {[tc.tool_name for tc in result.tool_calls] if result.tool_calls else 'None'}")

    # REVIEWER: Cannot use tools
    print("\n3. REVIEWER (can_use_tools=False):")
    reviewer = Agent(name="Reviewer", model="gpt4", role=AgentRole.REVIEWER, tools=[search_tool])
    print(f"   Tools available: {[t.name for t in reviewer.all_tools]}")
    print("   But system prompt says: 'focuses on judgment, not execution'")
    result = await reviewer.run(task)
    print(f"   Used tools: {[tc.tool_name for tc in result.tool_calls] if result.tool_calls else 'None'}")


async def main():
    print("\n" + "=" * 70)
    print("  Role Behavior - How Roles Affect LLM Responses")
    print("=" * 70)

    await show_role_prompts()
    await demo_worker_vs_autonomous()
    await demo_supervisor_delegation()
    await demo_reviewer_behavior()
    await demo_tool_access_control()

    print("\n" + "=" * 70)
    print("✅ Demo complete!")
    print("\n💡 Key Takeaway:")
    print("   Roles control CAPABILITIES + inject SYSTEM PROMPTS that guide LLM behavior")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    asyncio.run(main())
