"""
Human-in-the-Loop (HITL) for Tool Approval

Demonstrates human oversight for agent tool calls:
- interrupt_on rules for specific tools
- Handle InterruptedException
- Resume with HumanDecision (approve, reject, edit, abort, guide)

Usage: uv run python examples/advanced/human_in_the_loop.py
"""

import asyncio

from agenticflow import (
    AbortedException,
    Agent,
    HumanDecision,
    InterruptedException,
)
from agenticflow.agent.hitl import GuidanceResult, HumanResponse
from agenticflow.tools.base import tool


@tool
def read_file(path: str) -> str:
    """Read content from a file."""
    return f"[Mock] Content of {path}: Hello, World!"


@tool
def write_file(path: str, content: str) -> str:
    """Write content to a file."""
    return f"[Mock] Wrote to {path}: {content[:50]}..."


@tool
def delete_file(path: str) -> str:
    """Delete a file. DANGEROUS - requires approval."""
    return f"[Mock] Deleted {path}"


@tool
def send_email(to: str, subject: str, body: str) -> str:
    """Send an email."""
    return f"[Mock] Email sent to {to}: {subject}"


async def demo_basic_approval():
    """Basic tool approval workflow."""
    print("\n" + "=" * 60)
    print("1. Basic Tool Approval")
    print("=" * 60)

    agent = Agent(
        name="FileManager",
        model="gpt4",
        tools=[read_file, write_file, delete_file],
        instructions="Manage files safely.",
        interrupt_on={
            "delete_file": True,
            "write_file": True,
            "read_file": False,
        },
    )

    # Safe operation - auto-approved
    print("\nReading a file (auto-approved)...")
    result = await agent.act("read_file", {"path": "/data/readme.txt"})
    print(f"  Result: {result}")

    # Dangerous operation - requires approval
    print("\nDeleting a file (requires approval)...")
    try:
        await agent.act("delete_file", {"path": "/data/important.txt"})
    except InterruptedException as e:
        pending = e.state.pending_actions[0]
        print(f"  ⏸️  Interrupted! {pending.describe()}")

        # Human approves
        print("  [Human APPROVES]")
        decision = HumanDecision.approve(pending.action_id)
        result = await agent.resume_action(decision)
        print(f"  Result: {result}")


async def demo_reject_edit():
    """Reject and edit actions."""
    print("\n" + "=" * 60)
    print("2. Reject & Edit Actions")
    print("=" * 60)

    agent = Agent(
        name="EmailBot",
        model="gpt4",
        tools=[send_email],
        instructions="Send emails as requested.",
        interrupt_on={"*": True},
    )

    # Reject example
    print("\nSending email (will be rejected)...")
    try:
        await agent.act("send_email", {
            "to": "external@company.com",
            "subject": "Confidential Data",
            "body": "Here's the sensitive info..."
        })
    except InterruptedException as e:
        pending = e.state.pending_actions[0]
        print(f"  ⏸️  Pending: {pending.describe()}")
        print("  [Human REJECTS - too risky]")

        decision = HumanDecision.reject(pending.action_id, feedback="Don't send confidential data externally")
        result = await agent.resume_action(decision)
        print(f"  Result: {result}")

    # Edit example
    print("\nSending email (will be edited)...")
    try:
        await agent.act("send_email", {
            "to": "team@company.com",
            "subject": "Update",
            "body": "Quick update"
        })
    except InterruptedException as e:
        pending = e.state.pending_actions[0]
        print(f"  ⏸️  Pending: {pending.describe()}")
        print("  [Human EDITS - adds disclaimer]")

        new_args = dict(pending.args)
        new_args["body"] = f"{new_args['body']}\n\n[Disclaimer: This is an automated message]"

        decision = HumanDecision.edit(pending.action_id, new_args)
        result = await agent.resume_action(decision)
        print(f"  Result: {result}")


async def demo_abort():
    """Abort entire workflow."""
    print("\n" + "=" * 60)
    print("3. Abort Workflow")
    print("=" * 60)

    agent = Agent(
        name="DangerousAgent",
        model="gpt4",
        tools=[delete_file],
        instructions="Delete files as requested.",
        interrupt_on={"*": True},
    )

    print("\nAttempting dangerous operation...")
    try:
        await agent.act("delete_file", {"path": "/root/.bashrc"})
    except InterruptedException as e:
        pending = e.state.pending_actions[0]
        print(f"  ⏸️  Pending: {pending.describe()}")
        print("  [Human ABORTS - too risky!]")

        decision = HumanDecision.abort(
            pending.action_id,
            feedback="Operation too dangerous, stopping workflow"
        )

        try:
            await agent.resume_action(decision)
        except AbortedException as abort:
            print(f"  ❌ Workflow aborted: {abort.decision.feedback}")


async def demo_guidance():
    """Provide guidance instead of yes/no."""
    print("\n" + "=" * 60)
    print("4. Guidance & Response")
    print("=" * 60)

    agent = Agent(
        name="FileAssistant",
        model="gpt4",
        tools=[delete_file, write_file, read_file],
        instructions="Help manage files safely.",
        interrupt_on={"delete_file": True, "write_file": True},
    )

    # Guidance example
    print("\nAgent wants to delete - human provides guidance...")
    try:
        await agent.act("delete_file", {"path": "/important/data.csv"})
    except InterruptedException as e:
        pending = e.state.pending_actions[0]
        print(f"  ⏸️  Pending: {pending.describe()}")

        guidance = HumanDecision.guide(
            pending.action_id,
            guidance="Don't delete directly. First backup to /backup/, verify, then delete.",
            feedback="Critical data - be careful"
        )

        result = await agent.resume_action(guidance)

        if isinstance(result, GuidanceResult):
            print("  💡 Guidance received!")
            print(f"  📝 Instructions: {result.guidance}")
            print(f"  🔄 Should retry: {result.should_retry}")

    # Response example
    print("\nAgent asks a question - human responds...")
    try:
        await agent.act("write_file", {"path": "?", "content": "report data"})
    except InterruptedException as e:
        pending = e.state.pending_actions[0]
        print("  ⏸️  Agent asks: What path should I use?")

        response = HumanDecision.respond(
            pending.action_id,
            response={
                "path": "/reports/Q4-2024-Sales.csv",
                "format": "csv"
            },
            feedback="Use this naming convention"
        )

        result = await agent.resume_action(response)

        if isinstance(result, HumanResponse):
            print("  💬 Response received!")
            print(f"  📦 Data: {result.response}")


async def demo_interactive():
    """Interactive approval loop (simulated)."""
    print("\n" + "=" * 60)
    print("5. Interactive Approval Loop")
    print("=" * 60)

    agent = Agent(
        name="SystemAdmin",
        model="gpt4",
        tools=[delete_file, write_file, read_file],
        instructions="Perform system administration.",
        interrupt_on={"delete_file": True, "write_file": True},
    )

    # Simulated user decisions
    user_decisions = ["y", "n", "e"]  # approve, reject, edit
    decision_idx = 0

    async def run_with_approval(tool_name: str, args: dict) -> str | None:
        nonlocal decision_idx

        try:
            return await agent.act(tool_name, args)
        except InterruptedException as e:
            pending = e.state.pending_actions[0]
            print(f"\n  ⏸️  Approval required: {pending.describe()}")

            user_input = user_decisions[decision_idx] if decision_idx < len(user_decisions) else "y"
            decision_idx += 1
            print(f"  [Simulated input: '{user_input}']")

            if user_input == "y":
                decision = HumanDecision.approve(pending.action_id)
            elif user_input == "n":
                decision = HumanDecision.reject(pending.action_id)
            else:  # "e" for edit
                new_args = dict(pending.args)
                if "path" in new_args:
                    new_args["path"] = new_args["path"] + ".backup"
                decision = HumanDecision.edit(pending.action_id, new_args)

            return await agent.resume_action(decision)

    operations = [
        ("delete_file", {"path": "/tmp/old_logs.txt"}),
        ("write_file", {"path": "/etc/config.yml", "content": "new_setting: true"}),
        ("delete_file", {"path": "/var/data/important.db"}),
    ]

    print("\nRunning batch operations with approval...")
    for tool_name, args in operations:
        print(f"\n  → {tool_name}({args})")
        result = await run_with_approval(tool_name, args)
        print(f"  ← {result}")


async def main():
    """Run all HITL demos."""
    print("\n" + "=" * 60)
    print("HUMAN-IN-THE-LOOP (HITL) DEMONSTRATION")
    print("=" * 60)

    await demo_basic_approval()
    await demo_reject_edit()
    await demo_abort()
    await demo_guidance()
    await demo_interactive()

    print("\n" + "=" * 60)
    print("✓ All demos completed")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
