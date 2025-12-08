"""
Example 18: Human-in-the-Loop (HITL) for Tool Approval

This example demonstrates how to implement human oversight for agent tool calls:
- Configure interrupt_on rules for specific tools
- Handle InterruptedException when agent needs approval
- Resume execution with HumanDecision (approve, reject, edit, abort)

Human-in-the-Loop is essential for:
- Dangerous operations (delete, modify, send)
- Sensitive data access
- External API calls with side effects
- Any action requiring human judgment
"""

import asyncio

from config import get_model

from agenticflow.tools.base import tool

# Import HITL components
from agenticflow import (
    Agent,
    # HITL types
    HumanDecision,
    InterruptedException,
    AbortedException,
    DecisionType,
)


# =============================================================================
# Define Tools (some dangerous, some safe)
# =============================================================================

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
    """Send an email. Requires approval for external recipients."""
    return f"[Mock] Email sent to {to}: {subject}"


@tool
def search_web(query: str) -> str:
    """Search the web for information."""
    return f"[Mock] Search results for '{query}': AI is transforming industries..."


# =============================================================================
# Demo 1: Basic Tool Approval
# =============================================================================

async def demo_basic_approval():
    """Demonstrate basic tool approval workflow."""
    print("\n" + "="*60)
    print("Demo 1: Basic Tool Approval")
    print("="*60)
    
    # Get LLM model
    model = get_model()
    
    # Create agent with interrupt_on rules
    agent = Agent(
        name="FileManager",
        model=model,
        tools=[read_file, write_file, delete_file],
        instructions="You manage files safely. When asked to perform file operations, use your tools.",
        # Configure which tools require approval
        interrupt_on={
            "delete_file": True,  # Always require approval for delete
            "write_file": True,   # Always require approval for write
            "read_file": False,   # Auto-approve reads
        },
    )
    
    # Safe operation - auto-approved
    print("\n1. Reading a file (auto-approved)...")
    result = await agent.act("read_file", {"path": "/data/readme.txt"})
    print(f"   Result: {result}")
    
    # Dangerous operation - requires approval
    print("\n2. Trying to delete a file (requires approval)...")
    try:
        await agent.act("delete_file", {"path": "/data/important.txt"})
    except InterruptedException as e:
        pending = e.state.pending_actions[0]
        print(f"   ⏸️  Interrupted! Pending: {pending.describe()}")
        print(f"   Agent: {pending.agent_name}")
        print(f"   Reason: {pending.reason.value}")
        
        # Simulate human approval
        print("\n   [Human decides to APPROVE]")
        decision = HumanDecision.approve(pending.action_id, feedback="Verified safe to delete")
        
        result = await agent.resume_action(decision)
        print(f"   Result after approval: {result}")
    
    # Dangerous operation - human rejects
    print("\n3. Trying to delete another file (human rejects)...")
    try:
        await agent.act("delete_file", {"path": "/system/critical.conf"})
    except InterruptedException as e:
        pending = e.state.pending_actions[0]
        print(f"   ⏸️  Interrupted! Pending: {pending.describe()}")
        
        # Human rejects
        print("\n   [Human decides to REJECT]")
        decision = HumanDecision.reject(pending.action_id, feedback="Too dangerous!")
        
        result = await agent.resume_action(decision)
        print(f"   Result after rejection: {result}")  # None


# =============================================================================
# Demo 2: Dynamic Approval Rules
# =============================================================================

async def demo_dynamic_rules():
    """Demonstrate dynamic approval rules based on arguments."""
    print("\n" + "="*60)
    print("Demo 2: Dynamic Approval Rules")
    print("="*60)
    
    # Dynamic rule: only require approval for external emails
    def require_external_email_approval(tool_name: str, args: dict) -> bool:
        """Require approval for emails to non-company addresses."""
        to = args.get("to", "")
        is_external = not to.endswith("@company.com")
        if is_external:
            print(f"   📧 External email detected: {to}")
        return is_external
    
    model = get_model()
    
    agent = Agent(
        name="EmailAssistant",
        model=model,
        tools=[send_email, search_web],
        instructions="You help send emails. Use your tools when asked.",
        interrupt_on={
            "send_email": require_external_email_approval,  # Dynamic rule
            "search_web": False,  # Always auto-approve
        },
    )
    
    # Internal email - auto-approved
    print("\n1. Sending internal email (auto-approved)...")
    result = await agent.act("send_email", {
        "to": "alice@company.com",
        "subject": "Meeting",
        "body": "See you at 3pm"
    })
    print(f"   Result: {result}")
    
    # External email - requires approval
    print("\n2. Sending external email (requires approval)...")
    try:
        await agent.act("send_email", {
            "to": "client@external.org",
            "subject": "Contract",
            "body": "Please review the attached contract"
        })
    except InterruptedException as e:
        pending = e.state.pending_actions[0]
        print(f"   ⏸️  Interrupted! {pending.describe()}")
        
        # Human edits the email
        print("\n   [Human decides to EDIT - changing subject]")
        modified_args = {
            "to": "client@external.org",
            "subject": "Contract - CONFIDENTIAL",  # Modified!
            "body": "Please review the attached contract"
        }
        decision = HumanDecision.edit(pending.action_id, modified_args)
        
        result = await agent.resume_action(decision)
        print(f"   Result after edit: {result}")


# =============================================================================
# Demo 3: Abort Workflow
# =============================================================================

async def demo_abort():
    """Demonstrate aborting a workflow."""
    print("\n" + "="*60)
    print("Demo 3: Abort Workflow")
    print("="*60)
    
    model = get_model()
    
    agent = Agent(
        name="DangerousAgent",
        model=model,
        tools=[delete_file],
        instructions="Delete files as requested. Use your tools.",
        interrupt_on={
            "*": True,  # Wildcard: require approval for ALL tools
        },
    )
    
    print("\n1. Attempting dangerous operation...")
    try:
        await agent.act("delete_file", {"path": "/root/.bashrc"})
    except InterruptedException as e:
        pending = e.state.pending_actions[0]
        print(f"   ⏸️  Interrupted! {pending.describe()}")
        
        # Human aborts
        print("\n   [Human decides to ABORT - too risky!]")
        decision = HumanDecision.abort(pending.action_id, feedback="Operation too dangerous, stopping workflow")
        
        try:
            await agent.resume_action(decision)
        except AbortedException as abort:
            print(f"   ❌ Workflow aborted: {abort.decision.feedback}")


# =============================================================================
# Demo 4: Interactive Approval Loop
# =============================================================================

async def demo_interactive():
    """Demonstrate an interactive approval loop (simulated)."""
    print("\n" + "="*60)
    print("Demo 4: Interactive Approval Loop")
    print("="*60)
    
    model = get_model()
    
    agent = Agent(
        name="SystemAdmin",
        model=model,
        tools=[delete_file, write_file, read_file],
        instructions="Perform system administration using your tools.",
        interrupt_on={
            "delete_file": True,
            "write_file": True,
        },
    )
    
    # Simulated user input queue
    user_decisions = ["y", "n", "e", "s"]  # approve, reject, edit, skip
    decision_idx = 0
    
    async def run_with_approval(tool_name: str, args: dict) -> str | None:
        """Run a tool with interactive approval."""
        nonlocal decision_idx
        
        try:
            return await agent.act(tool_name, args)
        except InterruptedException as e:
            pending = e.state.pending_actions[0]
            print(f"\n   ⏸️  Approval required: {pending.describe()}")
            
            # Simulate user input
            if decision_idx < len(user_decisions):
                user_input = user_decisions[decision_idx]
                decision_idx += 1
            else:
                user_input = "y"
            
            print(f"   [Simulated user input: '{user_input}']")
            
            if user_input == "y":
                decision = HumanDecision.approve(pending.action_id)
            elif user_input == "n":
                decision = HumanDecision.reject(pending.action_id)
            elif user_input == "e":
                # Edit: add a backup suffix to the path
                new_args = dict(pending.args)
                if "path" in new_args:
                    new_args["path"] = new_args["path"] + ".backup"
                decision = HumanDecision.edit(pending.action_id, new_args)
            else:  # "s" for skip
                decision = HumanDecision.skip(pending.action_id)
            
            return await agent.resume_action(decision)
    
    # Run operations
    operations = [
        ("delete_file", {"path": "/tmp/old_logs.txt"}),
        ("delete_file", {"path": "/var/data/important.db"}),
        ("write_file", {"path": "/etc/config.yml", "content": "new_setting: true"}),
        ("write_file", {"path": "/tmp/test.txt", "content": "test data"}),
    ]
    
    print("\n   Running batch operations with approval...")
    for tool_name, args in operations:
        print(f"\n   → {tool_name}({args})")
        result = await run_with_approval(tool_name, args)
        print(f"   ← Result: {result}")


async def demo_guidance():
    """
    Demo 5: Guidance and Response
    
    Shows how humans can provide guidance/instructions instead of
    just approving or rejecting actions. This is useful when you
    want the agent to reconsider its approach.
    """
    print("\n" + "="*60)
    print("Demo 5: Guidance and Response")
    print("="*60)
    
    from agenticflow.agent.hitl import GuidanceResult, HumanResponse
    
    model = get_model()
    
    agent = Agent(
        name="FileAssistant",
        model=model,
        tools=[delete_file, write_file, read_file],
        instructions="Help manage files safely using your tools.",
        interrupt_on={
            "delete_file": True,
            "write_file": True,
        },
    )
    
    print("\n1. Agent wants to delete - human provides guidance...")
    try:
        result = await agent.act("delete_file", {"path": "/important/data.csv"})
    except InterruptedException as e:
        pending = e.state.pending_actions[0]
        print(f"   ⏸️  Pending: {pending.describe()}")
        
        # Human provides guidance instead of yes/no
        guidance = HumanDecision.guide(
            pending.action_id,
            guidance="Don't delete this file directly. First, create a backup in /backup/, "
                     "verify the backup is complete, then delete the original.",
            feedback="Critical data - be careful"
        )
        
        result = await agent.resume_action(guidance)
        
        if isinstance(result, GuidanceResult):
            print(f"   💡 Guidance received!")
            print(f"   📝 Instructions: {result.guidance}")
            print(f"   📋 Original action: {result.original_action.describe()}")
            print(f"   🔄 Should retry: {result.should_retry}")
            print(f"\n   Agent message format:")
            print(f"   {result.to_message()}")
    
    print("\n2. Agent asks a question - human responds directly...")
    try:
        # Simulating an agent asking "what should I name the output file?"
        result = await agent.act("write_file", {"path": "?", "content": "report data"})
    except InterruptedException as e:
        pending = e.state.pending_actions[0]
        print(f"   ⏸️  Agent asks: What path should I use for the file?")
        print(f"   (Pending: {pending.describe()})")
        
        # Human provides a direct response
        response = HumanDecision.respond(
            pending.action_id,
            response={
                "path": "/reports/Q4-2024-Sales.csv",
                "add_timestamp": True,
                "format": "csv"
            },
            feedback="Use this naming convention for all reports"
        )
        
        result = await agent.resume_action(response)
        
        if isinstance(result, HumanResponse):
            print(f"   💬 Response received!")
            print(f"   📦 Data: {result.response}")
            print(f"   📝 Feedback: {result.feedback}")
    
    print("\n3. Comparison of decision types:")
    print("""
   Decision Types:
   ┌─────────────┬─────────────────────────────────────────────────┐
   │ Type        │ Use Case                                        │
   ├─────────────┼─────────────────────────────────────────────────┤
   │ APPROVE     │ "Yes, do exactly what you proposed"             │
   │ REJECT      │ "No, don't do this at all"                      │
   │ EDIT        │ "Do it, but with these modified arguments"      │
   │ SKIP        │ "Skip this one, continue with the rest"         │
   │ ABORT       │ "Stop everything immediately"                   │
   │ GUIDE       │ "Here's how you should approach this instead"   │
   │ RESPOND     │ "Here's the information you asked for"          │
   └─────────────┴─────────────────────────────────────────────────┘
   """)


# =============================================================================
# Main
# =============================================================================

async def main():
    """Run all HITL demos."""
    print("="*60)
    print("Human-in-the-Loop (HITL) Demonstration")
    print("="*60)
    
    await demo_basic_approval()
    await demo_dynamic_rules()
    await demo_abort()
    await demo_interactive()
    await demo_guidance()
    
    print("\n✅ All HITL demos completed!")


if __name__ == "__main__":
    asyncio.run(main())
