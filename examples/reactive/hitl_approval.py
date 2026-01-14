"""
Human-in-the-Loop (HITL) Reactive Flow Example

Demonstrates how to require human approval for agent actions in reactive flows.
Useful for sensitive operations like file deletion, data modifications, or
high-stakes decisions.

Requirements:
    - Configure your LLM provider in .env (see examples/.env.example)
    - Set LLM_PROVIDER and appropriate API key (e.g., OPENAI_API_KEY)

Run:
    uv run python examples/reactive/hitl_approval.py

What happens:
    1. An analyst agent is triggered by "task.created" event
    2. You'll be prompted to approve the analyst's action
    3. If approved, the analyst analyzes the task and emits "analyst.completed"
    4. The executor agent is triggered by "analyst.completed"
    5. You'll be prompted to approve the executor's action
    6. If approved, the executor implements the plan

Try:
    - Approve both agents to see the full workflow
    - Reject the analyst to see early termination
    - Approve analyst but reject executor to see partial execution
"""

import asyncio
import sys
from dataclasses import dataclass
from pathlib import Path

# Add examples to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from agenticflow import Agent
from agenticflow.reactive.core import ReactionType, Trigger
from agenticflow.reactive.flow import EventFlow, EventFlowConfig
from config import get_model


@dataclass
class ApprovalBreakpoint:
    """Simple breakpoint for approval."""
    name: str
    prompt: str
    timeout: float | None = None
    on_timeout: str = "fail"


class ConsoleApprovalHandler:
    """Simple console-based approval handler."""
    
    async def request_approval(self, request) -> bool:
        """Prompt user for approval via console."""
        print(f"\n{'='*60}")
        print(f"🛑 Approval Required: {request.agent_name}")
        print(f"{'='*60}")
        
        if hasattr(request, 'breakpoint') and request.breakpoint:
            print(f"\n{request.breakpoint.prompt}\n")
        else:
            print(f"\nAgent '{request.agent_name}' wants to execute an action.")
        
        print(f"Event: {request.event.name}")
        if hasattr(request, 'pending_action'):
            print(f"Action: {request.pending_action.describe()}")
        
        # Get user input
        response = await self._get_input("Approve? (y/n): ")
        approved = response.lower().strip() in ('y', 'yes')
        
        print(f"{'✅ Approved' if approved else '❌ Rejected'}")
        print(f"{'='*60}\n")
        
        return approved
    
    async def _get_input(self, prompt: str) -> str:
        """Get console input in async-compatible way."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, input, prompt)


async def main():
    """Run HITL reactive flow example."""
    
    print("\n" + "="*60)
    print("Human-in-the-Loop Reactive Flow Example")
    print("="*60)
    print("\nThis example demonstrates requiring human approval")
    print("for agent actions in a reactive flow.\n")
    
    # Create model
    model = get_model()
    
    # Create agents
    analyst = Agent(
        name="analyst",
        model=model,
        system_prompt=(
            "You are a data analyst. When you receive a task, "
            "analyze what needs to be done and propose a plan."
        ),
    )
    
    executor = Agent(
        name="executor",
        model=model,
        system_prompt=(
            "You are an executor. You implement plans that have been approved. "
            "Be concise and actionable."
        ),
    )
    
    # Create HITL handler
    handler = ConsoleApprovalHandler()
    
    # Create flow with HITL support
    flow = EventFlow(
        config=EventFlowConfig(
            max_rounds=20,
            max_concurrent_agents=2,
        ),
        hitl_handler=handler,
    )
    
    # Register analyst - requires approval before running
    analyst_breakpoint = ApprovalBreakpoint(
        name="analyst_approval",
        prompt="The analyst wants to analyze the task. Approve?",
    )
    
    flow.register(
        analyst,
        [Trigger(
            on="task.created",
            reaction=ReactionType.AWAIT_HUMAN,
            breakpoint=analyst_breakpoint,
            emits="analyst.completed",
        )]
    )
    
    # Register executor - requires approval before executing
    executor_breakpoint = ApprovalBreakpoint(
        name="executor_approval",
        prompt="The executor wants to implement the plan. Approve?",
    )
    
    flow.register(
        executor,
        [Trigger(
            on="analyst.completed",
            reaction=ReactionType.AWAIT_HUMAN,
            breakpoint=executor_breakpoint,
        )]
    )
    
    # Run the flow
    print("Starting reactive flow with HITL...")
    print("You will be prompted to approve each agent action.\n")
    
    result = await flow.run(
        task="Analyze our Q4 sales data and create a summary report",
        initial_event="task.created",
    )
    
    # Show results
    print("\n" + "="*60)
    print("Flow Completed")
    print("="*60)
    print(f"\nEvents processed: {result.events_processed}")
    print(f"Agent reactions: {len(result.reactions)}")
    print(f"Execution time: {result.execution_time_ms:.0f}ms")
    
    print("\n" + "="*60)
    print("Reaction Details")
    print("="*60)
    for i, reaction in enumerate(result.reactions, 1):
        status = "✅ Success" if not reaction.error else f"❌ Error: {reaction.error}"
        print(f"\n{i}. {reaction.agent_name} - {status}")
        if reaction.output and len(reaction.output) > 0:
            output_preview = reaction.output[:200] + "..." if len(reaction.output) > 200 else reaction.output
            print(f"   Output: {output_preview}")
        if reaction.emitted_events:
            print(f"   Emitted: {', '.join(reaction.emitted_events)}")
    
    print(f"\n{'='*60}")
    print("Final Output")
    print(f"{'='*60}\n")
    print(result.output)
    print()


if __name__ == "__main__":
    asyncio.run(main())
