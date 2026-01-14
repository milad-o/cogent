"""
Checkpointing for Imperative Flows Example

Demonstrates how to use checkpointing with imperative Flow (Supervisor, Pipeline, Mesh)
to save/resume execution state for crash recovery and long-running workflows.

Requirements:
    - Configure your LLM provider in .env (see examples/.env.example)
    - Set LLM_PROVIDER and appropriate API key

Run:
    uv run python examples/flow/checkpointing_demo.py

What happens:
    1. Creates a pipeline flow (researcher → writer → editor)
    2. Enables checkpointing with FileCheckpointer
    3. Executes the flow and saves state after each agent
    4. Demonstrates resuming from a checkpoint
"""

import asyncio
import sys
from pathlib import Path

# Add examples to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from agenticflow import Agent, Flow
from agenticflow.flow.orchestrated import FlowConfig
from agenticflow.flow.checkpointer import FileCheckpointer
from config import get_model


async def main():
    """Run checkpointing demo."""
    
    print("\n" + "="*60)
    print("Flow Checkpointing Demo")
    print("="*60)
    print("\nThis demonstrates checkpointing for imperative flows")
    print("(Supervisor, Pipeline, Mesh patterns)\n")
    
    # Create model
    model = get_model()
    
    # Create agents for pipeline
    researcher = Agent(
        name="researcher",
        model=model,
        system_prompt="You are a researcher. Analyze topics and gather key information.",
    )
    
    writer = Agent(
        name="writer",
        model=model,
        system_prompt="You are a writer. Create clear, engaging content from research.",
    )
    
    editor = Agent(
        name="editor",
        model=model,
        system_prompt="You are an editor. Polish content for clarity and impact.",
    )
    
    # Setup checkpointer
    checkpoint_dir = Path("./checkpoints")
    checkpoint_dir.mkdir(exist_ok=True)
    
    checkpointer = FileCheckpointer(directory=checkpoint_dir)
    
    # Create flow with checkpointing enabled
    flow = Flow(
        name="content-pipeline",
        agents=[researcher, writer, editor],
        topology="pipeline",
        config=FlowConfig(
            flow_id="demo-flow-001",  # Fixed ID for resuming
            checkpoint_every=1,  # Checkpoint after each step
        ),
        checkpointer=checkpointer,
        verbose=True,
    )
    
    print("="*60)
    print("Running flow with checkpointing enabled...")
    print("Checkpoints will be saved to:", checkpoint_dir)
    print("="*60 + "\n")
    
    # Run the flow
    task = "Write a brief introduction to quantum computing for beginners"
    result = await flow.run(task)
    
    print("\n" + "="*60)
    print("Flow Completed Successfully")
    print("="*60)
    print(f"\nFinal Output ({len(result.output)} chars):\n")
    print(result.output[:500] + "..." if len(result.output) > 500 else result.output)
    
    # List checkpoints
    print("\n" + "="*60)
    print("Checkpoints Created")
    print("="*60)
    
    checkpoints = await checkpointer.list_checkpoints("demo-flow-001")
    print(f"\nFound {len(checkpoints)} checkpoint(s):")
    for i, cp_id in enumerate(checkpoints, 1):
        state = await checkpointer.load(cp_id)
        if state:
            step = state.metadata.get("step", "?")
            completed = state.metadata.get("completed", False)
            status = "✅ Completed" if completed else "🔄 In Progress"
            print(f"  {i}. {cp_id[:16]}... - Step {step} - {status}")
    
    # Demonstrate resume
    print("\n" + "="*60)
    print("Testing Resume from Checkpoint")
    print("="*60 + "\n")
    
    if checkpoints:
        latest_checkpoint = checkpoints[0]
        print(f"Resuming from checkpoint: {latest_checkpoint[:16]}...")
        
        # Create new flow instance
        flow2 = Flow(
            name="content-pipeline",
            agents=[researcher, writer, editor],
            topology="pipeline",
            config=FlowConfig(flow_id="demo-flow-001"),
            checkpointer=checkpointer,
        )
        
        # Resume from checkpoint
        resumed_result = await flow2.resume(latest_checkpoint)
        
        print("\n✅ Successfully resumed from checkpoint!")
        print(f"Output matches: {resumed_result.output == result.output}")
    
    print("\n" + "="*60)
    print("Demo Complete")
    print("="*60)
    print(f"\nCheckpoint files saved in: {checkpoint_dir}")
    print("You can delete them with: rm -rf ./checkpoints\n")


if __name__ == "__main__":
    asyncio.run(main())
