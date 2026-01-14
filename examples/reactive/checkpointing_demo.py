"""Example: Checkpointing for Long-Running Reactive Flows.

Demonstrates persistent checkpointing for crash recovery in multi-agent
workflows. Uses real LLM agents with tools to simulate a realistic
data processing pipeline.

Use Cases:
- Long-running batch processing jobs
- Workflows that may fail partway through
- Distributed systems needing coordination
- Audit trails for compliance

Usage:
    uv run python examples/reactive/checkpointing_demo.py
"""

import asyncio
import sys
from pathlib import Path

# Add examples to path for config import
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import get_model

from agenticflow import Agent, tool
from agenticflow.reactive import (
    ReactiveFlow,
    ReactiveFlowConfig,
    MemoryCheckpointer,
    FileCheckpointer,
    Observer,
    react_to,
)


# ═══════════════════════════════════════════════════════════════════════════════
# TOOLS - Simulated external services
# ═══════════════════════════════════════════════════════════════════════════════


@tool
def fetch_batch_data(batch_id: str) -> str:
    """Fetch data for a batch from the data warehouse."""
    # Simulated batch data
    batches = {
        "B-001": "Order #1001: 5 items, $250\nOrder #1002: 3 items, $150\nOrder #1003: 8 items, $400",
        "B-002": "Order #2001: 2 items, $100\nOrder #2002: 10 items, $500",
    }
    return batches.get(batch_id, f"Batch {batch_id}: 3 orders, $300 total")


@tool
def validate_records(data: str) -> str:
    """Validate records for data quality issues."""
    lines = data.strip().split("\n")
    issues = []
    valid = 0
    for line in lines:
        if "$" in line:
            valid += 1
        else:
            issues.append(f"Missing price in: {line[:30]}...")
    return f"Validated {valid} records. Issues: {len(issues)}"


@tool
def transform_to_report(data: str, format: str = "summary") -> str:
    """Transform validated data into a report format."""
    lines = len(data.strip().split("\n"))
    if format == "summary":
        return f"Report Summary:\n- {lines} records processed\n- Format: {format}\n- Status: Complete"
    return f"[{format.upper()} REPORT]\n{data}"


@tool
def notify_completion(message: str, channel: str = "slack") -> str:
    """Send notification about job completion."""
    return f"✓ Notification sent to {channel}: {message[:50]}..."


# ═══════════════════════════════════════════════════════════════════════════════
# CHECKPOINTING EXAMPLES
# ═══════════════════════════════════════════════════════════════════════════════


async def basic_checkpointing() -> None:
    """Basic checkpointing with MemoryCheckpointer."""
    print("\n" + "─" * 60)
    print("📦 BASIC CHECKPOINTING")
    print("─" * 60)
    print("Long-running ETL pipeline with automatic checkpoints.\n")

    observer = Observer.trace()
    checkpointer = MemoryCheckpointer()
    
    config = ReactiveFlowConfig(
        checkpoint_every=1,  # Save after every round
        flow_id="etl-pipeline-001",
    )

    # ETL Pipeline agents
    extractor = Agent(
        name="extractor",
        model=get_model(),
        system_prompt="You extract and fetch batch data. Use the fetch tool to get data.",
        tools=[fetch_batch_data],
    )
    
    validator = Agent(
        name="validator", 
        model=get_model(),
        system_prompt="You validate data quality. Use the validate tool to check records.",
        tools=[validate_records],
    )
    
    transformer = Agent(
        name="transformer",
        model=get_model(),
        system_prompt="You transform data into reports. Use the transform tool.",
        tools=[transform_to_report],
    )
    
    notifier = Agent(
        name="notifier",
        model=get_model(),
        system_prompt="You send completion notifications. Keep it brief.",
        tools=[notify_completion],
    )

    # Create flow with checkpointing
    flow = ReactiveFlow(config=config, checkpointer=checkpointer, observer=observer)
    
    # Register agents with event triggers
    flow.register(extractor, [react_to("task.created")])
    flow.register(validator, [react_to("extractor.completed")])
    flow.register(transformer, [react_to("validator.completed")])
    flow.register(notifier, [react_to("transformer.completed")])

    # Run the pipeline
    result = await flow.run(
        "Process batch B-001 through the ETL pipeline",
        initial_event="task.created",
        initial_data={"batch_id": "B-001"},
    )

    print(f"\n✓ Pipeline completed")
    print(f"  Flow ID: {result.flow_id}")
    print(f"  Last checkpoint: {result.checkpoint_id}")
    print(f"  Events processed: {result.events_processed}")
    print(f"  Agent reactions: {len(result.reactions)}")

    # Show checkpoint history
    checkpoints = await checkpointer.list_checkpoints("etl-pipeline-001")
    print(f"  Checkpoints saved: {len(checkpoints)}")


async def crash_recovery_simulation() -> None:
    """Simulate crash recovery using checkpoints."""
    print("\n" + "─" * 60)
    print("🔄 CRASH RECOVERY SIMULATION")
    print("─" * 60)
    print("Simulates resuming a flow from a saved checkpoint.\n")

    observer = Observer.trace()
    checkpointer = MemoryCheckpointer()
    
    # First, create a checkpoint manually (simulating a previous run)
    from agenticflow.flow.checkpointer import FlowState
    
    saved_state = FlowState(
        flow_id="data-sync-job",
        checkpoint_id="cp-before-crash",
        task="Sync customer data and validate records",
        events_processed=2,
        pending_events=[
            {"id": "ev-3", "name": "validation.complete", "data": {"task": "Sync customer data and validate records", "records": 150}}
        ],
        context={"source": "crm", "target": "warehouse"},
        last_output="Validation passed: 150 records ready",
        round=2,
    )
    await checkpointer.save(saved_state)

    print(f"💾 Simulated checkpoint saved: {saved_state.checkpoint_id}")
    print(f"   At round: {saved_state.round}")
    print(f"   Pending events: {len(saved_state.pending_events)}")

    # Create agents for the resumed flow
    sync_agent = Agent(
        name="sync_agent",
        model=get_model(),
        system_prompt="You handle data synchronization. Report completion status.",
    )
    
    finalizer = Agent(
        name="finalizer",
        model=get_model(),
        system_prompt="You finalize sync jobs and send notifications.",
        tools=[notify_completion],
    )

    # Create flow and register agents
    config = ReactiveFlowConfig(checkpoint_every=1)
    flow = ReactiveFlow(config=config, checkpointer=checkpointer, observer=observer)
    flow.register(sync_agent, [react_to("validation.complete")])
    flow.register(finalizer, [react_to("sync_agent.completed")])

    # Resume from checkpoint
    print("\n🔄 Resuming from checkpoint...")
    state = await checkpointer.load("cp-before-crash")
    if state:
        result = await flow.resume(state)
        print(f"\n✓ Flow resumed and completed")
        print(f"  Started from round: {saved_state.round}")
        print(f"  Final events processed: {result.events_processed}")
        print(f"  New reactions: {len(result.reactions)}")


async def file_based_checkpointing() -> None:
    """File-based checkpointing for persistence across restarts."""
    print("\n" + "─" * 60)
    print("💾 FILE-BASED CHECKPOINTING")
    print("─" * 60)
    print("Persists checkpoints to disk for cross-restart recovery.\n")

    import tempfile

    observer = Observer.trace()
    
    with tempfile.TemporaryDirectory() as tmpdir:
        checkpoint_dir = Path(tmpdir)
        checkpointer = FileCheckpointer(checkpoint_dir)
        
        config = ReactiveFlowConfig(
            checkpoint_every=1,
            flow_id="file-backed-flow",
        )

        processor = Agent(
            name="processor",
            model=get_model(),
            system_prompt="You process data batches. Report results briefly.",
            tools=[fetch_batch_data, validate_records],
        )
        
        reporter = Agent(
            name="reporter",
            model=get_model(),
            system_prompt="You create reports from processed data.",
            tools=[transform_to_report],
        )

        flow = ReactiveFlow(config=config, checkpointer=checkpointer, observer=observer)
        flow.register(processor, [react_to("task.created")])
        flow.register(reporter, [react_to("processor.completed")])

        result = await flow.run(
            "Process batch B-002 and create a report",
            initial_event="task.created",
            initial_data={"batch_id": "B-002"},
        )

        # Show saved checkpoint files
        checkpoint_files = list(checkpoint_dir.glob("*.json"))
        print(f"\n✓ Flow completed with file-based checkpoints")
        print(f"  Checkpoint directory: {checkpoint_dir}")
        print(f"  Checkpoint files: {len(checkpoint_files)}")
        for f in checkpoint_files:
            print(f"    - {f.name}")

        # Demonstrate loading from file
        latest = await checkpointer.load_latest("file-backed-flow")
        if latest:
            print(f"\n  Latest checkpoint: {latest.checkpoint_id}")
            print(f"  Task: {latest.task[:50]}...")
            print(f"  Round: {latest.round}")


async def multi_flow_coordination() -> None:
    """Multiple flows with independent checkpointing."""
    print("\n" + "─" * 60)
    print("🔗 MULTI-FLOW COORDINATION")
    print("─" * 60)
    print("Independent flows with separate checkpoint streams.\n")

    observer = Observer.off()  # Quiet mode for cleaner output
    checkpointer = MemoryCheckpointer()

    # Create two independent flows
    async def run_flow(flow_name: str, task: str) -> None:
        config = ReactiveFlowConfig(
            checkpoint_every=1,
            flow_id=flow_name,
        )
        
        agent = Agent(
            name="worker",
            model=get_model(),
            system_prompt="You process tasks efficiently. Be concise.",
        )
        
        flow = ReactiveFlow(config=config, checkpointer=checkpointer, observer=observer)
        flow.register(agent, [react_to("task.created")])
        
        result = await flow.run(task, initial_event="task.created")
        checkpoints = await checkpointer.list_checkpoints(flow_name)
        print(f"  {flow_name}: {len(checkpoints)} checkpoints")

    # Run flows concurrently
    await asyncio.gather(
        run_flow("flow-A", "Analyze sales data for Q4"),
        run_flow("flow-B", "Generate marketing report"),
        run_flow("flow-C", "Process customer feedback"),
    )

    print("\n✓ All flows completed with independent checkpoints")


async def main() -> None:
    """Run all checkpointing examples."""
    print("\n" + "═" * 60)
    print("CHECKPOINTING FOR REACTIVE FLOWS")
    print("═" * 60)
    print("""
Checkpointing enables:
• Crash recovery for long-running flows
• Resume execution from the last saved state
• Audit trail of flow execution progress
• Distributed coordination via shared checkpointers
""")

    await basic_checkpointing()
    await crash_recovery_simulation()
    await file_based_checkpointing()
    await multi_flow_coordination()

    print("\n" + "═" * 60)
    print("✅ All checkpointing examples completed!")
    print("═" * 60)


if __name__ == "__main__":
    asyncio.run(main())
