"""
Example: Shell Capability

Agent executes shell commands to analyze code and system info.

Usage:
    uv run python examples/capabilities/shell.py
"""

import asyncio
import tempfile
from pathlib import Path

from agenticflow import Agent, Flow
from agenticflow.capabilities import Shell


async def main() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create sample files for the agent to analyze
        (Path(tmpdir) / "main.py").write_text("def hello():\n    print('Hello')\n\nhello()\n")
        (Path(tmpdir) / "utils.py").write_text("def add(a, b):\n    return a + b\n")
        (Path(tmpdir) / "README.md").write_text("# Sample Project\n\nA demo project.\n")

        shell = Shell(
            allowed_commands=["ls", "cat", "wc", "grep", "find", "head", "tail"],
            allowed_paths=[tmpdir],
            working_dir=tmpdir,
            timeout_seconds=30,
        )

        model = "gpt4"
        devops = Agent(
            name="DevOps",
            model="gpt4",
            instructions="You help analyze codebases using shell commands. Be efficient with commands.",
            capabilities=[shell],
        )

        flow = Flow(
            name="code_analysis",
            agents=[devops],
            verbosity="debug",
        )

        result = await flow.run(
            f"Analyze the project in {tmpdir}. "
            "List all files, count lines of Python code, and summarize what the project does."
        )
        print(f"\n{result.output}")


if __name__ == "__main__":
    asyncio.run(main())
