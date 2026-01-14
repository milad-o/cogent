"""A2A delegation as tools for agents.

This module provides automatic tool injection for A2A delegation based on
agent roles in the flow topology.
"""

from typing import Any

from agenticflow.tools import tool


def create_delegate_tool(flow: Any, from_agent: str) -> Any:
    """Create a delegation tool for a coordinator agent.
    
    This tool is automatically injected into agents that coordinate work
    (registered with on="task.created" or similar coordination events).
    
    Args:
        flow: The ReactiveFlow instance
        from_agent: Name of the coordinating agent
        
    Returns:
        Tool function for delegating to specialists
    """
    
    @tool
    async def delegate_to(
        specialist: str,
        task: str,
    ) -> str:
        """Delegate a task to a specialist agent.
        
        Use this when you need to hand off specialized work to another team member.
        
        Args:
            specialist: Name of the specialist agent (e.g., "data_analyst", "writer", "researcher")
            task: Clear description of what you want them to do
            
        Returns:
            Confirmation that the task was delegated
            
        Example:
            delegate_to(
                specialist="data_analyst",
                task="Analyze Q4 sales data and identify trends"
            )
        """
        from agenticflow.reactive.a2a import create_request
        
        # Create agent request
        request = create_request(
            from_agent=from_agent,
            to_agent=specialist,
            task=task,
        )
        
        # Emit agent.request event to trigger the specialist
        await flow.emit("agent.request", request.to_event())
        
        return f"Task delegated to {specialist}"
    
    # Add metadata about available specialists
    available = [name for name, (_, cfg) in flow._agents_registry.items() 
                 if any(t.on == "agent.request" for t in cfg.triggers)]
    delegate_to.__doc__ = f"""Delegate a task to a specialist agent.
    
Available specialists in this flow: {', '.join(available)}

Use this when you need to hand off specialized work to another team member.

Args:
    specialist: Name of the specialist agent
    task: Clear description of what you want them to do
    
Returns:
    Confirmation that the task was delegated
"""
    
    return delegate_to


def create_reply_tool(flow: Any, specialist_agent: str) -> Any:
    """Create a reply tool for a specialist agent.
    
    This tool is automatically injected into agents that handle delegated work
    (registered with handles=True).
    
    Args:
        flow: The ReactiveFlow instance  
        specialist_agent: Name of the specialist agent
        
    Returns:
        Tool function for replying with results
    """
    
    @tool
    async def reply_with_result(
        result: str,
        success: bool = True,
    ) -> str:
        """Send your result back to the agent who delegated this task to you.
        
        Call this when you've completed the delegated work.
        
        Args:
            result: Your analysis, report, or output
            success: Whether the task completed successfully (default: True)
            
        Returns:
            Confirmation that the reply was sent
            
        Example:
            reply_with_result(
                result="Sales increased 15% in Q4, driven by holiday promotions",
                success=True
            )
        """
        from agenticflow.reactive.a2a import AgentResponse
        
        # Create response (correlation_id will be extracted from current event context)
        response = AgentResponse(
            from_agent=specialist_agent,
            to_agent="",  # Will be filled from request context
            result=result,
            data=None,
            correlation_id="",  # Will be filled from request
            success=success,
            error=None if success else "Task failed",
        )
        
        # Emit agent.response event
        await flow.emit("agent.response", response.to_event())
        
        return "Result sent back to coordinator"
    
    return reply_with_result

