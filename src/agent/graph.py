import time
from typing import Callable, Any
from langgraph.graph import StateGraph, END
from src.agent.state import AgentState
from src.agent.nodes import router_node, generate_node
from src.utils.logger import get_logger

logger = get_logger(__name__)

def execute_with_retry(func: Callable, max_retries: int = 3, initial_delay: float = 1.0) -> Any:
    """
    Executes a function with exponential backoff for resilience against transient errors.
    """
    delay = initial_delay
    for attempt in range(1, max_retries + 1):
        try:
            return func()
        except Exception as e:
            logger.warning(
                "Execution failed, retrying...",
                attempt=attempt,
                max_retries=max_retries,
                error=str(e)
            )
            if attempt == max_retries:
                logger.error("Max retries reached. Operation failed.", error=str(e))
                raise e
            time.sleep(delay)
            delay *= 2

def route_decision(state: AgentState) -> str:
    """Evaluates router output for conditional edges."""
    return state.get("next_node", "generate")

def build_agent_graph():
    """
    Constructs and compiles the LangGraph multi-step agent.
    """
    workflow = StateGraph(AgentState)

    # Add nodes
    workflow.add_node("router", router_node)
    workflow.add_node("generate", generate_node)

    # Set entry point
    workflow.set_entry_point("router")

    # Add conditional edge from router
    workflow.add_conditional_edges(
        "router",
        route_decision,
        {
            "retriever": "generate",
            "generate": "generate"
        }
    )

    # Add terminal edge
    workflow.add_edge("generate", END)

    return workflow.compile()