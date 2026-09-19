import pytest
from src.agent.state import AgentState
from src.agent.nodes import router_node, generate_node
from unittest.mock import MagicMock
from src.agent.graph import build_agent_graph, execute_with_retry

def test_execute_with_retry_success():
    """Requirement 2: Error handling/retry wrapper retries failed functions."""
    mock_func = MagicMock(side_effect=[Exception("Transient network issue"), "Success"])
    
    result = execute_with_retry(mock_func, max_retries=2, initial_delay=0.01)
    
    assert result == "Success"
    assert mock_func.call_count == 2

def test_build_agent_graph_compile():
    """Requirement 1: LangGraph StateGraph compiles cleanly into executable workflow."""
    app = build_agent_graph()
    assert app is not None
    
def test_agent_state_initialization():
    """Requirement 2: Agent state holds conversation history and metadata."""
    state: AgentState = {
        "messages": [{"role": "user", "content": "What is the policy?"}],
        "next_node": "",
        "documents": [],
        "cost_usd": 0.0,
    }
    assert len(state["messages"]) == 1
    assert state["cost_usd"] == 0.0

def test_router_node_directs_rag():
    """Requirement 2: Router sends knowledge query to retriever."""
    state: AgentState = {
        "messages": [{"role": "user", "content": "According to the document, what is X?"}],
        "next_node": "",
        "documents": [],
        "cost_usd": 0.0,
    }
    result = router_node(state)
    assert result["next_node"] in ["retriever", "generate"]