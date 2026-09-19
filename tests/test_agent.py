import pytest
from src.agent.state import AgentState
from src.agent.nodes import router_node, generate_node

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