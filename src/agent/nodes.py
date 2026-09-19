from typing import Dict, Any
from src.agent.state import AgentState
from src.utils.logger import get_logger

logger = get_logger(__name__)

def router_node(state: AgentState) -> Dict[str, Any]:
    """Determines whether the query requires external document retrieval."""
    last_message = state["messages"][-1]
    content = str(last_message.content if hasattr(last_message, "content") else last_message).lower()
    
    rag_keywords = ["document", "policy", "according to", "pdf", "file", "retrive", "context"]
    needs_rag = any(kw in content for kw in rag_keywords)
    
    next_step = "retriever" if needs_rag else "generate"
    logger.info("Router decision made", query=content, next_node=next_step)
    return {"next_node": next_step}

def generate_node(state: AgentState) -> Dict[str, Any]:
    """Generates response based on current messages and documents."""
    return {"next_node": "END"}