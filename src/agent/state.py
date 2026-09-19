from typing import TypedDict, List, Dict, Any, Sequence
from langchain_core.messages import BaseMessage

class AgentState(TypedDict):
    messages: Sequence[BaseMessage]
    next_node: str
    documents: List[Dict[str, Any]]
    cost_usd: float