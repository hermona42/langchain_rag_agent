import sys
from langchain_core.documents import Document
from config.settings import get_settings
from src.rag.vector_store import QdrantRAGStore
from src.rag.retriever import format_docs_with_citations
from src.agent.graph import build_agent_graph, execute_with_retry
from src.monitoring.cost_tracker import CostTrackerCallback
from src.utils.logger import get_logger

logger = get_logger("MainApp")

def initialize_sample_knowledge_base(store: QdrantRAGStore):
    """Populates the Qdrant vector database with sample operational policies."""
    sample_docs = [
        Document(
            page_content="Standard enterprise customer support SLA guarantees initial response within 2 hours for critical incidents.",
            metadata={"source": "support_sla_v2.pdf", "page": 3}
        ),
        Document(
            page_content="All enterprise software purchases include 24/7 dedicated engineering support and quarterly security audits.",
            metadata={"source": "enterprise_terms.pdf", "page": 12}
        ),
        Document(
            page_content="Refund requests must be submitted within 30 days of purchase through the billing account dashboard.",
            metadata={"source": "billing_policy.pdf", "page": 1}
        )
    ]
    logger.info("Indexing knowledge base sample documents into Qdrant...")
    store.add_documents(sample_docs)

def run():
    settings = get_settings()
    logger.info("Initializing LangChain RAG Agent System...", model=settings.MODEL_NAME)

    # Initialize Cost Tracker Callback
    cost_tracker = CostTrackerCallback(model_name=settings.MODEL_NAME)

    # Initialize Vector Store (using in-memory for zero-dependency execution)
    vector_store = QdrantRAGStore(collection_name="enterprise_knowledge", in_memory=True)
    initialize_sample_knowledge_base(vector_store)

    # Execute Sample Retrieval Query
    query = "According to the document, what is the customer support SLA response time?"
    logger.info("User Query Received", query=query)

    retrieved_docs = execute_with_retry(lambda: vector_store.similarity_search(query, k=2))
    formatted_context = format_docs_with_citations(retrieved_docs)

    print("\n" + "="*80)
    print("=== RETRIEVED CONTEXT WITH CITATIONS ===")
    print("="*80)
    print(formatted_context)

    # Build and Invoke Agent StateGraph
    app = build_agent_graph()
    initial_state = {
        "messages": [query],
        "next_node": "",
        "documents": [{"content": d.page_content, "metadata": d.metadata} for d in retrieved_docs],
        "cost_usd": 0.0
    }

    result = app.invoke(initial_state)

    print("\n" + "="*80)
    print("=== AGENT EXECUTION SUMMARY ===")
    print("="*80)
    print(f"Final Next Node Decision : {result.get('next_node')}")
    print(f"Total Retried Chunks     : {len(retrieved_docs)}")
    print(f"Total Model Usage Cost   : ${cost_tracker.total_cost_usd:.6f}")
    print("="*80 + "\n")

if __name__ == "__main__":
    run()