# Production Multi-Agent RAG System with LangGraph & Qdrant

A resilient, production-ready retrieval-augmented generation (RAG) agent framework built with Python, LangGraph, LangChain, Qdrant vector storage, and Pydantic. Designed with test-driven development (TDD), exponential backoff retry logic, structured logging, and token cost tracking.

---

## Key Features

- **Multi-Agent State Orchestration**: Controlled execution flow using compiled state graphs (`LangGraph`).
- **Citation-Backed RAG Engine**: Context processing with document metadata, source tracking, and citation formatting.
- **Cost & Token Monitoring**: Callbacks to calculate token expenditure per completion run.
- **Resilient Execution**: Exponential backoff wrappers (`execute_with_retry`) to handle transient network and API disruptions.
- **100% Test Coverage Strategy**: Complete unit test suite using `pytest` and `unittest.mock`.
- **CI/CD Integration**: Pre-configured GitHub Actions pipeline for automated test execution.

---

## Directory Structure

```text
langchain_rag_agent/
├── .github/
│   └── workflows/
│       └── ci.yml             # GitHub Actions CI pipeline
├── config/
│   ├── __init__.py
│   └── settings.py          # Centralized environment & app settings
├── src/
│   ├── agent/
│   │   ├── __init__.py
│   │   ├── graph.py          # Compiled StateGraph & retry handling
│   │   ├── nodes.py          # Decision & routing node functions
│   │   └── state.py          # TypedDict AgentState schema
│   ├── monitoring/
│   │   ├── __init__.py
│   │   └── cost_tracker.py   # Token usage & cost calculation callbacks
│   ├── rag/
│   │   ├── __init__.py
│   │   ├── retriever.py     # Document formatting & citation logic
│   │   └── vector_store.py  # Qdrant client & embedding integration
│   └── utils/
│       ├── __init__.py
│       └── logger.py         # Structlog logger initialization
├── tests/
│   ├── test_agent.py         # LangGraph graph & router tests
│   ├── test_monitoring.py    # Cost tracker & settings tests
│   └── test_rag.py           # Citation formatting & vector store tests
├── .env.example
├── main.py                   # System entry point
├── pytest.ini
└── requirements.txt
```

---

## Getting Started

### Prerequisites

- **Python**: 3.10 or 3.11
- **API Keys**: OpenAI API Key (or local equivalent)

### Installation

1. **Clone the Repository**:
   ```bash
   git clone [https://github.com/your-username/langchain_rag_agent.git](https://github.com/your-username/langchain_rag_agent.git)
   cd langchain_rag_agent
   ```

2. **Set Up Virtual Environment**:
   ```bash
   python -m venv venv
   # On Windows (Git Bash / MINGW64)
   source venv/Scripts/activate
   ```

3. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Environment Configuration**:
   Create a `.env` file based on `.env.example`:
   ```bash
   OPENAI_API_KEY=your-openai-api-key
   QDRANT_URL=http://localhost:6333
   MODEL_NAME=gpt-4o-mini
   EMBEDDING_MODEL=text-embedding-3-small
   ```

---

## Running the Application

To index sample documents into Qdrant in-memory mode, run the routing state graph, and view formatted citations:

```bash
python main.py
```

### Example Output

```text
================================================================================
=== RETRIEVED CONTEXT WITH CITATIONS ===
================================================================================
[1] Standard enterprise customer support SLA guarantees initial response within 2 hours for critical incidents.
    Source: support_sla_v2.pdf (Page 3)

================================================================================
=== AGENT EXECUTION SUMMARY ===
================================================================================
Final Next Node Decision : END
Total Retried Chunks     : 2
Total Model Usage Cost   : $0.000450
================================================================================
```

---

## Testing

Run the full automated test suite via `pytest`:

```bash
pytest -v
```