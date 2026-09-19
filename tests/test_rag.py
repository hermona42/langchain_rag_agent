from unittest.mock import MagicMock, patch
from langchain_core.documents import Document
from src.rag.retriever import format_citation_docs
from src.rag.vector_store import QdrantRAGStore


def test_format_citation_docs():
    docs = [
        Document(
            page_content="Response time is 2 hours.",
            metadata={"source": "sla.pdf", "page": 1},
        ),
        Document(
            page_content="24/7 engineering support included.",
            metadata={"source": "terms.pdf", "page": 5},
        ),
    ]

    formatted = format_citation_docs(docs)

    assert "[Source: sla.pdf | Page: 1]" in formatted
    assert "Response time is 2 hours." in formatted
    assert "[Source: terms.pdf | Page: 5]" in formatted


@patch("src.rag.vector_store.QdrantClient")
@patch("src.rag.vector_store.QdrantVectorStore")
def test_vector_store_in_memory_index(mock_qdrant_store, mock_qdrant_client):
    mock_client_instance = MagicMock()
    mock_client_instance.collection_exists.return_value = False
    mock_qdrant_client.return_value = mock_client_instance

    mock_store_instance = MagicMock()
    mock_qdrant_store.return_value = mock_store_instance

    store = QdrantRAGStore(collection_name="test_collection", in_memory=True)

    sample_docs = [Document(page_content="Test context", metadata={"source": "test.txt"})]
    store.add_documents(sample_docs)

    mock_client_instance.create_collection.assert_called_once()
    mock_store_instance.add_documents.assert_called_once_with(sample_docs)