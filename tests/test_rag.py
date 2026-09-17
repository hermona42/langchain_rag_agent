import pytest
from unittest.mock import MagicMock, patch
from langchain_core.documents import Document
from src.rag.vector_store import QdrantRAGStore
from src.rag.retriever import format_docs_with_citations

def test_format_docs_with_citations():
    """
    Requirement 4: RAG must produce structured text containing clear document citations.
    """
    docs = [
        Document(
            page_content="Company refund policy allows returns within 30 days.",
            metadata={"source": "policy_v2.pdf", "page": 4}
        ),
        Document(
            page_content="Subscriptions can be canceled at any time from billing settings.",
            metadata={"source": "faq.pdf", "page": 1}
        )
    ]
    formatted = format_docs_with_citations(docs)
    
    assert "[Source: policy_v2.pdf | Page: 4]" in formatted
    assert "Company refund policy allows returns within 30 days." in formatted
    assert "[Source: faq.pdf | Page: 1]" in formatted

@patch("src.rag.vector_store.OpenAIEmbeddings")
def test_vector_store_in_memory_index(mock_embeddings_class):
    """
    Requirement: Vector store initializes in memory and indexes document batches.
    """
    mock_embeddings = MagicMock()
    mock_embeddings.embed_documents.return_value = [[0.1] * 1536]
    mock_embeddings.embed_query.return_value = [0.1] * 1536
    mock_embeddings_class.return_value = mock_embeddings

    store = QdrantRAGStore(in_memory=True)
    docs = [Document(page_content="Test Qdrant indexing", metadata={"source": "test.pdf", "page": 1})]
    
    store.add_documents(docs)
    results = store.similarity_search("Test search", k=1)
    
    assert len(results) == 1
    assert results[0].page_content == "Test Qdrant indexing"