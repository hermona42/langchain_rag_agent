from typing import List
from langchain_core.documents import Document

def format_docs_with_citations(docs: List[Document]) -> str:
    """
    Formats retrieved documents into a structured prompt context with explicit citations.
    """
    formatted_chunks = []
    for doc in docs:
        source = doc.metadata.get("source", "Unknown Source")
        page = doc.metadata.get("page", "N/A")
        citation = f"[Source: {source} | Page: {page}]"
        chunk_text = f"{citation}\n{doc.page_content}"
        formatted_chunks.append(chunk_text)
    
    return "\n\n---\n\n".join(formatted_chunks)