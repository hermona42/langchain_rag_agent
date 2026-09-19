from typing import List
from langchain_core.documents import Document
from langchain_core.embeddings.fake import FakeEmbeddings
from langchain_openai import OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams

from config.settings import get_settings
from src.utils.logger import get_logger

logger = get_logger(__name__)


class QdrantRAGStore:
    def __init__(self, collection_name: str = "knowledge_base", in_memory: bool = False):
        self.settings = get_settings()
        self.collection_name = collection_name

        api_key = self.settings.OPENAI_API_KEY.strip()

        # Fall back to FakeEmbeddings if API key is missing or dummy
        if not api_key or api_key.startswith("dummy") or api_key == "your_openai_api_key_here":
            logger.info("Using FakeEmbeddings for zero-cost offline vector indexing")
            self.embeddings = FakeEmbeddings(size=1536)
        else:
            self.embeddings = OpenAIEmbeddings(
                model=self.settings.EMBEDDING_MODEL,
                openai_api_key=api_key,
            )

        if in_memory:
            self.client = QdrantClient(":memory:")
        else:
            if self.settings.QDRANT_API_KEY:
                self.client = QdrantClient(url=self.settings.QDRANT_URL, api_key=self.settings.QDRANT_API_KEY)
            else:
                self.client = QdrantClient(url=self.settings.QDRANT_URL)

        # Create collection if missing
        if not self.client.collection_exists(self.collection_name):
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(size=1536, distance=Distance.COSINE),
            )
            logger.info("Created new Qdrant collection", collection=self.collection_name)

        self.vector_store = QdrantVectorStore(
            client=self.client,
            collection_name=self.collection_name,
            embedding=self.embeddings,
        )

    def add_documents(self, documents: List[Document]) -> List[str]:
        logger.info("Adding documents to vector store", count=len(documents))
        return self.vector_store.add_documents(documents)

    def similarity_search(self, query: str, k: int = 3) -> List[Document]:
        logger.info("Executing similarity search", query=query, k=k)
        return self.vector_store.similarity_search(query=query, k=k)

    def as_retriever(self, k: int = 3):
        return self.vector_store.as_retriever(search_kwargs={"k": k})