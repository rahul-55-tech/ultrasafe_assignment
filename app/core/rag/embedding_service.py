from typing import List

from openai import OpenAI

from configurations.config import settings


class EmbeddingService:
    """Service for generating embeddings using OpenAI (SDK >= 1.0.0)"""

    def __init__(self):
        self.client = OpenAI(api_key=settings.openai_api_key)
        self.model = settings.embedding_model

    def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Synchronous embedding for a list of texts"""
        try:
            response = self.client.embeddings.create(model=self.model, input=texts)
            return [item.embedding for item in response.data]
        except Exception as e:
            raise Exception(f"OpenAI embedding error: {e}")

    def get_embedding(self, text: str) -> List[float]:
        """Single text embedding"""
        return self.get_embeddings([text])[0]

    def chunk_text(
        self, text: str, chunk_size: int = None, overlap: int = None
    ) -> List[str]:
        """Split text into chunks for embedding"""
        if chunk_size is None:
            chunk_size = settings.chunk_size
        if overlap is None:
            overlap = settings.chunk_overlap

        chunks = []
        start = 0

        while start < len(text):
            end = start + chunk_size
            chunks.append(text[start:end])
            start = end - overlap

        return chunks
