import uuid
from typing import Any, Dict, List, Optional, Tuple

from pinecone import Pinecone, ServerlessSpec

from configurations.config import settings

from .embedding_service import EmbeddingService


class VectorStore:
    def __init__(self):
        self.index_name = settings.pinecone_index_name
        # OpenAI ada-002 embedding dimension
        self.dimension = 1536

        pinecone = Pinecone(api_key=settings.pinecone_api_key)

        if self.index_name not in pinecone.list_indexes().names():
            print(f"Creating index: {self.index_name}...")
            pinecone.create_index(
                name=self.index_name,
                dimension=self.dimension,
                metric="cosine",
                spec=ServerlessSpec(cloud="aws", region="us-east-1"),
            )
            print(f"Index '{self.index_name}' created successfully.")
        else:
            print(f"Index '{self.index_name}' already exists.")

        self.index = pinecone.Index(self.index_name)

    def upsert_vectors(
        self,
        vectors: List[Tuple[str, List[float], Dict[str, Any]]],
        namespace: Optional[str] = None,
    ):
        """
        Upsert precomputed vectors with metadata.

        Args:
            vectors: List of tuples (id, embedding, metadata)
            namespace: The Pinecone namespace to upsert into.
        """
        pinecone_vectors = []
        for vector_id, embedding, metadata in vectors:
            pinecone_vectors.append(
                {"id": vector_id, "values": embedding, "metadata": metadata}
            )

        self.index.upsert(vectors=pinecone_vectors, namespace=namespace)

    async def similarity_search(
        self,
        query: str,
        k: int = 5,
        filter_dict: Optional[Dict[str, Any]] = None,
        namespace: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Search for similar texts by embedding a query string."""
        embedding_service = EmbeddingService()
        query_embedding = embedding_service.get_embedding(query)

        return await self.similarity_search_by_vector(
            vector=query_embedding, k=k, filter_dict=filter_dict, namespace=namespace
        )

    async def similarity_search_by_vector(
        self,
        vector: List[float],
        k: int = 5,
        filter_dict: Optional[Dict[str, Any]] = None,
        namespace: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Search for similar vectors using a pre-computed embedding."""
        search_results = self.index.query(
            vector=vector,
            top_k=k,
            filter=filter_dict,
            include_metadata=True,
            namespace=namespace,
        )

        results = []
        for match in search_results.matches:
            results.append(
                {"id": match.id, "score": match.score, "metadata": match.metadata}
            )
        return results

    async def delete_vectors(self, ids: List[str], namespace: Optional[str] = None):
        """Delete vectors by IDs"""
        self.index.delete(ids=ids, namespace=namespace)

    async def get_vector_count(self, namespace: Optional[str] = None) -> int:
        """Get total number of vectors in index/namespace"""
        stats = self.index.describe_index_stats()
        if namespace:
            return stats.namespaces.get(namespace, {}).get("vector_count", 0)
        return stats.total_vector_count

    async def add_texts(
        self,
        texts: List[str],
        metadatas: Optional[List[Dict[str, Any]]] = None,
        ids: Optional[List[str]] = None,
    ) -> List[str]:
        """Add texts to vector store"""
        if ids is None:
            ids = [str(uuid.uuid4()) for _ in texts]

        if metadatas is None:
            metadatas = [{} for _ in texts]

        embedding_service = EmbeddingService()
        embeddings = await embedding_service.get_embeddings(texts)

        vectors = []
        for i, (text, embedding, metadata, vector_id) in enumerate(
            zip(texts, embeddings, metadatas, ids)
        ):
            vectors.append(
                {
                    "id": vector_id,
                    "values": embedding,
                    "metadata": {"text": text, **metadata},
                }
            )

        self.index.upsert(vectors=vectors)
        return ids
