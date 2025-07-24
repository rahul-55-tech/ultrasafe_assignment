from typing import Any, Dict, List, Optional

import pandas as pd
from fastapi import Depends, HTTPException, status

from app.core.rag.reranker_service import RerankerService
from app.json_db import JSONDatabase, get_db

from app.core.rag.embedding_service import EmbeddingService
from app.core.rag.vector_store import VectorStore
from configurations.logger_config import logger

class RetrievalService:
    """Service for retrieving similar datasets and recommending analysis techniques."""

    def __init__(self, db: JSONDatabase):
        self.vector_store = VectorStore()
        self.embedding_service = EmbeddingService()
        self.reranker_service = RerankerService()
        self.db = db

    def _create_dataset_text_content(self, dataset: Dict, schemas: List[Dict]) -> str:
        """Helper to generate a consistent text representation of a dataset."""
        text_content = f"Dataset name: {dataset['name']}. "
        if dataset.get("description"):
            text_content += f"Description: {dataset['description']}. "

        text_content += f"File type: {dataset['file_type']}. "
        text_content += f"It has {dataset['row_count']} rows and {dataset['column_count']} columns. "

        numeric_cols = [
            s["column_name"]
            for s in schemas
            if "float" in s["data_type"] or "int" in s["data_type"]
        ]
        categorical_cols = [
            s["column_name"]
            for s in schemas
            if "object" in s["data_type"] or "bool" in s["data_type"]
        ]
        datetime_cols = [
            s["column_name"] for s in schemas if "datetime" in s["data_type"]
        ]

        if numeric_cols:
            text_content += f"Numerical columns include: {', '.join(numeric_cols)}. "
        if categorical_cols:
            text_content += (
                f"Categorical columns include: {', '.join(categorical_cols)}. "
            )
        if datetime_cols:
            text_content += f"Time-based columns include: {', '.join(datetime_cols)}."

        return text_content

    async def index_dataset(self, dataset: Dict, schemas: List[Dict]):
        """Index a dataset and its schema in the vector store."""
        metadata = {
            "dataset_id": dataset["id"],
            "name": dataset["name"],
            "description": dataset.get("description", ""),
            "file_type": dataset["file_type"],
            "row_count": dataset["row_count"],
            "column_count": dataset["column_count"],
            "schema_hash": dataset["schema_hash"],
            "column_names": [
                s["column_name"] for s in schemas
            ],
        }

        text_content = self._create_dataset_text_content(dataset, schemas)
        embedding = self.embedding_service.get_embedding(text_content)

        vector_id = f"dataset_{dataset['id']}"
        self.vector_store.upsert_vectors(
            [(vector_id, embedding, metadata)], namespace="datasets"
        )


    async def _rerank_with_model(
        self, query_text: str, candidates: List[Dict]
    ) -> List[Dict]:
        """Uses a reranker model to rank candidate datasets."""

        rerank_inputs = [
            {
                "id": candidate["id"],
                "text": self._create_dataset_text_content(
                    candidate["metadata"], candidate["metadata"].get("schema", [])
                ),
            }
            for candidate in candidates
        ]

        rerank_response = []
        reranked_scores = self.reranker_service.rerank(
            query=query_text, documents=rerank_inputs
        )
        logger.info(f'Reranking finished with res: {len(reranked_scores)}')
        # Attach scores to  candidates
        for item in reranked_scores:
            rerank_response.append({item["id"]: item["score"]})
        rerank_flat_dict = {k: v for d in rerank_response for k, v in d.items()}
        for candidate in candidates:
            dataset_id = candidate.get('id')
            reranked_score = rerank_flat_dict.get(dataset_id)
            candidate["rerank_score"] = reranked_score if reranked_score \
                else 0
        return sorted(candidates, key=lambda x: x["rerank_score"], reverse=True)

    async def find_similar_datasets(
        self, dataset_id: int, top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """Finds and reranks similar datasets."""

        target_dataset_info = self.db.find_one("datasets", {"id": dataset_id})
        if not target_dataset_info:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND, detail="Target dataset not found"
            )

        target_schemas = self.db.find("dataset_schemas", {"dataset_id": dataset_id})

        # Generate text and embedding for the target dataset
        text_content = self._create_dataset_text_content(
            target_dataset_info, target_schemas
        )
        target_embedding = self.embedding_service.get_embedding(text_content)

        # Fetch more results than needed for reranking
        fetch_k = top_k * 4
        initial_results = await self.vector_store.similarity_search_by_vector(
            vector=target_embedding,
            k=fetch_k,
            filter_dict={
                "dataset_id": {"$ne": dataset_id}
            },
            namespace="datasets",
        )

        if not initial_results:
            return []

        # Rerank the results
        target_metadata_for_rerank = {
            "metadata": {
                "column_names": [s["column_name"] for s in target_schemas],
                "column_count": target_dataset_info["column_count"],
                "file_type": target_dataset_info["file_type"],
            }
        }
        reranked_results = await self._rerank_with_model(text_content, initial_results)

        return reranked_results[:top_k]

    async def recommend_analysis_techniques(
        self, dataset_id: int, top_k: int = 3
    ) -> List[Dict[str, Any]]:
        """Recommends analysis techniques based on a dataset's schema."""
        dataset = self.db.find_one("datasets", {"id": dataset_id})
        if not dataset:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND, detail="Dataset not found"
            )

        schemas = self.db.find("dataset_schemas", {"dataset_id": dataset_id})

        query_text = self._create_dataset_text_content(dataset, schemas)

        # Query the knowledge base
        recommendations = await self.vector_store.similarity_search(
            query=query_text, k=top_k, namespace="analysis-techniques"
        )

        return recommendations

    async def get_dataset_recommendations(
        self, user_query: str, top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """Get dataset recommendations based on user query"""
        # Generate embedding for the query
        embeddings = self.embedding_service.get_embeddings([user_query])
        results = await self.vector_store.similarity_search(query=user_query)

        return results
