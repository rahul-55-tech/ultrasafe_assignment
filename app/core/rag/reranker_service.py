from typing import Dict, List

from pinecone import Pinecone
from configurations.config import settings


class RerankerService:
    def __init__(self):
        api_key = settings.pinecone_api_key
        self.pc = Pinecone(api_key=api_key)

    def rerank(
        self, query: str, documents: List[Dict[str, str]], threshold: float = 0.5
    ) -> List[Dict[str, float]]:
        result = self.pc.inference.rerank(
            model="bge-reranker-v2-m3",
            query=query,
            documents=documents,
            top_n=len(documents),
            return_documents=True,
            parameters={"truncate": "END"},
        )

        return [
            {"id": item["document"]["id"], "score": round(item["score"],
                                                          2) if item[
                "score"] else item["score"] }
            for item in result.data
            if item["score"] >= threshold
        ]
