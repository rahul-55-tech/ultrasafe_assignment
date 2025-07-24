import hashlib
from datetime import datetime
from app.core.rag.retrieval_service import RetrievalService
from app.json_db import JSONDatabase
from app.services.data_processor import DataProcessor
from configurations.config import settings
from langchain_openai import ChatOpenAI

import httpx

from typing import List, Dict
from configurations.config import settings


class UltrasafeLLMClient:
    def __init__(self, model: str = "usf1-mini", temperature: float = 0.7):
        self.api_key = settings.USF_OPENAPIKEY
        self.model = model
        self.temperature = temperature
        self.url = "https://api.us.inc/usf/v1/hiring/chat/completions"

    async def ainvoke(self, messages: List[Dict[str, str]]) -> str:
        headers = {
            "Authorization": self.api_key,
            "Content-Type": "application/json",
            "User-Agent": "python-requests/2.31.0",  # Mimic requests
            "Accept": "*/*"
        }

        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "web_search": True,
            "stream": False,
            "max_tokens": 1000
        }

        async with httpx.AsyncClient(follow_redirects=True,
                                     timeout=30.0) as client:
            response = await client.post(self.url, headers=headers,
                                         json=payload)
            response.raise_for_status()
            result = response.json()
            return result["choices"][0]["message"]["content"]


def load_llm():
    if settings.CONFIGURED_LLM == "ultrasafe":
        return UltrasafeLLMClient()
    else:
        return ChatOpenAI(
            api_key=settings.openai_api_key,
            model=settings.LLM_MODEL,
            temperature=0.1
        )


async def process_dataset_background(
    dataset_id: int, file_path: str, file_type: str, db: JSONDatabase
):
    """Background task to process uploaded dataset."""
    try:
        db.update(
            "datasets",
            {"id": dataset_id},
            {"status": "processing", "processing_progress": 0.1},
        )

        processor = DataProcessor()
        data = await processor.load_data(file_path, file_type)
        db.update("datasets", {"id": dataset_id}, {"processing_progress": 0.3})

        schema_info = await processor.analyze_schema(data)
        db.update("datasets", {"id": dataset_id}, {"processing_progress": 0.6})

        for col_info in schema_info:
            schema_data = {"dataset_id": dataset_id, **col_info}
            db.insert("dataset_schemas", schema_data)

        schema_hash = hashlib.md5(str(schema_info).encode()).hexdigest()

        dataset_update = {
            "row_count": len(data),
            "column_count": len(data.columns),
            "status": "processed",
            "processing_progress": 1.0,
            "schema_hash": schema_hash,
            "updated_at": datetime.utcnow(),
        }
        db.update("datasets", {"id": dataset_id}, dataset_update)

        dataset = db.find_one("datasets", {"id": dataset_id})
        schemas = db.find("dataset_schemas", {"dataset_id": dataset_id})

        # Pass the db instance to the retrieval service
        retrieval_service = RetrievalService(db=db)
        await retrieval_service.index_dataset(dataset, schemas)

    except Exception as e:
        db.update(
            "datasets",
            {"id": dataset_id},
            {
                "status": "error",
                "processing_progress": 1.0,
                "metadata": {"error": str(e)},
            },
        )
