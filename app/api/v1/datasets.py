import hashlib
import os
from datetime import datetime
from typing import Any, Dict, List, Optional

from fastapi import (APIRouter, BackgroundTasks, Depends, File, HTTPException,
                     Request, UploadFile, status)

from app.core.rag.retrieval_service import RetrievalService
from app.json_db import JSONDatabase, get_db
from app.schemas.dataset import DatasetResponse, DatasetSchemaResponse, UploadDatasetResponse
from app.services.data_processor import DataProcessor
from app.middleware.auth import require_auth, security
from fastapi.security import HTTPAuthorizationCredentials
from enum import Enum
from fastapi.responses import FileResponse
from app.utils.common_utility import process_dataset_background




router = APIRouter()


class ExportFormat(str, Enum):
    csv = "csv"
    json = "json"
    excel = "excel"
    parquet = "parquet"



@router.post("/upload", response_model=UploadDatasetResponse)
@require_auth()
async def upload_dataset(
    request: Request,
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    name: Optional[str] = None,
    description: Optional[str] = None,
    db: JSONDatabase = Depends(get_db),
    credentials: HTTPAuthorizationCredentials = Depends(security),

):
    """
    This endpoint allows uploading a dataset file.
    Once uploads, it will process the dataset in the background, and store
    the embedding in the pine cone database.
    """
    allowed_extensions = [".csv", ".xlsx", ".xls", ".json", ".parquet"]
    file_extension = os.path.splitext(file.filename)[1].lower()

    if file_extension not in allowed_extensions:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"File type not supported. Allowed types: {', '.join(allowed_extensions)}",
        )

    upload_dir = "uploads"
    os.makedirs(upload_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    sanitized_filename = os.path.basename(file.filename)
    filename = f"{timestamp}_{sanitized_filename}"
    file_path = os.path.join(upload_dir, filename)

    try:
        with open(file_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to save file: {str(e)}",
        )

    dataset_name = name or os.path.splitext(file.filename)[0]
    dataset_data = {
        "name": dataset_name,
        "description": description,
        "file_path": file_path,
        "file_size": len(content),
        "file_type": file_extension,
        "status": "uploaded",
        "processing_progress": 0.0,
        "created_at": datetime.utcnow(),
        "updated_at": datetime.utcnow(),
    }

    dataset = db.insert("datasets", dataset_data)
    # call the background task to process the dataset
    background_tasks.add_task(
        process_dataset_background, dataset["id"], file_path, file_extension, db
    )

    return dataset


@router.get("/", response_model=List[DatasetResponse])
@require_auth()
async def list_datasets(
    request: Request,
    db: JSONDatabase = Depends(get_db),
    credentials: HTTPAuthorizationCredentials = Depends(security)
):
    """This endpoint lists all datasets available in the database"""
    datasets = db.find("datasets")
    return datasets


@router.get("/{dataset_id}", response_model=DatasetResponse)
@require_auth()
async def get_dataset(
    request: Request,
    dataset_id: int, 
    db: JSONDatabase = Depends(get_db),
    credentials: HTTPAuthorizationCredentials = Depends(security)
):
    """
    This endpoint retrieves specific dataset by its ID.
    """
    dataset = db.find_one("datasets", {"id": dataset_id})
    if not dataset:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Dataset not found"
        )
    return dataset


@router.get("/{dataset_id}/schema", response_model=List[DatasetSchemaResponse])
@require_auth()
async def get_dataset_schema(
    request: Request,
    dataset_id: int, 
    db: JSONDatabase = Depends(get_db),
    credentials: HTTPAuthorizationCredentials = Depends(security)
):
    """Get dataset schema information."""
    schemas = db.find("dataset_schemas", {"dataset_id": dataset_id})
    if not schemas:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Dataset schema not found"
        )
    return schemas


@router.get("/{dataset_id}/similar", response_model=List[Dict[str, Any]])
@require_auth()
async def find_similar_datasets_endpoint(
    request: Request,
    dataset_id: int, 
    top_k: int = 5, 
    db: JSONDatabase = Depends(get_db),
    credentials: HTTPAuthorizationCredentials = Depends(security)
):
    """
    This APi used to find the datasets that are semantically similar to the
    given dataset,
    using a vector search followed by a reranking step.
    """
    retrieval_service = RetrievalService(db=db)
    similar_datasets = await retrieval_service.find_similar_datasets(
        dataset_id=dataset_id, top_k=top_k
    )
    return similar_datasets




@router.get("/{dataset_id}/export")
@require_auth()
async def export_dataset(
    request: Request,
    dataset_id: int,
    format: ExportFormat,
    background_tasks: BackgroundTasks,
    db: JSONDatabase = Depends(get_db),
    credentials: HTTPAuthorizationCredentials = Depends(security)
):
    """
    This api used to export the dataset in various file format
    """
    dataset = db.find_one("datasets", {"id": dataset_id})
    if not dataset:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Dataset not found"
        )
    processor = DataProcessor()
    try:
        data = await processor.load_data(dataset["file_path"], dataset["file_type"])
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to load original dataset file: {e}"
        )

    #  Provide the path and filename for the exported file
    export_dir = "exports"
    os.makedirs(export_dir, exist_ok=True)
    base_filename = "".join(c for c in dataset['name'] if c.isalnum() or c in (' ', '_')).rstrip()
    export_filename = f"{base_filename}.{format.value}"
    export_file_path = os.path.join(export_dir, export_filename)

    # call export data method from data processor
    try:
        await processor.export_data(data, file_format=format.value,
                                    file_path = export_file_path)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to export data to {format.value}: {e}"
        )

    # call the background task to delete the temporary file from the system.
    return FileResponse(
        path=export_file_path,
        filename=export_filename,
        media_type='application/octet-stream',
        background=background_tasks.add_task(os.remove, export_file_path)
    )



