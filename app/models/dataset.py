import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class Dataset(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str
    description: Optional[str] = None
    file_path: str
    file_size: Optional[int] = None
    file_type: Optional[str] = None
    row_count: Optional[int] = None
    column_count: Optional[int] = None

    # Processing status
    status: str = "uploaded"  # uploaded, processing, processed, error
    processing_progress: float = 0.0

    # Metadata
    dataset_metadata: Optional[Dict[str, Any]] = None
    schema_hash: Optional[str] = None

    # Relationships
    user_id: Optional[str] = None  # We'll store this as a UUID or string

    # Timestamps
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)

    # Vector embedding ID for RAG
    embedding_id: Optional[str] = None


class DatasetSchema(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    dataset_id: str  # Link back to Dataset.id

    # Schema information
    column_name: str
    data_type: str
    null_count: int = 0
    unique_count: Optional[int] = None
    sample_values: Optional[List[Any]] = None

    # Statistical information
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    mean_value: Optional[float] = None
    std_value: Optional[float] = None

    # Timestamps
    created_at: datetime = Field(default_factory=datetime.utcnow)
