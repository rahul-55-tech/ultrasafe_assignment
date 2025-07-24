from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel


class DatasetBase(BaseModel):
    name: str
    description: Optional[str] = None


class DatasetCreate(DatasetBase):
    pass


class DatasetUpdate(BaseModel):
    name: Optional[str] = None
    description: Optional[str] = None


class DatasetSchemaResponse(BaseModel):
    id: int
    column_name: str
    data_type: str
    null_count: int
    unique_count: Optional[int] = None
    sample_values: Optional[List[Any]] = None
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    mean_value: Optional[float] = None
    std_value: Optional[float] = None

    class Config:
        from_attributes = True


class DatasetResponse(DatasetBase):
    id: int
    file_path: str
    file_size: Optional[int] = None
    file_type: Optional[str] = None
    row_count: Optional[int] = None
    column_count: Optional[int] = None
    status: str
    processing_progress: float
    created_at: datetime
    updated_at: Optional[datetime] = None

    class Config:
        from_attributes = True


class UploadDatasetResponse(DatasetBase):
    id: int
    file_path: str
    file_size: Optional[int] = None
    file_type: Optional[str] = None
    status: str
    created_at: datetime
    updated_at: Optional[datetime] = None

class DatasetUploadResponse(BaseModel):
    dataset_id: int
    message: str
    status: str
