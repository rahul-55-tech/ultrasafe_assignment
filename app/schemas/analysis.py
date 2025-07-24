from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel


class AnalysisBase(BaseModel):
    name: str
    description: Optional[str] = None
    analysis_type: str
    parameters: Optional[Dict[str, Any]] = None


class AnalysisCreate(AnalysisBase):
    dataset_id: int


class AnalysisUpdate(BaseModel):
    name: Optional[str] = None
    description: Optional[str] = None
    parameters: Optional[Dict[str, Any]] = None


class AnalysisResultResponse(BaseModel):
    id: int
    result_type: str
    title: Optional[str] = None
    description: Optional[str] = None
    data: Optional[Dict[str, Any]] = None
    file_path: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None

    class Config:
        from_attributes = True


class AnalysisResponse(AnalysisBase):
    id: int
    status: str
    progress: float
    summary: Optional[str] = None
    insights: Optional[Dict[str, Any]] = None
    dataset_id: int
    created_at: Optional[datetime] = None

    class Config:
        from_attributes = True


class AnalysisRequest(BaseModel):
    dataset_id: int
    analysis_type: str
    parameters: Optional[Dict[str, Any]] = None
