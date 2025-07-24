import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class Analysis(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str
    description: Optional[str] = None

    # Configuration
    analysis_type: str  # "basic", "statistical", "ml", "agent"
    parameters: Optional[Dict[str, Any]] = {}

    # Status
    status: str = "pending"  # pending, running, completed, failed
    progress: float = 0.0

    # Results summary
    summary: Optional[str] = None
    insights: Optional[Dict[str, Any]] = {}

    # Relationships
    dataset_id: Optional[str] = None
    user_id: Optional[str] = None

    # Timestamps
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None


class AnalysisResult(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    analysis_id: str  # Reference to Analysis.id

    # Result details
    result_type: str  # chart, table, text, json
    title: Optional[str] = None
    description: Optional[str] = None

    # Data
    data: Optional[Dict[str, Any]] = {}
    file_path: Optional[str] = None

    # Metadata
    result_metadata: Optional[Dict[str, Any]] = {}

    # Timestamp
    created_at: datetime = Field(default_factory=datetime.utcnow)
