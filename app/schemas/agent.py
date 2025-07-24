from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel


class AgentWorkflowBase(BaseModel):
    name: str
    description: Optional[str] = None
    workflow_type: str
    agents: List[str]
    workflow_config: Optional[Dict[str, Any]] = None


class AgentWorkflowCreate(AgentWorkflowBase):
    dataset_id: int


class AgentTaskResponse(BaseModel):
    id: int
    agent_name: str
    task_type: str
    task_description: Optional[str] = None
    status: str
    progress: float
    input_data: Optional[Dict[str, Any]] = None
    output_data: Optional[Dict[str, Any]] = None
    error_message: Optional[str] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    execution_time: Optional[float] = None
    created_at: Optional[datetime] = None

    class Config:
        from_attributes = True


class AgentWorkflowResponse(AgentWorkflowBase):
    id: int
    status: str
    progress: float
    final_results: Optional[Dict[str, Any]] = None
    insights: Optional[Dict[str, Any]] = None
    dataset_id: int
    created_at: datetime
    tasks: Optional[List[AgentTaskResponse]] = []

    class Config:
        from_attributes = True


class AgentAnalysisRequest(BaseModel):
    dataset_id: int
    workflow_type: str
    agents: Optional[List[str]] = None
    workflow_config: Optional[Dict[str, Any]] = None
