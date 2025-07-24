import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class AgentWorkflow(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str
    description: Optional[str] = None

    # Workflow configuration
    workflow_type: (
        str  # data_exploration, statistical_analysis, visualization, insight_generation
    )
    agents: Optional[List[Dict[str, Any]]] = []
    workflow_config: Optional[Dict[str, Any]] = {}

    # Status
    status: str = "pending"  # pending, running, completed, failed
    progress: float = 0.0

    # Results
    final_results: Optional[Dict[str, Any]] = {}
    insights: Optional[Dict[str, Any]] = {}

    # Relationships
    dataset_id: Optional[str] = None
    user_id: Optional[str] = None

    # Timestamps
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None


class AgentTask(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    workflow_id: str  # Reference to AgentWorkflow.id

    # Task details
    agent_name: str
    task_type: str
    task_description: Optional[str] = None

    # Status
    status: str = "pending"  # pending, running, completed, failed
    progress: float = 0.0

    # Input/Output
    input_data: Optional[Dict[str, Any]] = {}
    output_data: Optional[Dict[str, Any]] = {}
    error_message: Optional[str] = None

    # Execution details
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    execution_time: Optional[float] = None  # in seconds

    # Timestamps
    created_at: datetime = Field(default_factory=datetime.utcnow)
