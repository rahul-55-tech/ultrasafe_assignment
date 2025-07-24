import os
from datetime import datetime
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException,Request
from fastapi.security import HTTPAuthorizationCredentials

from app.core.agents.workflow_manager import WorkflowManager
from app.json_db import JSONDatabase, get_db
from app.schemas.agent import (AgentTaskResponse, AgentWorkflowCreate,
                               AgentWorkflowResponse)
from app.middleware.auth import require_auth, security

router = APIRouter()


@router.post("/workflows", response_model=AgentWorkflowResponse)
@require_auth()
async def create_workflow(
    request: Request,
    workflow: AgentWorkflowCreate,
    background_tasks: BackgroundTasks,
    db: JSONDatabase = Depends(get_db),
    credentials: HTTPAuthorizationCredentials = Depends(security),
):
    """
    This endpoint will help you to create the workflow in the background
    """
    dataset = db.find_one("datasets", {"id": workflow.dataset_id})
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")

    workflow_id = db.generate_id("agent_workflows")
    db_workflow = {
        "id": workflow_id,
        "name": workflow.name,
        "description": workflow.description,
        "workflow_type": workflow.workflow_type,
        "agents": workflow.agents,
        "workflow_config": workflow.workflow_config,
        "dataset_id": workflow.dataset_id,
        "status": "pending",
        "progress": 0.0,
        "results": {},
        "summary": "",
        "created_at": datetime.utcnow().isoformat(),
    }

    db.insert("agent_workflows", db_workflow)

    background_tasks.add_task(
        run_workflow_background,
        workflow_id,
        dataset["file_path"],
        workflow.workflow_type,
        workflow.agents,
        workflow.workflow_config,
        db,
    )

    return db_workflow


@router.get("/workflows", response_model=List[AgentWorkflowResponse])
@require_auth()
async def list_workflows(
    request: Request,
    db: JSONDatabase = Depends(get_db),
    credentials: HTTPAuthorizationCredentials = Depends(security)
):
    """
    This endpoint will provide you the details of workflow list
    """
    workflows = db.find("agent_workflows")
    return workflows


@router.get("/workflows/{workflow_id}", response_model=AgentWorkflowResponse)
@require_auth()
async def get_workflow(
    request: Request,
    workflow_id: int, 
    db: JSONDatabase = Depends(get_db),
    credentials: HTTPAuthorizationCredentials = Depends(security)
):
    """
    This endpoint will allow getting specific workflow based on workflow id
    """
    workflow = db.find_one("agent_workflows", {"id": workflow_id})
    if not workflow:
        raise HTTPException(status_code=404, detail="Workflow not found")
    return workflow


@router.get("/workflows/{workflow_id}/tasks", response_model=List[AgentTaskResponse])
@require_auth()
async def get_workflow_tasks(
    request: Request,
    workflow_id: int, 
    db: JSONDatabase = Depends(get_db),
    credentials: HTTPAuthorizationCredentials = Depends(security)
):
    tasks = db.find("agent_tasks", {"workflow_id": workflow_id})
    return tasks


@router.get("/workflows/{workflow_id}/status")
@require_auth()
async def get_workflow_status(
    request: Request,
    workflow_id: int, 
    db: JSONDatabase = Depends(get_db),
    credentials: HTTPAuthorizationCredentials = Depends(security)
):
    """ This endpoint allows you get the status of the workflow """
    workflow = db.find_one("agent_workflows", {"id": workflow_id})
    if not workflow:
        raise HTTPException(status_code=404, detail="Workflow not found")

    tasks = db.find("agent_tasks", {"workflow_id": workflow_id})

    return {
        "workflow_id": workflow_id,
        "status": workflow["status"],
        "progress": workflow["progress"],
        "total_tasks": len(tasks),
        "completed_tasks": len([t for t in tasks if t["status"] == "completed"]),
        "failed_tasks": len([t for t in tasks if t["status"] == "failed"]),
        "tasks": tasks,
    }


async def run_workflow_background(
    workflow_id: int,
    dataset_path: str,
    workflow_type: str,
    agents: List[str],
    workflow_config: Dict[str, Any],
    db: JSONDatabase,
):
    try:
        workflow = db.find_one("agent_workflows", {"id": workflow_id})
        if not workflow:
            return

        workflow["status"] = "running"
        workflow["progress"] = 0.1
        db.update("agent_workflows", {"id": workflow_id}, workflow)

        workflow_manager = WorkflowManager()
        if workflow_type == "data_exploration":
            graph = workflow_manager.create_data_exploration_workflow()
        elif workflow_type == "custom":
            graph = workflow_manager.create_custom_workflow(
                workflow_config.get("steps", [])
            )
        else:
            raise ValueError(f"Unknown workflow type: {workflow_type}")

        workflow["progress"] = 0.2
        db.update("agent_workflows", {"id": workflow_id}, workflow)

        initial_state = {
            "dataset_path": dataset_path,
            "data": None,
            "exploration_results": None,
            "statistical_results": None,
            "visualization_results": None,
            "insights": None,
            "current_step": "initialized",
            "errors": [],
            "progress": 0.0,
        }

        result = await workflow_manager.run_workflow(graph, initial_state)

        workflow["progress"] = 0.9
        db.update("agent_workflows", {"id": workflow_id}, workflow)

        for step_name, agent_name in [
            ("exploration_results", "DataExplorationAgent"),
            ("statistical_results", "StatisticalAnalysisAgent"),
            ("visualization_results", "VisualizationAgent"),
            ("insights", "InsightGenerationAgent"),
        ]:
            if step_name in result:
                task = {
                    "id": db.generate_id("agent_tasks"),
                    "workflow_id": workflow_id,
                    "agent_name": agent_name,
                    "task_type": step_name,
                    "status": "completed",
                    "progress": 1.0,
                    "output_data": result[step_name],
                    "started_at": datetime.utcnow().isoformat(),
                    "completed_at": datetime.utcnow().isoformat(),
                }
                db.insert("agent_tasks", task)

        workflow["status"] = "completed"
        workflow["progress"] = 1.0
        workflow["results"] = result
        workflow["summary"] = f"Successfully completed {workflow_type} workflow"
        db.update("agent_workflows", {"id": workflow_id}, workflow)

    except Exception as e:
        workflow = db.find_one("agent_workflows", {"id": workflow_id})
        if workflow:
            workflow["status"] = "failed"
            workflow["summary"] = f"Workflow failed: {str(e)}"
            workflow["results"] = {"error": str(e)}
            db.update("agent_workflows", {"id": workflow_id}, workflow)
