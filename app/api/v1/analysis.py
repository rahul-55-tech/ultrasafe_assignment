import datetime
import os
from typing import Any, Dict, List

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException,Request,status
from fastapi.security import HTTPAuthorizationCredentials

from app.core.agents.agents import (DataExplorationAgent,
                                    InsightGenerationAgent,
                                    StatisticalAnalysisAgent,
                                    VisualizationAgent)
from app.json_db import JSONDatabase, get_db
from app.schemas.analysis import (AnalysisCreate, AnalysisResponse,
                                  AnalysisResultResponse)
from app.services.data_processor import DataProcessor
from app.middleware.auth import require_auth, security
from app.core.rag.retrieval_service import RetrievalService
from configurations.logger_config import logger

router = APIRouter()


@router.post("/", response_model=AnalysisResponse)
@require_auth()
async def create_analysis(
    request: Request,
    analysis: AnalysisCreate,
    background_tasks: BackgroundTasks,
    db: JSONDatabase = Depends(get_db),
    credentials: HTTPAuthorizationCredentials = Depends(security),
):
    """
    This endpoint allows you to create the different type of analysis fron
    the agent based on provided analysis_type.
    The supported analysis type is: [
                "exploration",
                "statistical",
                "visualization",
                "insights"

    """
    allowed_analysis_type =  [
                "exploration",
                "statistical",
                "visualization",
                "insights"]
    if analysis.analysis_type  not in allowed_analysis_type:
        raise HTTPException(status_code=400, detail=f"Please provide the "
                                                    f"analysis_type from "
                                                    f"the list :{allowed_analysis_type}")


    dataset = db.find_one("datasets", {"id": analysis.dataset_id})
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")

    db_analysis = {
        "name": analysis.name,
        "description": analysis.description,
        "analysis_type": analysis.analysis_type,
        "parameters": analysis.parameters,
        "dataset_id": analysis.dataset_id,
        "status": "pending",
        "progress": 0.0,
        "summary": "",
        "insights": {},
        "created_at":datetime.datetime.now()
    }

    db_analysis = db.insert("analysis", db_analysis)

    # Run analysis in background
    background_tasks.add_task(
        run_analysis_background,
        db_analysis["id"],
        dataset["file_path"],
        analysis.analysis_type,
        analysis.parameters,
        db
    )
    return db_analysis


@router.get("/", response_model=List[AnalysisResponse])
@require_auth()
async def list_analyss(
    request: Request,
    db: JSONDatabase = Depends(get_db),
    credentials: HTTPAuthorizationCredentials = Depends(security)
):
    return db.read_table("analysis")


@router.get("/{analysis_id}", response_model=AnalysisResponse)
@require_auth()
async def get_analysis(
    request: Request,
    analysis_id: int, 
    db: JSONDatabase = Depends(get_db),
    credentials: HTTPAuthorizationCredentials = Depends(security)
):
    """
    This api allows you get the analysis details for an specific dataset_id
    """
    analysis = db.find_one("analysis", {"id": analysis_id})
    if not analysis:
        raise HTTPException(status_code=404, detail="Analysis Details not "
                                                    "found")
    return analysis


@router.get("/{analysis_id}/results", response_model=List[AnalysisResultResponse])
@require_auth()
async def get_analysis_results(
    request: Request,
    analysis_id: int, 
    db: JSONDatabase = Depends(get_db),
    credentials: HTTPAuthorizationCredentials = Depends(security)
):
    """
    This API allow to get the analysis results
    """
    return db.find("analysis_results", {"analysis_id": analysis_id})


@router.post("/{dataset_id}/explore")
@require_auth()
async def explore_dataset(
    request: Request,
    dataset_id: int, 
    db: JSONDatabase = Depends(get_db),
    credentials: HTTPAuthorizationCredentials = Depends(security)
):
    """
    This endpoints allow to get the exploration details for an specific
    dataset
    """

    dataset = db.find_one("datasets", {"id": dataset_id})
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")

    processor = DataProcessor()
    data = await processor.load_data(dataset["file_path"], dataset["file_type"])

    agent = DataExplorationAgent()
    exploration_results = await agent.explore_dataset(data)

    return {"dataset_id": dataset_id, "exploration_results": exploration_results}


@router.post("/{dataset_id}/statistics")
@require_auth()
async def analyze_statistics(
    request: Request,
    dataset_id: int, 
    db: JSONDatabase = Depends(get_db),
    credentials: HTTPAuthorizationCredentials = Depends(security)
):
    """
    This APi will provide you the statistic analysis for specific dataset
    """
    dataset = db.find_one("datasets", {"id": dataset_id})
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")

    processor = DataProcessor()
    data = await processor.load_data(dataset["file_path"], dataset["file_type"])

    agent = StatisticalAnalysisAgent()
    results = await agent.analyze_statistics(data)

    return {"dataset_id": dataset_id, "statistical_results": results}


@router.post("/{dataset_id}/visualize")
@require_auth()
async def generate_visualizations(
    request: Request,
    dataset_id: int, 
    db: JSONDatabase = Depends(get_db),
    credentials: HTTPAuthorizationCredentials = Depends(security)
):
    """
    This APi wll generate the visulation chart from an specific dataset
    and store it in the file system
    """
    dataset = db.find_one("datasets", {"id": dataset_id})
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")

    processor = DataProcessor()
    data = await processor.load_data(dataset["file_path"], dataset["file_type"])

    agent = VisualizationAgent()
    visualizations = await agent.generate_visualizations(data, {})

    return {"dataset_id": dataset_id, "visualization_results": visualizations}


@router.post("/{dataset_id}/insights")
@require_auth()
async def generate_insights(
    request: Request,
    dataset_id: int, 
    db: JSONDatabase = Depends(get_db),
    credentials: HTTPAuthorizationCredentials = Depends(security)
):
    """
    This APi will give you insights details for the provided dataset
    """
    dataset = db.find_one("datasets", {"id": dataset_id})
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")

    processor = DataProcessor()
    data = await processor.load_data(dataset["file_path"], dataset["file_type"])

    agent = InsightGenerationAgent()
    insights = await agent.generate_insights(data, {})

    return {"dataset_id": dataset_id, "insight_results": insights}


async def run_analysis_background(
    analysis_id: int,
    dataset_path: str,
    analysis_type: str,
    parameters: Dict[str, Any],
    db: JSONDatabase
):
    """Background task to run analysis"""
    try:
        logger.info('started run analysis in background')
        analysis = db.find_one("analysis", {"id": analysis_id})
        if not analysis:
            return

        db.update(
            "analysis", {"id": analysis_id}, {"status": "running", "progress": 0.1}
        )

        processor = DataProcessor()
        data = await processor.load_data(
            dataset_path, os.path.splitext(dataset_path)[1]
        )
        db.update("analysis", {"id": analysis_id}, {"progress": 0.3})

        # Choose agent based on the provided analysis type
        if analysis_type == "exploration":
            agent = DataExplorationAgent()
            results = await agent.explore_dataset(data)
        elif analysis_type == "statistical":
            agent = StatisticalAnalysisAgent()
            results = await agent.analyze_statistics(data)
        elif analysis_type == "visualization":
            agent = VisualizationAgent()
            results = await agent.generate_visualizations(data, parameters)
        elif analysis_type == "insights":
            agent = InsightGenerationAgent()
            results = await agent.generate_insights(data, parameters)
        else:
            raise ValueError(f"Unknown analysis type: {analysis_type}")

        db.update("analysis", {"id": analysis_id}, {"progress": 0.8})

        result = {
            "analysis_id": analysis_id,
            "result_type": analysis_type,
            "title": f"{analysis_type.title()} Results",
            "data": results,
            "metadata": {"parameters": parameters},
        }
        db.insert("analysis_results", result)

        db.update(
            "analysis",
            {"id": analysis_id},
            {
                "status": "completed",
                "progress": 1.0,
                "summary": f"Successfully completed {analysis_type} analysis",
                "insights": {"status": "success", "results_count": 1},
            },
        )

    except Exception as e:
        db.update(
            "analysis",
            {"id": analysis_id},
            {
                "status": "failed",
                "summary": f"Analysis failed: {str(e)}",
                "insights": {"status": "error", "error": str(e)},
            },
        )


@router.post("/{dataset_id}/recommend_techniques", response_model=List[Dict[str, Any]])
@require_auth()
async def recommend_techniques(
    request: Request,
    dataset_id: int, 
    top_k: int = 3, 
    db: JSONDatabase = Depends(get_db),
    credentials: HTTPAuthorizationCredentials = Depends(security)
):
    """
    This endpoint will give you recommends relevant analysis techniques for a
    given dataset based on its
    schema, using a RAG system.
    """
    retrieval_service = RetrievalService(db=db)
    recommendations = await retrieval_service.recommend_analysis_techniques(
        dataset_id, top_k
    )
    return recommendations
