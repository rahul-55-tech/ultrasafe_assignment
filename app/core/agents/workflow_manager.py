import json
import numpy as np
from typing import Any, Dict, List, Optional, TypedDict

import pandas as pd
from langchain.schema import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph
from langgraph.prebuilt import ToolNode

from configurations.config import settings


def convert_numpy_types(obj):
    """Convert NumPy types to native Python types for JSON serialization"""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        # Handle infinite and NaN values
        if np.isinf(obj) or np.isnan(obj):
            return None
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    elif pd.isna(obj):  # Handle NaN values
        return None
    else:
        return obj


class NumpyEncoder(json.JSONEncoder):
    """Custom JSON encoder to handle NumPy types"""
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            # Handle infinite and NaN values
            if np.isinf(obj) or np.isnan(obj):
                return None
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif pd.isna(obj):
            return None
        return super().default(obj)
#
#
class WorkflowState(TypedDict, total=False):
    dataset_path: str
    data: Any  # Optional[pd.DataFrame], but Pandas isn't hashable/serializable, so use Any
    exploration_results: Dict[str, Any]
    statistical_results: Dict[str, Any]
    visualization_results: Dict[str, Any]
    insights: Dict[str, Any]
    current_step: str
    errors: List[str]
    progress: float
#
#
class WorkflowManager:
    """Manages multi-agent workflows using LangGraph"""

    def __init__(self):
        self.llm = ChatOpenAI(
            api_key=settings.openai_api_key, model=settings.LLM_MODEL, temperature=0.1
        )
        self.state_schema = WorkflowState  # Use TypedDict instead of raw dict

    def create_data_exploration_workflow(self) -> StateGraph:
        """Create a workflow for data exploration"""
        workflow = StateGraph(self.state_schema)

        workflow.add_node("explore_data", self._explore_data_node)
        workflow.add_node("analyze_statistics", self._analyze_statistics_node)
        workflow.add_node("generate_visualizations", self._generate_visualizations_node)
        workflow.add_node("generate_insights", self._generate_insights_node)

        workflow.set_entry_point("explore_data")
        workflow.add_edge("explore_data", "analyze_statistics")
        workflow.add_edge("analyze_statistics", "generate_visualizations")
        workflow.add_edge("generate_visualizations", "generate_insights")
        workflow.add_edge("generate_insights", END)

        return workflow.compile()

    def create_custom_workflow(self, steps: List[str]) -> StateGraph:
        """Create a custom workflow based on specified steps"""
        workflow = StateGraph(self.state_schema)

        # Validate and filter valid steps
        valid_steps = []
        for step in steps:
            if step in ["explore_data", "analyze_statistics", "generate_visualizations", "generate_insights"]:
                valid_steps.append(step)

        # If no valid steps provided, use default data exploration workflow
        if not valid_steps:
            return self.create_data_exploration_workflow()

        # Add nodes for valid steps
        for step in valid_steps:
            if step == "explore_data":
                workflow.add_node("explore_data", self._explore_data_node)
            elif step == "analyze_statistics":
                workflow.add_node("analyze_statistics", self._analyze_statistics_node)
            elif step == "generate_visualizations":
                workflow.add_node(
                    "generate_visualizations", self._generate_visualizations_node
                )
            elif step == "generate_insights":
                workflow.add_node("generate_insights", self._generate_insights_node)

        # Set entry point
        workflow.set_entry_point(valid_steps[0])

        # Add edges between steps
        for i in range(len(valid_steps) - 1):
            workflow.add_edge(valid_steps[i], valid_steps[i + 1])

        # Add final edge to END
        workflow.add_edge(valid_steps[-1], END)

        return workflow.compile()

    async def _explore_data_node(self, state: Dict[str, Any]) -> Dict[str, Any]:
        try:
            data = pd.read_csv(state["dataset_path"])
            state["data"] = data
            state["exploration_results"] = {
                "shape": data.shape,
                "columns": list(data.columns),
                "dtypes": data.dtypes.apply(str).to_dict(),
                "missing_values": convert_numpy_types(data.isnull().sum().to_dict()),
                "numeric_columns": list(data.select_dtypes(include=["number"]).columns),
                "categorical_columns": list(
                    data.select_dtypes(include=["object"]).columns
                ),
                "sample_data": convert_numpy_types(data.head().to_dict()),
            }
            state["current_step"] = "explore_data"
            state["progress"] = 0.25
        except Exception as e:
            state.setdefault("errors", []).append(f"Data exploration error: {str(e)}")
        return state

    async def _analyze_statistics_node(self, state: Dict[str, Any]) -> Dict[str, Any]:
        try:
            data = state["data"]
            if data is None:
                raise ValueError("No data available for analysis")
            numeric_cols = data.select_dtypes(include=["number"]).columns
            results = {
                col: {
                    "mean": convert_numpy_types(data[col].mean()),
                    "median": convert_numpy_types(data[col].median()),
                    "std": convert_numpy_types(data[col].std()),
                    "min": convert_numpy_types(data[col].min()),
                    "max": convert_numpy_types(data[col].max()),
                    "skewness": convert_numpy_types(data[col].skew()),
                    "kurtosis": convert_numpy_types(data[col].kurtosis()),
                }
                for col in numeric_cols
            }
            if len(numeric_cols) > 1:
                correlations = data[numeric_cols].corr().to_dict()
                results["correlations"] = convert_numpy_types(correlations)
            state["statistical_results"] = results
            state["current_step"] = "analyze_statistics"
            state["progress"] = 0.5
        except Exception as e:
            state.setdefault("errors", []).append(
                f"Statistical analysis error: {str(e)}"
            )
        return state

    async def _generate_visualizations_node(
        self, state: Dict[str, Any]
    ) -> Dict[str, Any]:
        try:
            data = state["data"]
            if data is None:
                raise ValueError("No data available for visualization")
            visualization_results = {"charts": [], "insights": []}
            for col in data.select_dtypes(include=["number"]).columns:
                visualization_results["charts"].append(
                    {
                        "type": "histogram",
                        "column": col,
                        "title": f"Distribution of {col}",
                        "file_path": f"charts/{col}_histogram.png",
                    }
                )
            state["visualization_results"] = visualization_results
            state["current_step"] = "generate_visualizations"
            state["progress"] = 0.75
        except Exception as e:
            state.setdefault("errors", []).append(f"Visualization error: {str(e)}")
        return state

    async def _generate_insights_node(self, state: Dict[str, Any]) -> Dict[str, Any]:
        try:
            # Convert all data to JSON-serializable format before creating prompt
            exploration_results = convert_numpy_types(state.get("exploration_results", {}))
            statistical_results = convert_numpy_types(state.get("statistical_results", {}))
            visualization_results = convert_numpy_types(state.get("visualization_results", {}))

            prompt = f"""
            Based on the following data analysis results, generate key insights:
            Data Exploration: {json.dumps(exploration_results, indent=2, cls=NumpyEncoder)}
            Statistical Analysis: {json.dumps(statistical_results, indent=2, cls=NumpyEncoder)}
            Visualizations: {json.dumps(visualization_results, indent=2, cls=NumpyEncoder)}
            Please provide:
            1. Key patterns and trends
            2. Anomalies or outliers
            3. Recommendations for further analysis
            4. Business implications
            """
            messages = [
                SystemMessage(
                    content="You are a data analyst expert. Provide clear, actionable insights."
                ),
                HumanMessage(content=prompt),
            ]
            response = await self.llm.ainvoke(messages)
            state["insights"] = {
                "summary": response.content,
                "key_findings": [],
                "recommendations": [],
                "business_implications": [],
            }
            state["current_step"] = "generate_insights"
            state["progress"] = 1.0
        except Exception as e:
            state.setdefault("errors", []).append(f"Insight generation error: {str(e)}")
        return state

    async def run_workflow(
        self, workflow, initial_state: Dict[str, Any]
    ) -> Dict[str, Any]:
        try:
            result = await workflow.ainvoke(initial_state)
            return result
        except Exception as e:
            initial_state.setdefault("errors", []).append(
                f"Workflow execution error: {str(e)}"
            )
            return initial_state
