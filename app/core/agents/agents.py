import json
from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
from langchain.schema import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from app.utils.common_utility import load_llm

from configurations.config import settings


class DataExplorationAgent:
    """Agent for data exploration and profiling"""

    def __init__(self):
        self.llm = load_llm()

    async def explore_dataset(self, data: pd.DataFrame) -> Dict[str, Any]:
        exploration_results = {
            "basic_info": {
                "num_rows": int(data.shape[0]),
                "num_columns": int(data.shape[1]),
                "column_names": list(data.columns),
            },
            "column_types": data.dtypes.astype(str).to_dict(),
            "missing_values": data.isnull().sum().to_dict(),
            "missing_percentage": (
                        (data.isnull().sum() / len(data)) * 100).to_dict(),
            "summary_statistics": data.describe(include="all").to_dict(),
            "sample_data": {
                "head": data.head().to_dict(),
                "tail": data.tail().to_dict(),
                "random_sample": data.sample(min(5, len(data))).to_dict(),
            },
        }

        return self.clean_nans(exploration_results)

    @staticmethod
    def clean_nans(obj):
        if isinstance(obj, dict):
            return {k: DataExplorationAgent.clean_nans(v) for k, v in
                    obj.items()}
        elif isinstance(obj, list):
            return [DataExplorationAgent.clean_nans(x) for x in obj]
        elif isinstance(obj, float):
            if np.isnan(obj) or np.isinf(obj):
                return None
            return float(obj)
        elif isinstance(obj, (np.floating, np.integer)):
            if np.isnan(obj) or np.isinf(obj):
                return None
            return obj.item()
        return obj

    async def explore_dataset(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Perform comprehensive data exploration"""
        try:
            basic_info = {
                "shape": data.shape,
                "columns": list(data.columns),
                "dtypes": {col: str(dtype) for col, dtype in
                           data.dtypes.items()},
                "memory_usage": int(data.memory_usage(deep=True).sum()),
            }

            missing_analysis = {
                "total_missing": data.isnull().sum().sum(),
                "missing_per_column": data.isnull().sum().to_dict(),
                "missing_percentage": (
                            data.isnull().sum() / len(data) * 100).to_dict(),
            }

            numeric_cols = data.select_dtypes(include=["number"]).columns
            categorical_cols = data.select_dtypes(include=["object"]).columns
            datetime_cols = data.select_dtypes(include=["datetime"]).columns

            type_analysis = {
                "numeric_columns": list(numeric_cols),
                "categorical_columns": list(categorical_cols),
                "datetime_columns": list(datetime_cols),
            }

            sample_data = {
                "head": data.head().to_dict(),
                "tail": data.tail().to_dict(),
                "random_sample": data.sample(min(5, len(data))).to_dict(),
            }

            duplicate_analysis = {
                "total_duplicates": data.duplicated().sum(),
                "duplicate_percentage": (
                            data.duplicated().sum() / len(data) * 100),
            }

            insights = await self._generate_exploration_insights(
                basic_info, missing_analysis, type_analysis,
                duplicate_analysis
            )

            return self.clean_nans(
                {
                    "basic_info": basic_info,
                    "missing_analysis": missing_analysis,
                    "type_analysis": type_analysis,
                    "sample_data": sample_data,
                    "duplicate_analysis": duplicate_analysis,
                    "insights": insights,
                }
            )

        except Exception as e:
            return {"error": str(e)}



    async def _generate_exploration_insights(
        self, basic_info, missing_analysis, type_analysis, duplicate_analysis
    ):
        """Generate insights about the data exploration results"""
        prompt = f"""
        Based on the following data exploration results, provide key insights:
        
        Basic Information:
        - Shape: {basic_info['shape']}
        - Columns: {basic_info['columns']}
        - Data Types: {basic_info['dtypes']}
        
        Missing Values:
        - Total Missing: {missing_analysis['total_missing']}
        - Missing by Column: {missing_analysis['missing_per_column']}
        
        Data Types:
        - Numeric Columns: {type_analysis['numeric_columns']}
        - Categorical Columns: {type_analysis['categorical_columns']}
        - Datetime Columns: {type_analysis['datetime_columns']}
        
        Duplicates:
        - Total Duplicates: {duplicate_analysis['total_duplicates']}
        - Duplicate Percentage: {duplicate_analysis['duplicate_percentage']:.2f}%
        
        Please provide:
        1. Data quality assessment
        2. Potential issues to address
        3. Recommendations for data cleaning
        4. Suggestions for analysis approach
        """

        messages = [
            {
                "role": "system",
                "content": "You are a data quality expert. Provide clear, actionable insights about data exploration results."},
            {"role": "user", "content": prompt}
        ]
        response = await self.llm.ainvoke(messages)
        return response


def convert_numpy(obj):
    """Recursively convert numpy types to native Python types"""
    import numpy as np

    if isinstance(obj, dict):
        return {k: convert_numpy(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy(i) for i in obj]
    elif isinstance(obj, (np.integer, np.int64, np.int32)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64, np.float32)):
        return float(obj)
    elif isinstance(obj, (np.bool_)):
        return bool(obj)
    elif isinstance(obj, (np.ndarray,)):
        return obj.tolist()
    else:
        return obj


class StatisticalAnalysisAgent:
    """Agent for statistical analysis"""

    def __init__(self):
        self.llm = ChatOpenAI(
            api_key=settings.openai_api_key, model=settings.LLM_MODEL, temperature=0.1
        )

    @staticmethod
    def convert_numpy(obj):
        if isinstance(obj, dict):
            return {k: StatisticalAnalysisAgent.convert_numpy(v) for k, v in
                    obj.items()}
        elif isinstance(obj, list):
            return [StatisticalAnalysisAgent.convert_numpy(x) for x in obj]
        elif isinstance(obj, float):
            if np.isnan(obj) or np.isinf(obj):
                return None
            return float(obj)
        elif isinstance(obj, (np.integer, np.floating)):
            if np.isnan(obj) or np.isinf(obj):
                return None
            return obj.item()
        elif isinstance(obj, (np.bool_, bool)):
            return bool(obj)
        elif isinstance(obj, pd.Series):
            return obj.to_dict()
        elif isinstance(obj, pd.DataFrame):
            return obj.to_dict()
        return obj

    async def analyze_statistics(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Perform comprehensive statistical analysis"""
        try:
            numeric_cols = data.select_dtypes(include=["number"]).columns

            if len(numeric_cols) == 0:
                return {"error": "No numeric columns found for statistical analysis"}

            # Descriptive statistics
            descriptive_stats = {}
            for col in numeric_cols:
                descriptive_stats[col] = {
                    "count": data[col].count(),
                    "mean": data[col].mean(),
                    "median": data[col].median(),
                    "std": data[col].std(),
                    "min": data[col].min(),
                    "max": data[col].max(),
                    "q25": data[col].quantile(0.25),
                    "q75": data[col].quantile(0.75),
                    "skewness": data[col].skew(),
                    "kurtosis": data[col].kurtosis(),
                }

            # Correlation analysis
            correlation_analysis = {}
            if len(numeric_cols) > 1:
                correlation_matrix = data[numeric_cols].corr()
                correlation_analysis = {
                    "correlation_matrix": correlation_matrix.to_dict(),
                    "high_correlations": self._find_high_correlations(
                        correlation_matrix
                    ),
                }

            # Outlier analysis
            outlier_analysis = {}
            for col in numeric_cols:
                outliers = self._detect_outliers(data[col])
                outlier_analysis[col] = {
                    "outlier_count": len(outliers),
                    "outlier_percentage": (len(outliers) / len(data) * 100),
                    "outlier_indices": outliers.index.tolist(),
                }

            # Distribution analysis
            distribution_analysis = {}
            for col in numeric_cols:
                distribution_analysis[col] = {
                    "is_normal": self._test_normality(data[col]),
                    "distribution_type": self._classify_distribution(data[col]),
                }

            # Generate insights
            insights = await self._generate_statistical_insights(
                descriptive_stats,
                correlation_analysis,
                outlier_analysis,
                distribution_analysis,
            )

            result = {
                "descriptive_stats": descriptive_stats,
                "correlation_analysis": correlation_analysis,
                "outlier_analysis": outlier_analysis,
                "distribution_analysis": distribution_analysis,
                "insights": insights,
            }
            return StatisticalAnalysisAgent.convert_numpy(result)

        except Exception as e:
            return {"error": str(e)}

    def _detect_outliers(self, series: pd.Series, method: str = "iqr") -> pd.Series:
        """Detect outliers using IQR method"""
        if method == "iqr":
            Q1 = series.quantile(0.25)
            Q3 = series.quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            return series[(series < lower_bound) | (series > upper_bound)]
        return pd.Series()

    def _test_normality(self, series: pd.Series) -> bool:
        """Simple normality test using skewness and kurtosis"""
        skewness = series.skew()
        kurtosis = series.kurtosis()
        return abs(skewness) < 1 and abs(kurtosis) < 3

    def _classify_distribution(self, series: pd.Series) -> str:
        """Classify distribution type"""
        skewness = series.skew()
        if abs(skewness) < 0.5:
            return "approximately_normal"
        elif skewness > 0.5:
            return "right_skewed"
        else:
            return "left_skewed"

    def _find_high_correlations(
        self, corr_matrix: pd.DataFrame, threshold: float = 0.7
    ) -> List[Dict[str, Any]]:
        """Find high correlations above threshold"""
        high_correlations = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i + 1, len(corr_matrix.columns)):
                corr_value = corr_matrix.iloc[i, j]
                if abs(corr_value) >= threshold:
                    high_correlations.append(
                        {
                            "column1": corr_matrix.columns[i],
                            "column2": corr_matrix.columns[j],
                            "correlation": corr_value,
                        }
                    )
        return high_correlations

    async def _generate_statistical_insights(
        self,
        descriptive_stats,
        correlation_analysis,
        outlier_analysis,
        distribution_analysis,
    ):
        """Generate insights about statistical analysis results"""
        prompt = f"""
        Based on the following statistical analysis results, provide key insights:
        
        Descriptive Statistics Summary:
        {descriptive_stats}
        
        Correlation Analysis:
        {correlation_analysis}
        
        Outlier Analysis:
        {outlier_analysis}
        
        Distribution Analysis:
        {distribution_analysis}
        
        Please provide:
        1. Key statistical patterns
        2. Notable correlations and their implications
        3. Outlier assessment and recommendations
        4. Distribution characteristics
        5. Statistical insights for decision making
        """

        messages = [
            SystemMessage(
                content="You are a statistical analysis expert. Provide clear, actionable insights about statistical results."
            ),
            HumanMessage(content=prompt),
        ]

        response = await self.llm.ainvoke(messages)
        return response.content


class VisualizationAgent:
    """Agent for generating visualizations"""

    def __init__(self):
        self.output_dir = "static/charts"

    async def generate_visualizations(
        self, data: pd.DataFrame, analysis_results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate comprehensive visualizations"""
        try:
            numeric_cols = data.select_dtypes(include=["number"]).columns
            categorical_cols = data.select_dtypes(include=["object"]).columns

            charts = []

            # Distribution plots for numeric columns
            for col in numeric_cols:
                chart_info = await self._create_distribution_chart(data, col)
                charts.append(chart_info)

            # Correlation heatmap
            if len(numeric_cols) > 1:
                correlation_chart = await self._create_correlation_heatmap(
                    data, numeric_cols
                )
                charts.append(correlation_chart)

            # Categorical analysis
            for col in categorical_cols:
                chart_info = await self._create_categorical_chart(data, col)
                charts.append(chart_info)

            # Scatter plots for highly correlated variables
            if "correlation_analysis" in analysis_results:
                high_correlations = analysis_results["correlation_analysis"].get(
                    "high_correlations", []
                )
                for corr in high_correlations[:3]:  # Limit to top 3
                    scatter_chart = await self._create_scatter_plot(
                        data, corr["column1"], corr["column2"]
                    )
                    charts.append(scatter_chart)

            return {
                "charts": charts,
                "total_charts": len(charts),
                "chart_types": list(set([chart["type"] for chart in charts])),
            }

        except Exception as e:
            return {"error": str(e)}

    async def _create_distribution_chart(
        self, data: pd.DataFrame, column: str
    ) -> Dict[str, Any]:
        """Create distribution chart for a numeric column"""
        try:
            fig = px.histogram(data, x=column, title=f"Distribution of {column}")
            file_path = f"{self.output_dir}/{column}_distribution.html"
            fig.write_html(file_path)

            return {
                "type": "distribution",
                "column": column,
                "title": f"Distribution of {column}",
                "file_path": file_path,
                "chart_type": "histogram",
            }
        except Exception as e:
            return {
                "error": f"Failed to create distribution chart for {column}: {str(e)}"
            }

    async def _create_correlation_heatmap(
        self, data: pd.DataFrame, numeric_cols
    ) -> Dict[str, Any]:
        """Create correlation heatmap"""
        try:
            corr_matrix = data[numeric_cols].corr()
            fig = px.imshow(
                corr_matrix,
                title="Correlation Heatmap",
                color_continuous_scale="RdBu",
                aspect="auto",
            )
            file_path = f"{self.output_dir}/correlation_heatmap.html"
            fig.write_html(file_path)

            return {
                "type": "correlation",
                "title": "Correlation Heatmap",
                "file_path": file_path,
                "chart_type": "heatmap",
            }
        except Exception as e:
            return {"error": f"Failed to create correlation heatmap: {str(e)}"}

    async def _create_categorical_chart(
        self, data: pd.DataFrame, column: str
    ) -> Dict[str, Any]:
        """Create chart for categorical column"""
        try:
            value_counts = data[column].value_counts()
            fig = px.bar(
                x=value_counts.index,
                y=value_counts.values,
                title=f"Distribution of {column}",
                labels={"x": column, "y": "Count"},
            )
            file_path = f"{self.output_dir}/{column}_categorical.html"
            fig.write_html(file_path)

            return {
                "type": "categorical",
                "column": column,
                "title": f"Distribution of {column}",
                "file_path": file_path,
                "chart_type": "bar",
            }
        except Exception as e:
            return {
                "error": f"Failed to create categorical chart for {column}: {str(e)}"
            }

    async def _create_scatter_plot(
        self, data: pd.DataFrame, col1: str, col2: str
    ) -> Dict[str, Any]:
        """Create scatter plot for two variables"""
        try:
            fig = px.scatter(
                data, x=col1, y=col2, title=f"Scatter Plot: {col1} vs {col2}"
            )
            file_path = f"{self.output_dir}/{col1}_{col2}_scatter.html"
            fig.write_html(file_path)

            return {
                "type": "scatter",
                "columns": [col1, col2],
                "title": f"Scatter Plot: {col1} vs {col2}",
                "file_path": file_path,
                "chart_type": "scatter",
            }
        except Exception as e:
            return {"error": f"Failed to create scatter plot: {str(e)}"}


class InsightGenerationAgent:
    """Agent for generating business insights"""

    def __init__(self):
        self.llm = ChatOpenAI(
            api_key=settings.openai_api_key, model=settings.LLM_MODEL, temperature=0.3
        )

    async def generate_insights(
        self, data: pd.DataFrame, analysis_results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate comprehensive business insights"""
        try:
            # Combine all analysis results
            combined_results = {
                "data_shape": data.shape,
                "columns": list(data.columns),
                "analysis_results": analysis_results,
            }

            # Generate different types of insights
            business_insights = await self._generate_business_insights(combined_results)
            technical_insights = await self._generate_technical_insights(
                combined_results
            )
            recommendations = await self._generate_recommendations(combined_results)

            return {
                "business_insights": business_insights,
                "technical_insights": technical_insights,
                "recommendations": recommendations,
                "summary": await self._generate_executive_summary(combined_results),
            }

        except Exception as e:
            return {"error": str(e)}

    async def _generate_business_insights(self, results: Dict[str, Any]) -> str:
        """Generate business-focused insights"""
        prompt = f"""
        Based on the following data analysis results, provide business insights:
        
        Data Overview:
        - Shape: {results['data_shape']}
        - Columns: {results['columns']}
        
        Analysis Results:
        {results['analysis_results']}
        
        Please provide:
        1. Key business patterns and trends
        2. Market insights and opportunities
        3. Risk factors and concerns
        4. Strategic recommendations for business decisions
        5. Competitive advantages or disadvantages
        """

        messages = [
            SystemMessage(
                content="You are a business analyst expert. Provide strategic business insights based on data analysis."
            ),
            HumanMessage(content=prompt),
        ]

        response = await self.llm.ainvoke(messages)
        return response.content

    async def _generate_technical_insights(self, results: Dict[str, Any]) -> str:
        """Generate technical insights"""
        prompt = f"""
        Based on the following data analysis results, provide technical insights:
        
        Data Overview:
        - Shape: {results['data_shape']}
        - Columns: {results['columns']}
        
        Analysis Results:
        {results['analysis_results']}
        
        Please provide:
        1. Data quality assessment
        2. Technical patterns and anomalies
        3. Model performance implications
        4. Data engineering recommendations
        5. Technical debt considerations
        """

        messages = [
            SystemMessage(
                content="You are a data scientist expert. Provide technical insights based on data analysis."
            ),
            HumanMessage(content=prompt),
        ]

        response = await self.llm.ainvoke(messages)
        return response.content

    async def _generate_recommendations(self, results: Dict[str, Any]) -> str:
        """Generate actionable recommendations"""
        prompt = f"""
        Based on the following data analysis results, provide actionable recommendations:
        
        Data Overview:
        - Shape: {results['data_shape']}
        - Columns: {results['columns']}
        
        Analysis Results:
        {results['analysis_results']}
        
        Please provide:
        1. Immediate actions to take
        2. Short-term recommendations (1-3 months)
        3. Long-term strategic recommendations (6-12 months)
        4. Resource allocation suggestions
        5. Success metrics and KPIs to track
        """

        messages = [
            SystemMessage(
                content="You are a strategic consultant. Provide actionable recommendations based on data analysis."
            ),
            HumanMessage(content=prompt),
        ]

        response = await self.llm.ainvoke(messages)
        return response.content

    async def _generate_executive_summary(self, results: Dict[str, Any]) -> str:
        """Generate executive summary"""
        prompt = f"""
        Based on the following data analysis results, provide an executive summary:
        
        Data Overview:
        - Shape: {results['data_shape']}
        - Columns: {results['columns']}
        
        Analysis Results:
        {results['analysis_results']}
        
        Please provide a concise executive summary that includes:
        1. Key findings (3-5 bullet points)
        2. Business impact
        3. Recommended next steps
        4. Risk assessment
        Keep it under 200 words and suitable for executive presentation.
        """

        messages = [
            SystemMessage(
                content="You are an executive consultant. Provide a concise executive summary based on data analysis."
            ),
            HumanMessage(content=prompt),
        ]

        response = await self.llm.ainvoke(messages)
        return response.content
