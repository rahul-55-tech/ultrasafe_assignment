import json
from datetime import datetime
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd


class DataProcessor:
    """Service for processing and analyzing data files"""

    def __init__(self):
        self.supported_formats = {
            ".csv": self._load_csv,
            ".xlsx": self._load_excel,
            ".xls": self._load_excel,
            ".json": self._load_json,
            ".parquet": self._load_parquet,
        }

    async def load_data(self, file_path: str, file_type: str) -> pd.DataFrame:
        """Load data from file based on file type"""
        if file_type not in self.supported_formats:
            raise ValueError(f"Unsupported file type: {file_type}")

        loader = self.supported_formats[file_type]
        return await loader(file_path)

    async def _load_csv(self, file_path: str) -> pd.DataFrame:
        """Load CSV file"""
        try:
            return pd.read_csv(file_path)
        except Exception as e:
            raise ValueError(f"Failed to load CSV file: {str(e)}")

    async def _load_excel(self, file_path: str) -> pd.DataFrame:
        """Load Excel file"""
        try:
            return pd.read_excel(file_path)
        except Exception as e:
            raise ValueError(f"Failed to load Excel file: {str(e)}")

    async def _load_json(self, file_path: str) -> pd.DataFrame:
        """Load JSON file"""
        try:
            return pd.read_json(file_path)
        except Exception as e:
            raise ValueError(f"Failed to load JSON file: {str(e)}")

    async def _load_parquet(self, file_path: str) -> pd.DataFrame:
        """Load Parquet file"""
        try:
            return pd.read_parquet(file_path)
        except Exception as e:
            raise ValueError(f"Failed to load Parquet file: {str(e)}")

    async def analyze_schema(self, data: pd.DataFrame) -> List[Dict[str, Any]]:
        """Analyze data schema and generate column information"""
        schema_info = []

        for column in data.columns:
            col_data = data[column]
            col_info = {
                "column_name": column,
                "data_type": str(col_data.dtype),
                "null_count": int(col_data.isnull().sum()),
                "unique_count": int(col_data.nunique()),
            }

            # Add sample values for categorical data
            if col_data.dtype == "object" or col_data.nunique() < 50:
                sample_values = col_data.dropna().unique()[:10].tolist()
                col_info["sample_values"] = sample_values

            # Add statistics for numeric data
            if pd.api.types.is_numeric_dtype(col_data):
                col_info.update(
                    {
                        "min_value": (
                            float(col_data.min()) if not col_data.empty else None
                        ),
                        "max_value": (
                            float(col_data.max()) if not col_data.empty else None
                        ),
                        "mean_value": (
                            float(col_data.mean()) if not col_data.empty else None
                        ),
                        "std_value": (
                            float(col_data.std()) if not col_data.empty else None
                        ),
                    }
                )

            schema_info.append(col_info)

        return schema_info

    async def clean_data(
        self, data: pd.DataFrame, cleaning_options: Dict[str, Any]
    ) -> pd.DataFrame:
        """Clean data based on specified options"""
        cleaned_data = data.copy()

        # Handle missing values
        if cleaning_options.get("handle_missing"):
            strategy = cleaning_options.get("missing_strategy", "drop")
            if strategy == "drop":
                cleaned_data = cleaned_data.dropna()
            elif strategy == "fill_mean":
                numeric_cols = cleaned_data.select_dtypes(include=[np.number]).columns
                cleaned_data[numeric_cols] = cleaned_data[numeric_cols].fillna(
                    cleaned_data[numeric_cols].mean()
                )
            elif strategy == "fill_median":
                numeric_cols = cleaned_data.select_dtypes(include=[np.number]).columns
                cleaned_data[numeric_cols] = cleaned_data[numeric_cols].fillna(
                    cleaned_data[numeric_cols].median()
                )
            elif strategy == "fill_mode":
                for col in cleaned_data.columns:
                    if cleaned_data[col].dtype == "object":
                        mode_value = cleaned_data[col].mode()
                        if not mode_value.empty:
                            cleaned_data[col] = cleaned_data[col].fillna(mode_value[0])

        # Remove duplicates
        if cleaning_options.get("remove_duplicates"):
            cleaned_data = cleaned_data.drop_duplicates()

        # Handle outliers
        if cleaning_options.get("handle_outliers"):
            method = cleaning_options.get("outlier_method", "iqr")
            if method == "iqr":
                cleaned_data = self._remove_outliers_iqr(cleaned_data)

        return cleaned_data

    def _remove_outliers_iqr(self, data: pd.DataFrame) -> pd.DataFrame:
        """Remove outliers using IQR method"""
        cleaned_data = data.copy()
        numeric_cols = cleaned_data.select_dtypes(include=[np.number]).columns

        for col in numeric_cols:
            Q1 = cleaned_data[col].quantile(0.25)
            Q3 = cleaned_data[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR

            cleaned_data = cleaned_data[
                (cleaned_data[col] >= lower_bound) & (cleaned_data[col] <= upper_bound)
            ]

        return cleaned_data

    async def export_data(self, data: pd.DataFrame, file_format: str,
                          file_path:
    str) -> str:
        """Export data to different formats"""
        try:
            if file_format.lower() == "csv":
                data.to_csv(file_path, index=False)
            elif file_format.lower() == "excel":
                data.to_excel(file_path, index=False)
            elif file_format.lower() == "json":
                data.to_json(file_path, orient="records", indent=2)
            elif file_format.lower() == "parquet":
                data.to_parquet(file_path, index=False)
            else:
                raise ValueError(f"Unsupported export format: {file_format}")

            return file_path

        except Exception as e:
            raise ValueError(f"Failed to export data: {str(e)}")

    async def get_basic_statistics(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Get basic statistics for the dataset"""
        stats = {
            "shape": data.shape,
            "memory_usage": data.memory_usage(deep=True).sum(),
            "missing_values": data.isnull().sum().sum(),
            "duplicate_rows": data.duplicated().sum(),
            "numeric_columns": list(data.select_dtypes(include=[np.number]).columns),
            "categorical_columns": list(data.select_dtypes(include=["object"]).columns),
            "datetime_columns": list(data.select_dtypes(include=["datetime"]).columns),
        }

        # Add summary statistics for numeric columns
        if len(stats["numeric_columns"]) > 0:
            stats["numeric_summary"] = (
                data[stats["numeric_columns"]].describe().to_dict()
            )

        return stats

    async def validate_data(
        self, data: pd.DataFrame, validation_rules: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Validate data against specified rules"""
        validation_results = {"passed": True, "errors": [], "warnings": []}

        # Check required columns
        if "required_columns" in validation_rules:
            missing_columns = set(validation_rules["required_columns"]) - set(
                data.columns
            )
            if missing_columns:
                validation_results["passed"] = False
                validation_results["errors"].append(
                    f"Missing required columns: {missing_columns}"
                )

        # Check data types
        if "column_types" in validation_rules:
            for col, expected_type in validation_rules["column_types"].items():
                if col in data.columns:
                    actual_type = str(data[col].dtype)
                    if actual_type != expected_type:
                        validation_results["warnings"].append(
                            f"Column {col} has type {actual_type}, expected {expected_type}"
                        )

        # Check value ranges
        if "value_ranges" in validation_rules:
            for col, range_info in validation_rules["value_ranges"].items():
                if col in data.columns and pd.api.types.is_numeric_dtype(data[col]):
                    min_val = data[col].min()
                    max_val = data[col].max()

                    if "min" in range_info and min_val < range_info["min"]:
                        validation_results["warnings"].append(
                            f"Column {col} has values below minimum: {min_val} < {range_info['min']}"
                        )

                    if "max" in range_info and max_val > range_info["max"]:
                        validation_results["warnings"].append(
                            f"Column {col} has values above maximum: {max_val} > {range_info['max']}"
                        )

        return validation_results
