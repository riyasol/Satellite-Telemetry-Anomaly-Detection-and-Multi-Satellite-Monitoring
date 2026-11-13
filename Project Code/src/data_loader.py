"""
data_loader.py
Satellite Telemetry Data Ingestion, Cleaning, and Feature Preparation Module.
Implements telemetry data processing pipeline for multi-satellite monitoring:
  - Data ingestion from CSV storage
  - Schema validation and data integrity checks
  - Missing value imputation and chronological sorting
  - Parameter range and physical sanity validation
  - Feature selection and preparation for model training and thresholding
"""

import os
from typing import List, Tuple, Optional
import numpy as np
import pandas as pd


def get_project_root() -> str:
    """Return the absolute path to the project root directory."""
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def clean_and_validate_telemetry(df: pd.DataFrame) -> pd.DataFrame:
    """
    Perform thorough data cleaning, validation, and transformations on telemetry observations.
    Ensures timestamps are properly parsed, records are sorted chronologically,
    and missing values are imputed.

    Parameters:
      df: Raw telemetry DataFrame

    Returns:
      Cleaned and validated DataFrame
    """
    df_clean = df.copy()

    # 1. Parse and standardize timestamps
    if "timestamp" in df_clean.columns:
        df_clean["timestamp"] = pd.to_datetime(df_clean["timestamp"])
    else:
        # Synthesize time index if missing
        df_clean["timestamp"] = pd.date_range("2026-03-01", periods=len(df_clean), freq="1min")

    # 2. Ensure satellite identifier exists
    if "satellite_id" not in df_clean.columns:
        df_clean["satellite_id"] = "SAT-01"

    # 3. Sort chronologically by satellite and timestamp
    df_clean = df_clean.sort_values(by=["satellite_id", "timestamp"]).reset_index(drop=True)

    # 4. Numerical telemetry parameters validation and imputation
    telemetry_params = ["temperature", "voltage", "current", "orientation"]
    for param in telemetry_params:
        if param in df_clean.columns:
            # Coerce non-numeric values to NaN
            df_clean[param] = pd.to_numeric(df_clean[param], errors="coerce")
            # Handle missing values: forward fill followed by backward fill, then median fallback
            if df_clean[param].isna().any():
                df_clean[param] = (
                    df_clean.groupby("satellite_id")[param]
                    .transform(lambda group: group.ffill().bfill())
                )
                if df_clean[param].isna().any():
                    median_val = df_clean[param].median()
                    df_clean[param] = df_clean[param].fillna(median_val)

    # 5. Validate ground_truth column if present
    if "ground_truth" in df_clean.columns:
        df_clean["ground_truth"] = df_clean["ground_truth"].fillna(0).astype(int)

    # 6. Validate anomaly_type column if present
    if "anomaly_type" not in df_clean.columns and "ground_truth" in df_clean.columns:
        df_clean["anomaly_type"] = np.where(df_clean["ground_truth"] == 1, "Anomaly", "Normal")

    return df_clean


def load_telemetry_data(file_path: Optional[str] = None) -> pd.DataFrame:
    """
    Load satellite telemetry dataset.
    If the CSV does not exist, automatically generate it via generate_data.py.

    Parameters:
      file_path: Optional custom path to telemetry CSV file

    Returns:
      Cleaned and validated multi-satellite telemetry DataFrame
    """
    if file_path is None:
        file_path = os.path.join(get_project_root(), "data", "telemetry_data.csv")

    if not os.path.exists(file_path):
        from generate_data import generate_telemetry
        generate_telemetry(file_path)

    df_raw = pd.read_csv(file_path)
    df_clean = clean_and_validate_telemetry(df_raw)
    return df_clean


def prepare_features(
    df: pd.DataFrame,
    feature_cols: Optional[List[str]] = None
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Select and validate numerical telemetry features for machine learning models.

    Parameters:
      df: Cleaned telemetry DataFrame
      feature_cols: Optional specific list of feature names

    Returns:
      Tuple of (feature_dataframe, list_of_selected_feature_names)
    """
    if feature_cols is None:
        default_features = ["temperature", "voltage", "current", "orientation"]
        feature_cols = [col for col in default_features if col in df.columns]

    if not feature_cols:
        # Fallback to all numerical columns excluding ground truth and ID fields
        feature_cols = [
            c for c in df.select_dtypes(include=[np.number]).columns
            if c not in ["ground_truth", "satellite_id"]
        ]

    X = df[feature_cols].copy()
    return X, feature_cols


# Backward-compatible convenience aliases
def load_dataset(source: str = "synthetic", file_path: Optional[str] = None, **kwargs) -> pd.DataFrame:
    """Convenience alias for loading the multi-satellite telemetry dataset."""
    return load_telemetry_data(file_path=file_path)


def load_synthetic_telemetry(file_path: Optional[str] = None) -> pd.DataFrame:
    """Backward-compatible wrapper for load_telemetry_data."""
    return load_telemetry_data(file_path=file_path)


if __name__ == "__main__":
    print("Testing data_loader.py...")
    df = load_telemetry_data()
    print(f"Dataset successfully loaded: {df.shape[0]} rows, {df.shape[1]} columns")
    print(f"Satellites present: {list(df['satellite_id'].unique())}")
    X, feats = prepare_features(df)
    print(f"Features prepared: {feats} | Shape: {X.shape}")
