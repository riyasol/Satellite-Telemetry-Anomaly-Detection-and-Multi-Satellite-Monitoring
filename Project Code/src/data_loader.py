"""
data_loader.py
Unified Data Ingestion, Cleaning, and Feature Preparation Module.
Supports dual telemetry data sources:
  1. Actual ESA Mission 1 spacecraft telemetry dataset (multi-channel zip extraction,
     timestamp alignment, time-based missing value interpolation, and ground-truth label integration).
  2. Controlled multi-satellite telemetry dataset (SAT-01, SAT-02, SAT-03) with physics-based
     orbital cycles and injected fault categories for controlled benchmarking.
"""

import os
import zipfile
from typing import List, Tuple, Optional
import numpy as np
import pandas as pd


def get_project_root() -> str:
    """Return the absolute path to the project root directory."""
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def get_esa_channels_info(esa_dir: Optional[str] = None) -> pd.DataFrame:
    """
    Load the ESA Mission 1 channel metadata table.
    Returns DataFrame containing Channel, Subsystem, Physical Unit, Group, Target.
    """
    if esa_dir is None:
        esa_dir = os.path.join(get_project_root(), "ESA-Mission1", "ESA-Mission1")

    channels_csv = os.path.join(esa_dir, "channels.csv")
    if not os.path.exists(channels_csv):
        raise FileNotFoundError(f"ESA channels metadata not found at: {channels_csv}")

    return pd.read_csv(channels_csv)


def load_esa_telemetry(
    channels: Optional[List[str]] = None,
    start_time: str = "2004-11-25",
    end_time: str = "2004-12-15",
    step: int = 1,
    max_records: int = 10000,
    esa_dir: Optional[str] = None
) -> pd.DataFrame:
    """
    Load actual European Space Agency (ESA) Mission 1 telemetry data from channel zip files,
    align timestamps, clean missing values, and match operational anomaly intervals from labels.csv.

    Parameters:
      channels: List of channel names (defaults to Subsystem 6 Group 3: channel_12, 13, 19, 20)
      start_time: Start datetime string (e.g. '2004-11-25')
      end_time: End datetime string (e.g. '2004-12-15')
      step: Subsampling step interval
      max_records: Maximum rows to return for responsive model execution
      esa_dir: Root directory of ESA dataset
    """
    if esa_dir is None:
        esa_dir = os.path.join(get_project_root(), "ESA-Mission1", "ESA-Mission1")

    if channels is None:
        # Default to high-priority Subsystem 6 (Group 3) target telemetry channels
        channels = ["channel_12", "channel_13", "channel_19", "channel_20"]

    channel_dfs = []
    channels_folder = os.path.join(esa_dir, "channels")

    for ch in channels:
        zip_path = os.path.join(channels_folder, f"{ch}.zip")
        if not os.path.exists(zip_path):
            raise FileNotFoundError(f"ESA channel zip not found: {zip_path}")

        with zipfile.ZipFile(zip_path, "r") as z:
            with z.open(ch) as f:
                ch_df = pd.read_pickle(f)
                if start_time and end_time:
                    ch_df = ch_df.loc[start_time:end_time]
                channel_dfs.append(ch_df)

    # Merge channels along time index
    merged = pd.concat(channel_dfs, axis=1, join="outer").sort_index()

    # Handle missing values via time interpolation and edge boundary filling
    merged = merged.interpolate(method="time").bfill().ffill()

    if step > 1:
        merged = merged.iloc[::step]

    if max_records and len(merged) > max_records:
        merged = merged.iloc[:max_records]

    # Map true operational anomaly intervals from labels.csv
    labels_csv = os.path.join(esa_dir, "labels.csv")
    ground_truth = np.zeros(len(merged), dtype=int)

    if os.path.exists(labels_csv):
        labels_df = pd.read_csv(labels_csv)
        labels_df = labels_df[labels_df["Channel"].isin(channels)].copy()
        if not labels_df.empty:
            labels_df["StartTime"] = pd.to_datetime(labels_df["StartTime"]).dt.tz_localize(None)
            labels_df["EndTime"] = pd.to_datetime(labels_df["EndTime"]).dt.tz_localize(None)

            idx_vals = merged.index
            for _, row in labels_df.iterrows():
                mask = (idx_vals >= row["StartTime"]) & (idx_vals <= row["EndTime"])
                ground_truth[mask] = 1

    df_out = merged.reset_index().rename(columns={"datetime": "timestamp"})
    df_out["satellite_id"] = "ESA-Mission1"
    df_out["ground_truth"] = ground_truth
    df_out["data_source"] = "ESA Mission 1 Telemetry"
    df_out["anomaly_type"] = np.where(df_out["ground_truth"] == 1, "ESA Operational Anomaly", "Normal")

    return df_out


def clean_and_validate_telemetry(df: pd.DataFrame) -> pd.DataFrame:
    """
    Perform thorough data cleaning, validation, and transformations on telemetry observations.
    Ensures timestamps are properly parsed, records are sorted chronologically,
    and missing values are imputed.
    """
    df_clean = df.copy()

    if "timestamp" in df_clean.columns:
        df_clean["timestamp"] = pd.to_datetime(df_clean["timestamp"])
    else:
        df_clean["timestamp"] = pd.date_range("2026-03-01", periods=len(df_clean), freq="1min")

    if "satellite_id" not in df_clean.columns:
        df_clean["satellite_id"] = "SAT-01"

    df_clean = df_clean.sort_values(by=["satellite_id", "timestamp"]).reset_index(drop=True)

    telemetry_params = ["temperature", "voltage", "current", "orientation"]
    for param in telemetry_params:
        if param in df_clean.columns:
            df_clean[param] = pd.to_numeric(df_clean[param], errors="coerce")
            if df_clean[param].isna().any():
                df_clean[param] = (
                    df_clean.groupby("satellite_id")[param]
                    .transform(lambda group: group.ffill().bfill())
                )
                if df_clean[param].isna().any():
                    median_val = df_clean[param].median()
                    df_clean[param] = df_clean[param].fillna(median_val)

    if "ground_truth" in df_clean.columns:
        df_clean["ground_truth"] = df_clean["ground_truth"].fillna(0).astype(int)

    if "anomaly_type" not in df_clean.columns and "ground_truth" in df_clean.columns:
        df_clean["anomaly_type"] = np.where(df_clean["ground_truth"] == 1, "Anomaly", "Normal")

    return df_clean


def load_synthetic_telemetry(file_path: Optional[str] = None) -> pd.DataFrame:
    """
    Load multi-satellite telemetry dataset.
    If the file does not exist, automatically generate it via generate_data.py.
    """
    if file_path is None:
        file_path = os.path.join(get_project_root(), "data", "telemetry_data.csv")

    if not os.path.exists(file_path):
        from generate_data import generate_telemetry
        generate_telemetry(file_path)

    df_raw = pd.read_csv(file_path)
    df_clean = clean_and_validate_telemetry(df_raw)
    return df_clean


def load_telemetry_data(file_path: Optional[str] = None) -> pd.DataFrame:
    """Convenience alias for loading the multi-satellite telemetry dataset."""
    return load_synthetic_telemetry(file_path=file_path)


def load_dataset(source: str = "synthetic", **kwargs) -> pd.DataFrame:
    """
    Convenience factory to load either 'synthetic' or 'esa' telemetry data.
    """
    if source.lower() in ["esa", "actual", "mission1", "esa-mission1"]:
        return load_esa_telemetry(**kwargs)
    else:
        return load_synthetic_telemetry(kwargs.get("file_path"))


def prepare_features(
    df: pd.DataFrame,
    feature_cols: Optional[List[str]] = None
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Select and validate numerical telemetry features for machine learning models.
    """
    if feature_cols is None:
        default_features = ["temperature", "voltage", "current", "orientation"]
        feature_cols = [col for col in default_features if col in df.columns]

    if not feature_cols:
        feature_cols = [
            c for c in df.select_dtypes(include=[np.number]).columns
            if c not in ["ground_truth", "satellite_id"]
        ]

    X = df[feature_cols].copy()
    return X, feature_cols


if __name__ == "__main__":
    print("Testing data_loader.py...")
    synth_df = load_dataset("synthetic")
    print(f"Synthetic loaded: shape={synth_df.shape}, cols={list(synth_df.columns)}")

    esa_df = load_dataset("esa", max_records=2000)
    print(f"ESA loaded: shape={esa_df.shape}, cols={list(esa_df.columns)}")
    print(f"ESA ground truth anomalies: {esa_df['ground_truth'].sum()} / {len(esa_df)}")
