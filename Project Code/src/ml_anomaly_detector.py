"""
ml_anomaly_detector.py
Machine Learning Telemetry Anomaly Detector using genuine scikit-learn Isolation Forest.
Implements model training (fit), inference on unseen data (predict),
anomaly score computation, configurable contamination tuning, and model persistence (joblib).
"""

import os
import joblib
import numpy as np
import pandas as pd
from typing import List, Tuple, Optional, Union
from sklearn.ensemble import IsolationForest


class SatelliteIsolationForest:
    """
    Isolation Forest Anomaly Detector tailored for satellite telemetry health monitoring.

    Underlying Algorithm:
      scikit-learn IsolationForest builds an ensemble of isolation trees (iTrees).
      Because anomalous observations have attribute values distinct from nominal patterns,
      they require fewer random partitions to be isolated near the tree roots.
      Normal observations reside deep in the trees.
    """

    def __init__(
        self,
        contamination: Union[float, str] = 0.05,
        n_estimators: int = 100,
        max_samples: Union[int, float, str] = "auto",
        random_state: int = 42
    ):
        """
        Initialize Isolation Forest.
        Parameters:
          contamination: Expected proportion of outliers in the data (e.g., 0.01, 0.03, 0.05, 0.10).
          n_estimators: Number of isolation trees in the ensemble (default 100).
          max_samples: Number of samples drawn to train each base tree.
          random_state: Seed for reproducible tree partitioning.
        """
        self.contamination = contamination
        self.n_estimators = n_estimators
        self.max_samples = max_samples
        self.random_state = random_state

        self.model = IsolationForest(
            contamination=self.contamination,
            n_estimators=self.n_estimators,
            max_samples=self.max_samples,
            random_state=self.random_state
        )
        self.feature_names: List[str] = []
        self.is_trained: bool = False

    def fit(self, X: Union[pd.DataFrame, np.ndarray], feature_names: Optional[List[str]] = None) -> "SatelliteIsolationForest":
        """
        Fit Isolation Forest model on training telemetry features.
        Never touches ground-truth labels to prevent data leakage.
        """
        if isinstance(X, pd.DataFrame):
            self.feature_names = list(X.columns)
            X_mat = X.values
        else:
            self.feature_names = feature_names if feature_names else [f"feature_{i}" for i in range(X.shape[1])]
            X_mat = X

        # Genuine scikit-learn fit
        self.model.fit(X_mat)
        self.is_trained = True
        return self

    def predict(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        Predict anomaly classification on observations.

        scikit-learn convention:
          1  = Inlier (Normal)
         -1  = Outlier (Anomaly)

        Conversion to standard evaluation format:
          0 = Normal
          1 = Anomaly
        """
        if not self.is_trained:
            raise RuntimeError("Model must be trained with .fit() before calling .predict()")

        X_mat = X.values if isinstance(X, pd.DataFrame) else X
        raw_preds = self.model.predict(X_mat)

        # 0 = Normal, 1 = Anomaly
        binary_preds = (raw_preds == -1).astype(int)
        return binary_preds

    def decision_function(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        Compute scikit-learn raw decision function.
        Positive values indicate nominal inliers; negative values indicate anomalies.
        """
        if not self.is_trained:
            raise RuntimeError("Model must be trained with .fit() before calling .decision_function()")

        X_mat = X.values if isinstance(X, pd.DataFrame) else X
        return self.model.decision_function(X_mat)

    def compute_anomaly_scores(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        Compute user-friendly ML anomaly scores.
        Inverted from decision_function: higher values = higher likelihood of anomaly.
        """
        # Negate decision function: outliers have negative decision scores -> become high positive
        raw_dec = self.decision_function(X)
        scores = -raw_dec
        return np.round(scores, 4)

    def predict_with_scores(self, X: Union[pd.DataFrame, np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Convenience method returning both binary predictions (0=Normal, 1=Anomaly)
        and anomaly scores.
        """
        preds = self.predict(X)
        scores = self.compute_anomaly_scores(X)
        return preds, scores

    def save(self, file_path: str):
        """Save the trained Isolation Forest model to disk using joblib."""
        os.makedirs(os.path.dirname(os.path.abspath(file_path)), exist_ok=True)
        joblib.dump(self, file_path)
        print(f"✅ Saved Isolation Forest model to: {file_path}")

    @classmethod
    def load(cls, file_path: str) -> "SatelliteIsolationForest":
        """Load a saved SatelliteIsolationForest instance from disk."""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Model file not found: {file_path}")
        instance = joblib.load(file_path)
        return instance


def train_test_split_telemetry(
    df: pd.DataFrame,
    feature_cols: List[str],
    test_size: float = 0.30,
    time_based: bool = True
) -> Tuple[pd.DataFrame, pd.DataFrame, Optional[pd.Series], Optional[pd.Series]]:
    """
    Perform rigorous train/test splitting on telemetry data to prevent data leakage.

    Parameters:
      df: Input DataFrame containing telemetry features and optional ground_truth
      feature_cols: List of numerical columns to use as ML training features
      test_size: Fraction of data reserved for testing (default 0.30)
      time_based: If True, uses temporal chronologic split (essential for time series)

    Returns:
      (X_train, X_test, y_train, y_test)
    """
    if time_based and "timestamp" in df.columns:
        df_sorted = df.sort_values("timestamp").reset_index(drop=True)
    else:
        df_sorted = df.copy()

    split_idx = int(len(df_sorted) * (1.0 - test_size))

    train_df = df_sorted.iloc[:split_idx]
    test_df = df_sorted.iloc[split_idx:]

    X_train = train_df[feature_cols].copy()
    X_test = test_df[feature_cols].copy()

    y_train = train_df["ground_truth"].copy() if "ground_truth" in train_df.columns else None
    y_test = test_df["ground_truth"].copy() if "ground_truth" in test_df.columns else None

    return X_train, X_test, y_train, y_test


# Backward-compatible function for existing code
def apply_ml_detection(df: pd.DataFrame, contamination: float = 0.05) -> pd.DataFrame:
    """
    Backward-compatible wrapper that applies Isolation Forest detection
    to a telemetry DataFrame.
    """
    candidate_features = ["temperature", "voltage", "current", "orientation"]
    available_features = [f for f in candidate_features if f in df.columns]

    if not available_features:
        # Fall back to any numeric columns except timestamp / ground truth
        available_features = [
            c for c in df.select_dtypes(include=[np.number]).columns
            if c not in ["ground_truth", "satellite_id"]
        ]

    detector = SatelliteIsolationForest(contamination=contamination, random_state=42)
    detector.fit(df[available_features])

    preds, scores = detector.predict_with_scores(df[available_features])

    df_out = df.copy()
    df_out["ml_prediction"] = preds  # 0 = Normal, 1 = Anomaly
    df_out["ml_score"] = scores
    df_out["ml_result"] = np.where(preds == 1, "ML Anomaly", "Normal")

    return df_out


if __name__ == "__main__":
    from generate_data import generate_synthetic_telemetry

    df = generate_synthetic_telemetry(num_records_per_satellite=200)
    features = ["temperature", "voltage", "current", "orientation"]

    X_train, X_test, y_train, y_test = train_test_split_telemetry(df, features, test_size=0.3)
    print(f"X_train shape: {X_train.shape}, X_test shape: {X_test.shape}")

    model = SatelliteIsolationForest(contamination=0.05)
    model.fit(X_train)
    test_preds, test_scores = model.predict_with_scores(X_test)

    print(f"Predictions: {np.bincount(test_preds)} (0=Normal, 1=Anomaly)")
    print(f"Mean Anomaly Score: {test_scores.mean():.4f}")
