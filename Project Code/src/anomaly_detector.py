"""
anomaly_detector.py
Rule-Based Telemetry Anomaly Detection Module for Multi-Satellite Health Monitoring.
Fulfills baseline threshold anomaly detection requirements:
  - Documented physical safety thresholds for critical telemetry parameters
  - Parameter-level violation identification and reason tracking
  - Row-by-row and batch-level anomaly classification
  - Statistical threshold estimation (n-sigma) for uncalibrated parameters
"""

from typing import Dict, Any, Tuple, Optional, List
import pandas as pd
import numpy as np


# Documented physical nominal safety thresholds for satellite telemetry parameters
DEFAULT_TELEMETRY_THRESHOLDS = {
    "temperature": {
        "min": None,
        "max": 80.0,
        "description": "Temperature Upper Thermal Limit (80.0°C)"
    },
    "voltage": {
        "min": 3.0,
        "max": 4.2,
        "description": "Battery Bus Operating Voltage Range (3.0V - 4.2V)"
    },
    "current": {
        "min": None,
        "max": 2.5,
        "description": "Bus Current Overload Limit (2.5A)"
    },
    "orientation": {
        "min": -45.0,
        "max": 45.0,
        "description": "Reaction Wheel Attitude Pointing Limits (±45.0°)"
    }
}


class RuleBasedDetector:
    """
    Configurable Rule-Based Anomaly Detector for satellite telemetry.
    Evaluates telemetry against defined upper/lower safety limits
    and logs which specific parameter(s) triggered an anomaly flag.
    """

    def __init__(
        self,
        thresholds: Optional[Dict[str, Dict[str, Any]]] = None,
        rule_name: str = "Satellite Telemetry Safety Threshold Rules"
    ):
        self.rule_name = rule_name
        self.thresholds = thresholds.copy() if thresholds else DEFAULT_TELEMETRY_THRESHOLDS.copy()

    def set_threshold(
        self,
        parameter: str,
        min_val: Optional[float] = None,
        max_val: Optional[float] = None,
        description: str = ""
    ):
        """Set or update the operating safety threshold for a specific telemetry parameter."""
        self.thresholds[parameter] = {
            "min": min_val,
            "max": max_val,
            "description": description or f"Threshold for {parameter}"
        }

    def fit_statistical_limits(
        self,
        baseline_df: pd.DataFrame,
        feature_cols: List[str],
        n_sigma: float = 3.0
    ):
        """
        Derive empirical n-sigma statistical limits from baseline nominal telemetry data.
        Useful when physical engineering limits are uncalibrated or for comparison.
        """
        self.thresholds = {}
        for col in feature_cols:
            if col in baseline_df.columns:
                mean = baseline_df[col].mean()
                std = baseline_df[col].std()
                min_val = mean - n_sigma * std
                max_val = mean + n_sigma * std
                self.thresholds[col] = {
                    "min": float(min_val),
                    "max": float(max_val),
                    "description": f"Empirical {n_sigma}-Sigma Limits [{min_val:.3f}, {max_val:.3f}]"
                }

    def predict_row(self, row: pd.Series) -> Tuple[int, str]:
        """
        Evaluate a single telemetry observation against configured safety thresholds.

        Returns:
          Tuple of (is_anomaly: int [0=Normal, 1=Anomaly], reason: str)
        """
        violations = []

        for param, bounds in self.thresholds.items():
            if param not in row or pd.isna(row[param]):
                continue

            val = row[param]
            min_val = bounds.get("min")
            max_val = bounds.get("max")

            if min_val is not None and val < min_val:
                violations.append(f"{param}={val:.2f} < min({min_val:.2f})")
            elif max_val is not None and val > max_val:
                violations.append(f"{param}={val:.2f} > max({max_val:.2f})")

        if violations:
            return 1, "; ".join(violations)
        return 0, "Nominal"

    def predict(
        self,
        df: pd.DataFrame,
        feature_cols: Optional[List[str]] = None
    ) -> Tuple[pd.Series, pd.Series]:
        """
        Apply rule-based threshold evaluation across all observations in a DataFrame.

        Returns:
          Tuple of (predictions: pd.Series [0 or 1], violation_reasons: pd.Series)
        """
        preds = []
        violations = []

        for _, row in df.iterrows():
            is_anomaly, reason = self.predict_row(row)
            preds.append(is_anomaly)
            violations.append(reason)

        return pd.Series(preds, index=df.index), pd.Series(violations, index=df.index)


# Backward-compatible function for existing scripts
def rule_based_detection(row: pd.Series) -> bool:
    """
    Evaluates whether a telemetry reading breaches nominal safety boundaries.
    Returns True if an anomaly is detected, False otherwise.
    """
    if "temperature" in row and row["temperature"] > 80:
        return True
    if "voltage" in row and (row["voltage"] < 3.0 or row["voltage"] > 4.2):
        return True
    if "current" in row and row["current"] > 2.5:
        return True
    if "orientation" in row and abs(row["orientation"]) > 45:
        return True
    return False


if __name__ == "__main__":
    detector = RuleBasedDetector()
    test_normal = pd.Series({"temperature": 71.0, "voltage": 3.9, "current": 1.9, "orientation": 12.0})
    test_anomaly = pd.Series({"temperature": 86.5, "voltage": 2.7, "current": 2.1, "orientation": 15.0})

    print("Test Normal Reading:", detector.predict_row(test_normal))
    print("Test Anomalous Reading:", detector.predict_row(test_anomaly))
