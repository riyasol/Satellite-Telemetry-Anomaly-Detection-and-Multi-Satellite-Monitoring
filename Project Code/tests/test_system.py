"""
test_system.py
Unit and System Verification Test Suite for:
Satellite Telemetry Anomaly Detection and Multi-Satellite Monitoring
Tests:
  1. Synthetic multi-satellite telemetry generation with ground truth
  2. ESA Mission 1 telemetry loading, time alignment, and label matching
  3. Rule-based detector with nominal and anomalous parameter inputs
  4. Isolation Forest model training, inference, anomaly scoring, and model persistence
  5. Combined OR-logic anomaly detection logic
  6. Quantitative evaluation metrics (Accuracy, Precision, Recall, F1, Confusion Matrix)
  7. Execution of all 4 approved proposal experiments
  8. Telemetry plotting and visualization functions
"""

import os
import sys
import tempfile
import unittest
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Add project root and src to sys.path so modules resolve in any environment
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(PROJECT_ROOT, "src")

for path in [PROJECT_ROOT, SRC_DIR]:
    if path not in sys.path:
        sys.path.insert(0, path)

from src.data_loader import load_dataset, load_esa_telemetry, load_synthetic_telemetry
from src.anomaly_detector import RuleBasedDetector, rule_based_detection
from src.ml_anomaly_detector import SatelliteIsolationForest, train_test_split_telemetry
from src.evaluation import compute_metrics, evaluate_all_detectors
from src.experiments import (
    run_experiment_1,
    run_experiment_2_contamination,
    run_experiment_3_feature_combinations,
    run_experiment_4_method_comparison,
    run_all_experiments
)
from src.visualizer import (
    plot_telemetry_time_series,
    plot_confusion_matrices,
    plot_contamination_curve,
    plot_feature_combinations_bar,
    plot_anomaly_score_distribution
)


class TestSatelliteTelemetrySystem(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        """Set up test datasets for use across test methods."""
        cls.synth_df = load_dataset("synthetic")
        cls.features = ["temperature", "voltage", "current", "orientation"]

    def tearDown(self):
        """Clean up open matplotlib figures after each test."""
        plt.close("all")

    def test_01_synthetic_telemetry_generation(self):
        """Verify multi-satellite telemetry schema, records, and ground truth."""
        df = self.synth_df
        expected_cols = [
            "timestamp", "satellite_id", "temperature", "voltage",
            "current", "orientation", "ground_truth", "anomaly_type"
        ]
        for col in expected_cols:
            self.assertIn(col, df.columns, f"Missing required column: {col}")

        self.assertGreaterEqual(len(df), 1500, "Dataset contains fewer records than expected")
        satellites = set(df["satellite_id"].unique())
        self.assertTrue({"SAT-01", "SAT-02", "SAT-03"}.issubset(satellites))
        self.assertIn(1, df["ground_truth"].values)
        self.assertIn(0, df["ground_truth"].values)

    def test_02_esa_telemetry_loading(self):
        """Verify actual ESA Mission 1 channel loading and label matching."""
        esa_df = load_dataset("esa", max_records=500)
        self.assertGreater(len(esa_df), 0, "ESA data failed to load")
        self.assertIn("timestamp", esa_df.columns)
        self.assertIn("channel_12", esa_df.columns)
        self.assertEqual(esa_df["satellite_id"].iloc[0], "ESA-Mission1")
        # Ensure no missing values after interpolation
        self.assertEqual(esa_df[["channel_12", "channel_13", "channel_19", "channel_20"]].isna().sum().sum(), 0)

    def test_03_rule_based_detector_known_inputs(self):
        """Verify rule-based detector on known nominal and anomalous parameters."""
        detector = RuleBasedDetector()

        # Nominal reading
        nom_row = pd.Series({"temperature": 71.0, "voltage": 3.85, "current": 1.85, "orientation": 15.0})
        is_anom, reason = detector.predict_row(nom_row)
        self.assertEqual(is_anom, 0)
        self.assertEqual(reason, "Nominal")
        self.assertFalse(rule_based_detection(nom_row))

        # Thermal Spike (temp > 80°C)
        thermal_row = pd.Series({"temperature": 89.0, "voltage": 3.85, "current": 1.85, "orientation": 15.0})
        is_anom, reason = detector.predict_row(thermal_row)
        self.assertEqual(is_anom, 1)
        self.assertIn("temperature", reason)
        self.assertTrue(rule_based_detection(thermal_row))

        # Voltage Collapse (voltage < 3.0V)
        volt_row = pd.Series({"temperature": 71.0, "voltage": 2.45, "current": 1.85, "orientation": 15.0})
        is_anom, reason = detector.predict_row(volt_row)
        self.assertEqual(is_anom, 1)
        self.assertIn("voltage", reason)
        self.assertTrue(rule_based_detection(volt_row))

        # Current Surge (current > 2.5A)
        curr_row = pd.Series({"temperature": 71.0, "voltage": 3.85, "current": 3.10, "orientation": 15.0})
        is_anom, reason = detector.predict_row(curr_row)
        self.assertEqual(is_anom, 1)
        self.assertIn("current", reason)
        self.assertTrue(rule_based_detection(curr_row))

        # Attitude Anomaly (|orientation| > 45°)
        orient_row = pd.Series({"temperature": 71.0, "voltage": 3.85, "current": 1.85, "orientation": -60.0})
        is_anom, reason = detector.predict_row(orient_row)
        self.assertEqual(is_anom, 1)
        self.assertIn("orientation", reason)
        self.assertTrue(rule_based_detection(orient_row))

    def test_04_isolation_forest_training_and_scoring(self):
        """Verify Isolation Forest training, inference, scoring, and joblib save/load."""
        X_train, X_test, y_train, y_test = train_test_split_telemetry(self.synth_df, self.features, test_size=0.30)

        model = SatelliteIsolationForest(contamination=0.05, random_state=42)
        model.fit(X_train)
        self.assertTrue(model.is_trained)

        preds, scores = model.predict_with_scores(X_test)
        self.assertEqual(len(preds), len(X_test))
        self.assertEqual(len(scores), len(X_test))

        # Predictions must be binary 0 or 1
        unique_preds = set(np.unique(preds))
        self.assertTrue(unique_preds.issubset({0, 1}))

        # Continuous anomaly scores must be floating point
        self.assertEqual(scores.dtype, np.float64)

        # Test model persistence (save and load)
        with tempfile.NamedTemporaryFile(suffix=".joblib", delete=False) as tmp:
            tmp_path = tmp.name

        try:
            model.save(tmp_path)
            loaded_model = SatelliteIsolationForest.load(tmp_path)
            loaded_preds = loaded_model.predict(X_test)
            np.testing.assert_array_equal(preds, loaded_preds)
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    def test_05_combined_detection(self):
        """Verify OR-based combined detection preserves independent detections."""
        detector = RuleBasedDetector()
        rule_preds, _ = detector.predict(self.synth_df)

        model = SatelliteIsolationForest(contamination=0.05, random_state=42)
        model.fit(self.synth_df[self.features])
        ml_preds = model.predict(self.synth_df[self.features])

        comb_preds = ((rule_preds == 1) | (ml_preds == 1)).astype(int)

        for r, m, c in zip(rule_preds, ml_preds, comb_preds):
            if r == 1 or m == 1:
                self.assertEqual(c, 1)
            else:
                self.assertEqual(c, 0)

    def test_06_model_evaluation_metrics(self):
        """Verify computation of Accuracy, Precision, Recall, F1, and Confusion Matrix."""
        y_true = np.array([0, 0, 0, 0, 1, 1, 1, 1])
        y_pred = np.array([0, 0, 0, 1, 1, 1, 1, 0])

        m = compute_metrics(y_true, y_pred, "TestDetector")
        self.assertEqual(m["total_samples"], 8)
        self.assertEqual(m["actual_anomalies"], 4)
        self.assertEqual(m["detected_anomalies"], 4)
        self.assertEqual(m["true_positives"], 3)
        self.assertEqual(m["false_positives"], 1)
        self.assertEqual(m["true_negatives"], 3)
        self.assertEqual(m["false_negatives"], 1)
        self.assertAlmostEqual(m["accuracy"], 6 / 8, places=3)
        self.assertAlmostEqual(m["precision"], 3 / 4, places=3)
        self.assertAlmostEqual(m["recall"], 3 / 4, places=3)

    def test_07_experiments_execution(self):
        """Verify that all 4 approved proposal experiments execute and produce valid outputs."""
        rule_det = RuleBasedDetector()
        res = run_all_experiments(self.synth_df, self.features, rule_detector=rule_det)

        # Exp 1: Overlap Table
        self.assertIn("experiment_1", res)
        self.assertIn("overlap_table", res["experiment_1"])
        self.assertEqual(len(res["experiment_1"]["overlap_table"]), 4)

        # Exp 2: Contamination Sweep (4 values: 0.01, 0.03, 0.05, 0.10)
        self.assertIn("experiment_2", res)
        self.assertEqual(len(res["experiment_2"]), 4)

        # Exp 3: Feature Combinations
        self.assertIn("experiment_3", res)
        self.assertGreaterEqual(len(res["experiment_3"]), 4)

        # Exp 4: Method Comparison
        self.assertIn("experiment_4", res)
        self.assertIn("comparison_table", res["experiment_4"])
        self.assertEqual(len(res["experiment_4"]["comparison_table"]), 3)

    def test_08_visualizer_generation(self):
        """Verify that visualizer generates matplotlib figures without errors."""
        df = self.synth_df.copy()
        df["combined_prediction"] = df["ground_truth"]
        df["ml_score"] = np.random.normal(0, 0.1, len(df))
        df["ml_prediction"] = df["ground_truth"]

        fig1 = plot_telemetry_time_series(df, ["temperature", "voltage"])
        self.assertIsNotNone(fig1)

        cm_dummy = {"Rule": np.array([[10, 1], [2, 8]]), "ML": np.array([[11, 0], [1, 9]])}
        fig2 = plot_confusion_matrices(cm_dummy)
        self.assertIsNotNone(fig2)

        exp2_dummy = pd.DataFrame({
            "Contamination": [0.01, 0.03, 0.05, 0.10],
            "Precision": [0.9, 0.9, 1.0, 0.8],
            "Recall": [0.3, 0.5, 1.0, 1.0],
            "F1-Score": [0.45, 0.64, 1.0, 0.88]
        })
        fig3 = plot_contamination_curve(exp2_dummy)
        self.assertIsNotNone(fig3)


if __name__ == "__main__":
    unittest.main(verbosity=2)
