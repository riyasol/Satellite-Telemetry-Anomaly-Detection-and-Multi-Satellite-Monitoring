"""
main.py
Satellite Telemetry Anomaly Detection and Multi-Satellite Monitoring
Main execution pipeline fulfilling approved IML project proposal requirements:
  1. Multi-satellite telemetry ingestion, cleaning, validation, and feature preparation
  2. Rule-based safety threshold anomaly detection with parameter violation tracking
  3. Genuine scikit-learn Isolation Forest model training, inference, and anomaly scoring
  4. Combined (OR-logic) detection strategy integrating rules and ML
  5. Rigorous time-series train/test evaluation (Accuracy, Precision, Recall, F1, Confusion Matrix)
  6. Automated execution of all 4 approved proposal experiments and CSV report generation
  7. Automated generation and export of all visual figures to reports/figures/
"""

import os
import sys
import numpy as np
import pandas as pd

# Add src to path for direct invocation
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data_loader import load_telemetry_data, prepare_features, get_project_root
from anomaly_detector import RuleBasedDetector
from ml_anomaly_detector import SatelliteIsolationForest, train_test_split_telemetry
from evaluation import evaluate_all_detectors, format_confusion_matrix_text, compute_metrics
from experiments import run_all_experiments
from visualizer import save_all_figures


def run_pipeline(contamination: float = 0.05, test_size: float = 0.30):
    root_dir = get_project_root()
    reports_dir = os.path.join(root_dir, "reports")
    figures_dir = os.path.join(reports_dir, "figures")
    models_dir = os.path.join(root_dir, "models")

    os.makedirs(reports_dir, exist_ok=True)
    os.makedirs(figures_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)

    print("\n" + "=" * 80)
    print("🛰️  SATELLITE TELEMETRY ANOMALY DETECTION & MULTI-SATELLITE MONITORING")
    print("   Course: Introduction to Machine Learning (IML) | Team Project Implementation")
    print("=" * 80)

    # 1. Ingestion & Preprocessing (Riya's contribution)
    print("\n[Step 1] Ingesting and Preprocessing Telemetry Data...")
    df = load_telemetry_data()
    _, features = prepare_features(df)
    satellites = list(df["satellite_id"].unique())
    print(f"   ✅ Loaded {len(df)} telemetry observations across {len(satellites)} satellites: {satellites}")
    print(f"   ✅ Monitored Physical Parameters: {features}")

    if "ground_truth" in df.columns:
        n_gt = int(df["ground_truth"].sum())
        print(f"   🎯 Ground-Truth Anomalies: {n_gt} ({100.0 * n_gt / len(df):.2f}% of observations)")

    # 2. Chronological Train/Test Split (Time-based to prevent leakage)
    print(f"\n[Step 2] Splitting Telemetry Chronologically (Test Size = {int(test_size*100)}%)...")
    X_train, X_test, y_train, y_test = train_test_split_telemetry(df, features, test_size=test_size)
    print(f"   - Training Baseline:  {len(X_train)} samples (Nominal operating pattern learning)")
    print(f"   - Testing Partition:  {len(X_test)} samples (Unseen evaluation period)")

    # 3. Model Training (Isolation Forest - Shreya's contribution)
    print(f"\n[Step 3] Training Isolation Forest Model (contamination={contamination})...")
    ml_model = SatelliteIsolationForest(contamination=contamination, random_state=42)
    ml_model.fit(X_train)
    print("   ✅ Isolation Forest successfully trained on nominal baseline features.")

    # Save model artifacts
    model_path = os.path.join(models_dir, "isolation_forest_model.joblib")
    ml_model.save(model_path)
    # Also save backward-compatible name
    ml_model.save(os.path.join(models_dir, "isolation_forest_synthetic.joblib"))

    # 4. Inference & Detection Pipeline (Jayanth's integration)
    print("\n[Step 4] Running Multi-Satellite Anomaly Detection...")

    # A. Rule-Based Threshold Detection (Gagan's contribution)
    rule_detector = RuleBasedDetector()
    rule_preds, rule_violations = rule_detector.predict(df)
    df["rule_prediction"] = rule_preds
    df["rule_violations"] = rule_violations

    # B. Machine Learning Isolation Forest (Shreya's contribution)
    ml_preds, ml_scores = ml_model.predict_with_scores(df[features])
    df["ml_prediction"] = ml_preds
    df["ml_score"] = ml_scores
    df["ml_result"] = np.where(ml_preds == 1, "ML Anomaly", "Normal")

    # C. Combined Detection (OR-Logic)
    df["combined_prediction"] = ((df["rule_prediction"] == 1) | (df["ml_prediction"] == 1)).astype(int)
    df["combined_status"] = np.where(
        (df["rule_prediction"] == 1) & (df["ml_prediction"] == 1), "Both (Rule + ML)",
        np.where(df["rule_prediction"] == 1, "Rule Anomaly",
                 np.where(df["ml_prediction"] == 1, "ML Anomaly", "Normal"))
    )

    print(f"   - Rule-Based Detections:       {df['rule_prediction'].sum():>4} ({100*df['rule_prediction'].mean():.2f}%)")
    print(f"   - Isolation Forest Detections: {df['ml_prediction'].sum():>4} ({100*df['ml_prediction'].mean():.2f}%)")
    print(f"   - Combined (OR-Logic) Flags:   {df['combined_prediction'].sum():>4} ({100*df['combined_prediction'].mean():.2f}%)")

    # 5. Performance Evaluation on Unseen Test Partition
    test_start = len(X_train)
    test_df = df.iloc[test_start:]
    if y_test is not None:
        print("\n" + "-" * 80)
        print("📈 TEST SET EVALUATION METRICS (Chronologically Unseen Data)")
        print("-" * 80)
        comp_df = evaluate_all_detectors(
            y_test.values,
            test_df["rule_prediction"].values,
            test_df["ml_prediction"].values,
            test_df["combined_prediction"].values
        )
        print(comp_df.to_string(index=False))

        print("\nConfusion Matrices:")
        for m_name, preds in [
            ("Rule-Based", test_df["rule_prediction"]),
            ("Isolation Forest", test_df["ml_prediction"]),
            ("Combined (OR)", test_df["combined_prediction"])
        ]:
            cm = compute_metrics(y_test.values, preds.values)["confusion_matrix"]
            print("\n" + format_confusion_matrix_text(cm, m_name))

    # 6. Proposal Experiments Execution
    print("\n" + "-" * 80)
    print("🔬 EXECUTING 4 APPROVED PROPOSAL EXPERIMENTS")
    print("-" * 80)
    exp_results = run_all_experiments(
        df,
        features,
        rule_detector=rule_detector,
        contamination=contamination,
        test_size=test_size,
        save_reports_dir=reports_dir
    )

    print("\n[Experiment 1] Rule-Based vs Isolation Forest Overlap:")
    print(exp_results["experiment_1"]["overlap_table"].to_string(index=False))

    print("\n[Experiment 2] Contamination Sweep (0.01, 0.03, 0.05, 0.10):")
    print(exp_results["experiment_2"].to_string(index=False))

    print("\n[Experiment 3] Feature Combination Ablation Study:")
    print(exp_results["experiment_3"].to_string(index=False))

    print("\n[Experiment 4] Side-by-Side Method Performance Summary:")
    if "comparison_table" in exp_results["experiment_4"]:
        print(exp_results["experiment_4"]["comparison_table"].to_string(index=False))

    # 7. Generate and Export Visual Figures
    print("\n[Step 7] Generating High-Resolution Telemetry & Experiment Figures...")
    save_all_figures(df, exp_results, figures_dir)

    # 8. Save Comprehensive Analysis Report
    report_file = os.path.join(reports_dir, "telemetry_analysis_report.csv")
    df.to_csv(report_file, index=False)
    # Also save backward-compatible name
    df.to_csv(os.path.join(reports_dir, "telemetry_analysis_report_synthetic.csv"), index=False)
    print(f"💾 Full telemetry analysis report saved to: {report_file}")

    print("\n" + "=" * 80)
    print("✅ PIPELINE EXECUTION COMPLETED SUCCESSFULLY!")
    print("   To launch the Streamlit monitoring interface, run:")
    print("   streamlit run src/dashboard.py")
    print("=" * 80)

    return df, exp_results


def main():
    run_pipeline(contamination=0.05, test_size=0.30)


if __name__ == "__main__":
    main()
