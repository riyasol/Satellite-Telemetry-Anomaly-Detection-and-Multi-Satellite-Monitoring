"""
main.py
Satellite Telemetry Anomaly Detection and Multi-Satellite Monitoring
Main execution pipeline fulfilling approved IML project proposal requirements:
  1. Multi-source telemetry ingestion: Controlled multi-satellite data (SAT-01..03)
     and actual ESA Mission 1 spacecraft telemetry channels.
  2. Rule-based threshold anomaly detection with parameter violation tracking.
  3. Genuine scikit-learn Isolation Forest model training, inference, and anomaly scoring.
  4. Combined (OR-logic) detection strategy.
  5. Rigorous train/test evaluation (Accuracy, Precision, Recall, F1, Confusion Matrix).
  6. Automated execution of all 4 approved proposal experiments and CSV report generation.
  7. Automated generation and export of visual figures to reports/figures/.
"""

import os
import sys
import numpy as np
import pandas as pd

# Add src to path for direct invocation
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data_loader import load_dataset, get_project_root
from anomaly_detector import RuleBasedDetector
from ml_anomaly_detector import SatelliteIsolationForest, train_test_split_telemetry
from evaluation import evaluate_all_detectors, format_confusion_matrix_text, compute_metrics
from experiments import run_all_experiments
from visualizer import save_all_figures, plot_telemetry_time_series
import matplotlib.pyplot as plt


def run_pipeline(source: str = "synthetic", contamination: float = 0.05, test_size: float = 0.30):
    root_dir = get_project_root()
    reports_dir = os.path.join(root_dir, "reports")
    figures_dir = os.path.join(reports_dir, "figures")
    models_dir = os.path.join(root_dir, "models")

    os.makedirs(reports_dir, exist_ok=True)
    os.makedirs(figures_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)

    print("\n" + "=" * 80)
    print(f"🛰️  RUNNING PIPELINE FOR DATA SOURCE: {source.upper()}")
    print("=" * 80)

    # 1. Load Data
    if source.lower() == "synthetic":
        df = load_dataset("synthetic")
        features = ["temperature", "voltage", "current", "orientation"]
        rule_detector = RuleBasedDetector()
    else:
        df = load_dataset("esa", start_time="2004-11-25", end_time="2004-12-05", max_records=10000)
        features = [c for c in df.columns if c.startswith("channel_")]
        rule_detector = RuleBasedDetector()
        # Derive 3-sigma statistical thresholds from baseline training period
        split_baseline = int(len(df) * (1.0 - test_size))
        rule_detector.fit_statistical_limits(df.iloc[:split_baseline], features, n_sigma=3.0)

    print(f"📊 Dataset Loaded: {len(df)} records | Numerical Features: {features}")
    if "ground_truth" in df.columns:
        n_gt = int(df["ground_truth"].sum())
        print(f"🎯 Ground-Truth Anomalies Present: {n_gt} ({100.0 * n_gt / len(df):.2f}%)")

    # 2. Chronological Train/Test Split
    X_train, X_test, y_train, y_test = train_test_split_telemetry(df, features, test_size=test_size)
    print(f"\n📂 Train/Test Split (Time-series chronological):")
    print(f"   - Training Baseline: {len(X_train)} samples")
    print(f"   - Testing Partition: {len(X_test)} samples (Unseen evaluation period)")

    # 3. Model Training (Isolation Forest)
    print(f"\n🤖 Training Isolation Forest Model (contamination={contamination})...")
    ml_model = SatelliteIsolationForest(contamination=contamination, random_state=42)
    ml_model.fit(X_train)
    print("   ✅ Isolation Forest successfully trained on nominal baseline telemetry.")

    # Save trained model
    model_save_path = os.path.join(models_dir, f"isolation_forest_{source}.joblib")
    ml_model.save(model_save_path)
    if source == "synthetic":
        ml_model.save(os.path.join(models_dir, "isolation_forest_model.joblib"))

    # 4. Inference on Full Dataset
    print("\n🔍 Running Anomaly Detection...")
    # A. Rule-Based
    rule_preds, rule_violations = rule_detector.predict(df)
    df["rule_prediction"] = rule_preds
    df["rule_violations"] = rule_violations

    # B. Isolation Forest ML
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

    print(f"   - Rule-Based Anomalies Detected:    {df['rule_prediction'].sum():>4} ({100*df['rule_prediction'].mean():.2f}%)")
    print(f"   - Isolation Forest Anomalies:       {df['ml_prediction'].sum():>4} ({100*df['ml_prediction'].mean():.2f}%)")
    print(f"   - Combined (OR-Logic) Anomalies:    {df['combined_prediction'].sum():>4} ({100*df['combined_prediction'].mean():.2f}%)")

    # 5. Model Evaluation (on unseen test set)
    test_start = len(X_train)
    test_df = df.iloc[test_start:]
    if y_test is not None:
        print("\n" + "-" * 70)
        print("📈 TEST SET EVALUATION METRICS (Unseen Data)")
        print("-" * 70)
        comp_df = evaluate_all_detectors(
            y_test.values,
            test_df["rule_prediction"].values,
            test_df["ml_prediction"].values,
            test_df["combined_prediction"].values
        )
        print(comp_df.to_string(index=False))

        for m_name, preds in [
            ("Rule-Based", test_df["rule_prediction"]),
            ("Isolation Forest", test_df["ml_prediction"]),
            ("Combined", test_df["combined_prediction"])
        ]:
            cm = compute_metrics(y_test.values, preds.values)["confusion_matrix"]
            print("\n" + format_confusion_matrix_text(cm, m_name))

    # 6. Run Proposal Experiments
    print("\n" + "-" * 70)
    print("🔬 EXECUTING APPROVED IML PROPOSAL EXPERIMENTS")
    print("-" * 70)
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

    print("\n[Experiment 2] Contamination Parameter Sweep (0.01, 0.03, 0.05, 0.10):")
    print(exp_results["experiment_2"].to_string(index=False))

    print("\n[Experiment 3] Feature Combination Comparison:")
    print(exp_results["experiment_3"].to_string(index=False))

    print("\n[Experiment 4] Final Method Comparison Table:")
    if "comparison_table" in exp_results["experiment_4"]:
        print(exp_results["experiment_4"]["comparison_table"].to_string(index=False))

    # 7. Generate and Export Figures
    if source == "synthetic":
        save_all_figures(df, exp_results, figures_dir)
    else:
        # Save ESA telemetry figure
        fig_esa = plot_telemetry_time_series(df.iloc[:2000], feature_cols=features, title_suffix="(ESA Mission 1 Subsystem 6)")
        fig_esa.savefig(os.path.join(figures_dir, "fig6_esa_mission1_telemetry.png"), dpi=200)
        plt.close(fig_esa)

    # 8. Save Analysis Report
    report_file = os.path.join(reports_dir, f"telemetry_analysis_report_{source}.csv")
    df.to_csv(report_file, index=False)
    if source == "synthetic":
        df.to_csv(os.path.join(reports_dir, "telemetry_analysis_report.csv"), index=False)

    print(f"\n💾 Full analysis report saved to: {report_file}")
    return df, exp_results


def main():
    print("=" * 80)
    print("🚀 SATELLITE TELEMETRY ANOMALY DETECTION & MULTI-SATELLITE MONITORING")
    print("   Course: Introduction to Machine Learning (IML) | Team Project Implementation")
    print("=" * 80)

    # 1. Run Controlled Multi-Satellite Synthetic Telemetry
    df_synth, exp_synth = run_pipeline("synthetic", contamination=0.05)

    # 2. Run Actual ESA Mission 1 Spacecraft Telemetry Benchmark
    df_esa, exp_esa = run_pipeline("esa", contamination=0.05)

    print("\n" + "=" * 80)
    print("✅ BOTH PIPELINES EXECUTED SUCCESSFULLY!")
    print("   To launch the interactive dashboard, run:")
    print("   streamlit run src/dashboard.py")
    print("=" * 80)


if __name__ == "__main__":
    main()
