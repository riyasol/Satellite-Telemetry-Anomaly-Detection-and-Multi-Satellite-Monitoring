"""
dashboard.py
Interactive Streamlit Dashboard for:
Satellite Telemetry Anomaly Detection and Multi-Satellite Monitoring
Course: Introduction to Machine Learning (IML)
Team Members:
  - Riya (IIT2024134)
  - Shreya (IIT2024137)
  - Gagan (IIT2024133)
  - Jayanth (IIT2024155)

Implements the 5 core monitoring and evaluation sections:
  1. Overview (Total observations, normal readings, detected anomalies, anomaly rate, detector breakdown)
  2. Dataset (Preview, numerical telemetry parameters, descriptive statistics, min/max ranges)
  3. Anomaly Detection (Rule-based, Isolation Forest, Combined OR-logic, parameter violations)
  4. Experiments & Evaluation (Experiments 1-4: overlap, contamination sweep, feature ablation, confusion matrices)
  5. Visualizations & Reports (Time-series plots with anomaly markers, anomaly score distribution, CSV reports)
"""

import os
import sys
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

# Ensure local imports work reliably
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data_loader import load_telemetry_data, prepare_features, get_project_root
from anomaly_detector import RuleBasedDetector
from ml_anomaly_detector import SatelliteIsolationForest, train_test_split_telemetry
from evaluation import evaluate_all_detectors, compute_metrics
from experiments import (
    run_experiment_1,
    run_experiment_2_contamination,
    run_experiment_3_feature_combinations,
    run_experiment_4_method_comparison
)
from visualizer import (
    plot_telemetry_time_series,
    plot_confusion_matrices,
    plot_contamination_curve,
    plot_feature_combinations_bar,
    plot_anomaly_score_distribution
)

# Page configuration
st.set_page_config(
    page_title="Satellite Telemetry Monitoring System",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom UI styling
st.markdown("""
<style>
    .main-title { font-size: 2.1rem; font-weight: 700; color: #1E3A8A; margin-bottom: 0.1rem; }
    .sub-title { font-size: 1.0rem; color: #4B5563; margin-bottom: 1.2rem; }
    .badge-sat { background-color: #DBEAFE; color: #1E40AF; padding: 4px 10px; border-radius: 12px; font-weight: 600; font-size: 0.85rem; }
</style>
""", unsafe_allow_html=True)


@st.cache_data(show_spinner=False)
def get_cached_dataset():
    """Load telemetry dataset with Streamlit caching for responsive interaction."""
    df = load_telemetry_data()
    _, features = prepare_features(df)
    return df, features


# ==========================================
# SIDEBAR CONTROLS
# ==========================================
st.sidebar.image("https://img.icons8.com/fluency/96/satellite.png", width=64)
st.sidebar.title("Mission Controls")

# Satellite Filter
df_raw, feature_cols = get_cached_dataset()
all_satellites = list(df_raw["satellite_id"].unique())
selected_sat = st.sidebar.selectbox(
    "Active Satellite Monitor:",
    ["All Satellites"] + all_satellites,
    help="Select an individual satellite or monitor constellation aggregate."
)

st.sidebar.markdown("---")
st.sidebar.subheader("Isolation Forest Hyperparameters")

contamination = st.sidebar.slider(
    "Contamination (Expected Outlier Ratio):",
    min_value=0.01,
    max_value=0.15,
    value=0.05,
    step=0.01,
    help="Assumed anomaly ratio used by scikit-learn IsolationForest"
)

n_estimators = st.sidebar.slider(
    "Number of Trees (n_estimators):",
    min_value=50,
    max_value=200,
    value=100,
    step=25,
    help="Number of isolation trees in the ensemble"
)

test_size = st.sidebar.slider(
    "Evaluation Test Size (%):",
    min_value=10,
    max_value=50,
    value=30,
    step=5,
    help="Fraction of chronological observations reserved for unseen testing"
) / 100.0

st.sidebar.markdown("---")
st.sidebar.markdown("""
**Project Team:**
- Riya (IIT2024134)
- Shreya (IIT2024137)
- Gagan (IIT2024133)
- Jayanth (IIT2024155)
""")

# ==========================================
# PIPELINE EXECUTION
# ==========================================
# Filter by satellite if selected
if selected_sat != "All Satellites":
    df = df_raw[df_raw["satellite_id"] == selected_sat].copy().reset_index(drop=True)
else:
    df = df_raw.copy()

# 1. Setup Rule-Based Detector
rule_detector = RuleBasedDetector()

# 2. Chronological Train/Test Split
X_train, X_test, y_train, y_test = train_test_split_telemetry(df, feature_cols, test_size=test_size)

# 3. Fit Isolation Forest
ml_model = SatelliteIsolationForest(
    contamination=contamination,
    n_estimators=n_estimators,
    random_state=42
)
ml_model.fit(X_train)

# 4. Inferences
rule_preds, rule_violations = rule_detector.predict(df)
df["rule_prediction"] = rule_preds
df["rule_violations"] = rule_violations

ml_preds, ml_scores = ml_model.predict_with_scores(df[feature_cols])
df["ml_prediction"] = ml_preds
df["ml_score"] = ml_scores
df["ml_result"] = np.where(ml_preds == 1, "ML Anomaly", "Normal")

df["combined_prediction"] = ((df["rule_prediction"] == 1) | (df["ml_prediction"] == 1)).astype(int)
df["combined_status"] = np.where(
    (df["rule_prediction"] == 1) & (df["ml_prediction"] == 1), "Both (Rule + ML)",
    np.where(df["rule_prediction"] == 1, "Rule Anomaly",
             np.where(df["ml_prediction"] == 1, "ML Anomaly", "Normal"))
)

# ==========================================
# MAIN DASHBOARD INTERFACE
# ==========================================
st.markdown('<div class="main-title">🛰️ Satellite Telemetry Anomaly Detection & Monitoring</div>', unsafe_allow_html=True)
st.markdown(f'<span class="badge-sat">MONITORING CONTEXT: {selected_sat.upper()}</span> &nbsp; | &nbsp; <b>Features:</b> {", ".join(feature_cols)}', unsafe_allow_html=True)
st.write("")

# 5 Core Tabs fulfilling proposal specifications
tab_overview, tab_dataset, tab_detection, tab_experiments, tab_viz_reports = st.tabs([
    "1. 📊 Overview",
    "2. 📁 Dataset",
    "3. 🔍 Anomaly Detection",
    "4. 🔬 Experiments & Evaluation",
    "5. 📈 Visualizations & Reports"
])

# -------------------------------------------------------------
# TAB 1: OVERVIEW
# -------------------------------------------------------------
with tab_overview:
    st.subheader("System Health Overview & Telemetry Metrics")

    total_records = len(df)
    total_anomalies = int(df["combined_prediction"].sum())
    normal_readings = total_records - total_anomalies
    anomaly_pct = (total_anomalies / total_records) * 100.0 if total_records > 0 else 0.0

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Observations", f"{total_records:,}")
    with col2:
        st.metric("Normal Readings", f"{normal_readings:,}", delta=f"{100-anomaly_pct:.1f}%")
    with col3:
        st.metric("Detected Anomalies", f"{total_anomalies:,}", delta=f"-{anomaly_pct:.1f}%", delta_color="inverse")
    with col4:
        st.metric("Anomaly Rate", f"{anomaly_pct:.2f}%")

    st.markdown("---")
    st.subheader("Anomaly Detection Breakdown")

    rule_count = int(df["rule_prediction"].sum())
    ml_count = int(df["ml_prediction"].sum())
    both_count = int(((df["rule_prediction"] == 1) & (df["ml_prediction"] == 1)).sum())

    m_col1, m_col2, m_col3 = st.columns(3)
    with m_col1:
        st.info(f"**Rule-Based Baseline**\n\n• Triggered: **{rule_count}** observations ({100*rule_count/total_records:.2f}%)\n\n• Fixed safety bounds on individual telemetry parameters (temperature, voltage, current, orientation).")
    with m_col2:
        st.success(f"**Isolation Forest (ML)**\n\n• Triggered: **{ml_count}** observations ({100*ml_count/total_records:.2f}%)\n\n• Unsupervised ensemble isolating multivariate and non-linear telemetry deviations.")
    with m_col3:
        st.warning(f"**Combined (OR-Logic)**\n\n• Triggered: **{total_anomalies}** total anomalies ({anomaly_pct:.2f}%)\n\n• Agreement (flagged by both): **{both_count}** observations.")

# -------------------------------------------------------------
# TAB 2: DATASET
# -------------------------------------------------------------
with tab_dataset:
    st.subheader("Telemetry Dataset Preview & Descriptive Statistics")
    st.markdown(f"**Monitored Satellites:** `{all_satellites}` | **Observations:** `{len(df)}` | **Numerical Parameters:** `{len(feature_cols)}`")

    st.write("##### Sample Telemetry Observations (First 15 Rows)")
    preview_cols = ["timestamp", "satellite_id"] + feature_cols
    if "anomaly_type" in df.columns:
        preview_cols.append("anomaly_type")
    st.dataframe(df[preview_cols].head(15), use_container_width=True)

    st.markdown("---")
    col_stat1, col_stat2 = st.columns([2, 1])

    with col_stat1:
        st.write("##### Numerical Parameter Summary Statistics")
        st.dataframe(df[feature_cols].describe().round(3), use_container_width=True)

    with col_stat2:
        st.write("##### Telemetry Channel Specifications")
        param_data = []
        for col in feature_cols:
            param_data.append({
                "Parameter": col,
                "Data Type": str(df[col].dtype),
                "Min Value": f"{df[col].min():.2f}",
                "Max Value": f"{df[col].max():.2f}"
            })
        st.dataframe(pd.DataFrame(param_data), use_container_width=True)

# -------------------------------------------------------------
# TAB 3: ANOMALY DETECTION
# -------------------------------------------------------------
with tab_detection:
    st.subheader("Rule-Based vs ML Anomaly Detection Outputs")

    filter_col1, filter_col2 = st.columns([1, 1])
    with filter_col1:
        view_filter = st.selectbox(
            "Filter Observations to Display:",
            ("All Observations", "Only Combined Anomalies", "Only Rule-Based Anomalies", "Only ML Anomalies")
        )
    with filter_col2:
        max_rows_display = st.slider("Max Rows to Render:", 10, 500, 50)

    # Apply filter
    df_filtered = df.copy()
    if view_filter == "Only Combined Anomalies":
        df_filtered = df_filtered[df_filtered["combined_prediction"] == 1]
    elif view_filter == "Only Rule-Based Anomalies":
        df_filtered = df_filtered[df_filtered["rule_prediction"] == 1]
    elif view_filter == "Only ML Anomalies":
        df_filtered = df_filtered[df_filtered["ml_prediction"] == 1]

    st.write(f"Showing **{len(df_filtered)}** matching telemetry observations:")

    display_cols = ["timestamp", "satellite_id"] + feature_cols + [
        "rule_prediction", "rule_violations", "ml_prediction", "ml_score", "combined_status"
    ]
    if "anomaly_type" in df.columns:
        display_cols.append("anomaly_type")

    st.dataframe(df_filtered[display_cols].head(max_rows_display), use_container_width=True)

    st.markdown("---")
    st.subheader("Configured Safety Thresholds (Rule-Based Baseline)")
    thresh_records = []
    for param, bounds in rule_detector.thresholds.items():
        min_str = f"{bounds.get('min'):.2f}" if bounds.get("min") is not None else "None"
        max_str = f"{bounds.get('max'):.2f}" if bounds.get("max") is not None else "None"
        thresh_records.append({
            "Parameter": str(param),
            "Lower Safety Bound": min_str,
            "Upper Safety Bound": max_str,
            "Engineering Limit Description": str(bounds.get("description", "Safety Threshold"))
        })
    st.dataframe(pd.DataFrame(thresh_records), use_container_width=True)

# -------------------------------------------------------------
# TAB 4: EXPERIMENTS & EVALUATION
# -------------------------------------------------------------
with tab_experiments:
    st.subheader("Approved Proposal Experiments & Performance Evaluation")

    exp_tab1, exp_tab2, exp_tab3, exp_tab4 = st.tabs([
        "Exp 1: Rule vs ML",
        "Exp 2: Contamination Study",
        "Exp 3: Feature Combinations",
        "Exp 4: Method Comparison & Confusion Matrices"
    ])

    # EXP 1
    with exp_tab1:
        st.write("#### Experiment 1: Comparison of Rule-Based Detection vs Isolation Forest")
        st.caption("Analyzes agreement rate and detection differences on chronologically unseen test partition.")
        exp1_res = run_experiment_1(df, feature_cols, rule_detector, ml_model, test_size=test_size)
        st.dataframe(exp1_res["overlap_table"], use_container_width=True)
        st.info(f"Overall agreement rate between Rule-Based and Isolation Forest: **{exp1_res['agreement_percentage']}%**")

    # EXP 2
    with exp_tab2:
        st.write("#### Experiment 2: Contamination Parameter Sensitivity Sweep")
        st.caption("Testing contamination parameter values: 0.01, 0.03, 0.05, 0.10 on unseen test set.")
        exp2_df = run_experiment_2_contamination(df, feature_cols, contaminations=[0.01, 0.03, 0.05, 0.10], test_size=test_size)
        st.dataframe(exp2_df, use_container_width=True)

        fig_exp2 = plot_contamination_curve(exp2_df)
        st.pyplot(fig_exp2)
        plt.close(fig_exp2)

    # EXP 3
    with exp_tab3:
        st.write("#### Experiment 3: Telemetry Feature Combination Ablation Study")
        st.caption("Comparing univariate sensors, bivariate sensor pairs, and full multivariate feature sets.")
        exp3_df = run_experiment_3_feature_combinations(df, feature_cols=feature_cols, contamination=contamination, test_size=test_size)
        st.dataframe(exp3_df, use_container_width=True)

        fig_exp3 = plot_feature_combinations_bar(exp3_df)
        st.pyplot(fig_exp3)
        plt.close(fig_exp3)

    # EXP 4
    with exp_tab4:
        st.write("#### Experiment 4: Comprehensive Comparison of Rule-Based vs ML vs Combined")
        st.caption("Side-by-side performance evaluation on unseen test partition.")
        exp4_res = run_experiment_4_method_comparison(df, feature_cols, rule_detector, contamination=contamination, test_size=test_size)

        if "comparison_table" in exp4_res:
            st.dataframe(exp4_res["comparison_table"], use_container_width=True)

        if "confusion_matrices" in exp4_res:
            st.write("##### Confusion Matrices (Unseen Test Set)")
            fig_cm = plot_confusion_matrices(exp4_res["confusion_matrices"])
            st.pyplot(fig_cm)
            plt.close(fig_cm)

# -------------------------------------------------------------
# TAB 5: VISUALIZATIONS & REPORTS
# -------------------------------------------------------------
with tab_viz_reports:
    st.subheader("Telemetry Time-Series Visualizations & Report Export")

    # Time series visualization
    st.write("##### Telemetry Health Monitoring Over Time (Normal vs Anomaly)")
    fig_ts = plot_telemetry_time_series(
        df,
        feature_cols=feature_cols,
        anomaly_col="combined_prediction",
        title_suffix=f"({selected_sat})"
    )
    st.pyplot(fig_ts)
    plt.close(fig_ts)

    st.markdown("---")
    st.write("##### Isolation Forest Continuous Anomaly Score Distribution")
    fig_dist = plot_anomaly_score_distribution(df["ml_score"].values, df["ml_prediction"].values)
    st.pyplot(fig_dist)
    plt.close(fig_dist)

    st.markdown("---")
    st.subheader("Export Project Reports")
    exp_col1, exp_col2 = st.columns(2)

    with exp_col1:
        st.write("##### Full Telemetry Analysis Dataset")
        csv_telemetry = df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="📥 Download Telemetry Analysis Report (CSV)",
            data=csv_telemetry,
            file_name="telemetry_analysis_report.csv",
            mime="text/csv"
        )

    with exp_col2:
        st.write("##### Experiment 4 Method Comparison Report")
        if "comparison_table" in exp4_res:
            csv_exp = exp4_res["comparison_table"].to_csv(index=False).encode("utf-8")
            st.download_button(
                label="📥 Download Experiment 4 Comparison (CSV)",
                data=csv_exp,
                file_name="experiment_4_method_comparison.csv",
                mime="text/csv"
            )

st.markdown("---")
st.caption("Satellite Telemetry Anomaly Detection and Multi-Satellite Monitoring | Introduction to Machine Learning (IML)")
