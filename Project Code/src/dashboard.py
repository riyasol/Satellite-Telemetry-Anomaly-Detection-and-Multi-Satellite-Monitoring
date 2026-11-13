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

Dual Dataset Capabilities:
  1. Synthetic Telemetry (Controlled): Multi-satellite simulation (SAT-01, SAT-02, SAT-03)
     monitoring temperature, voltage, current, orientation with orbital cycles and injected faults.
  2. ESA Mission 1 Benchmark: Actual European Space Agency spacecraft telemetry channels
     (Subsystem 6 Group 3) with real operational anomaly labels from labels.csv.
"""

import os
import sys
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

# Ensure local imports work reliably
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data_loader import load_dataset, get_project_root
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
    page_title="Satellite Telemetry Anomaly Detection",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom UI styling
st.markdown("""
<style>
    .main-title { font-size: 2.1rem; font-weight: 700; color: #1E3A8A; margin-bottom: 0.1rem; }
    .sub-title { font-size: 1.0rem; color: #4B5563; margin-bottom: 1.2rem; }
    .badge-synth { background-color: #DBEAFE; color: #1E40AF; padding: 4px 10px; border-radius: 12px; font-weight: 600; font-size: 0.85rem; }
    .badge-esa { background-color: #FEF3C7; color: #92400E; padding: 4px 10px; border-radius: 12px; font-weight: 600; font-size: 0.85rem; }
</style>
""", unsafe_allow_html=True)


@st.cache_data(show_spinner=False)
def get_cached_dataset(source: str):
    """Load dataset with Streamlit caching for responsive UI interaction."""
    if source == "Synthetic Telemetry (Controlled)":
        df = load_dataset("synthetic")
        features = ["temperature", "voltage", "current", "orientation"]
    else:
        df = load_dataset("esa", start_time="2004-11-25", end_time="2004-12-05", max_records=10000)
        features = [c for c in df.columns if c.startswith("channel_")]
    return df, features


# ==========================================
# SIDEBAR CONTROLS
# ==========================================
st.sidebar.image("https://img.icons8.com/fluency/96/satellite.png", width=64)
st.sidebar.title("Telemetry Controls")

data_source = st.sidebar.radio(
    "Select Telemetry Source:",
    ("Synthetic Telemetry (Controlled)", "ESA Mission 1 Benchmark"),
    help="Switch between synthetic multi-satellite demonstration and real ESA Mission 1 spacecraft telemetry."
)

st.sidebar.markdown("---")
st.sidebar.subheader("Isolation Forest Parameters")

contamination = st.sidebar.slider(
    "Contamination (Expected Outlier Ratio):",
    min_value=0.01,
    max_value=0.15,
    value=0.05,
    step=0.01,
    help="Assumed anomaly ratio used by scikit-learn IsolationForest"
)

test_size = st.sidebar.slider(
    "Evaluation Test Size (%):",
    min_value=10,
    max_value=50,
    value=30,
    step=5,
    help="Fraction of data reserved for unseen testing"
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
# DATA LOADING & MODEL EXECUTION
# ==========================================
df_raw, feature_cols = get_cached_dataset(data_source)
df = df_raw.copy()

# Setup Rule-based detector
rule_detector = RuleBasedDetector()
if data_source != "Synthetic Telemetry (Controlled)":
    # Derive statistical thresholds from baseline period
    split_idx = int(len(df) * (1.0 - test_size))
    rule_detector.fit_statistical_limits(df.iloc[:split_idx], feature_cols, n_sigma=3.0)

# Train/Test Split
X_train, X_test, y_train, y_test = train_test_split_telemetry(df, feature_cols, test_size=test_size)

# Train Isolation Forest ML Model
ml_model = SatelliteIsolationForest(contamination=contamination, random_state=42)
ml_model.fit(X_train)

# Inference on Full Dataset
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
st.markdown('<div class="main-title">🛰️ Satellite Telemetry Health & Anomaly Monitoring</div>', unsafe_allow_html=True)

if data_source == "Synthetic Telemetry (Controlled)":
    st.markdown('<span class="badge-synth">SOURCE: Synthetic Controlled Telemetry (SAT-01, SAT-02, SAT-03)</span>', unsafe_allow_html=True)
else:
    st.markdown('<span class="badge-esa">SOURCE: Actual ESA Mission 1 Benchmark (Subsystem 6 Telemetry Channels)</span>', unsafe_allow_html=True)

st.write("")

# 5 Main Tabs per Approved IML Proposal Requirements
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
    st.subheader("Detection Methods Summary Breakdown")

    rule_count = int(df["rule_prediction"].sum())
    ml_count = int(df["ml_prediction"].sum())
    both_count = int(((df["rule_prediction"] == 1) & (df["ml_prediction"] == 1)).sum())

    m_col1, m_col2, m_col3 = st.columns(3)
    with m_col1:
        st.info(f"**Rule-Based Detector**\n\n• Triggered: **{rule_count}** observations ({100*rule_count/total_records:.2f}%)\n\n• Safety thresholds evaluated on individual telemetry parameters.")
    with m_col2:
        st.success(f"**Isolation Forest (ML)**\n\n• Triggered: **{ml_count}** observations ({100*ml_count/total_records:.2f}%)\n\n• Unsupervised multivariate isolation trees detecting multi-dimensional outliers.")
    with m_col3:
        st.warning(f"**Combined (OR-Logic)**\n\n• Triggered: **{total_anomalies}** total anomalies ({anomaly_pct:.2f}%)\n\n• Overlap agreement count: **{both_count}** observations.")

# -------------------------------------------------------------
# TAB 2: DATASET
# -------------------------------------------------------------
with tab_dataset:
    st.subheader("Telemetry Dataset Preview & Descriptive Statistics")

    st.markdown(f"**Active Source:** `{data_source}` | **Total Records:** `{len(df)}` | **Available Features:** `{len(feature_cols)}`")

    st.write("##### Sample Observations (First 15 Rows)")
    preview_cols = ["timestamp", "satellite_id"] + feature_cols
    if "anomaly_type" in df.columns:
        preview_cols.append("anomaly_type")
    st.dataframe(df[preview_cols].head(15), use_container_width=True)

    st.markdown("---")
    col_stat1, col_stat2 = st.columns([2, 1])

    with col_stat1:
        st.write("##### Numerical Feature Summary Statistics")
        st.dataframe(df[feature_cols].describe().round(4), use_container_width=True)

    with col_stat2:
        st.write("##### Telemetry Parameters Specification")
        param_data = []
        for col in feature_cols:
            param_data.append({
                "Parameter": col,
                "Data Type": str(df[col].dtype),
                "Min Value": f"{df[col].min():.3f}",
                "Max Value": f"{df[col].max():.3f}"
            })
        st.dataframe(pd.DataFrame(param_data), use_container_width=True)

# -------------------------------------------------------------
# TAB 3: ANOMALY DETECTION
# -------------------------------------------------------------
with tab_detection:
    st.subheader("Independent and Combined Anomaly Predictions")

    filter_col1, filter_col2, filter_col3 = st.columns([1, 1, 1])
    with filter_col1:
        sat_options = ["All Satellites"] + list(df["satellite_id"].unique())
        selected_sat = st.selectbox("Filter Satellite / Channel Group:", sat_options)
    with filter_col2:
        view_filter = st.selectbox(
            "Filter Records Displayed:",
            ("All Observations", "Only Combined Anomalies", "Only Rule-Based Anomalies", "Only ML Anomalies")
        )
    with filter_col3:
        max_rows_display = st.slider("Max Rows to Render:", 10, 500, 50)

    # Apply filter
    df_filtered = df.copy()
    if selected_sat != "All Satellites":
        df_filtered = df_filtered[df_filtered["satellite_id"] == selected_sat]

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
    st.subheader("Parameter Thresholds Configuration")
    thresh_records = []
    for param, bounds in rule_detector.thresholds.items():
        min_str = f"{bounds.get('min'):.3f}" if bounds.get("min") is not None else "N/A"
        max_str = f"{bounds.get('max'):.3f}" if bounds.get("max") is not None else "N/A"
        thresh_records.append({
            "Parameter": str(param),
            "Lower Bound (Min)": min_str,
            "Upper Bound (Max)": max_str,
            "Description": str(bounds.get("description", "Safety Threshold"))
        })
    st.dataframe(pd.DataFrame(thresh_records), use_container_width=True)

# -------------------------------------------------------------
# TAB 4: EXPERIMENTS & EVALUATION
# -------------------------------------------------------------
with tab_experiments:
    st.subheader("Approved IML Proposal Experiments & Model Evaluation")

    exp_tab1, exp_tab2, exp_tab3, exp_tab4 = st.tabs([
        "Exp 1: Rule vs ML",
        "Exp 2: Contamination Study",
        "Exp 3: Feature Combinations",
        "Exp 4: Method Comparison & Confusion Matrices"
    ])

    # EXP 1
    with exp_tab1:
        st.write("#### Experiment 1: Comparison of Rule-Based Detection vs Isolation Forest")
        st.caption("Analyzes agreement and discrepancies between rule thresholding and ML isolation partitioning.")
        exp1_res = run_experiment_1(df, feature_cols, rule_detector, ml_model, test_size=test_size)
        st.dataframe(exp1_res["overlap_table"], use_container_width=True)
        st.info(f"Overall agreement rate on test partition: **{exp1_res['agreement_percentage']}%**")

    # EXP 2
    with exp_tab2:
        st.write("#### Experiment 2: Contamination Parameter Sensitivity Analysis")
        st.caption("Testing contamination parameter values: 0.01, 0.03, 0.05, 0.10 on unseen test set.")
        exp2_df = run_experiment_2_contamination(df, feature_cols, contaminations=[0.01, 0.03, 0.05, 0.10], test_size=test_size)
        st.dataframe(exp2_df, use_container_width=True)

        fig_exp2 = plot_contamination_curve(exp2_df)
        st.pyplot(fig_exp2)
        plt.close(fig_exp2)

    # EXP 3
    with exp_tab3:
        st.write("#### Experiment 3: Feature Combination Ablation Study")
        st.caption("Evaluating Isolation Forest across univariate, bivariate, and multivariate telemetry subsets.")
        exp3_df = run_experiment_3_feature_combinations(df, feature_cols=feature_cols, contamination=contamination, test_size=test_size)
        st.dataframe(exp3_df, use_container_width=True)

        fig_exp3 = plot_feature_combinations_bar(exp3_df)
        st.pyplot(fig_exp3)
        plt.close(fig_exp3)

    # EXP 4
    with exp_tab4:
        st.write("#### Experiment 4: Comprehensive Comparison of Rule-Based vs ML vs Combined")
        st.caption("Full side-by-side performance evaluation on unseen test partition.")
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
    st.subheader("Telemetry Time-Series Graphs & Report Generation")

    selected_viz_sat = df["satellite_id"].iloc[0]
    if len(df["satellite_id"].unique()) > 1:
        selected_viz_sat = st.selectbox("Select Satellite for Time-Series Plot:", df["satellite_id"].unique())

    df_viz_sat = df[df["satellite_id"] == selected_viz_sat].copy()

    st.write("##### Telemetry Health Time Series (Nominal vs Anomalies)")
    fig_ts = plot_telemetry_time_series(
        df_viz_sat,
        feature_cols=feature_cols,
        anomaly_col="combined_prediction",
        title_suffix=f"({selected_viz_sat})"
    )
    st.pyplot(fig_ts)
    plt.close(fig_ts)

    st.markdown("---")
    st.write("##### ML Anomaly Score Distribution")
    fig_score = plot_anomaly_score_distribution(df["ml_score"].values, df["ml_prediction"].values)
    st.pyplot(fig_score)
    plt.close(fig_score)

    st.markdown("---")
    st.subheader("Export Analysis Reports & Experiment Results")
    exp_col1, exp_col2 = st.columns(2)

    with exp_col1:
        st.write("##### Telemetry Analysis Report")
        csv_telemetry = df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="📥 Download Telemetry Analysis Report (CSV)",
            data=csv_telemetry,
            file_name=f"telemetry_analysis_report_{data_source.split()[0].lower()}.csv",
            mime="text/csv"
        )

    with exp_col2:
        st.write("##### Proposal Experiment Results Summary")
        if "comparison_table" in exp4_res:
            csv_exp = exp4_res["comparison_table"].to_csv(index=False).encode("utf-8")
            st.download_button(
                label="📥 Download Experiment 4 Comparison (CSV)",
                data=csv_exp,
                file_name="experiment_4_method_comparison.csv",
                mime="text/csv"
            )
        elif "overlap_table" in exp1_res:
            csv_exp = exp1_res["overlap_table"].to_csv(index=False).encode("utf-8")
            st.download_button(
                label="📥 Download Experiment 1 Overlap (CSV)",
                data=csv_exp,
                file_name="experiment_1_overlap.csv",
                mime="text/csv"
            )

st.markdown("---")
st.caption("Satellite Telemetry Anomaly Detection and Multi-Satellite Monitoring | Introduction to Machine Learning (IML)")
