"""
visualizer.py
Telemetry Visualization and Experimental Plotting Module.
Generates comprehensive visual indicators for multi-satellite health monitoring:
  - Telemetry time-series plots with detected anomalies highlighted
  - Confusion matrix heatmaps for Rule-Based, Isolation Forest, and Combined detection
  - Contamination parameter sensitivity curves
  - Feature combination comparison bar charts
  - Isolation Forest continuous anomaly score distribution histograms
"""

import os
from typing import List, Dict, Optional, Any
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_telemetry_time_series(
    df: pd.DataFrame,
    feature_cols: Optional[List[str]] = None,
    anomaly_col: str = "combined_prediction",
    time_col: str = "timestamp",
    max_points: int = 1500,
    title_suffix: str = ""
) -> plt.Figure:
    """
    Plot telemetry parameters over time with normal observations in blue
    and detected anomalies highlighted with red markers.
    """
    if feature_cols is None:
        numeric_cols = [
            c for c in df.select_dtypes(include=[np.number]).columns
            if c not in ["ground_truth", "rule_prediction", "ml_prediction", "combined_prediction", "ml_score"]
        ]
        feature_cols = numeric_cols[:4]

    plot_df = df.iloc[:max_points].copy() if len(df) > max_points else df.copy()

    time_vals = plot_df[time_col] if time_col in plot_df.columns else np.arange(len(plot_df))
    has_anomalies = anomaly_col in plot_df.columns
    anomaly_mask = (plot_df[anomaly_col] == 1) if has_anomalies else np.zeros(len(plot_df), dtype=bool)

    n_features = len(feature_cols)
    fig, axes = plt.subplots(n_features, 1, figsize=(12, 2.6 * n_features), sharex=True)
    if n_features == 1:
        axes = [axes]

    for ax, col in zip(axes, feature_cols):
        # Nominal telemetry stream
        ax.plot(time_vals, plot_df[col], label=f"{col} (Nominal)", color="#1f77b4", linewidth=1.2, alpha=0.85)

        # Highlight anomalies
        if has_anomalies and anomaly_mask.any():
            ax.scatter(
                time_vals[anomaly_mask],
                plot_df[col][anomaly_mask],
                color="#d62728",
                s=40,
                zorder=5,
                label=f"Detected Anomaly ({anomaly_mask.sum()})",
                marker="o",
                edgecolors="black",
                linewidth=0.5
            )

        ax.set_ylabel(col, fontsize=10, fontweight="bold")
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.legend(loc="upper right", fontsize=8)

    axes[-1].set_xlabel("Observation Time" if time_col in plot_df.columns else "Sample Index", fontsize=10)
    plt.xticks(rotation=25)
    title = f"Satellite Telemetry Health Monitoring {title_suffix}".strip()
    fig.suptitle(title, fontsize=13, fontweight="bold")
    plt.tight_layout()
    return fig


def plot_confusion_matrices(cm_dict: Dict[str, np.ndarray]) -> plt.Figure:
    """
    Plot side-by-side confusion matrix heatmaps for multiple detectors.
    """
    n_models = len(cm_dict)
    fig, axes = plt.subplots(1, n_models, figsize=(4.2 * n_models, 3.8))
    if n_models == 1:
        axes = [axes]

    for ax, (name, cm) in zip(axes, cm_dict.items()):
        im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
        ax.set_title(name, fontsize=11, fontweight="bold")

        tick_marks = [0, 1]
        ax.set_xticks(tick_marks)
        ax.set_yticks(tick_marks)
        ax.set_xticklabels(["Normal", "Anomaly"], fontsize=9)
        ax.set_yticklabels(["Normal", "Anomaly"], fontsize=9)

        # Annotate cells
        thresh = cm.max() / 2.0 if cm.max() > 0 else 1.0
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                val = cm[i, j]
                color = "white" if val > thresh else "black"
                label = f"{val}"
                if (i, j) == (0, 0):
                    label += "\n(TN)"
                elif (i, j) == (0, 1):
                    label += "\n(FP)"
                elif (i, j) == (1, 0):
                    label += "\n(FN)"
                elif (i, j) == (1, 1):
                    label += "\n(TP)"
                ax.text(j, i, label, ha="center", va="center", color=color, fontsize=10)

        ax.set_ylabel("True Ground-Truth", fontsize=9)
        ax.set_xlabel("Predicted Label", fontsize=9)

    plt.tight_layout()
    return fig


def plot_contamination_curve(exp2_df: pd.DataFrame) -> plt.Figure:
    """
    Plot Experiment 2 metrics (Precision, Recall, F1) across contamination levels.
    """
    fig, ax = plt.subplots(figsize=(8, 4.2))

    contaminations = exp2_df["Contamination"].values

    if "F1-Score" in exp2_df.columns:
        ax.plot(contaminations, exp2_df["Precision"], marker="o", linewidth=2, label="Precision", color="#2ca02c")
        ax.plot(contaminations, exp2_df["Recall"], marker="s", linewidth=2, label="Recall", color="#ff7f0e")
        ax.plot(contaminations, exp2_df["F1-Score"], marker="^", linewidth=2.5, label="F1-Score", color="#1f77b4")
        ax.set_ylabel("Metric Value (0.0 - 1.0)", fontsize=10)
        ax.set_ylim(-0.05, 1.08)
    else:
        ax.plot(contaminations, exp2_df["Anomalies Flagged"], marker="o", linewidth=2, label="Flagged Count", color="#1f77b4")
        ax.set_ylabel("Anomalies Flagged", fontsize=10)

    ax.set_xlabel("Contamination Parameter (c)", fontsize=10)
    ax.set_xticks(contaminations)
    ax.set_title("Experiment 2: Isolation Forest Contamination Sensitivity", fontsize=12, fontweight="bold")
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(loc="best", fontsize=9)
    plt.tight_layout()
    return fig


def plot_feature_combinations_bar(exp3_df: pd.DataFrame) -> plt.Figure:
    """
    Plot Experiment 3 comparison bar chart across feature combinations.
    """
    fig, ax = plt.subplots(figsize=(9, 4.5))

    names = exp3_df["Feature Combination"]
    y_pos = np.arange(len(names))

    if "F1-Score" in exp3_df.columns:
        bars = ax.barh(y_pos, exp3_df["F1-Score"], color="#1f77b4", alpha=0.85, height=0.55)
        ax.set_xlabel("F1-Score", fontsize=10)
        ax.set_xlim(0, 1.1)
        for bar in bars:
            width = bar.get_width()
            ax.text(width + 0.02, bar.get_y() + bar.get_height() / 2, f"{width:.3f}", va="center", fontsize=9)
    else:
        bars = ax.barh(y_pos, exp3_df["Anomalies Flagged"], color="#1f77b4", alpha=0.85, height=0.55)
        ax.set_xlabel("Anomalies Flagged", fontsize=10)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(names, fontsize=9)
    ax.invert_yaxis()
    ax.set_title("Experiment 3: Feature Combination Performance Comparison", fontsize=12, fontweight="bold")
    ax.grid(True, axis="x", linestyle="--", alpha=0.5)
    plt.tight_layout()
    return fig


def plot_anomaly_score_distribution(scores: np.ndarray, predictions: Optional[np.ndarray] = None) -> plt.Figure:
    """
    Plot histogram of Isolation Forest continuous anomaly scores.
    """
    fig, ax = plt.subplots(figsize=(8, 3.8))

    if predictions is not None:
        normal_scores = scores[predictions == 0]
        anomaly_scores = scores[predictions == 1]
        ax.hist(normal_scores, bins=30, alpha=0.6, color="#1f77b4", label=f"Normal (n={len(normal_scores)})", density=True)
        if len(anomaly_scores) > 0:
            ax.hist(anomaly_scores, bins=20, alpha=0.7, color="#d62728", label=f"Anomaly (n={len(anomaly_scores)})", density=True)
    else:
        ax.hist(scores, bins=35, alpha=0.7, color="#1f77b4", label="All Scores", density=True)

    ax.set_xlabel("ML Anomaly Score (Higher = More Anomalous)", fontsize=10)
    ax.set_ylabel("Density", fontsize=10)
    ax.set_title("Isolation Forest Anomaly Score Distribution", fontsize=12, fontweight="bold")
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(loc="upper right", fontsize=9)
    plt.tight_layout()
    return fig


def save_all_figures(df: pd.DataFrame, exp_results: Dict[str, Any], output_dir: str):
    """
    Generate and save all proposal figures to the specified directory.
    """
    os.makedirs(output_dir, exist_ok=True)

    # 1. Telemetry Time Series
    first_sat = df["satellite_id"].iloc[0] if "satellite_id" in df.columns else "SAT-01"
    sat_df = df[df["satellite_id"] == first_sat] if "satellite_id" in df.columns else df
    fig_ts = plot_telemetry_time_series(sat_df, title_suffix=f"({first_sat})")
    fig_ts.savefig(os.path.join(output_dir, "fig1_synthetic_telemetry.png"), dpi=200)
    plt.close(fig_ts)

    # 2. Confusion Matrices (Exp 4)
    if "confusion_matrices" in exp_results.get("experiment_4", {}):
        fig_cm = plot_confusion_matrices(exp_results["experiment_4"]["confusion_matrices"])
        fig_cm.savefig(os.path.join(output_dir, "fig2_confusion_matrices.png"), dpi=200)
        plt.close(fig_cm)

    # 3. Contamination Sensitivity Curve (Exp 2)
    if "experiment_2" in exp_results:
        fig_cont = plot_contamination_curve(exp_results["experiment_2"])
        fig_cont.savefig(os.path.join(output_dir, "fig3_contamination_curve.png"), dpi=200)
        plt.close(fig_cont)

    # 4. Feature Combination Bar Chart (Exp 3)
    if "experiment_3" in exp_results:
        fig_feat = plot_feature_combinations_bar(exp_results["experiment_3"])
        fig_feat.savefig(os.path.join(output_dir, "fig4_feature_combinations.png"), dpi=200)
        plt.close(fig_feat)

    # 5. Anomaly Score Distribution
    if "ml_score" in df.columns:
        fig_dist = plot_anomaly_score_distribution(df["ml_score"].values, df.get("ml_prediction", None))
        fig_dist.savefig(os.path.join(output_dir, "fig5_anomaly_scores_distribution.png"), dpi=200)
        plt.close(fig_dist)

    print(f"✅ Generated and saved all visual figures to: {output_dir}")


def plot_telemetry(df: pd.DataFrame):
    """Backward-compatible plotting function."""
    fig = plot_telemetry_time_series(df)
    plt.show()
    return fig


if __name__ == "__main__":
    from generate_data import generate_synthetic_telemetry
    test_df = generate_synthetic_telemetry(num_records_per_satellite=150)
    test_df["combined_prediction"] = test_df["ground_truth"]
    fig = plot_telemetry_time_series(test_df, ["temperature", "voltage", "current"])
    print("Visualizer functions initialized successfully.")
