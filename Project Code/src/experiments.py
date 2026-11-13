"""
experiments.py
Implementation of the 4 approved IML proposal experiments:
  Experiment 1: Comparison between Rule-Based detection and Isolation Forest (agreement & overlap).
  Experiment 2: Parameter sensitivity analysis over contamination values (0.01, 0.03, 0.05, 0.10).
  Experiment 3: Feature combination ablation study comparing telemetry subsets (univariate vs bivariate vs multivariate).
  Experiment 4: Comprehensive performance evaluation of Rule-Based vs Isolation Forest vs Combined (OR) approach.
"""

import os
from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd
from anomaly_detector import RuleBasedDetector
from ml_anomaly_detector import SatelliteIsolationForest, train_test_split_telemetry
from evaluation import compute_metrics, evaluate_all_detectors


def run_experiment_1(
    df: pd.DataFrame,
    feature_cols: List[str],
    rule_detector: RuleBasedDetector,
    ml_detector: SatelliteIsolationForest,
    test_size: float = 0.30
) -> Dict[str, Any]:
    """
    Experiment 1: Compare Rule-Based detection and Isolation Forest.
    Analyzes agreement rate, hard outliers flagged only by rules,
    and multivariate anomalies flagged only by ML.
    """
    X_train, X_test, y_train, y_test = train_test_split_telemetry(df, feature_cols, test_size=test_size)

    # Rule predictions on full dataset, slice test partition
    rule_preds, _ = rule_detector.predict(df)
    test_rule_preds = rule_preds.iloc[len(X_train):].values

    # ML predictions on test partition
    if not ml_detector.is_trained:
        ml_detector.fit(X_train)
    test_ml_preds = ml_detector.predict(X_test)

    # Agreement and overlap categorization
    both_anomaly = int(np.sum((test_rule_preds == 1) & (test_ml_preds == 1)))
    rule_only = int(np.sum((test_rule_preds == 1) & (test_ml_preds == 0)))
    ml_only = int(np.sum((test_rule_preds == 0) & (test_ml_preds == 1)))
    both_normal = int(np.sum((test_rule_preds == 0) & (test_ml_preds == 0)))
    total_test = len(X_test)

    agreement_pct = round(100.0 * (both_anomaly + both_normal) / total_test, 2)

    overlap_table = pd.DataFrame([
        {"Category": "Flagged by Both (Agreement)", "Count": both_anomaly, "Percentage": f"{100*both_anomaly/total_test:.2f}%"},
        {"Category": "Rule-Based Only (Hard Outlier)", "Count": rule_only, "Percentage": f"{100*rule_only/total_test:.2f}%"},
        {"Category": "Isolation Forest Only (Multivariate Outlier)", "Count": ml_only, "Percentage": f"{100*ml_only/total_test:.2f}%"},
        {"Category": "Agreed Normal", "Count": both_normal, "Percentage": f"{100*both_normal/total_test:.2f}%"}
    ])

    results = {
        "total_test_samples": total_test,
        "rule_total_flagged": int(np.sum(test_rule_preds == 1)),
        "ml_total_flagged": int(np.sum(test_ml_preds == 1)),
        "agreement_percentage": agreement_pct,
        "overlap_table": overlap_table
    }

    if y_test is not None:
        m_rule = compute_metrics(y_test, test_rule_preds, model_name="Rule-Based")
        m_ml = compute_metrics(y_test, test_ml_preds, model_name="Isolation Forest")
        results["metrics_comparison"] = pd.DataFrame([
            {
                "Detector": "Rule-Based",
                "Precision": m_rule["precision"],
                "Recall": m_rule["recall"],
                "F1-Score": m_rule["f1_score"],
                "Accuracy": m_rule["accuracy"]
            },
            {
                "Detector": "Isolation Forest",
                "Precision": m_ml["precision"],
                "Recall": m_ml["recall"],
                "F1-Score": m_ml["f1_score"],
                "Accuracy": m_ml["accuracy"]
            }
        ])

    return results


def run_experiment_2_contamination(
    df: pd.DataFrame,
    feature_cols: List[str],
    contaminations: Optional[List[float]] = None,
    test_size: float = 0.30
) -> pd.DataFrame:
    """
    Experiment 2: Test contamination values (0.01, 0.03, 0.05, 0.10).
    Evaluates sensitivity of Isolation Forest to assumed outlier proportions.
    """
    if contaminations is None:
        contaminations = [0.01, 0.03, 0.05, 0.10]

    X_train, X_test, y_train, y_test = train_test_split_telemetry(df, feature_cols, test_size=test_size)

    rows = []
    for c in contaminations:
        model = SatelliteIsolationForest(contamination=c, random_state=42)
        model.fit(X_train)
        preds = model.predict(X_test)

        flagged = int(np.sum(preds == 1))
        flagged_rate = round(100.0 * flagged / len(X_test), 2)

        row = {
            "Contamination": c,
            "Anomalies Flagged": flagged,
            "Flagged Rate (%)": flagged_rate
        }

        if y_test is not None:
            m = compute_metrics(y_test, preds, model_name=f"IF (c={c})")
            row["Accuracy"] = m["accuracy"]
            row["Precision"] = m["precision"]
            row["Recall"] = m["recall"]
            row["F1-Score"] = m["f1_score"]

        rows.append(row)

    return pd.DataFrame(rows)


def run_experiment_3_feature_combinations(
    df: pd.DataFrame,
    feature_cols: Optional[List[str]] = None,
    feature_sets: Optional[Dict[str, List[str]]] = None,
    contamination: float = 0.05,
    test_size: float = 0.30
) -> pd.DataFrame:
    """
    Experiment 3: Compare different combinations of telemetry features.
    Demonstrates multivariate benefits over univariate sensor subsets.
    """
    if feature_cols is None:
        raw_candidates = ["temperature", "voltage", "current", "orientation"]
        feature_cols = [c for c in raw_candidates if c in df.columns]

    if feature_sets is None:
        feature_sets = {}
        # Single features
        for col in feature_cols[:3]:
            feature_sets[f"Univariate: {col}"] = [col]
        # Bivariate pair
        if len(feature_cols) >= 2:
            feature_sets[f"Pair: {feature_cols[0]} + {feature_cols[1]}"] = [feature_cols[0], feature_cols[1]]
        # Full multivariate set
        feature_sets["Full Feature Set"] = feature_cols

    rows = []
    for name, cols in feature_sets.items():
        valid_cols = [c for c in cols if c in df.columns]
        if not valid_cols:
            continue

        X_train, X_test, y_train, y_test = train_test_split_telemetry(df, valid_cols, test_size=test_size)

        model = SatelliteIsolationForest(contamination=contamination, random_state=42)
        model.fit(X_train)
        preds = model.predict(X_test)

        flagged = int(np.sum(preds == 1))
        row = {
            "Feature Combination": name,
            "Features Used": ", ".join(valid_cols),
            "Num Features": len(valid_cols),
            "Anomalies Flagged": flagged
        }

        if y_test is not None:
            m = compute_metrics(y_test, preds, model_name=name)
            row["Precision"] = m["precision"]
            row["Recall"] = m["recall"]
            row["F1-Score"] = m["f1_score"]
            row["Accuracy"] = m["accuracy"]

        rows.append(row)

    return pd.DataFrame(rows)


def run_experiment_4_method_comparison(
    df: pd.DataFrame,
    feature_cols: List[str],
    rule_detector: RuleBasedDetector,
    contamination: float = 0.05,
    test_size: float = 0.30
) -> Dict[str, Any]:
    """
    Experiment 4: Compare performance of Rule-Based vs Isolation Forest vs Combined (OR) approach.
    Computes side-by-side performance metrics and full confusion matrices.
    """
    X_train, X_test, y_train, y_test = train_test_split_telemetry(df, feature_cols, test_size=test_size)

    # Rule predictions
    full_rule_preds, _ = rule_detector.predict(df)
    test_rule_preds = full_rule_preds.iloc[len(X_train):].values

    # ML predictions
    model = SatelliteIsolationForest(contamination=contamination, random_state=42)
    model.fit(X_train)
    test_ml_preds = model.predict(X_test)

    # Combined predictions (OR-based logic)
    test_comb_preds = ((test_rule_preds == 1) | (test_ml_preds == 1)).astype(int)

    results = {
        "test_size": len(X_test),
        "rule_flagged": int(np.sum(test_rule_preds == 1)),
        "ml_flagged": int(np.sum(test_ml_preds == 1)),
        "combined_flagged": int(np.sum(test_comb_preds == 1))
    }

    if y_test is not None:
        comp_df = evaluate_all_detectors(y_test.values, test_rule_preds, test_ml_preds, test_comb_preds)
        results["comparison_table"] = comp_df

        m_rule = compute_metrics(y_test.values, test_rule_preds, "Rule-Based")
        m_ml = compute_metrics(y_test.values, test_ml_preds, "Isolation Forest")
        m_comb = compute_metrics(y_test.values, test_comb_preds, "Combined")

        results["confusion_matrices"] = {
            "Rule-Based": m_rule["confusion_matrix"],
            "Isolation Forest": m_ml["confusion_matrix"],
            "Combined": m_comb["confusion_matrix"]
        }

    return results


def run_all_experiments(
    df: pd.DataFrame,
    feature_cols: List[str],
    rule_detector: Optional[RuleBasedDetector] = None,
    contamination: float = 0.05,
    test_size: float = 0.30,
    save_reports_dir: Optional[str] = None
) -> Dict[str, Any]:
    """
    Execute all 4 approved experiments and optionally save CSV summaries.
    """
    if rule_detector is None:
        rule_detector = RuleBasedDetector()

    ml_detector = SatelliteIsolationForest(contamination=contamination, random_state=42)

    exp1 = run_experiment_1(df, feature_cols, rule_detector, ml_detector, test_size=test_size)
    exp2 = run_experiment_2_contamination(df, feature_cols, test_size=test_size)
    exp3 = run_experiment_3_feature_combinations(df, feature_cols=feature_cols, contamination=contamination, test_size=test_size)
    exp4 = run_experiment_4_method_comparison(df, feature_cols, rule_detector, contamination=contamination, test_size=test_size)

    all_res = {
        "experiment_1": exp1,
        "experiment_2": exp2,
        "experiment_3": exp3,
        "experiment_4": exp4
    }

    if save_reports_dir:
        os.makedirs(save_reports_dir, exist_ok=True)
        exp1["overlap_table"].to_csv(os.path.join(save_reports_dir, "experiment_1_overlap.csv"), index=False)
        if "metrics_comparison" in exp1:
            exp1["metrics_comparison"].to_csv(os.path.join(save_reports_dir, "experiment_1_metrics.csv"), index=False)
        exp2.to_csv(os.path.join(save_reports_dir, "experiment_2_contamination.csv"), index=False)
        exp3.to_csv(os.path.join(save_reports_dir, "experiment_3_features.csv"), index=False)
        if "comparison_table" in exp4:
            exp4["comparison_table"].to_csv(os.path.join(save_reports_dir, "experiment_4_method_comparison.csv"), index=False)
        print(f"✅ Saved all experiment reports to: {save_reports_dir}")

    return all_res
