"""
evaluation.py
Comprehensive model evaluation metrics and confusion matrix computations for:
1. Rule-Based Detector
2. Machine Learning Isolation Forest
3. Combined (OR-Logic) Detector
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Optional
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, model_name: str = "Detector") -> Dict[str, Any]:
    """
    Compute Accuracy, Precision, Recall, F1-Score, and Confusion Matrix (TN, FP, FN, TP).
    """
    y_true_arr = np.asarray(y_true).astype(int)
    y_pred_arr = np.asarray(y_pred).astype(int)

    acc = float(accuracy_score(y_true_arr, y_pred_arr))
    prec = float(precision_score(y_true_arr, y_pred_arr, zero_division=0))
    rec = float(recall_score(y_true_arr, y_pred_arr, zero_division=0))
    f1 = float(f1_score(y_true_arr, y_pred_arr, zero_division=0))

    cm = confusion_matrix(y_true_arr, y_pred_arr, labels=[0, 1])
    # Extract TN, FP, FN, TP
    tn, fp, fn, tp = cm.ravel() if cm.shape == (2, 2) else (0, 0, 0, 0)

    total = len(y_true_arr)
    detected_anomalies = int(np.sum(y_pred_arr == 1))
    actual_anomalies = int(np.sum(y_true_arr == 1))

    return {
        "model_name": model_name,
        "total_samples": total,
        "actual_anomalies": actual_anomalies,
        "detected_anomalies": detected_anomalies,
        "accuracy": round(acc, 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1_score": round(f1, 4),
        "confusion_matrix": cm,
        "true_positives": int(tp),
        "false_positives": int(fp),
        "true_negatives": int(tn),
        "false_negatives": int(fn)
    }


def evaluate_all_detectors(
    y_true: np.ndarray,
    rule_preds: np.ndarray,
    ml_preds: np.ndarray,
    combined_preds: np.ndarray
) -> pd.DataFrame:
    """
    Evaluate Rule-Based, Isolation Forest, and Combined detection side-by-side.
    Returns comparison summary DataFrame.
    """
    m_rule = compute_metrics(y_true, rule_preds, model_name="Rule-Based Detector")
    m_ml = compute_metrics(y_true, ml_preds, model_name="Isolation Forest (ML)")
    m_comb = compute_metrics(y_true, combined_preds, model_name="Combined (OR-Logic)")

    records = [
        {
            "Method": m["model_name"],
            "Accuracy": f"{m['accuracy']:.4f}",
            "Precision": f"{m['precision']:.4f}",
            "Recall": f"{m['recall']:.4f}",
            "F1-Score": f"{m['f1_score']:.4f}",
            "TP": m["true_positives"],
            "FP": m["false_positives"],
            "TN": m["true_negatives"],
            "FN": m["false_negatives"],
            "Total Anomalies Flagged": m["detected_anomalies"]
        }
        for m in [m_rule, m_ml, m_comb]
    ]

    return pd.DataFrame(records)


def format_confusion_matrix_text(cm: np.ndarray, model_name: str = "") -> str:
    """Return formatted plain-text representation of a confusion matrix."""
    tn, fp, fn, tp = cm.ravel() if cm.shape == (2, 2) else (0, 0, 0, 0)
    lines = [
        f"--- Confusion Matrix: {model_name} ---",
        f"                Predicted Normal (0)   Predicted Anomaly (1)",
        f"Actual Normal (0)      TN={tn:<8}            FP={fp:<8}",
        f"Actual Anomaly(1)      FN={fn:<8}            TP={tp:<8}"
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    y_t = np.array([0, 0, 0, 0, 1, 1, 1, 0, 1, 0])
    y_r = np.array([0, 0, 0, 0, 1, 1, 0, 0, 0, 0])
    y_m = np.array([0, 0, 1, 0, 1, 1, 1, 0, 1, 0])
    y_c = np.array([0, 0, 1, 0, 1, 1, 1, 0, 1, 0])

    comp = evaluate_all_detectors(y_t, y_r, y_m, y_c)
    print(comp.to_string(index=False))

