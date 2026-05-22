"""
Full metrics suite for classification evaluation.
Covers accuracy, precision, recall, F1, AUC-ROC, confusion matrix, Cohen's kappa.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    cohen_kappa_score,
    matthews_corrcoef,
    classification_report,
)


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray = None,
) -> dict:
    """
    Compute full classification metrics.

    Parameters
    ----------
    y_true  : ground-truth labels
    y_pred  : predicted labels
    y_proba : predicted probabilities, shape (n_samples, n_classes).
              Required for AUC-ROC. Pass None to skip.

    Returns
    -------
    dict with keys:
      accuracy, precision, recall, f1,
      auc_roc (if y_proba given), confusion_matrix,
      cohen_kappa, matthews_corrcoef
    """
    n_classes = len(np.unique(y_true))
    avg = "binary" if n_classes == 2 else "macro"

    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, average=avg, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, average=avg, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, average=avg, zero_division=0)),
        "cohen_kappa": float(cohen_kappa_score(y_true, y_pred)),
        "matthews_corrcoef": float(matthews_corrcoef(y_true, y_pred)),
        "confusion_matrix": confusion_matrix(y_true, y_pred),
    }

    if y_proba is not None:
        try:
            if n_classes == 2:
                auc = roc_auc_score(y_true, y_proba[:, 1])
            else:
                auc = roc_auc_score(y_true, y_proba, multi_class="ovr", average="macro")
            metrics["auc_roc"] = float(auc)
        except Exception:
            metrics["auc_roc"] = float("nan")
    else:
        metrics["auc_roc"] = float("nan")

    return metrics


def aggregate_cv_metrics(fold_metrics: list) -> dict:
    """
    Given a list of per-fold metric dicts, return {metric: (mean, std)}.
    Skips 'confusion_matrix' in aggregation.
    """
    scalar_keys = [k for k in fold_metrics[0] if k != "confusion_matrix"]
    agg = {}
    for k in scalar_keys:
        vals = [m[k] for m in fold_metrics]
        agg[k] = {"mean": float(np.mean(vals)), "std": float(np.std(vals))}
    return agg


def format_results_table(cv_results: dict) -> pd.DataFrame:
    """
    cv_results: {model_name: {"mean": {...}, "std": {...}}}
    Returns a DataFrame with columns [model, accuracy, f1, auc_roc, ...]
    and values formatted as "mean ± std".
    """
    rows = []
    for model, stats in cv_results.items():
        row = {"model": model}
        for metric in ["accuracy", "f1", "auc_roc", "cohen_kappa"]:
            if metric in stats["mean"]:
                m = stats["mean"][metric]
                s = stats["std"][metric]
                row[metric] = f"{m:.4f} ± {s:.4f}"
        rows.append(row)
    return pd.DataFrame(rows).set_index("model")


def print_metrics(metrics: dict, model_name: str = "") -> None:
    header = f"  {model_name}" if model_name else ""
    print(f"\n{'='*50}{header}")
    for k, v in metrics.items():
        if k == "confusion_matrix":
            print(f"  confusion_matrix:\n{v}")
        else:
            print(f"  {k:<25}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")
    print("=" * 50)
