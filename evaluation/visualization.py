"""
Publication-quality plotting for VQC research results.
All figures saved at 300 DPI as PNG and PDF.
"""

import os
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

matplotlib.rcParams.update({
    "font.size": 12,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
    "legend.fontsize": 10,
    "figure.dpi": 150,
})

_OUTPUT_DIR = "results"


def _save(fig: plt.Figure, name: str, output_dir: str = None) -> str:
    d = output_dir or _OUTPUT_DIR
    os.makedirs(d, exist_ok=True)
    path_png = os.path.join(d, f"{name}.png")
    path_pdf = os.path.join(d, f"{name}.pdf")
    fig.savefig(path_png, dpi=300, bbox_inches="tight")
    fig.savefig(path_pdf, bbox_inches="tight")
    return path_png


def plot_convergence(
    histories: list,
    labels: list = None,
    title: str = "Training Convergence",
    output_dir: str = None,
) -> plt.Figure:
    """
    Plot loss/objective curves across CV folds.

    histories : list of lists, each inner list is loss values per SPSA step.
    labels    : optional per-history label strings.
    """
    fig, ax = plt.subplots(figsize=(8, 4))

    palette = sns.color_palette("tab10", len(histories))
    for i, hist in enumerate(histories):
        lbl = labels[i] if labels else f"Fold {i+1}"
        ax.plot(hist, color=palette[i], alpha=0.7, linewidth=1.2, label=lbl)

    # Mean curve
    min_len = min(len(h) for h in histories)
    arr = np.array([h[:min_len] for h in histories])
    mean_curve = arr.mean(axis=0)
    std_curve = arr.std(axis=0)
    steps = np.arange(min_len)
    ax.plot(steps, mean_curve, color="black", linewidth=2, label="Mean")
    ax.fill_between(steps, mean_curve - std_curve, mean_curve + std_curve,
                    color="black", alpha=0.15)

    ax.set_xlabel("Optimizer Step")
    ax.set_ylabel("Objective Value")
    ax.set_title(title)
    ax.legend(loc="upper right", ncol=2)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    _save(fig, "convergence_curves", output_dir)
    return fig


def plot_noise_sweep(
    df: pd.DataFrame,
    noise_col: str = "noise_level",
    output_dir: str = None,
) -> plt.Figure:
    """
    Line plot of accuracy vs quantum noise level.

    Expected DataFrame columns (at minimum):
        noise_level, vqc_mean, vqc_std, svm_mean, svm_std
    Optional: vqc_zne_mean, vqc_zne_std
    """
    fig, ax = plt.subplots(figsize=(8, 5))

    x = df[noise_col].values

    # VQC baseline
    ax.errorbar(x, df["vqc_mean"], yerr=df["vqc_std"],
                marker="o", linewidth=2, capsize=4, label="VQC (noisy)", color="royalblue")

    # VQC + ZNE
    if "vqc_zne_mean" in df.columns:
        ax.errorbar(x, df["vqc_zne_mean"], yerr=df.get("vqc_zne_std", 0),
                    marker="^", linewidth=2, linestyle="--", capsize=4,
                    label="VQC + ZNE", color="darkorange")

    # Classical SVM
    ax.errorbar(x, df["svm_mean"], yerr=df["svm_std"],
                marker="s", linewidth=2, capsize=4, label="Classical SVM", color="green")

    ax.set_xlabel("2-Qubit Depolarizing Error Rate")
    ax.set_ylabel("Test Accuracy")
    ax.set_title("Noise Resilience: VQC vs Classical SVM")
    ax.legend()
    ax.set_ylim(0.3, 1.05)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    _save(fig, "noise_sweep", output_dir)
    return fig


def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: list = None,
    title: str = "Confusion Matrix",
    output_dir: str = None,
) -> plt.Figure:
    """Seaborn heatmap confusion matrix."""
    if class_names is None:
        class_names = [f"Class {i}" for i in range(cm.shape[0])]

    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(
        cm_norm, annot=True, fmt=".2f", cmap="Blues",
        xticklabels=class_names, yticklabels=class_names,
        linewidths=0.5, ax=ax, vmin=0, vmax=1
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    fig.tight_layout()
    _save(fig, f"confusion_matrix_{title.replace(' ', '_').lower()}", output_dir)
    return fig


def plot_roc_curves(
    y_true: np.ndarray,
    y_proba_dict: dict,
    output_dir: str = None,
) -> plt.Figure:
    """
    Overlaid ROC curves for multiple models (binary classification).

    y_proba_dict : {model_name: y_proba_array} where y_proba is (n, 2) or (n,) for class-1 prob.
    """
    from sklearn.metrics import roc_curve, auc

    fig, ax = plt.subplots(figsize=(7, 6))
    palette = sns.color_palette("tab10", len(y_proba_dict))

    for idx, (name, y_proba) in enumerate(y_proba_dict.items()):
        if y_proba.ndim == 2:
            scores = y_proba[:, 1]
        else:
            scores = y_proba
        fpr, tpr, _ = roc_curve(y_true, scores)
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, color=palette[idx], linewidth=2,
                label=f"{name} (AUC={roc_auc:.3f})")

    ax.plot([0, 1], [0, 1], "k--", linewidth=1, alpha=0.5)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves — All Models")
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    _save(fig, "roc_curves", output_dir)
    return fig


def plot_model_comparison(
    df: pd.DataFrame,
    metrics: list = None,
    output_dir: str = None,
) -> plt.Figure:
    """
    Grouped bar chart with error bars.

    df must have columns: model, accuracy_mean, accuracy_std, f1_mean, f1_std, ...
    metrics: list of base metric names to include (e.g. ['accuracy', 'f1', 'auc_roc'])
    """
    if metrics is None:
        metrics = ["accuracy", "f1", "auc_roc"]

    available = [m for m in metrics if f"{m}_mean" in df.columns]
    n_metrics = len(available)
    n_models = len(df)

    x = np.arange(n_models)
    width = 0.8 / n_metrics
    palette = sns.color_palette("tab10", n_metrics)

    fig, ax = plt.subplots(figsize=(max(8, n_models * 1.5), 5))

    for i, metric in enumerate(available):
        means = df[f"{metric}_mean"].values
        stds = df[f"{metric}_std"].values if f"{metric}_std" in df.columns else np.zeros(n_models)
        offset = (i - n_metrics / 2 + 0.5) * width
        bars = ax.bar(x + offset, means, width, yerr=stds, capsize=3,
                      label=metric.replace("_", " ").title(),
                      color=palette[i], alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels(df["model"].values, rotation=20, ha="right")
    ax.set_ylabel("Score")
    ax.set_ylim(0.0, 1.1)
    ax.set_title("Model Comparison (5-Fold CV)")
    ax.legend(loc="lower right")
    ax.axhline(0.5, color="gray", linestyle=":", alpha=0.5, label="Random baseline")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    _save(fig, "model_comparison", output_dir)
    return fig


def plot_noise_type_ablation(
    df: pd.DataFrame,
    output_dir: str = None,
) -> plt.Figure:
    """
    Bar chart showing accuracy degradation per noise channel type.
    df columns: noise_type, accuracy_mean, accuracy_std
    """
    fig, ax = plt.subplots(figsize=(7, 4))
    colors = sns.color_palette("coolwarm", len(df))
    ax.bar(df["noise_type"], df["accuracy_mean"], yerr=df["accuracy_std"],
           capsize=4, color=colors, alpha=0.85)
    ax.set_ylabel("Test Accuracy")
    ax.set_title("VQC Accuracy by Noise Channel Type")
    ax.set_xticklabels(df["noise_type"], rotation=15, ha="right")
    ax.axhline(0.5, color="gray", linestyle=":", alpha=0.5)
    ax.set_ylim(0.3, 1.05)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    _save(fig, "noise_type_ablation", output_dir)
    return fig
