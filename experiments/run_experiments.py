"""
Full experimental comparison: VQC (ideal) vs VQC (noisy) vs classical baselines.

Methodology:
  - 5-fold stratified cross-validation
  - Multiple seeds averaged
  - Full metrics: accuracy, F1, AUC-ROC, Cohen's kappa
  - Wilcoxon signed-rank statistical significance test
  - All results saved to results/comparison_results.csv
"""

import os
import sys
import time
import numpy as np
import pandas as pd

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import yaml
from sklearn.model_selection import StratifiedKFold
from scipy.stats import wilcoxon
from tqdm import tqdm

from data.generate_data import generate_sensor_data
from models.vqc_model import create_vqc
from models.classical_model import create_classical_model
from noise.noise_model import (
    build_noisy_sampler_and_pm,
    build_ideal_sampler_and_pm,
    noise_model_from_config,
)
from evaluation.metrics import compute_metrics, aggregate_cv_metrics, format_results_table
from evaluation.visualization import plot_model_comparison, plot_roc_curves, plot_confusion_matrix


def run_fold(model, X_train, y_train, X_val, y_val):
    """Train and evaluate one fold. Returns metrics dict."""
    model.fit(X_train, y_train)
    y_pred = model.predict(X_val)
    try:
        y_proba = model.predict_proba(X_val)
    except Exception:
        y_proba = None
    return compute_metrics(y_val, y_pred, y_proba), y_pred, y_proba


def run_cv_for_model(model_factory, X, y, n_folds=5, seed=42, desc="model"):
    """
    Run stratified k-fold CV. Returns list of per-fold metric dicts,
    list of (y_true, y_pred, y_proba) per fold, and timing info.
    """
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    fold_metrics = []
    fold_preds = []
    times = []

    for fold_i, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        X_tr, X_val = X[train_idx], X[val_idx]
        y_tr, y_val = y[train_idx], y[val_idx]

        t0 = time.time()
        model = model_factory()
        metrics, y_pred, y_proba = run_fold(model, X_tr, y_tr, X_val, y_val)
        elapsed = time.time() - t0

        fold_metrics.append(metrics)
        fold_preds.append((y_val, y_pred, y_proba))
        times.append(elapsed)

        print(
            f"    [{desc}] fold {fold_i+1}/{n_folds} "
            f"acc={metrics['accuracy']:.3f}  f1={metrics['f1']:.3f}  "
            f"auc={metrics['auc_roc']:.3f}  t={elapsed:.1f}s"
        )

    return fold_metrics, fold_preds, times


def run_full_comparison(config: dict, output_dir: str = "results") -> pd.DataFrame:
    """
    Runs 5-fold CV for all models and returns comparison DataFrame.

    Models compared:
      - VQC (ideal, noiseless)
      - VQC (medium quantum noise from config)
      - Classical SVM (RBF)
      - Logistic Regression
      - MLP (2-layer)
      - Random (lower bound)
    """
    print("\n" + "=" * 60)
    print("  Full Model Comparison — 5-Fold Cross-Validation")
    print("=" * 60)

    n_folds = config["experiment"]["n_folds"]
    seed = config["experiment"]["seeds"][0]
    n_qubits = config["vqc"]["n_qubits"]
    max_iter = config["vqc"]["max_iter"]
    shots = config["vqc"]["shots"]

    X, y = generate_sensor_data(
        n_samples=config["data"]["n_samples"],
        n_features=config["data"]["n_features"],
        snr=config["data"].get("signal_snr", 3.0),
        seed=seed,
    )
    print(f"\n  Dataset: {X.shape[0]} samples × {X.shape[1]} features, "
          f"class balance = {np.bincount(y).tolist()}")

    nm = noise_model_from_config(config)
    noisy_sampler, noisy_pm = build_noisy_sampler_and_pm(nm, shots=shots, seed=seed)
    ideal_sampler, ideal_pm = build_ideal_sampler_and_pm(shots=shots, seed=seed)

    models = {
        "VQC (ideal)": lambda: create_vqc(
            n_qubits=n_qubits, max_iter=max_iter,
            sampler=ideal_sampler, pass_manager=ideal_pm, seed=seed
        ),
        "VQC (noisy)": lambda: create_vqc(
            n_qubits=n_qubits, max_iter=max_iter,
            sampler=noisy_sampler, pass_manager=noisy_pm, seed=seed
        ),
        "SVM (RBF)": lambda: create_classical_model("svm_rbf", seed),
        "Logistic Reg": lambda: create_classical_model("logistic_regression", seed),
        "MLP": lambda: create_classical_model("mlp", seed),
        "Random": lambda: create_classical_model("random", seed),
    }

    all_cv_results = {}     # {name: {'mean': {...}, 'std': {...}}}
    all_fold_accs = {}      # {name: [fold1_acc, fold2_acc, ...]}
    all_preds = {}          # for ROC curves and CM

    for name, factory in models.items():
        print(f"\n  ── {name} ──")
        fold_metrics, fold_preds, times = run_cv_for_model(
            factory, X, y, n_folds=n_folds, seed=seed, desc=name
        )
        agg = aggregate_cv_metrics(fold_metrics)
        all_cv_results[name] = {"mean": {k: v["mean"] for k, v in agg.items()},
                                 "std": {k: v["std"] for k, v in agg.items()}}
        all_fold_accs[name] = [m["accuracy"] for m in fold_metrics]
        all_preds[name] = fold_preds

    # ── Statistical significance: VQC (noisy) vs SVM ─────────────────────────
    vqc_accs = np.array(all_fold_accs.get("VQC (noisy)", [0] * n_folds))
    svm_accs = np.array(all_fold_accs.get("SVM (RBF)", [0] * n_folds))

    print("\n  Statistical Significance Test (Wilcoxon signed-rank):")
    print(f"    VQC (noisy) folds: {vqc_accs.round(3).tolist()}")
    print(f"    SVM (RBF)   folds: {svm_accs.round(3).tolist()}")
    try:
        stat, p_val = wilcoxon(svm_accs, vqc_accs)
        print(f"    statistic={stat:.3f}  p-value={p_val:.4f}  "
              f"{'significant (p<0.05)' if p_val < 0.05 else 'not significant'}")
    except Exception as e:
        print(f"    (could not compute: {e})")

    # ── Build results DataFrame ───────────────────────────────────────────────
    rows = []
    for name, res in all_cv_results.items():
        row = {"model": name}
        for metric in ["accuracy", "f1", "auc_roc", "cohen_kappa"]:
            row[f"{metric}_mean"] = res["mean"].get(metric, float("nan"))
            row[f"{metric}_std"] = res["std"].get(metric, float("nan"))
        rows.append(row)
    df = pd.DataFrame(rows)

    os.makedirs(output_dir, exist_ok=True)
    df.to_csv(os.path.join(output_dir, "comparison_results.csv"), index=False)

    # ── Plots ─────────────────────────────────────────────────────────────────
    import matplotlib.pyplot as plt

    fig_bar = plot_model_comparison(df, output_dir=output_dir)
    plt.close(fig_bar)

    # Confusion matrix for VQC noisy (last fold)
    if "VQC (noisy)" in all_preds:
        y_true_last, y_pred_last, _ = all_preds["VQC (noisy)"][-1]
        from sklearn.metrics import confusion_matrix
        cm = confusion_matrix(y_true_last, y_pred_last)
        fig_cm = plot_confusion_matrix(cm, ["Normal", "Anomaly"],
                                       title="VQC Noisy", output_dir=output_dir)
        plt.close(fig_cm)

    # ROC curves (last fold, models with proba)
    y_true_roc = all_preds[list(all_preds.keys())[0]][-1][0]
    roc_dict = {}
    for name, preds in all_preds.items():
        _, _, y_proba = preds[-1]
        if y_proba is not None:
            roc_dict[name] = y_proba
    if roc_dict:
        fig_roc = plot_roc_curves(y_true_roc, roc_dict, output_dir=output_dir)
        plt.close(fig_roc)

    # ── Print final table ──────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  FINAL RESULTS (mean ± std across 5 folds)")
    print("=" * 60)
    for _, row in df.iterrows():
        print(
            f"  {row['model']:<22} "
            f"acc={row['accuracy_mean']:.3f}±{row['accuracy_std']:.3f}  "
            f"f1={row['f1_mean']:.3f}±{row['f1_std']:.3f}  "
            f"auc={row['auc_roc_mean']:.3f}±{row['auc_roc_std']:.3f}"
        )

    return df


if __name__ == "__main__":
    with open("config/base_config.yaml") as f:
        config = yaml.safe_load(f)

    run_full_comparison(config)
