"""
Evaluation script — loads saved VQC and produces full metrics report.

Fixes over v1:
  - Removes SimpleNamespace hack for loading model weights
  - Full metrics: accuracy, F1, AUC-ROC, confusion matrix, Cohen's kappa
  - Can test under a different noise profile than training (transfer noise)
  - CLI flags: --noise-profile, --no-noise

Usage:
  python evaluate.py                   # load saved model, test on fresh data
  python evaluate.py --no-noise        # evaluate under ideal conditions
  python evaluate.py --noise-profile high  # test under higher noise
"""

import os
import sys
import pickle
import argparse
import numpy as np

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import yaml
from data.generate_data import generate_sensor_data
from noise.noise_model import (
    build_noise_model,
    build_noisy_sampler_and_pm,
    build_ideal_sampler_and_pm,
    noise_model_from_config,
)
from evaluation.metrics import compute_metrics, print_metrics


NOISE_PROFILES = {
    "low":    dict(depolarizing_1q=0.0001, depolarizing_2q=0.001,
                   t1_us=200, t2_us=150, readout_error=0.005),
    "medium": dict(depolarizing_1q=0.001,  depolarizing_2q=0.01,
                   t1_us=100, t2_us=80,   readout_error=0.02),
    "high":   dict(depolarizing_1q=0.01,   depolarizing_2q=0.05,
                   t1_us=30,  t2_us=20,   readout_error=0.10),
}


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate saved VQC model")
    parser.add_argument("--config", default="config/base_config.yaml")
    parser.add_argument("--no-noise", action="store_true")
    parser.add_argument("--noise-profile", choices=list(NOISE_PROFILES.keys()),
                        default=None, help="Override noise profile for evaluation")
    parser.add_argument("--model-path", default="saved_models/vqc_model.pkl")
    return parser.parse_args()


def evaluate(
    model_path: str = "saved_models/vqc_model.pkl",
    config: dict = None,
    use_noise: bool = True,
    noise_profile_override: str = None,
) -> dict:
    print(f"\n{'='*60}")
    print(f"  VQC Evaluation  |  model={model_path}")
    print("=" * 60)

    # ── Load model (no SimpleNamespace hack) ──────────────────────────────────
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model not found at {model_path}.\n"
            "Run `python train.py` first to train and save a model."
        )

    with open(model_path, "rb") as f:
        vqc = pickle.load(f)
    print(f"  Loaded VQC from {model_path}")

    # Load training metadata if available
    meta_path = "saved_models/metadata.pkl"
    if os.path.exists(meta_path):
        with open(meta_path, "rb") as f:
            meta = pickle.load(f)
        cv_acc = meta.get('cv_mean_accuracy', 'n/a')
        if isinstance(cv_acc, float):
            print(f"  Training metadata: n_qubits={meta.get('n_qubits')}, "
                  f"CV mean accuracy={cv_acc:.4f}")

    # ── Generate fresh test data ───────────────────────────────────────────────
    seed = config["experiment"]["seeds"][0] if config else 42
    n_features = config["data"]["n_features"] if config else 8
    snr = config["data"].get("signal_snr", 3.0) if config else 3.0

    X_test, y_test = generate_sensor_data(
        n_samples=400,
        n_features=n_features,
        snr=snr,
        seed=seed + 9999,   # different seed from training
    )
    print(f"  Test set: {len(X_test)} samples  |  "
          f"class balance: {np.bincount(y_test).tolist()}")

    # ── Noise config for evaluation ────────────────────────────────────────────
    shots = config["vqc"]["shots"] if config else 1024
    if not use_noise:
        sampler, pm = build_ideal_sampler_and_pm(shots=shots, seed=seed)
        print("  Evaluating under: IDEAL (no noise)")
    elif noise_profile_override:
        profile = NOISE_PROFILES[noise_profile_override]
        nm = build_noise_model(**profile)
        sampler, pm = build_noisy_sampler_and_pm(nm, shots=shots, seed=seed)
        print(f"  Evaluating under: '{noise_profile_override}' noise profile")
    elif config:
        nm = noise_model_from_config(config)
        sampler, pm = build_noisy_sampler_and_pm(nm, shots=shots, seed=seed)
        print(f"  Evaluating under: config noise (dep_2q={config['noise']['depolarizing_2q']})")
    else:
        sampler, pm = build_ideal_sampler_and_pm(shots=shots, seed=seed)

    # Swap sampler (and pass_manager if applicable) on the loaded VQC (best-effort)
    for attr in ("_sampler", "sampler"):
        if hasattr(vqc, attr):
            setattr(vqc, attr, sampler)
            break
    if pm is not None:
        for attr in ("_pass_manager", "pass_manager"):
            if hasattr(vqc, attr):
                setattr(vqc, attr, pm)
                break

    # ── Predict ───────────────────────────────────────────────────────────────
    y_pred = vqc.predict(X_test)
    try:
        y_proba = vqc.predict_proba(X_test)
    except Exception:
        y_proba = None

    # ── Metrics ───────────────────────────────────────────────────────────────
    metrics = compute_metrics(y_test, y_pred, y_proba)
    print_metrics(metrics, "VQC Evaluation")

    # ── Confusion matrix plot ──────────────────────────────────────────────────
    import matplotlib.pyplot as plt
    from evaluation.visualization import plot_confusion_matrix
    fig = plot_confusion_matrix(
        metrics["confusion_matrix"],
        class_names=["Normal", "Anomaly"],
        title="VQC Evaluation",
        output_dir="results",
    )
    plt.close(fig)

    # ── Save report ───────────────────────────────────────────────────────────
    import pandas as pd
    os.makedirs("results", exist_ok=True)
    report = {k: v for k, v in metrics.items() if k != "confusion_matrix"}
    pd.DataFrame([report]).to_csv("results/evaluation_report.csv", index=False)
    print("\n  Report saved to results/evaluation_report.csv")

    return metrics


if __name__ == "__main__":
    args = parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    evaluate(
        model_path=args.model_path,
        config=config,
        use_noise=not args.no_noise,
        noise_profile_override=args.noise_profile,
    )
